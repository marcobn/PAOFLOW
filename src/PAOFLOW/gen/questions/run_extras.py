"""Questions of the run properties added for the example cases (16 November 2026).

Each block is asked only when its entry of the property menu is chosen, so the earlier prompts of
the ``run`` workflow, and the files they generate, are unchanged:

==========================  ============================================================================
menu key                    ``main.py`` calls
==========================  ============================================================================
``site_projected_bands``    :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.site_projected_bands` after the band path
``jdos``                    :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.jdos` on the full k grid
``rashba_edelstein``        :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.rashba_edelstein` on the energy window
``weyl_points``             :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.write_Hamiltonian` and
                            :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.find_weyl_points`
``z2_invariant``            :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.z2_invariant`
``mirror_chern``            :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.mirror_chern_number`
``doped_transport``         :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.doped_transport` with the scattering models
``hamiltonian_changes``     ``modify_hamiltonian(p)`` after every ``pao_hamiltonian``: Hubbard shifts and
                            electric field (:meth:`~PAOFLOW.PAOFLOW.PAOFLOW.add_external_fields`), ad hoc
                            spin-orbit, Zeeman field, supercell doubling
``restart``                 :meth:`~PAOFLOW.PAOFLOW.PAOFLOW.restart_dump` after the gradients;
                            the next run loads it and starts from the adaptive smearing
==========================  ============================================================================

The spin Hall conductivity projected on atoms (``shc_proj``) is a GUI-only refinement of the
``spin_Hall`` property.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable

from .common import save_summary
from .model import Answers, Context, Note, Question

SITE_BANDS = 'Site-projected bands'
JDOS = 'Joint density of states'
TOPOLOGY_SEARCH = 'Topological invariants and Weyl points'
DOPING = 'Doped semiconductor'
HAMILTONIAN = 'Hamiltonian changes'

#: Menu entries appended to :data:`PAOFLOW.gen.paoflow_driver.PROPERTY_MENU`.
EXTRA_PROPERTY_MENU = [
    ('site_projected_bands', 'Site-projected bands (weight of chosen atoms along the band path)'),
    ('jdos', 'Joint density of states'),
    ('rashba_edelstein', 'Rashba-Edelstein response (current-induced spin density)'),
    ('weyl_points', 'Weyl points (band touchings; chirality with Z2Pack)'),
    ('z2_invariant', 'Z2 invariant (Wannier charge centres, Z2Pack)'),
    ('mirror_chern', 'Mirror Chern number of a 2D layer (Z2Pack)'),
    (
        'doped_transport',
        'Doped semiconductor: chemical potential, conductivity and Seebeck vs temperature',
    ),
    (
        'hamiltonian_changes',
        'Change the Hamiltonian first: Hubbard shifts, electric or Zeeman field, '
        'ad hoc spin-orbit, supercell doubling',
    ),
    ('restart', 'Restart point: keep the Hamiltonian and gradients for the next run'),
]

#: Built-in relaxation-time models of :func:`PAOFLOW.boltzmann.do_tau_models.builtin_tau_model`.
SCATTERING_CHANNELS = [
    ('acoustic', 'acoustic phonons (deformation potential)'),
    ('optical', 'optical phonons (deformation potential)'),
    ('polar_optical', 'polar optical phonons (Froehlich)'),
    ('polar_acoustic', 'piezoelectric acoustic phonons'),
    ('impurity', 'ionized impurities (Brooks-Herring)'),
]

#: Material parameters of the built-in models: key, prompt, default (GaAs, as in qe_examples
#: example10), unit and the channels that read it.
TAU_PARAMETERS = (
    ('ms', 'effective mass', 0.7, 'm_e', SCATTERING_CHANNELS),
    ('rho', 'mass density', 5.31e3, 'kg/m³', ('acoustic', 'optical', 'polar_acoustic')),
    ('v', 'sound velocity', 5.2e3, 'm/s', ('acoustic', 'polar_acoustic')),
    ('D_ac', 'acoustic deformation potential', 7.0, 'eV', ('acoustic',)),
    ('D_op', 'optical deformation potential', 3.0e10, 'eV/m', ('optical',)),
    ('hwlo', 'optical phonon energies', '0.03536', 'eV', ('optical', 'polar_optical')),
    ('eps_0', 'static dielectric constant', 13.5, '', ('polar_acoustic', 'polar_optical', 'impurity')),
    ('eps_inf', 'high-frequency dielectric constant', 11.6, '',
     ('polar_acoustic', 'polar_optical', 'impurity')),
    ('piezo', 'piezoelectric constant', 0.16, 'C/m²', ('polar_acoustic',)),
    ('nI', 'ionized impurity concentration', None, 'cm⁻³', ('impurity',)),
    ('Zi', 'impurity charge', 1.0, 'e', ('impurity',)),
)  # fmt: skip


def _properties(answers: Answers) -> list[str]:
    return answers.get('properties') or []


def _internal(answers: Answers) -> bool:
    return answers.get('projector', 'internal') != 'projwfc'


def _chosen(key: str) -> Callable[[Answers, Context], bool]:
    return lambda answers, context: key in _properties(answers)


# ---------------------------------------------------------------------------------- parsers
def hubbard_row(line: str) -> tuple[str, float]:
    """``'Al-3p 0.1'`` -> ``('Al-3p', 0.1)``: a manifold (species and shell, as ACBN0 names it) and U.

    Raises
    ------
    ValueError
        Unless a ``<species>-<n><l>`` label and a number are given.
    """
    parts = line.replace(',', ' ').split()
    if len(parts) != 2:
        raise ValueError('a row is the manifold (e.g. Al-3p) and U (eV)')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*-[0-9][spdfSPDF]', parts[0]):
        raise ValueError('a manifold is <species>-<shell>, e.g. Al-3p or Mn-3d: %r' % parts[0])
    try:
        return (parts[0], float(parts[1]))
    except ValueError:
        raise ValueError('U of %s is not a number' % parts[0]) from None


def soc_row(line: str) -> tuple[str, float, float]:
    """``'Pt 0.0 0.553'`` -> ``('Pt', 0.0, 0.553)``: species, lambda_p and lambda_d (eV).

    Raises
    ------
    ValueError
        Unless a species and two numbers are given.
    """
    parts = line.replace(',', ' ').split()
    if len(parts) != 3:
        raise ValueError('a row is the species, lambda_p and lambda_d (eV)')
    try:
        return (parts[0], float(parts[1]), float(parts[2]))
    except ValueError:
        raise ValueError('the strengths of %s are not numbers' % parts[0]) from None


def optional_vector(text: str) -> tuple[float, float, float] | None:
    """``'0 0 0.1'`` -> ``(0.0, 0.0, 0.1)``; blank (or all zeros) -> None.

    Raises
    ------
    ValueError
        Unless blank or three numbers.
    """
    parts = str(text).replace(',', ' ').split()
    if not parts:
        return None
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        raise ValueError('three numbers, or blank') from None
    if len(values) != 3:
        raise ValueError('three numbers, or blank')
    return values if any(values) else None


def int_triplet(text: str) -> tuple[int, int, int]:
    """``'1 0 0'`` -> ``(1, 0, 0)``.

    Raises
    ------
    ValueError
        Unless three non-negative integers.
    """
    parts = str(text).replace(',', ' ').split()
    try:
        values = tuple(int(part) for part in parts)
    except ValueError:
        raise ValueError('three integers') from None
    if len(values) != 3 or min(values) < 0:
        raise ValueError('three non-negative integers')
    return values


def float_tuple(text: str) -> tuple[float, ...]:
    """``'0, 0.5'`` -> ``(0.0, 0.5)``.

    Raises
    ------
    ValueError
        Unless one or more numbers.
    """
    parts = str(text).replace(',', ' ').split()
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        raise ValueError('numbers separated by commas or spaces') from None
    if not values:
        raise ValueError('at least one number')
    return values


def angle_pair(text: str) -> tuple[float, float]:
    """``'0 0'`` -> ``(0.0, 0.0)``: polar and azimuthal angle (radians).

    Raises
    ------
    ValueError
        Unless two numbers.
    """
    values = float_tuple(text)
    if len(values) != 2:
        raise ValueError('two angles (radians): theta and phi')
    return values


# --------------------------------------------------------------------------------- defaults
def _atoms(answers: Answers, context: Context) -> list[str]:
    """Species of every atom: from the save, else from the pw.x input of the folder."""
    summary = save_summary(answers, context)
    if summary:
        return list(summary['atoms'])

    def from_input() -> list[str]:
        from ..crystal_structure import structure_from_pw_input
        from ..paoflow_driver import detect_pw_input

        name = detect_pw_input(context.workdir)
        structure = structure_from_pw_input(os.path.join(context.workdir, name)) if name else None
        return list(structure['species']) if structure else []

    return context.cached('pw-input-atoms', from_input)


def _atom_choices(answers: Answers, context: Context) -> list[tuple[str, str]]:
    return [(str(i), '%d: %s' % (i, name)) for i, name in enumerate(_atoms(answers, context), 1)]


def _some_atom(value, answers: Answers, context: Context) -> str | None:
    """At least one atom, when the folder lists them (otherwise main.py projects on atom 1)."""
    return None if value or not _atoms(answers, context) else 'choose at least one atom'


def _energy_zero(summary: dict) -> float | None:
    """The PAOFLOW energy zero of the save: the Fermi level, or the top of the valence band."""
    for key in ('fermi_energy_ev', 'homo_ev'):
        if summary.get(key) is not None:
            return summary[key]
    return None


def _default_doping_emin(answers: Answers, context: Context) -> float:
    summary = save_summary(answers, context)
    if summary and summary.get('lowest_band_ev') is not None and _energy_zero(summary) is not None:
        return float(math.floor(summary['lowest_band_ev'] - _energy_zero(summary) - 1.0))
    return -20.0


def _default_doping_ne(answers: Answers, context: Context) -> int:
    """Energies 3 meV apart (a third of the 0.01 eV Gaussians of the carrier count)."""
    emin = answers.get('doping_emin', _default_doping_emin(answers, context))
    emax = answers.get('doping_emax', 4.0)
    return int(math.ceil((float(emax) - float(emin)) / 0.003)) + 1


def _default_planes(answers: Answers, context: Context) -> str:
    return '0' if answers.get('is_2d') else '0, 0.5'


def _channels(answers: Answers) -> list[str]:
    return answers.get('scattering') or []


def _tau_question(key: str, prompt: str, default, unit: str, channels) -> Question:
    names = [c[0] if isinstance(c, tuple) else c for c in channels]

    def when(answers: Answers, context: Context) -> bool:
        return 'doped_transport' in _properties(answers) and any(
            c in names for c in _channels(answers)
        )

    if key == 'hwlo':
        return Question(
            'tau_hwlo', '  %s (%s, comma separated)' % (prompt, unit), default=default,
            parse=float_tuple, when=when, unit=unit, section=DOPING,
            doc='PAOFLOW.boltzmann.do_tau_models.optical_model:params',
        )  # fmt: skip
    if key == 'nI':
        default = lambda a, c: abs(float(a.get('doping_conc', -1.0e18)))  # noqa: E731
    return Question(
        'tau_' + key, '  %s%s' % (prompt, ' (%s)' % unit if unit else ''), kind='float',
        default=default, when=when, unit=unit, section=DOPING,
    )  # fmt: skip


# -------------------------------------------------------------------------------- questions
RUN_EXTRA_ITEMS = (
    # ---- site-projected bands
    Question(
        'site_atoms',
        'Atoms to project the bands on (numbers of the save):',
        kind='menu',
        choices=_atom_choices,
        when=_chosen('site_projected_bands'),
        validate=_some_atom,
        label='Atoms',
        help='The weight of these atoms in every state of the band path (edge or surface atoms '
        'of a ribbon or a slab, one layer of a heterostructure).',
        doc='PAOFLOW.PAOFLOW.PAOFLOW.site_projected_bands:site_proj',
        section=SITE_BANDS,
    ),
    # ---- joint density of states
    Question(
        'jdos_emax', '  JDOS: highest photon energy', kind='float', default=10.0, unit='eV',
        when=_chosen('jdos'), section=JDOS, doc='PAOFLOW.PAOFLOW.PAOFLOW.jdos:emax',
    ),
    Question(
        'jdos_ne', '  JDOS: number of photon energies', kind='int', default=501,
        when=_chosen('jdos'), section=JDOS, doc='PAOFLOW.PAOFLOW.PAOFLOW.jdos:ne',
    ),
    Question(
        'jdos_delta', '  JDOS: broadening', kind='float', default=0.1, unit='eV',
        when=_chosen('jdos'), section=JDOS, doc='PAOFLOW.PAOFLOW.PAOFLOW.jdos:delta',
    ),
    # ---- Weyl points, Z2, mirror Chern
    Question(
        'weyl_grid', 'Weyl search: boxes along each reciprocal vector', default='8 8 8',
        parse=int_triplet, when=_chosen('weyl_points'), label='Search grid',
        section=TOPOLOGY_SEARCH, doc='PAOFLOW.PAOFLOW.PAOFLOW.find_weyl_points:search_grid',
    ),
    Question(
        'weyl_symmetrize', 'Weyl search: add the symmetry images of each point?', kind='bool',
        default=True, when=_chosen('weyl_points'), label='Symmetry images',
        section=TOPOLOGY_SEARCH, doc='PAOFLOW.PAOFLOW.PAOFLOW.find_weyl_points:symmetrize',
    ),
    Question(
        'z2_planes', 'Z2: planes k3 = const (crystal units)', default=_default_planes,
        parse=float_tuple, when=_chosen('z2_invariant'), label='Planes k3',
        help='0 for a layer with vacuum along c; 0 and 0.5 for a 3D crystal. The run needs '
        'spin-orbit coupling.',
        section=TOPOLOGY_SEARCH, doc='PAOFLOW.PAOFLOW.PAOFLOW.z2_invariant:k3_planes',
    ),
    Note(
        '  The mirror Chern number needs a fully relativistic save of a 2D layer with a '
        'horizontal mirror.',
        when=_chosen('mirror_chern'), section=TOPOLOGY_SEARCH,
    ),
    # ---- doped semiconductor
    Question(
        'doping_conc', 'Carrier concentration (cm^-3; negative: electrons, n type)',
        kind='float', default=-1.0e18, unit='cm⁻³', when=_chosen('doped_transport'),
        label='Carrier concentration', section=DOPING,
        doc='PAOFLOW.PAOFLOW.PAOFLOW.doped_transport:doping_conc',
    ),
    Question(
        'doping_tmin', '  lowest temperature', kind='float', default=300.0, unit='K',
        when=_chosen('doped_transport'), section=DOPING,
    ),
    Question(
        'doping_tmax', '  highest temperature', kind='float', default=600.0, unit='K',
        when=_chosen('doped_transport'), section=DOPING,
    ),
    Question(
        'doping_nt', '  number of temperatures', kind='int', default=7,
        when=_chosen('doped_transport'), section=DOPING,
    ),
    Question(
        'doping_emin', '  carrier count: lowest energy (every valence band)', kind='float',
        default=_default_doping_emin, unit='eV', when=_chosen('doped_transport'),
        help='Below the lowest band of the save (default: 1 eV under it).',
        section=DOPING, level='advanced', doc='PAOFLOW.PAOFLOW.PAOFLOW.doping:emin',
    ),
    Question(
        'doping_emax', '  carrier count: highest energy', kind='float', default=4.0, unit='eV',
        when=_chosen('doped_transport'), section=DOPING, level='advanced',
        doc='PAOFLOW.PAOFLOW.PAOFLOW.doping:emax',
    ),
    Question(
        'doping_ne', '  carrier count: number of energies', kind='int', default=_default_doping_ne,
        when=_chosen('doped_transport'), section=DOPING, level='advanced',
        doc='PAOFLOW.PAOFLOW.PAOFLOW.doping:ne',
    ),
    Question(
        'scattering', 'Scattering channels (none: constant relaxation time):', kind='menu',
        choices=SCATTERING_CHANNELS, when=_chosen('doped_transport'),
        label='Scattering channels', section=DOPING,
        doc='PAOFLOW.PAOFLOW.PAOFLOW.doped_transport:scattering_channels',
    ),
    *(_tau_question(*parameter) for parameter in TAU_PARAMETERS),
    # ---- Hamiltonian changes
    Question(
        'hubbard_shifts', 'Hubbard shifts: manifold (e.g. Al-3p) and U (eV)', kind='rows',
        default=[], parse=hubbard_row, row_prompt='  manifold and U (blank to end)',
        columns=('manifold', 'U (eV)'),
        when=lambda a, c: 'hamiltonian_changes' in _properties(a) and _internal(a),
        label='Hubbard shifts', help='-U/2 on every orbital of the manifold, resolved on the basis of '
        'each run (the standard and the extended basis shift the same shells).', section=HAMILTONIAN,
        doc='PAOFLOW.PAOFLOW.PAOFLOW.add_external_fields:HubbardU',
    ),
    Note(
        '  Hubbard shifts by manifold need the internal projections (projwfc.x has no shell labels).',
        when=lambda a, c: 'hamiltonian_changes' in _properties(a) and not _internal(a),
        section=HAMILTONIAN,
    ),
    Question(
        'efield', 'Uniform electric field Ex Ey Ez (V/A; blank: none)', default='',
        parse=optional_vector, when=_chosen('hamiltonian_changes'), unit='V/Å',
        label='Electric field', section=HAMILTONIAN,
        doc='PAOFLOW.PAOFLOW.PAOFLOW.add_external_fields:Efield',
    ),
    Question(
        'adhoc_soc', 'Ad hoc spin-orbit: species, lambda_p, lambda_d (eV)', kind='rows', default=[],
        parse=soc_row, row_prompt='  species and strengths (blank to end)',
        columns=('species', 'λ_p (eV)', 'λ_d (eV)'),
        when=lambda a, c: 'hamiltonian_changes' in _properties(a) and not a.get('spin_orbit'),
        label='Ad hoc spin-orbit', help='Spin-orbit coupling added to a scalar-relativistic '
        'Hamiltonian, shell by shell.', section=HAMILTONIAN,
        doc='PAOFLOW.PAOFLOW.PAOFLOW.adhoc_spin_orbit:soc_strengh',
    ),
    Question(
        'adhoc_soc_angles', '  spin quantization axis: theta phi (radians)', default='0 0',
        parse=angle_pair, when=lambda a, c: 'hamiltonian_changes' in _properties(a)
        and not a.get('spin_orbit') and bool(a.get('adhoc_soc')),
        label='Spin axis (θ, φ)', section=HAMILTONIAN, level='advanced',
        doc='PAOFLOW.PAOFLOW.PAOFLOW.adhoc_spin_orbit:theta',
    ),
    Question(
        'zeeman', 'Zeeman (exchange) field Bx By Bz (eV; blank: none)', default='',
        parse=optional_vector, when=_chosen('hamiltonian_changes'), unit='eV',
        label='Zeeman field', help='-B·S on every site; needs spinors (a fully relativistic save '
        'or ad hoc spin-orbit).', section=HAMILTONIAN,
        doc='PAOFLOW.PAOFLOW.PAOFLOW.add_external_fields:Bfield',
    ),
    Question(
        'doubling', 'Supercell doubling nx ny nz (2**n cells along each vector)', default='0 0 0',
        parse=int_triplet, when=_chosen('hamiltonian_changes'), label='Doubling',
        section=HAMILTONIAN, doc='PAOFLOW.PAOFLOW.PAOFLOW.doubling_Hamiltonian:nx',
    ),
    Note(
        '  Restart: the first run writes paoflow_dump_<rank>.json after the gradients; the next '
        'run, with the same number of MPI ranks, starts from them (delete them to rebuild).',
        when=_chosen('restart'), section=HAMILTONIAN,
    ),
)  # fmt: skip


def top_valence_band(cfg: dict) -> int:
    """Index (from 0) of the highest occupied band of the save.

    Parameters
    ----------
    cfg : dict
        Run configuration with ``workdir``, ``savedir`` and ``spin_orbit``.

    Returns
    -------
    int
        ``nelec - 1`` with spin-orbit (one electron per band), ``nelec / 2 - 1`` otherwise;
        0 without a readable save.

    Notes
    -----
    With Hubbard shifts :meth:`PAOFLOW.PAOFLOW.PAOFLOW.pao_eigh` puts the top of band ``bval`` at
    zero; its default (0, the lowest band) moves an insulator's spectrum by the width of that band.
    """
    from ..system_summary import read_save_summary

    summary = read_save_summary(os.path.join(cfg.get('workdir', '.'), cfg.get('savedir', '')))
    if not summary or not summary.get('nelec'):
        return 0
    electrons = int(round(summary['nelec']))
    return max(electrons - 1 if cfg.get('spin_orbit') else electrons // 2 - 1, 0)


def assemble_run_extras(cfg: dict, answers: Answers) -> dict:
    """Add the keys of the chosen extra properties to ``cfg``.

    Parameters
    ----------
    cfg : dict
        Run configuration (``properties`` set); modified in place.
    answers : dict
        Answers of :data:`RUN_EXTRA_ITEMS`.

    Returns
    -------
    dict
        ``cfg``.
    """
    props = cfg['properties']
    if 'site_projected_bands' in props:
        cfg['site_atoms'] = [int(atom) - 1 for atom in answers['site_atoms']] or [0]
    if 'jdos' in props:
        for key in ('jdos_emax', 'jdos_ne', 'jdos_delta'):
            cfg[key] = answers[key]
    if 'weyl_points' in props:
        cfg['weyl_grid'] = tuple(answers['weyl_grid'])
        cfg['weyl_symmetrize'] = answers['weyl_symmetrize']
    if 'z2_invariant' in props:
        cfg['z2_planes'] = tuple(answers['z2_planes'])
    if 'doped_transport' in props:
        for key in ('doping_conc', 'doping_tmin', 'doping_tmax', 'doping_nt', 'doping_emin',
                    'doping_emax', 'doping_ne'):  # fmt: skip
            cfg[key] = answers[key]
        cfg['scattering'] = list(answers.get('scattering') or [])
        tau = {}
        for key, _prompt, _default, _unit, channels in TAU_PARAMETERS:
            names = [c[0] if isinstance(c, tuple) else c for c in channels]
            if any(channel in names for channel in cfg['scattering']):
                value = answers['tau_' + key]
                tau[key] = list(value) if key == 'hwlo' else value
        cfg['tau_parameters'] = tau
    if 'hamiltonian_changes' in props:
        rows = (answers.get('hubbard_shifts') or []) if _internal(cfg) else []
        cfg['hubbard_shifts'] = [tuple(row) for row in rows]
        cfg['efield'] = answers.get('efield')
        cfg['zeeman'] = answers.get('zeeman')
        rows = [] if cfg.get('spin_orbit') else answers.get('adhoc_soc') or []
        cfg['adhoc_soc'] = [tuple(row) for row in rows]
        cfg['adhoc_soc_angles'] = tuple(answers.get('adhoc_soc_angles') or (0.0, 0.0))
        cfg['doubling'] = tuple(answers['doubling'])
        cfg['top_valence_band'] = top_valence_band(cfg)
    if 'spin_Hall' in props and answers.get('shc_atoms'):
        cfg['shc_atoms'] = [int(atom) - 1 for atom in answers['shc_atoms']]
    return cfg


#: Spin Hall conductivity of chosen atoms (GUI-only refinement of the ``spin_Hall`` property).
SHC_ATOMS = Question(
    'shc_atoms',
    'Project the spin Hall conductivity on atoms (none: the whole cell):',
    kind='menu',
    choices=_atom_choices,
    when=_chosen('spin_Hall'),
    label='Spin Hall: atoms',
    help='The contribution of these atoms (one layer of a slab, say); none gives the total.',
    doc='PAOFLOW.PAOFLOW.PAOFLOW.spin_Hall:shc_proj',
    section='Grids and run settings',
    level='advanced',
    cli=False,
)
