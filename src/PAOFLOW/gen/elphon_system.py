"""``paoflow-gen elphon``: write an electron-phonon system folder from a ``pw.x`` input.

The folder has the layout of ``examples/elphon/<System>``: ``system.py`` (the only file with
values, an :class:`~PAOFLOW.elphon.workflows.ElphonSystem` plus the per-workflow defaults),
``qe/`` (scf, ``ph.x``, nscf and, optionally, optics-nscf inputs with ``run_qe.sh``) and one
subfolder per selected workflow (``phonons/``, ``coupling/``, ``superconductivity/``,
``optics/``, ``transport/``, ``arpes/``) holding copies of the drivers of
``PAOFLOW/gen/elphon_templates`` (the same files as the examples).
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Sequence

import numpy as np

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elphon_templates')

#: Driver files of every workflow folder (relative to the template directory).
WORKFLOWS: dict[str, tuple[str, ...]] = {
    'phonons': ('phonons/phonons.py',),
    'coupling': ('coupling/quadrupoles.py', 'coupling/long_range_check.py'),
    'superconductivity': (
        'superconductivity/eliashberg.py',
        'superconductivity/me.py',
        'superconductivity/me_aniso.py',
        'superconductivity/plot_me.py',
    ),
    'optics': (
        'optics/absorption.py',
        'optics/emissivity.py',
        'optics/dielectric.py',
        'optics/plot_absorption.py',
    ),
    'optics-metal': ('optics/dielectric.py',),
    'transport': ('transport/mobility.py', 'transport/plot_mobility.py'),
    'transport-metal': ('transport/resistivity.py', 'transport/plot_resistivity.py'),
    'arpes': ('arpes/arpes.py',),
}
QE_HELPERS = ('qe/kmesh.py',)

WRITE_UKK_TEMPLATE = '''#!/usr/bin/env python3
"""Write {ukk} for epw.x with wannierize = .false. (run in this directory before epw.x).

EPW reads the band bookkeeping from this file only: the {nbnd} nscf bands, {exclusion}
No Wannier functions are built; the identity rotations only feed
EPW's Wannier stage, which PAOFLOW does not use.  The empty {stem}.bvec /
{stem}.mmn written alongside let that stage finish.
"""

from PAOFLOW.gen.epw_inputs import write_placeholder_ukk

write_placeholder_ukk('{ukk}', nbnd={nbnd}, nk_total={nk_total}, exclude_bands={excluded}, nelec={nelec})
print('Wrote {ukk} (and the empty wannier90 stubs {stem}.bvec, {stem}.mmn)')
'''


# ----------------------------------------------------------------------------- rendering
def _value(value: Any) -> str:
    """``repr`` of a configuration value (tuples for grids, lists for sequences)."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, float) and value == float('inf'):
        return "float('inf')"
    return repr(value)


def _format_dict(name: str, values: dict[str, Any], comments: dict[str, str] | None = None) -> str:
    """``NAME = dict(\\n    key=value,  # comment\\n)`` with one entry per line."""
    comments = comments or {}
    if not values:
        return '%s = {}\n' % name
    lines = ['%s = dict(' % name]
    for key, value in values.items():
        if isinstance(value, dict):
            inner = ', '.join('%s=%s' % (k, _value(v)) for k, v in value.items())
            entry = '    %s=dict(%s),' % (key, inner)
        else:
            entry = '    %s=%s,' % (key, _value(value))
        comment = comments.get(key)
        lines.append(entry + ('  # %s' % comment if comment else ''))
    lines.append(')')
    return '\n'.join(lines) + '\n'


SYSTEM_COMMENTS = {
    'coupling_source': "'dfpt': PAOFLOW's vertex from ph.x dvscf + the nscf save; 'epw': .epb files",
    'qe_dir': 'qe/phonon (scf, ph.x), qe/nscf (<prefix>.save), qe/optics',
    'coarse_grid': 'nscf k-grid',
    'qgrid': 'ph.x q-grid (divides the k-grid)',
    'masses_amu': 'one per species, ATOMIC_SPECIES order',
    'nelec': 'valence electrons',
    'basis_dir': 'paoflow-genbasis-ps tables',
    'pthr': 'projectability threshold',
    'long_range': 'dipole (Fröhlich) + quadrupole model when polar or quadrupoles is set',
    'quadrupoles': "'coupling/quadrupole.fmt' from coupling/quadrupoles.py",
    'cache_dir': 'coarse vertex g(R_e, R_p), shared by every workflow',
    'spin_degeneracy': '2 without spin-orbit coupling, 1 with it (None: from the save)',
}
SUPERCONDUCTIVITY_COMMENTS = {
    'nk': 'dense electron grid per axis (EPW nkf)',
    'nq': 'dense phonon grid per axis (EPW nqf); nk % nq == 0',
    'sigma_ev': 'Fermi-surface smearing (EPW degaussw)',
    'adaptive_smearing': 'Yates prefactor of the adaptive widths (None: fixed smearing)',
    'mu_star': 'Coulomb pseudopotential',
    'transport': 'also alpha^2F_tr (transport/resistivity.py)',
    'fs_coupling': 'Fermi-surface pair coupling for me_aniso.py (needs nk == nq)',
    'fsthick_ev': 'Fermi window of the anisotropic states (EPW fsthick)',
    'me_temps': 'temperatures (K) of the isotropic Migdal-Eliashberg equations',
    'me_linear_temps': '(Tmin, Tmax, nstemp) of the linearised kernel',
    'me_aniso_temps': 'temperatures (K) of the anisotropic equations',
    'wscut': 'Matsubara cutoff (eV, EPW wscut)',
    'wscut_aniso': 'Matsubara cutoff of the anisotropic equations (eV)',
    'degaussq_mev': 'phonon smearing of alpha^2F for the equations (meV, EPW degaussq)',
    'degaussq_linear_mev': '... for the linearised kernel (meV)',
    'npade': 'percentage of Matsubara points of the Padé approximant',
    'real_axis_max_mev': 'upper limit of the real-axis panels of plot_me.py',
}
OPTICS_COMMENTS = {
    'nk': 'dense electron grid per axis (EPW nkf)',
    'nq': 'dense phonon grid per axis (EPW nqf); nk % nq == 0',
    'omega_ev': 'photon energies (min, max, step)',
    'temps_k': 'temperatures (K)',
    'degauss_ev': 'delta smearing (EPW degaussw); the floor with adaptive_smearing',
    'adaptive_smearing': 'Yates prefactor of the adaptive widths (None: fixed smearing)',
    'fsthick_ev': 'states within this window of the Fermi level (EPW fsthick)',
    'refractive_index': 'constant n_r of alpha (None: n(omega) of the direct run, else 3.4)',
    'position_dipoles': 'position dipoles of the PAO orbitals in the velocity',
    'target_gap_ev': 'scissor target (eV); None: the Hamiltonian gap',
    'gap_reference_save': 'QE save of an eACBN0 run whose PAO gap is the target',
    'varshni': '(alpha eV/K, beta K) of a temperature-dependent gap',
    'electrons_save': 'QE save of an eACBN0 nscf in the same basis (eACBN0 electrons)',
    'screening_eps': 'target dielectric constant of the Poncé screening correction',
    'emissivity': 'also the thermal emissivity of a slab',
    'thickness_um': 'slab thickness (um)',
    'angles': 'emission angles (deg)',
    'direct_dielectric_dir': 'output of optics/dielectric.py (extended basis)',
    'kk_omega_max_ev': 'Kramers-Kronig range of the direct term without an external eps',
    'dielectric': 'optics/dielectric.py: interpolated grid, window and broadening',
}
TRANSPORT_COMMENTS = {
    'nk': 'dense electron grid per axis (mobilities need >= 48)',
    'nq': 'dense phonon grid per axis; nk % nq == 0',
    'temps_k': 'temperatures (K)',
    'doping_cm3': 'net doping (cm^-3; > 0 n type, < 0 p type, 0 intrinsic)',
    'window_ev': 'states within this window of the band edges (EPW fsthick)',
    'smearing_ev': 'delta smearing (EPW degaussw); the floor with adaptive_smearing',
    'adaptive_smearing': 'Yates prefactor of the adaptive widths (None: fixed smearing)',
    'target_gap_ev': 'scissor of the carrier statistics (eV)',
    'varshni': '(alpha eV/K, beta K) of a temperature-dependent gap',
    'electrons_save': 'QE save of an eACBN0 nscf in the same basis (eACBN0 electrons)',
    'screening_eps': 'target dielectric constant of the Poncé screening correction',
    'rho0_uohm_cm': 'residual resistivity (micro-ohm cm)',
    'b_ee_per_ev': 'Fermi-liquid coefficient b (1/eV)',
    'omega_ev': 'photon energies (min, max, step) of the metal optics',
    'thickness_um': 'film thickness (um; inf = opaque)',
    'angles': 'emission angles (deg)',
    'interband_dielectric_dir': 'output of optics/dielectric.py --interband-only',
    'extended_drude': 'frequency-dependent rate + mass enhancement (False: DC rate)',
}
ARPES_COMMENTS = {
    'photon_ev': 'photon energy (He I: 21.2)',
    'work_function_ev': 'work function (eV)',
    'polarization': 'light polarisation (Cartesian)',
    'direction': 'end point of the line from Gamma (crystal coordinates)',
    'direction_label': 'its name',
    'window': '(E_min, E_max, n) of the energy window (eV, relative to E_F)',
    'resolution': '(energy, momentum) resolution (eV, 1/A)',
    'self_energy': "any of 'isotropic', 'anisotropic', 'fan-migdal'",
    'temps_k': 'temperatures (K)',
    'segment_ev': 'segment of +- this energy across the Fermi crossing (None: the full line)',
    'npts': 'points of the path',
    'nq': 'dense q-grid of the Fan-Migdal run',
    'eta_ev': 'Lorentzian width / floor (eV) of the Fan-Migdal run',
    'adaptive_smearing': 'adaptive widths a |v| dq of the Fan-Migdal run (metals)',
    'intensity_power': 'power scale of the band maps (weak bands next to strong ones)',
}
PHONONS_COMMENTS = {
    'units': "'cm-1' or 'THz'",
    'mesh': 'q-mesh of the DOS',
    'sigma': 'DOS broadening',
}
COUPLING_COMMENTS = {
    'nk': 'dense grid of the path points of long_range_check.py',
    'direction': 'Cartesian direction of q (2 pi / alat)',
}


def render_system_py(cfg: dict[str, Any]) -> str:
    """Text of ``system.py``: ``NAME``, ``SYSTEM`` and the per-workflow default dictionaries.

    Parameters
    ----------
    cfg : dict
        ``name``, ``prefix``, ``coupling_source``, ``kgrid``, ``qgrid``, ``masses_amu``,
        ``nelec``, ``basisdir``, ``configuration``, ``pthr``, ``long_range``,
        ``quadrupoles`` (path or ``None``), ``cache_dir``, ``spin_degeneracy`` and the
        optional dictionaries ``superconductivity``, ``optics``, ``transport``, ``arpes``,
        ``phonons``, ``coupling`` (missing ones are written empty).

    Returns
    -------
    str
        The module text.
    """
    system = dict(
        coupling_source=cfg.get('coupling_source', 'dfpt'),
        qe_dir='qe',
        coarse_grid=tuple(int(n) for n in cfg['kgrid']),
        qgrid=tuple(int(n) for n in cfg['qgrid']),
        masses_amu=[float(m) for m in cfg['masses_amu']],
        nelec=cfg['nelec'],
        basis_dir=cfg.get('basisdir', 'BASIS_PS'),
        configuration=cfg.get('configuration', 'standard'),
        pthr=float(cfg.get('pthr', 0.95)),
        long_range=bool(cfg.get('long_range', True)),
        quadrupoles=cfg.get('quadrupoles'),
        cache_dir=cfg.get('cache_dir', 'cache'),
        spin_degeneracy=cfg.get('spin_degeneracy'),
    )
    lines = [
        '#!/usr/bin/env python3',
        '"""%s: configuration of the electron-phonon workflows (one folder per property).'
        % cfg['name'],
        '',
        'Edit the values here; every driver of the workflow folders reads them through',
        '``PAOFLOW.elphon.workflows.load_system``.  Paths are relative to this directory.',
        '"""',
        '',
        'import os',
        '',
        'from PAOFLOW.elphon.workflows import ElphonSystem',
        '',
        'HERE = os.path.dirname(os.path.abspath(__file__))',
        'NAME = %r' % cfg['name'],
        '',
        'SYSTEM = ElphonSystem(',
        '    prefix=%r,' % cfg['prefix'],
        '    root=HERE,',
    ]
    for key, value in system.items():
        comment = SYSTEM_COMMENTS.get(key)
        lines.append('    %s=%s,%s' % (key, _value(value), '  # %s' % comment if comment else ''))
    lines += [')', '', '# Defaults of the workflow drivers (command-line options override them).']
    for name, comments in (
        ('SUPERCONDUCTIVITY', SUPERCONDUCTIVITY_COMMENTS),
        ('OPTICS', OPTICS_COMMENTS),
        ('TRANSPORT', TRANSPORT_COMMENTS),
        ('ARPES', ARPES_COMMENTS),
        ('PHONONS', PHONONS_COMMENTS),
        ('COUPLING', COUPLING_COMMENTS),
    ):
        lines.append(_format_dict(name, cfg.get(name.lower(), {}) or {}, comments).rstrip('\n'))
    return '\n'.join(lines) + '\n'


def render_run_qe(cfg: dict[str, Any]) -> str:
    """Text of ``qe/run_qe.sh``: scf, ``ph.x``, the nscf (and the EPW / optics steps).

    Parameters
    ----------
    cfg : dict
        ``prefix``, ``coupling_source``, ``optics_nscf`` (bool), ``metal`` (bool).

    Returns
    -------
    str
        The shell script (``NP`` processes, run from ``qe/``).
    """
    prefix = cfg['prefix']
    epw = cfg.get('coupling_source', 'dfpt') == 'epw'
    lines = [
        '#!/bin/bash',
        '# QE steps of the electron-phonon workflows (run from this directory): NP=8 bash run_qe.sh',
        'set -e',
        'NP=${NP:-8}',
        'PW=${PW:-pw.x}; PH=${PH:-ph.x}',
        'log() { echo "$(date +%T) $*"; }',
        '',
        'cd phonon',
        'log scf;  mpirun -np $NP $PW -in scf.in > scf.out',
        'log ph.x; mpirun -np $NP $PH -in ph.in > ph.out        # irreducible q, fildvscf -> _ph0/',
    ]
    if epw:
        lines.append(
            'log pp.py; python3 ${EPW_PP:-pp.py}                 # <q-e>/EPW/bin/pp.py: collects '
            'dvscf, patterns, dyn into save/'
        )
    lines += [
        'cd ../nscf',
        'mkdir -p %s.save' % prefix,
        'cp ../phonon/%s.save/charge-density.dat ../phonon/%s.save/data-file-schema.xml %s.save/'
        % (prefix, prefix, prefix),
        'log nscf; mpirun -np $NP $PW -in nscf.in > nscf.out',
    ]
    if epw:
        lines += [
            'log ukk;  python3 write_ukk.py                         # %s.ukk for wannierize = .false.'
            % prefix,
            'log epw;  mpirun -np $NP ${EPW:-epw.x} -nk $NP -in epw.in > epw.out   # writes %s.epb*'
            % prefix,
        ]
    if cfg.get('optics_nscf'):
        lines += [
            'cd ../optics',
            'mkdir -p %s.save' % prefix,
            'cp ../phonon/%s.save/charge-density.dat ../phonon/%s.save/data-file-schema.xml '
            '%s.save/' % (prefix, prefix, prefix),
            'log optics nscf; mpirun -np $NP $PW -nk $NP -in nscf.in > nscf.out',
        ]
    lines += ['cd ..', 'log done']
    return '\n'.join(lines) + '\n'


def write_ukk_script(cfg: dict[str, Any]) -> str:
    """Text of ``qe/nscf/write_ukk.py`` (placeholder ``<prefix>.ukk`` for ``wannierize = .false.``)."""
    from .epw_inputs import exclude_bands_string

    excluded = [int(b) for b in cfg.get('exclude_bands', [])]
    if excluded:
        exclusion = (
            'of which\nbands %s are excluded from the electron-phonon calculation, matching\n'
            '``bands_skipped`` in epw.in.' % exclude_bands_string(excluded).split('= ')[1]
        )
    else:
        exclusion = 'all of them kept in the\nelectron-phonon calculation.'
    return WRITE_UKK_TEMPLATE.format(
        ukk=cfg['prefix'] + '.ukk',
        stem=cfg['prefix'],
        nbnd=int(cfg['nbnd']),
        exclusion=exclusion,
        nk_total=' * '.join(str(int(n)) for n in cfg['kgrid']),
        excluded=repr(excluded),
        nelec=int(cfg['nelec']),
    )


def qe_inputs(cfg: dict[str, Any]) -> dict[str, str]:
    """QE (and EPW) inputs of the system folder, ``{relative path: text}``.

    Parameters
    ----------
    cfg : dict
        ``prefix``, ``kgrid``, ``qgrid``, ``nbnd``, ``nelec``, ``masses_amu``,
        ``pw_template_text`` (a ``pw.x`` input; without it only ``qe/nscf/nscf.kpoints`` is
        written), ``pseudo_dir``, ``coupling_source``, ``irreducible`` (nscf on the
        irreducible wedge), ``metal`` (no ``epsil``), ``exclude_bands``, ``optics_nscf``,
        ``optics_kgrid``, ``optics_nbnd``.

    Returns
    -------
    dict
        ``qe/phonon/scf.in``, ``qe/phonon/ph.in``, ``qe/nscf/nscf.in`` (or ``nscf.kpoints``),
        ``qe/run_qe.sh`` and, for the EPW source, ``qe/nscf/epw.in`` and
        ``qe/nscf/write_ukk.py``; with ``optics_nscf`` also ``qe/optics/nscf.in``.
    """
    from .epw_inputs import (
        automatic_kpoints_card,
        epw_input,
        kpoints_card,
        nscf_input,
        optics_nscf_input,
        ph_input,
        scf_input,
    )

    prefix = cfg['prefix']
    kgrid, qgrid = tuple(cfg['kgrid']), tuple(cfg['qgrid'])
    epw = cfg.get('coupling_source', 'dfpt') == 'epw'
    explicit = epw or not cfg.get('irreducible', True)
    template = cfg.get('pw_template_text')
    pseudo_dir = cfg.get('pseudo_dir') or None
    files = {}
    if template:
        files['qe/phonon/scf.in'] = scf_input(template, prefix, pseudo_dir)
        files['qe/nscf/nscf.in'] = nscf_input(
            template, prefix, kgrid, int(cfg['nbnd']), pseudo_dir, explicit=explicit
        )
        if cfg.get('optics_nscf'):
            files['qe/optics/nscf.in'] = optics_nscf_input(
                template, prefix, tuple(cfg.get('optics_kgrid', (8, 8, 8))),
                int(cfg.get('optics_nbnd', 60)), pseudo_dir,
            )  # fmt: skip
    else:
        files['qe/nscf/nscf.kpoints'] = (
            kpoints_card(kgrid) if explicit else automatic_kpoints_card(kgrid)
        )
    files['qe/phonon/ph.in'] = ph_input(
        prefix, qgrid, epsil=not cfg.get('metal', False), tr2_ph=cfg.get('tr2_ph', 1.0e-16)
    )
    if epw:
        excluded = [int(b) for b in cfg.get('exclude_bands', [])]
        files['qe/nscf/epw.in'] = epw_input(
            prefix, cfg['masses_amu'], kgrid, qgrid, int(cfg['nbnd']) - len(excluded),
            dvscf_dir='../phonon/save', exclude_bands=excluded,
        )  # fmt: skip
        files['qe/nscf/write_ukk.py'] = write_ukk_script(cfg)
    files['qe/run_qe.sh'] = render_run_qe(cfg)
    return files


def workflow_files(cfg: dict[str, Any]) -> list[str]:
    """Template files (relative paths) of the workflows selected in ``cfg['workflows']``."""
    files = list(QE_HELPERS)
    for name in cfg.get('workflows', ()):
        key = name
        if name in ('transport', 'optics') and cfg.get('metal'):
            key = name + '-metal'  # resistivity instead of mobility; the interband eps only
        files += list(WORKFLOWS[key])
    return files


def write_system_folder(cfg: dict[str, Any], out: str, force: bool = False) -> list[str]:
    """Write ``system.py``, the QE inputs and the workflow drivers of a system folder.

    Parameters
    ----------
    cfg : dict
        The generator configuration (:func:`render_system_py`, :func:`qe_inputs`,
        ``workflows``, ``metal``).
    out : str
        The system folder (created).
    force : bool, optional
        Overwrite existing files (otherwise they are skipped with a message).

    Returns
    -------
    list of str
        Written files.
    """
    out = os.path.abspath(out)
    texts = {'system.py': render_system_py(cfg)}
    texts.update(qe_inputs(cfg))
    written = []
    for rel, text in texts.items():
        path = os.path.join(out, rel)
        if os.path.exists(path) and not force:
            print('  %s exists (use --force to overwrite)' % path)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        if rel.endswith('.sh'):
            os.chmod(path, 0o755)
        written.append(path)
    for rel in workflow_files(cfg):
        path = os.path.join(out, rel)
        if os.path.exists(path) and not force:
            print('  %s exists (use --force to overwrite)' % path)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(os.path.join(TEMPLATE_DIR, rel), path)
        written.append(path)
    for rel in ('cache', 'figures'):
        os.makedirs(os.path.join(out, rel), exist_ok=True)
    return written


# ------------------------------------------------------------------------------ prompts
def _floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(',', ' ').split() if x.strip()]


def ask_divisible(ask_int, prompt: str, default: int, divisor: int, what: str) -> int:
    """Ask for an integer that is a positive multiple of ``divisor`` (re-prompting otherwise)."""
    while True:
        value = ask_int(prompt, default)
        if value > 0 and value % divisor == 0:
            return value
        print('  %s must be a positive multiple of %d.' % (what, divisor))


def ask_workflows(ask, choices: Sequence[str], default: Sequence[str]) -> list[str]:
    """Comma-separated multi-select among ``choices`` (re-prompts on an unknown name)."""
    while True:
        answer = ask('Workflows (comma-separated: %s)' % ', '.join(choices), ', '.join(default))
        names = [x.strip() for x in answer.replace(',', ' ').split() if x.strip()]
        unknown = [n for n in names if n not in choices]
        if not unknown:
            return list(dict.fromkeys(names))
        print('  unknown workflow(s): %s' % ', '.join(unknown))


def collect_elphon_system(args, workdir: str) -> dict[str, Any]:
    """Prompt for the electron-phonon system configuration.

    Parameters
    ----------
    args : argparse.Namespace
        ``prefix`` and ``out`` overrides of ``paoflow-gen``.
    workdir : str
        Directory scanned for a ``pw.x`` input.

    Returns
    -------
    dict
        The configuration of :func:`write_system_folder`.
    """
    from . import paoflow_driver as pd
    from .epw_inputs import parse_exclude_bands, species_masses

    cfg: dict[str, Any] = {'workdir': workdir}
    template = pd.ask(
        'pw.x input of the system (gives qe/phonon/scf.in and the nscf inputs; blank = none)',
        pd.detect_pw_input(workdir) or '',
    )
    cfg['pw_template'] = template
    cfg['pw_template_text'] = None
    template_pseudo_dir = None
    if template:
        path = os.path.join(workdir, os.path.expanduser(template))
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                cfg['pw_template_text'] = handle.read()
        except OSError:
            print('  (%s not readable; the scf and nscf inputs are not written)' % path)
        else:
            template_pseudo_dir = pd._read_namelist_value(cfg['pw_template_text'], 'pseudo_dir')
            if template_pseudo_dir is not None:
                template_pseudo_dir = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(path)), template_pseudo_dir)
                )
    default_prefix = (
        pd._read_namelist_value(cfg['pw_template_text'], 'prefix')
        if cfg['pw_template_text']
        else None
    )
    cfg['prefix'] = args.prefix or pd.ask('Prefix', (default_prefix or 'pwscf').strip('\'"'))
    cfg['name'] = pd.ask('System name (titles of the figures)', cfg['prefix'])
    out = os.path.abspath(os.path.expanduser(args.out or workdir))
    cfg['out'] = out
    if template_pseudo_dir is None:
        template_pseudo_dir = workdir
    # qe/phonon, qe/nscf and qe/optics are siblings, so one relative path serves all three.
    default_pseudo = os.path.relpath(template_pseudo_dir, os.path.join(out, 'qe', 'phonon'))
    cfg['pseudo_dir'] = pd.ask(
        'pseudo_dir of the QE run directories', default_pseudo.rstrip('/') + '/'
    )
    source = pd.ask_choice(
        "Coupling source (dfpt = PAOFLOW's vertex from ph.x dvscf, norm-conserving; "
        'epw = EPW epbwrite, any pseudopotential)',
        ['dfpt', 'epw'],
        'dfpt',
    )
    cfg['coupling_source'] = source
    kg = pd.ask_int('Coarse k-grid per axis (nscf)', 6)
    while True:
        qg = pd.ask_int('Coarse q-grid per axis (ph.x; must divide the k-grid)', kg)
        if qg > 0 and kg % qg == 0:
            break
        print('  The q-grid must divide the k-grid (%d).' % kg)
    cfg['kgrid'], cfg['qgrid'] = [kg] * 3, [qg] * 3
    cfg['irreducible'] = True
    if source == 'dfpt':
        cfg['irreducible'] = pd.ask_yes_no(
            'nscf on the irreducible wedge (K_POINTS automatic, unfolded by symmetry)? '
            '(No = explicit full list)',
            True,
        )
    else:
        print('  (EPW needs the explicit full k list: K_POINTS crystal)')
        cfg['irreducible'] = False
    cfg['nbnd'] = pd.ask_int('Bands in the nscf run (nbnd, > number of PAO orbitals)', 0)
    cfg['exclude_bands'] = []
    if source == 'epw':
        while True:
            try:
                cfg['exclude_bands'] = parse_exclude_bands(
                    pd.ask(
                        'Bands to exclude from the coupling (EPW exclude_bands, e.g. 1:5; blank = none)',
                        '',
                    )
                )
                break
            except ValueError as err:
                print('  %s' % err)
    default_masses = species_masses(cfg['pw_template_text'] or '')
    masses = pd.ask(
        'Atomic mass of each species (amu), comma-separated',
        ', '.join('%g' % m for m in default_masses),
    )
    cfg['masses_amu'] = _floats(masses)
    cfg['nelec'] = pd.ask_int('Valence electrons (nelec)', 0)
    cfg['metal'] = pd.ask_yes_no('Is the system a metal?', False)
    cfg['basisdir'] = pd.ask('Basis directory name (paoflow-genbasis-ps)', 'BASIS_PS')
    cfg['configuration'] = 'standard'
    cfg['pthr'] = pd.ask_float('Projectability threshold (pthr)', 0.95)
    cfg['spin_degeneracy'] = None
    cfg['long_range'] = True
    cfg['quadrupoles'] = None
    cfg['cache_dir'] = 'cache'
    if not cfg['metal']:
        print(
            '  (ph.in gets epsil = .true.: the dielectric tensor and Born charges of the '
            'long-range model; quadrupoles come from coupling/quadrupoles.py)'
        )
    default = (
        ['superconductivity', 'transport', 'arpes']
        if cfg['metal']
        else ['optics', 'transport', 'arpes']
    )
    choices = ['phonons', 'coupling', 'superconductivity', 'optics', 'transport', 'arpes']
    cfg['workflows'] = ask_workflows(pd.ask, choices, default)
    cfg['optics_nscf'] = False
    if 'superconductivity' in cfg['workflows']:
        _collect_superconductivity(cfg, pd, kg)
    if 'optics' in cfg['workflows']:
        _collect_optics(cfg, pd, kg)
    if 'transport' in cfg['workflows']:
        _collect_transport(cfg, pd, kg)
    if 'arpes' in cfg['workflows']:
        _collect_arpes(cfg, pd, kg)
    if 'phonons' in cfg['workflows']:
        cfg['phonons'] = dict(units='cm-1', mesh=(24, 24, 24), sigma=3.0)
    if 'coupling' in cfg['workflows']:
        cfg['coupling'] = dict(nk=24, direction=(1.0, 0.0, 0.0))
    if cfg['optics_nscf']:
        cfg['optics_kgrid'] = [pd.ask_int('Optics nscf: Monkhorst-Pack grid per axis', 8)] * 3
        cfg['optics_nbnd'] = pd.ask_int(
            'Optics nscf: bands (more than the orbitals of the extended PAO basis)', 60
        )
    if cfg['nbnd'] <= 0 or not cfg['masses_amu'] or cfg['nelec'] <= 0:
        print(
            '\nNote: nbnd, masses_amu and nelec are system-specific; edit system.py and qe/ before running.'
        )
    return cfg


def _collect_superconductivity(cfg, pd, kg: int) -> None:
    nq = pd.ask_int('Superconductivity: dense phonon q-grid per axis (NQ_DENSE)', 4 * kg)
    nk = ask_divisible(
        pd.ask_int,
        'Superconductivity: dense electron k-grid per axis (NK_DENSE)',
        8 * kg,
        nq,
        'NK_DENSE',
    )
    sigma = pd.ask_float('Fermi-surface smearing (eV, EPW degaussw)', 0.05)
    mu_star = pd.ask_float('Coulomb pseudopotential mu* (for Tc)', 0.10)
    transport = pd.ask_yes_no(
        'Also the transport Eliashberg function (resistivity, infrared optics, emissivity)?',
        cfg.get('metal', False) and 'transport' in cfg['workflows'],
    )
    fs_coupling = pd.ask_yes_no(
        'Store the Fermi-surface pair coupling for the anisotropic Migdal-Eliashberg equations '
        '(needs nk == nq)?',
        nk == nq,
    )
    fsthick = (
        pd.ask_float('Fermi window of the anisotropic states (eV, EPW fsthick)', 0.2)
        if fs_coupling
        else None
    )
    temps = _floats(pd.ask('Migdal-Eliashberg temperatures (K): Tmin, Tmax, nstemp', '0.5, 10, 20'))
    if len(temps) == 3 and temps[2] >= 2:
        me_temps = [round(float(t), 3) for t in np.linspace(temps[0], temps[1], int(temps[2]))]
    else:
        me_temps = temps or [1.0, 2.0, 4.0]
    cfg['superconductivity'] = dict(
        nk=nk,
        nq=nq,
        sigma_ev=sigma,
        adaptive_smearing=None,
        mu_star=mu_star,
        transport=transport,
        fs_coupling=fs_coupling,
        fsthick_ev=fsthick,
        me_temps=me_temps,
        me_linear_temps=(me_temps[0], me_temps[-1], len(me_temps)),
        me_aniso_temps=me_temps,
        wscut=0.1,
        wscut_aniso=0.5,
        degaussq_mev=0.15,
        degaussq_linear_mev=0.5,
        npade=90.0,
    )


def _collect_optics(cfg, pd, kg: int) -> None:
    nq = pd.ask_int('Optics: dense phonon q-grid per axis (EPW nqf)', kg)
    nk = ask_divisible(
        pd.ask_int, 'Optics: dense electron k-grid per axis (EPW nkf)', 2 * nq, nq, 'NK_DENSE'
    )
    electrons_save = (
        pd.ask(
            'QE save of an eACBN0 (DFT+U+V, ortho-atomic) nscf in the same PAO basis, whose PAO '
            'Hamiltonian replaces the PBE one (blank = PBE electrons)',
            '',
        )
        or None
    )
    gap = pd.ask('Target indirect gap for a scissor (eV; blank = the Hamiltonian gap)', '')
    gap_reference = None
    if not electrons_save:
        gap_reference = (
            pd.ask(
                'QE save of an eACBN0 (DFT+U+V) run whose PAO gap is the target (blank = none)', ''
            )
            or None
        )
    varshni = _floats(
        pd.ask('Temperature-dependent gap: Varshni alpha (eV/K), beta (K) (blank = fixed)', '')
    )
    eps = pd.ask(
        'Screening correction of the coupling (Poncé 2018): target static dielectric constant '
        '(Si 11.94; blank = DFPT screening)',
        '',
    )
    emissivity = pd.ask_yes_no(
        'Also compute the thermal emissivity of a slab vs temperature (Kirchhoff)?', False
    )
    thickness, angles, direct_dir = 500.0, (0.0, 30.0, 60.0), None
    if emissivity:
        thickness = pd.ask_float('Slab thickness (um)', 500.0)
        angles = tuple(
            _floats(pd.ask('Emission angles (deg), comma-separated', '0, 30, 60')) or [0.0]
        )
        cfg['optics_nscf'] = pd.ask_yes_no(
            'Direct dielectric function from an extended-basis run (qe/optics nscf + optics/dielectric.py)?',
            True,
        )
        direct_dir = '../qe/optics/output' if cfg['optics_nscf'] else None
    omega = (
        pd.ask_float('Lowest photon energy (eV, EPW omegamin)', 0.01 if emissivity else 0.05),
        pd.ask_float('Highest photon energy (eV, EPW omegamax)', 3.0),
        pd.ask_float('Photon-energy step (eV, EPW omegastep)', 0.01 if emissivity else 0.05),
    )
    temps = _floats(
        pd.ask(
            'Temperatures (K), comma-separated',
            '300, 500, 700, 900, 1100, 1300, 1500' if emissivity else '300',
        )
    ) or [300.0]
    degauss = pd.ask_float('Delta smearing (eV, EPW degaussw)', 0.01 if emissivity else 0.05)
    fsthick = pd.ask_float('Energy window around E_F (eV, EPW fsthick)', 4.0)
    n_r = pd.ask(
        'Constant refractive index n_r for alpha (blank = n(omega) of the direct run, else 3.4)', ''
    )
    cfg['optics'] = dict(
        nk=nk,
        nq=nq,
        omega_ev=omega,
        temps_k=temps,
        degauss_ev=degauss,
        adaptive_smearing=None,
        fsthick_ev=fsthick,
        refractive_index=float(n_r) if n_r.strip() else None,
        position_dipoles=pd.ask_yes_no(
            'Position dipoles of the PAO orbitals in the velocity operator?', True
        ),
        target_gap_ev=float(gap) if gap.strip() else None,
        gap_reference_save=gap_reference,
        varshni=tuple(varshni) if len(varshni) == 2 else None,
        electrons_save=electrons_save,
        screening_eps=float(eps) if eps.strip() else None,
        emissivity=emissivity,
        thickness_um=thickness,
        angles=angles,
        direct_dielectric_dir=direct_dir,
        kk_omega_max_ev=10.0,
        dielectric=dict(nfft=24, emin=0.01, emax=10.0, ne=1000, delta=0.1, intraband=True),
    )


def _collect_transport(cfg, pd, kg: int) -> None:
    if cfg.get('metal'):
        temps = _floats(
            pd.ask('Resistivity: temperatures (K)', '20, 50, 100, 200, 300, 400, 500, 600')
        )
        rho0 = pd.ask_float('Residual resistivity (micro-ohm cm)', 0.0)
        b_ee = pd.ask_float('Electron-electron (Fermi-liquid) coefficient b (1/eV)', 0.0)
        interband = pd.ask_yes_no(
            'Interband dielectric function from an extended-basis run (qe/optics nscf + '
            'optics/dielectric.py --interband-only) for the emissivity?',
            False,
        )
        if interband:
            cfg['optics_nscf'] = True
        cfg['transport'] = dict(
            temps_k=temps or [300.0],
            rho0_uohm_cm=rho0,
            b_ee_per_ev=b_ee,
            omega_ev=(0.005, 2.0, 0.005),
            thickness_um=float('inf'),
            angles=(0.0, 30.0, 60.0),
            interband_dielectric_dir='../qe/optics/output' if interband else None,
            extended_drude=True,
        )
        if interband and 'optics' not in cfg['workflows']:
            cfg['optics'] = dict(
                dielectric=dict(
                    nfft=36,
                    emin=0.002,
                    emax=6.0,
                    ne=3000,
                    delta=0.1,
                    intraband=False,
                    smearing='gauss',
                )
            )
        return
    nq = pd.ask_int('Mobility: dense phonon q-grid per axis (mobilities need >= 48)', 8 * kg)
    nk = ask_divisible(pd.ask_int, 'Mobility: dense electron k-grid per axis', nq, nq, 'NK_DENSE')
    temps = _floats(
        pd.ask('Temperatures (K), comma-separated', '200, 250, 300, 350, 400, 500')
    ) or [300.0]
    doping = pd.ask_float('Net doping (cm^-3; > 0 n type, < 0 p type, 0 intrinsic)', 0.0)
    window = pd.ask_float('Energy window from the band edges (eV, EPW fsthick)', 0.3)
    smearing = pd.ask_float('Delta smearing (eV, EPW degaussw)', 0.01)
    electrons_save = (
        pd.ask(
            'QE save of an eACBN0 (DFT+U+V, ortho-atomic) nscf in the same PAO basis (blank = PBE electrons)',
            '',
        )
        or None
    )
    gap = pd.ask(
        'Target indirect gap for the carrier statistics (eV; blank = the Hamiltonian gap)', ''
    )
    varshni = []
    if gap.strip() or electrons_save:
        varshni = _floats(
            pd.ask('Temperature-dependent gap: Varshni alpha (eV/K), beta (K) (blank = fixed)', '')
        )
    eps = pd.ask(
        'Screening correction of the coupling (Poncé 2018): target static dielectric constant '
        '(Si 11.94; blank = DFPT screening)',
        '',
    )
    cfg['transport'] = dict(
        nk=nk,
        nq=nq,
        temps_k=temps,
        doping_cm3=doping,
        window_ev=window,
        smearing_ev=smearing,
        adaptive_smearing=None,
        target_gap_ev=float(gap) if gap.strip() else None,
        varshni=tuple(varshni) if len(varshni) == 2 else None,
        electrons_save=electrons_save,
        screening_eps=float(eps) if eps.strip() else None,
    )


def _collect_arpes(cfg, pd, kg: int) -> None:
    metal = cfg.get('metal', False)
    photon = pd.ask_float('ARPES: photon energy (eV; He I 21.2)', 21.2)
    work = pd.ask_float('Work function (eV)', 4.5)
    direction = _floats(
        pd.ask('Path: end point of the line from Gamma (crystal coordinates)', '0.5 0 0.5')
    )
    label = pd.ask('Its label', 'X')
    sc = cfg.get('superconductivity', {})
    if metal:
        kinds = ['isotropic'] if 'superconductivity' in cfg['workflows'] else ['fan-migdal']
        if sc.get('fs_coupling'):
            kinds.append('anisotropic')
    else:
        kinds = ['fan-migdal']
    cfg['arpes'] = dict(
        photon_ev=photon,
        work_function_ev=work,
        polarization=(1.0, 0.0, 1.0),
        direction=tuple(direction) if len(direction) == 3 else (0.5, 0.0, 0.5),
        direction_label=label,
        window=(-0.1, 0.02, 401) if metal else (-13.0, 1.5, 581),
        resolution=(0.002, 0.0) if metal else (0.05, 0.0),
        self_energy=kinds,
        temps_k=[10.0] if metal else [10.0, 300.0, 600.0],
        segment_ev=0.12 if metal else None,
        npts=81 if metal else 41,
        nq=18 if metal else 12,
        eta_ev=0.005 if metal else 0.02,
        adaptive_smearing=0.7 if metal else None,
        intensity_power=1.0 if metal else 0.3,
    )


def main_elphon(args, workdir: str) -> int:
    """Run the electron-phonon generator: prompts, then the system folder.

    Parameters
    ----------
    args : argparse.Namespace
        ``paoflow-gen`` arguments (``prefix``, ``out``, ``force``).
    workdir : str
        Directory scanned for QE inputs.

    Returns
    -------
    int
        Exit status.
    """
    cfg = collect_elphon_system(args, workdir)
    written = write_system_folder(cfg, cfg['out'], force=args.force)
    print('\nWrote %d files under %s:' % (len(written), cfg['out']))
    for path in written:
        print('  %s' % os.path.relpath(path, cfg['out']))
    print('\nNext steps:')
    if not cfg.get('pw_template_text'):
        print(
            '  0) write qe/phonon/scf.in and qe/nscf/nscf.in (K_POINTS card of qe/nscf/nscf.kpoints, nbnd = %d)'
            % cfg['nbnd']
        )
    print(
        '  1) copy the pseudopotentials into qe/ and run the QE steps: (cd qe && NP=8 bash run_qe.sh)'
    )
    print('  2) paoflow-genbasis-ps --pseudo-dir qe --out %s --preset standard' % cfg['basisdir'])
    for name in cfg['workflows']:
        if name == 'phonons':
            print('  3) python phonons/phonons.py')
        elif name == 'coupling':
            print(
                '  3) python coupling/quadrupoles.py --write-inputs; (cd qe/phonon/quadrupoles && bash run.sh); '
                'python coupling/quadrupoles.py --extract; then set quadrupoles in system.py'
            )
        elif name == 'superconductivity':
            print(
                '  3) mpirun -np N python superconductivity/eliashberg.py; python superconductivity/me.py'
                + (
                    '; python superconductivity/me_aniso.py'
                    if cfg['superconductivity'].get('fs_coupling')
                    else ''
                )
                + '; python superconductivity/plot_me.py'
            )
        elif name == 'optics':
            print(
                '  3) mpirun -np N python optics/absorption.py; python optics/plot_absorption.py'
                + (
                    '   (optics/dielectric.py first, for --emissivity)'
                    if cfg['optics'].get('emissivity')
                    else ''
                )
            )
        elif name == 'transport' and cfg.get('metal'):
            print(
                '  3) python transport/resistivity.py (after eliashberg.py --transport); python transport/plot_resistivity.py'
            )
        elif name == 'transport':
            print(
                '  3) mpirun -np N python transport/mobility.py; python transport/plot_mobility.py'
            )
        elif name == 'arpes':
            print('  3) mpirun -np N python arpes/arpes.py')
    print(
        '  (every driver reads %s/system.py; outputs go to <workflow>/output/, the vertex cache to cache/)'
        % os.path.basename(cfg['out'])
    )
    return 0
