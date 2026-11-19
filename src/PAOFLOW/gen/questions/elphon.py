"""Questions of the ``elphon`` workflow: the electron-phonon system folder (``system.py``, ``qe/``, drivers)."""

from __future__ import annotations

import os

import numpy as np

from .model import (
    Answers,
    Context,
    Note,
    Question,
    Workflow,
    divisible_by,
    float_list,
    gated,
    optional_number,
    optional_text,
)

SYSTEM = 'System'
COARSE = 'Coarse grids and bands'
WORKFLOWS = 'Workflows'
SC = 'Superconductivity'
OPTICS = 'Optics'
TRANSPORT = 'Transport'
ARPES = 'ARPES'
THERMAL = 'Thermal (special displacements)'
OPTICS_NSCF = 'Extended-basis optics nscf'
FD = 'Finite differences'

#: The workflow folders a system can hold, in the order of the prompt.
ELPHON_WORKFLOWS = (
    'phonons',
    'coupling',
    'superconductivity',
    'optics',
    'transport',
    'arpes',
    'thermal',
)

#: The workflows of the finite-difference source: the Eliashberg route (Gamma of the supercell
#: carries every mode; the other dense drivers zero the modes at Gamma) and, for metals, the
#: resistivity that reads its transport Eliashberg function.
FD_WORKFLOWS = ('superconductivity', 'transport')


# ---------------------------------------------------------------------------- the template
def template_path(answers: Answers, context: Context) -> str | None:
    """Absolute path of the ``pw.x`` template, or ``None`` when no template is given.

    Parameters
    ----------
    answers : dict
        Answers so far (``pw_template``).
    context : Context
        The session context.

    Returns
    -------
    str or None
    """
    template = answers.get('pw_template') or ''
    if not template:
        return None
    return os.path.join(context.workdir, os.path.expanduser(template))


def template_text(answers: Answers, context: Context) -> str | None:
    """Text of the ``pw.x`` template (``None`` if not given or not readable).

    Parameters
    ----------
    answers : dict
        Answers so far (``pw_template``).
    context : Context
        The session context.

    Returns
    -------
    str or None
    """
    path = template_path(answers, context)
    if path is None:
        return None

    def read() -> str | None:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                return handle.read()
        except OSError:
            return None

    return context.cached('template:' + path, read)


def template_summary(answers: Answers, context: Context) -> dict | None:
    """Summary of the ``pw.x`` template (:func:`PAOFLOW.gen.system_summary.read_pw_input_summary`).

    Parameters
    ----------
    answers : dict
        Answers so far (``pw_template``).
    context : Context
        The session context.

    Returns
    -------
    dict or None
    """
    from ..system_summary import read_pw_input_summary

    path = template_path(answers, context)
    if path is None or template_text(answers, context) is None:
        return None
    return context.cached('template-summary:' + path, lambda: read_pw_input_summary(path))


def from_template(key: str, fallback, convert=None):
    """Default taken from the template summary (``fallback`` when absent or unknown).

    Parameters
    ----------
    key : str
        Key of :func:`PAOFLOW.gen.system_summary.read_pw_input_summary`.
    fallback : Any
        The default without a template or value.
    convert : callable, optional
        Applied to the found value.

    Returns
    -------
    callable
        ``default(answers, context)``.
    """

    def default(answers: Answers, context: Context):
        summary = template_summary(answers, context)
        value = summary.get(key) if summary is not None else None
        if value is None:
            return fallback
        return convert(value) if convert is not None else value

    return default


def _template_unreadable(answers: Answers, context: Context) -> bool:
    return template_path(answers, context) is not None and template_text(answers, context) is None


def _namelist(answers: Answers, context: Context, name: str) -> str | None:
    from ..paoflow_driver import _read_namelist_value

    text = template_text(answers, context)
    return _read_namelist_value(text, name) if text else None


def output_folder(context: Context) -> str:
    """The system folder (``-o/--out``, else the working directory), absolute.

    Parameters
    ----------
    context : Context
        The session context.

    Returns
    -------
    str
    """
    return os.path.abspath(os.path.expanduser(context.overrides.get('out') or context.workdir))


def _default_pseudo_dir(answers: Answers, context: Context) -> str:
    pseudo_dir = None
    path = template_path(answers, context)
    if path is not None and template_text(answers, context) is not None:
        pseudo_dir = _namelist(answers, context, 'pseudo_dir')
        if pseudo_dir is not None:
            pseudo_dir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(path)), pseudo_dir)
            )
    if pseudo_dir is None:
        pseudo_dir = context.workdir
    # qe/phonon, qe/nscf and qe/optics are siblings, so one relative path serves all three.
    relative = os.path.relpath(pseudo_dir, os.path.join(output_folder(context), 'qe', 'phonon'))
    return relative.rstrip('/') + '/'


def _default_prefix(answers: Answers, context: Context) -> str:
    return (_namelist(answers, context, 'prefix') or 'pwscf').strip('\'"')


def _default_masses(answers: Answers, context: Context) -> str:
    from ..epw_inputs import species_masses

    masses = species_masses(template_text(answers, context) or '')
    return ', '.join('%g' % m for m in masses)


def _exclude_bands(text: str) -> list[int]:
    from ..epw_inputs import parse_exclude_bands

    return parse_exclude_bands(text)


# ------------------------------------------------------------------------- shared helpers
def _kg(answers: Answers) -> int:
    return int(answers['kgrid_axis'])


def _workflows(answers: Answers) -> list[str]:
    return list(answers.get('workflows') or [])


def _selected(name: str):
    return lambda a, c: name in _workflows(a)


def _metal(answers: Answers) -> bool:
    return bool(answers.get('metal'))


def _fd(answers: Answers) -> bool:
    return answers.get('coupling_source') == 'fd'


def _nat(answers: Answers, context: Context) -> int:
    """Atoms of the primitive cell (``nat`` of the template; 1 without a template)."""
    nat = _namelist(answers, context, 'nat')
    try:
        return max(1, int(float(str(nat).strip(' ,\'"'))))
    except (TypeError, ValueError):
        return 1


def _fd_atoms(answers: Answers, context: Context) -> int:
    """Atoms of the finite-difference supercell."""
    return _nat(answers, context) * int(answers.get('fd_supercell', 1)) ** 3


def _validate_fd_supercell(value: int, answers: Answers, context: Context) -> str | None:
    if value < 1:
        return '  The supercell must have at least one cell per axis.'
    if _kg(answers) % value:
        return (
            '  The coarse k-grid (%d) must be a multiple of the supercell: the supercell runs '
            'use k/%d per axis.' % (_kg(answers), value)
        )
    return None


def _fd_cost(answers: Answers, context: Context) -> str:
    atoms = _fd_atoms(answers, context)
    if answers.get('fd_mode') == 'random':
        cells = 2 * int(answers.get('fd_nconfig') or 3 * atoms)
    else:
        cells = 6 * atoms
    return (
        '  (%d pw.x runs of %d atoms: the reference supercell and %d displaced cells, on a '
        '%d^3 k-grid; Gamma of the supercell holds the primitive phonons of a %d^3 q-grid)'
        % (cells + 1, atoms, cells, _kg(answers) // max(1, answers.get('fd_supercell', 1)),
           answers.get('fd_supercell', 1))
    )  # fmt: skip


def _validate_qgrid(value: int, answers: Answers, context: Context) -> str | None:
    kg = _kg(answers)
    if value > 0 and kg % value == 0:
        return None
    return '  The q-grid must divide the k-grid (%d).' % kg


def _validate_workflows(value: list[str], answers: Answers, context: Context) -> str | None:
    unknown = [name for name in value if name not in ELPHON_WORKFLOWS]
    if unknown:
        return '  unknown workflow(s): %s' % ', '.join(unknown)
    if _fd(answers):
        allowed = FD_WORKFLOWS if _metal(answers) else ()
        other = [name for name in value if name not in allowed]
        if other:
            return (
                '  The finite-difference source gives superconductivity and, for a metal, the '
                'resistivity; %s need the dfpt or epw source.' % ', '.join(other)
            )
    return None


def _default_workflows(answers: Answers, context: Context) -> list[str]:
    if _fd(answers):
        return list(FD_WORKFLOWS) if _metal(answers) else []
    if _metal(answers):
        return ['superconductivity', 'transport', 'arpes']
    return ['optics', 'transport', 'arpes']


def _emissivity(answers: Answers) -> bool:
    return bool(answers.get('optics_emissivity'))


def _optics_nscf(answers: Answers) -> bool:
    """Whether the extended-basis optics nscf is written (optics emissivity or metal interband)."""
    workflows = _workflows(answers)
    direct = 'optics' in workflows and _emissivity(answers) and bool(answers.get('optics_direct'))
    interband = (
        'transport' in workflows and _metal(answers) and bool(answers.get('resistivity_interband'))
    )
    return direct or interband


def _thermal_default_bands(answers: Answers, context: Context) -> int:
    nat = _namelist(answers, context, 'nat')
    try:
        nat = max(1, int(float(str(nat).strip(' ,\'"'))))
    except (TypeError, ValueError):
        nat = 1
    nbnd = answers.get('nbnd', 0)
    default_bands = int(np.ceil(nbnd / nat)) if nbnd > 0 else 12
    return max(4, default_bands)


# ------------------------------------------------------------------------------ the items
SYSTEM_ITEMS = (
    Question(
        'pw_template',
        'pw.x input of the system (gives qe/phonon/scf.in and the nscf inputs; blank = none)',
        default=lambda a, c: c.cached('pw_input', lambda: _detect_pw_input(c)) or '',
        label='pw.x input of the system',
        help='Template of qe/phonon/scf.in and of the nscf inputs; blank writes only the '
        'k-point list and ph.in.',
        section=SYSTEM,
    ),
    Note(
        lambda a, c: '  (%s not readable; the scf and nscf inputs are not written)'
        % template_path(a, c),
        when=_template_unreadable,
        section=SYSTEM,
    ),
    Question('prefix', 'Prefix', default=_default_prefix, override='prefix', section=SYSTEM),
    Question(
        'name',
        'System name (titles of the figures)',
        default=lambda a, c: a['prefix'],
        section=SYSTEM,
    ),
    Question(
        'pseudo_dir',
        'pseudo_dir of the QE run directories',
        default=_default_pseudo_dir,
        section=SYSTEM,
    ),
    Question(
        'coupling_source',
        "Coupling source (dfpt = PAOFLOW's vertex from ph.x dvscf, norm-conserving; "
        'epw = EPW epbwrite, any pseudopotential; fd = finite differences of a supercell, '
        'pw.x only)',
        kind='choice',
        choices=['dfpt', 'epw', 'fd'],
        default=from_template(
            'augmented', 'dfpt', lambda augmented: 'epw' if augmented else 'dfpt'
        ),
        label='Coupling source',
        help='Default epw when a pseudopotential of the template is ultrasoft or PAW (the '
        'native DFPT vertex is norm-conserving only). fd displaces the atoms of a supercell and '
        'differentiates its PAO Hamiltonian: no ph.x, so any pseudopotential, DFT+U or hybrid '
        'that pw.x runs; it gives superconductivity and the metal resistivity.',
        section=SYSTEM,
    ),
    Question(
        'kgrid_axis',
        'Coarse k-grid per axis (nscf)',
        kind='int',
        default=6,
        section=COARSE,
    ),
    Question(
        'qgrid_axis',
        'Coarse q-grid per axis (ph.x; must divide the k-grid)',
        kind='int',
        default=lambda a, c: _kg(a),
        validate=_validate_qgrid,
        when=lambda a, c: not _fd(a),
        section=COARSE,
    ),
    Question(
        'irreducible',
        'nscf on the irreducible wedge (K_POINTS automatic, unfolded by symmetry)? '
        '(No = explicit full list)',
        kind='bool',
        default=True,
        when=lambda a, c: a.get('coupling_source') == 'dfpt',
        section=COARSE,
    ),
    Note(
        '  (EPW needs the explicit full k list: K_POINTS crystal)',
        when=lambda a, c: a.get('coupling_source') == 'epw',
        section=COARSE,
    ),
    Question(
        'fd_supercell',
        'Supercell of the finite differences (primitive cells per axis)',
        kind='int',
        default=2,
        validate=_validate_fd_supercell,
        label='Supercell',
        help='The ideal n x n x n supercell is the unit cell of the finite differences: its '
        'Gamma phonons are the primitive phonons on the n^3 q-grid, and its runs use the coarse '
        'k-grid divided by n.',
        when=lambda a, c: _fd(a),
        section=FD,
    ),
    Question(
        'fd_mode',
        'Displacements (cartesian = every atom along +-x, +-y, +-z, 6 N cells; random = all '
        'atoms at once in random patterns, 2 M cells, least squares)',
        kind='choice',
        choices=['cartesian', 'random'],
        default=lambda a, c: 'cartesian' if 6 * _fd_atoms(a, c) <= 300 else 'random',
        label='Displacements',
        when=lambda a, c: _fd(a),
        section=FD,
    ),
    Question(
        'fd_distance',
        'Displacement amplitude (bohr)',
        kind='float',
        default=0.06,
        unit='bohr',
        when=lambda a, c: _fd(a),
        section=FD,
    ),
    Question(
        'fd_nconfig',
        'Random configurations M (3 N gives a full-rank fit without a locality cutoff)',
        kind='int',
        default=lambda a, c: 3 * _fd_atoms(a, c),
        when=lambda a, c: _fd(a) and a.get('fd_mode') == 'random',
        section=FD,
    ),
    Question(
        'fd_cutoff',
        'Locality cutoff of the fit (bohr; blank = every atom; with a cutoff M can be smaller)',
        default='',
        parse=optional_number,
        unit='bohr',
        when=lambda a, c: _fd(a) and a.get('fd_mode') == 'random',
        section=FD,
        level='advanced',
    ),
    Question(
        'fd_seed',
        'Seed of the random directions',
        kind='int',
        default=0,
        when=lambda a, c: _fd(a) and a.get('fd_mode') == 'random',
        section=FD,
        level='advanced',
    ),
    Note(_fd_cost, when=lambda a, c: _fd(a), section=FD),
    Question(
        'nbnd',
        'Bands in the nscf run (nbnd, > number of PAO orbitals)',
        kind='int',
        default=from_template('nbnd', 0),
        section=COARSE,
    ),
    Note(
        '  (fd: nbnd is per primitive cell, the supercell runs get n^3 times it; 0 sizes them '
        'to the PAO basis)',
        when=lambda a, c: _fd(a),
        section=COARSE,
    ),
    Question(
        'exclude_bands',
        'Bands to exclude from the coupling (EPW exclude_bands, e.g. 1:5; blank = none)',
        default='',
        parse=_exclude_bands,
        when=lambda a, c: a.get('coupling_source') == 'epw',
        section=COARSE,
    ),
    Question(
        'masses_amu',
        'Atomic mass of each species (amu), comma-separated',
        default=_default_masses,
        parse=float_list,
        unit='amu',
        section=SYSTEM,
    ),
    Question(
        'nelec',
        'Valence electrons (nelec)',
        kind='int',
        default=from_template('nelec', 0, lambda nelec: int(round(nelec))),
        help='Default from the valence charges of the template pseudopotentials.',
        section=SYSTEM,
    ),
    Question(
        'metal',
        'Is the system a metal?',
        kind='bool',
        default=from_template('metal', False),
        help="Default from the template occupations ('smearing': metal).",
        section=SYSTEM,
    ),
    Question(
        'basisdir',
        'Basis directory name (paoflow-genbasis-ps)',
        default='BASIS_PS',
        section=SYSTEM,
    ),
    Question(
        'pthr',
        'Projectability threshold (pthr)',
        kind='float',
        default=0.95,
        section=SYSTEM,
    ),
    Question(
        'tr2_ph',
        'ph.x self-consistency threshold (tr2_ph)',
        kind='float',
        default=1.0e-16,
        when=lambda a, c: not _fd(a),
        section=COARSE,
        level='advanced',
        cli=False,
    ),
    Note(
        '  (ph.in gets epsil = .true.: the dielectric tensor and Born charges of the '
        'long-range model; quadrupoles come from coupling/quadrupoles.py)',
        when=lambda a, c: not _metal(a) and not _fd(a),
        section=SYSTEM,
    ),
    Note(
        '  (fd gives superconductivity and the resistivity of a metal; an insulator needs the '
        'dfpt or epw source)',
        when=lambda a, c: _fd(a) and not _metal(a),
        section=SYSTEM,
    ),
    Question(
        'workflows',
        'Workflows (comma-separated: %s)' % ', '.join(ELPHON_WORKFLOWS),
        kind='names',
        choices=list(ELPHON_WORKFLOWS),
        default=_default_workflows,
        parse=lambda text: list(
            dict.fromkeys(x.strip() for x in text.replace(',', ' ').split() if x.strip())
        ),
        validate=_validate_workflows,
        label='Workflows',
        section=WORKFLOWS,
    ),
)

SUPERCONDUCTIVITY_ITEMS = gated(
    (
        Question(
            'sc_nq',
            'Superconductivity: dense phonon q-grid per axis (NQ_DENSE)',
            kind='int',
            default=lambda a, c: 4 * _kg(a),
            when=lambda a, c: not _fd(a),
            section=SC,
        ),
        Question(
            'sc_nk',
            'Superconductivity: dense electron k-grid per axis (NK_DENSE)',
            kind='int',
            default=lambda a, c: 8 * _kg(a) // (a.get('fd_supercell', 1) if _fd(a) else 1),
            validate=divisible_by(lambda a: 1 if _fd(a) else a['sc_nq'], 'NK_DENSE'),
            section=SC,
        ),
        Note(
            '  (fd: the k-grid is of the supercell; q is its Gamma point, and eliashberg.py runs '
            'on one MPI rank)',
            when=lambda a, c: _fd(a),
            section=SC,
        ),
        Question(
            'sc_adaptive',
            'Adaptive Fermi-surface smearing prefactor a (Yates; 0.7 production, 0 = fixed width)',
            kind='float',
            default=0.7,
            section=SC,
        ),
        Question(
            'sc_sigma',
            'Fermi-surface smearing (eV): floor of the adaptive widths, or the fixed width '
            '(EPW degaussw)',
            kind='float',
            default=lambda a, c: 0.001 if a.get('sc_adaptive') else 0.05,
            unit='eV',
            section=SC,
        ),
        Question(
            'sc_mu_star',
            'Coulomb pseudopotential mu* (for Tc)',
            kind='float',
            default=0.10,
            section=SC,
        ),
        Question(
            'sc_transport',
            'Also the transport Eliashberg function (resistivity, infrared optics, emissivity)?',
            kind='bool',
            default=lambda a, c: _metal(a) and 'transport' in _workflows(a),
            section=SC,
        ),
        Question(
            'sc_fs_coupling',
            'Store the Fermi-surface pair coupling for the anisotropic Migdal-Eliashberg '
            'equations (needs nk == nq)?',
            kind='bool',
            default=lambda a, c: a['sc_nk'] == a['sc_nq'],
            when=lambda a, c: not _fd(a),
            section=SC,
        ),
        Question(
            'sc_fsthick',
            'Fermi window of the anisotropic states (eV, EPW fsthick)',
            kind='float',
            default=0.2,
            when=lambda a, c: bool(a.get('sc_fs_coupling')),
            unit='eV',
            section=SC,
        ),
        Question(
            'sc_me_temps',
            'Migdal-Eliashberg temperatures (K): Tmin, Tmax, nstemp',
            default='0.5, 10, 20',
            parse=float_list,
            unit='K',
            section=SC,
        ),
    ),
    _selected('superconductivity'),
)

OPTICS_ITEMS = gated(
    (
        Question(
            'optics_nq',
            'Optics: dense phonon q-grid per axis (EPW nqf)',
            kind='int',
            default=lambda a, c: _kg(a),
            section=OPTICS,
        ),
        Question(
            'optics_nk',
            'Optics: dense electron k-grid per axis (EPW nkf)',
            kind='int',
            default=lambda a, c: 2 * a['optics_nq'],
            validate=divisible_by(lambda a: a['optics_nq'], 'NK_DENSE'),
            section=OPTICS,
        ),
        Question(
            'optics_electrons_save',
            'QE save of an eACBN0 (DFT+U+V, ortho-atomic) nscf in the same PAO basis, whose PAO '
            'Hamiltonian replaces the PBE one (blank = PBE electrons)',
            default='',
            parse=optional_text,
            section=OPTICS,
            level='advanced',
        ),
        Question(
            'optics_gap',
            'Target indirect gap for a scissor (eV; blank = the Hamiltonian gap)',
            default='',
            parse=optional_number,
            unit='eV',
            section=OPTICS,
        ),
        Question(
            'optics_gap_reference',
            'QE save of an eACBN0 (DFT+U+V) run whose PAO gap is the target (blank = none)',
            default='',
            parse=optional_text,
            when=lambda a, c: not a.get('optics_electrons_save'),
            section=OPTICS,
            level='advanced',
        ),
        Question(
            'optics_varshni',
            'Temperature-dependent gap: Varshni alpha (eV/K), beta (K) (blank = fixed)',
            default='',
            parse=float_list,
            section=OPTICS,
            level='advanced',
        ),
        Question(
            'optics_screening_eps',
            'Screening correction of the coupling (Poncé 2018): target static dielectric '
            'constant (Si 11.94; blank = DFPT screening)',
            default='',
            parse=optional_number,
            section=OPTICS,
            level='advanced',
        ),
        Question(
            'optics_emissivity',
            'Also compute the thermal emissivity of a slab vs temperature (Kirchhoff)?',
            kind='bool',
            default=False,
            section=OPTICS,
        ),
        Question(
            'optics_thickness',
            'Slab thickness (um)',
            kind='float',
            default=500.0,
            when=lambda a, c: _emissivity(a),
            unit='µm',
            section=OPTICS,
        ),
        Question(
            'optics_angles',
            'Emission angles (deg), comma-separated',
            default='0, 30, 60',
            parse=float_list,
            when=lambda a, c: _emissivity(a),
            unit='deg',
            section=OPTICS,
        ),
        Question(
            'optics_direct',
            'Direct dielectric function from an extended-basis run (qe/optics nscf + '
            'optics/dielectric.py)?',
            kind='bool',
            default=True,
            when=lambda a, c: _emissivity(a),
            section=OPTICS,
        ),
        Question(
            'optics_omega_min',
            'Lowest photon energy (eV, EPW omegamin)',
            kind='float',
            default=lambda a, c: 0.01 if _emissivity(a) else 0.05,
            unit='eV',
            section=OPTICS,
        ),
        Question(
            'optics_omega_max',
            'Highest photon energy (eV, EPW omegamax)',
            kind='float',
            default=3.0,
            unit='eV',
            section=OPTICS,
        ),
        Question(
            'optics_omega_step',
            'Photon-energy step (eV, EPW omegastep)',
            kind='float',
            default=lambda a, c: 0.01 if _emissivity(a) else 0.05,
            unit='eV',
            section=OPTICS,
        ),
        Question(
            'optics_temps',
            'Temperatures (K), comma-separated',
            default=lambda a, c: '300, 500, 700, 900, 1100, 1300, 1500'
            if _emissivity(a)
            else '300',
            parse=float_list,
            unit='K',
            section=OPTICS,
        ),
        Question(
            'optics_degauss',
            'Delta smearing (eV, EPW degaussw)',
            kind='float',
            default=lambda a, c: 0.01 if _emissivity(a) else 0.05,
            unit='eV',
            section=OPTICS,
        ),
        Question(
            'optics_fsthick',
            'Energy window around E_F (eV, EPW fsthick)',
            kind='float',
            default=4.0,
            unit='eV',
            section=OPTICS,
        ),
        Question(
            'optics_refractive_index',
            'Constant refractive index n_r for alpha (blank = n(omega) of the direct run, else 3.4)',
            default='',
            parse=optional_number,
            section=OPTICS,
            level='advanced',
        ),
        Question(
            'optics_position_dipoles',
            'Position dipoles of the PAO orbitals in the velocity operator?',
            kind='bool',
            default=True,
            section=OPTICS,
        ),
    ),
    _selected('optics'),
)

_RESISTIVITY = lambda a, c: 'transport' in _workflows(a) and _metal(a)  # noqa: E731
_MOBILITY = lambda a, c: 'transport' in _workflows(a) and not _metal(a)  # noqa: E731

TRANSPORT_ITEMS = (
    *gated(
        (
            Question(
                'resistivity_temps',
                'Resistivity: temperatures (K)',
                default='20, 50, 100, 200, 300, 400, 500, 600',
                parse=float_list,
                unit='K',
                section=TRANSPORT,
            ),
            Question(
                'resistivity_rho0',
                'Residual resistivity (micro-ohm cm)',
                kind='float',
                default=0.0,
                unit='µΩ cm',
                section=TRANSPORT,
            ),
            Question(
                'resistivity_b_ee',
                'Electron-electron (Fermi-liquid) coefficient b (1/eV)',
                kind='float',
                default=0.0,
                unit='1/eV',
                section=TRANSPORT,
            ),
            Question(
                'resistivity_interband',
                'Interband dielectric function from an extended-basis run (qe/optics nscf + '
                'optics/dielectric.py --interband-only) for the emissivity?',
                kind='bool',
                default=False,
                section=TRANSPORT,
            ),
        ),
        _RESISTIVITY,
    ),
    *gated(
        (
            Question(
                'mobility_nq',
                'Mobility: dense phonon q-grid per axis (mobilities need >= 48)',
                kind='int',
                default=lambda a, c: 8 * _kg(a),
                section=TRANSPORT,
            ),
            Question(
                'mobility_nk',
                'Mobility: dense electron k-grid per axis',
                kind='int',
                default=lambda a, c: a['mobility_nq'],
                validate=divisible_by(lambda a: a['mobility_nq'], 'NK_DENSE'),
                section=TRANSPORT,
            ),
            Question(
                'mobility_temps',
                'Temperatures (K), comma-separated',
                default='200, 250, 300, 350, 400, 500',
                parse=float_list,
                unit='K',
                section=TRANSPORT,
            ),
            Question(
                'mobility_doping',
                'Net doping (cm^-3; > 0 n type, < 0 p type, 0 intrinsic)',
                kind='float',
                default=0.0,
                unit='cm⁻³',
                section=TRANSPORT,
            ),
            Question(
                'mobility_window',
                'Energy window from the band edges (eV, EPW fsthick)',
                kind='float',
                default=0.3,
                unit='eV',
                section=TRANSPORT,
            ),
            Question(
                'mobility_smearing',
                'Delta smearing (eV, EPW degaussw)',
                kind='float',
                default=0.01,
                unit='eV',
                section=TRANSPORT,
            ),
            Question(
                'mobility_electrons_save',
                'QE save of an eACBN0 (DFT+U+V, ortho-atomic) nscf in the same PAO basis '
                '(blank = PBE electrons)',
                default='',
                parse=optional_text,
                section=TRANSPORT,
                level='advanced',
            ),
            Question(
                'mobility_gap',
                'Target indirect gap for the carrier statistics (eV; blank = the Hamiltonian gap)',
                default='',
                parse=optional_number,
                unit='eV',
                section=TRANSPORT,
            ),
            Question(
                'mobility_varshni',
                'Temperature-dependent gap: Varshni alpha (eV/K), beta (K) (blank = fixed)',
                default='',
                parse=float_list,
                when=lambda a, c: a.get('mobility_gap') is not None
                or bool(a.get('mobility_electrons_save')),
                section=TRANSPORT,
                level='advanced',
            ),
            Question(
                'mobility_screening_eps',
                'Screening correction of the coupling (Poncé 2018): target static dielectric '
                'constant (Si 11.94; blank = DFPT screening)',
                default='',
                parse=optional_number,
                section=TRANSPORT,
                level='advanced',
            ),
        ),
        _MOBILITY,
    ),
)

ARPES_ITEMS = gated(
    (
        Question(
            'arpes_photon',
            'ARPES: photon energy (eV; He I 21.2)',
            kind='float',
            default=21.2,
            unit='eV',
            section=ARPES,
        ),
        Question(
            'arpes_work_function',
            'Work function (eV)',
            kind='float',
            default=4.5,
            unit='eV',
            section=ARPES,
        ),
        Question(
            'arpes_direction',
            'Path: end point of the line from Gamma (crystal coordinates)',
            default='0.5 0 0.5',
            parse=float_list,
            section=ARPES,
        ),
        Question('arpes_label', 'Its label', default='X', section=ARPES),
    ),
    _selected('arpes'),
)

THERMAL_ITEMS = gated(
    (
        Question(
            'thermal_temps',
            'Thermal (ZG supercells): temperatures (K), comma-separated',
            default='0, 300',
            parse=float_list,
            unit='K',
            section=THERMAL,
        ),
        Question(
            'thermal_supercell',
            'ZG supercell per axis (3 for tests, 4 and more for production)',
            kind='int',
            default=3,
            section=THERMAL,
        ),
        Question(
            'thermal_kgrid',
            'k-grid per axis of the supercell scf',
            kind='int',
            default=lambda a, c: max(1, -(-_kg(a) // a['thermal_supercell'])),
            section=THERMAL,
        ),
        Question(
            'thermal_bands_per_atom',
            'Bands per atom of the supercell scf (above the PAO orbitals per atom)',
            kind='int',
            default=_thermal_default_bands,
            section=THERMAL,
        ),
        Question(
            'thermal_configurations',
            'Configurations per temperature (seeds of the sign search)',
            kind='int',
            default=1,
            section=THERMAL,
        ),
    ),
    _selected('thermal'),
)

OPTICS_NSCF_ITEMS = gated(
    (
        Question(
            'optics_kgrid_axis',
            'Optics nscf: Monkhorst-Pack grid per axis',
            kind='int',
            default=8,
            section=OPTICS_NSCF,
        ),
        Question(
            'optics_nbnd',
            'Optics nscf: bands (more than the orbitals of the extended PAO basis)',
            kind='int',
            default=60,
            section=OPTICS_NSCF,
        ),
    ),
    lambda a, c: _optics_nscf(a),
)

FINAL_NOTE = Note(
    '\nNote: nbnd, masses_amu and nelec are system-specific; edit system.py and qe/ before running.',
    when=lambda a, c: (a.get('nbnd', 0) <= 0 and not _fd(a))
    or not a.get('masses_amu')
    or a.get('nelec', 0) <= 0,
)

ELPHON_ITEMS = (
    SYSTEM_ITEMS
    + SUPERCONDUCTIVITY_ITEMS
    + OPTICS_ITEMS
    + TRANSPORT_ITEMS
    + ARPES_ITEMS
    + THERMAL_ITEMS
    + OPTICS_NSCF_ITEMS
    + (FINAL_NOTE,)
)


def _detect_pw_input(context: Context) -> str | None:
    from ..paoflow_driver import detect_pw_input

    return detect_pw_input(context.workdir)


# --------------------------------------------------------------------------- the assembly
def _superconductivity(answers: Answers) -> dict:
    temps = answers['sc_me_temps']
    if len(temps) == 3 and temps[2] >= 2:
        me_temps = [round(float(t), 3) for t in np.linspace(temps[0], temps[1], int(temps[2]))]
    else:
        me_temps = temps or [1.0, 2.0, 4.0]
    fs_coupling = answers.get('sc_fs_coupling', False)  # not with the fd source (q = Gamma)
    return dict(
        nk=answers['sc_nk'],
        nq=answers.get('sc_nq', 1),
        sigma_ev=answers['sc_sigma'],
        adaptive_smearing=answers['sc_adaptive'] or None,
        mu_star=answers['sc_mu_star'],
        transport=answers['sc_transport'],
        fs_coupling=fs_coupling,
        fsthick_ev=answers['sc_fsthick'] if fs_coupling else None,
        me_temps=me_temps,
        me_linear_temps=(me_temps[0], me_temps[-1], len(me_temps)),
        me_aniso_temps=me_temps,
        wscut=0.1,
        wscut_aniso=0.5,
        degaussq_mev=0.15,
        degaussq_linear_mev=0.5,
        npade=90.0,
    )


def _optics(answers: Answers) -> dict:
    emissivity = _emissivity(answers)
    thickness, angles, direct_dir = 500.0, (0.0, 30.0, 60.0), None
    if emissivity:
        thickness = answers['optics_thickness']
        angles = tuple(answers['optics_angles'] or [0.0])
        direct_dir = '../qe/optics/output' if answers['optics_direct'] else None
    varshni = answers['optics_varshni']
    electrons_save = answers['optics_electrons_save']
    return dict(
        nk=answers['optics_nk'],
        nq=answers['optics_nq'],
        omega_ev=(
            answers['optics_omega_min'],
            answers['optics_omega_max'],
            answers['optics_omega_step'],
        ),
        temps_k=answers['optics_temps'] or [300.0],
        degauss_ev=answers['optics_degauss'],
        adaptive_smearing=None,
        fsthick_ev=answers['optics_fsthick'],
        refractive_index=answers['optics_refractive_index'],
        position_dipoles=answers['optics_position_dipoles'],
        target_gap_ev=answers['optics_gap'],
        gap_reference_save=None if electrons_save else answers['optics_gap_reference'],
        varshni=tuple(varshni) if len(varshni) == 2 else None,
        electrons_save=electrons_save,
        screening_eps=answers['optics_screening_eps'],
        emissivity=emissivity,
        thickness_um=thickness,
        angles=angles,
        direct_dielectric_dir=direct_dir,
        kk_omega_max_ev=10.0,
        dielectric=dict(nfft=24, emin=0.01, emax=10.0, ne=1000, delta=0.1, intraband=True),
    )


def _resistivity(cfg: dict, answers: Answers) -> None:
    interband = answers['resistivity_interband']
    if interband:
        cfg['optics_nscf'] = True
    cfg['transport'] = dict(
        temps_k=answers['resistivity_temps'] or [300.0],
        rho0_uohm_cm=answers['resistivity_rho0'],
        b_ee_per_ev=answers['resistivity_b_ee'],
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


def _mobility(answers: Answers) -> dict:
    varshni = answers.get('mobility_varshni', [])
    return dict(
        nk=answers['mobility_nk'],
        nq=answers['mobility_nq'],
        temps_k=answers['mobility_temps'] or [300.0],
        doping_cm3=answers['mobility_doping'],
        window_ev=answers['mobility_window'],
        smearing_ev=answers['mobility_smearing'],
        adaptive_smearing=None,
        target_gap_ev=answers['mobility_gap'],
        varshni=tuple(varshni) if len(varshni) == 2 else None,
        electrons_save=answers['mobility_electrons_save'],
        screening_eps=answers['mobility_screening_eps'],
    )


def _arpes(cfg: dict, answers: Answers) -> dict:
    metal = cfg.get('metal', False)
    direction = answers['arpes_direction']
    sc = cfg.get('superconductivity', {})
    if metal:
        kinds = ['isotropic'] if 'superconductivity' in cfg['workflows'] else ['fan-migdal']
        if sc.get('fs_coupling'):
            kinds.append('anisotropic')
    else:
        kinds = ['fan-migdal']
    return dict(
        photon_ev=answers['arpes_photon'],
        work_function_ev=answers['arpes_work_function'],
        polarization=(1.0, 0.0, 1.0),
        direction=tuple(direction) if len(direction) == 3 else (0.5, 0.0, 0.5),
        direction_label=answers['arpes_label'],
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


def _thermal(cfg: dict, answers: Answers) -> dict:
    metal = cfg.get('metal', False)
    dims = answers['thermal_supercell']
    return dict(
        temps_k=answers['thermal_temps'] or [0.0, 300.0],
        supercell=(dims,) * 3,
        kgrid=(answers['thermal_kgrid'],) * 3,
        nbnd_per_atom=answers['thermal_bands_per_atom'],
        force_constants=None,
        seeds=list(range(max(1, answers['thermal_configurations']))),
        error_thresh=0.05,
        reference=True,
        path='G-X' if metal else 'L-G-X',
        sym_points=None,
        nk_per_seg=100,
        energy_window=(-0.3, 0.3, 601) if metal else (-2.0, 3.0, 1001),
        broadening_ev=0.005 if metal else 0.02,
        crossing_ev=0.0,
        half_window_ev=0.25 if metal else 0.5,
        dielectric=None if metal else dict(nfft=24, emin=0.05, emax=6.0, ne=1200, delta=0.05),
        fd_mode='random',
        fd_nconfig=60,
        fd_distance=0.03,
        lambda_nk=12,
        lambda_sigma_ev=0.05,
    )


def _finite_differences(cfg: dict, answers: Answers) -> None:
    """The supercell runs of the ``'fd'`` source (``FINITE_DIFFERENCES`` of ``system.py``)."""
    n = answers['fd_supercell']
    random = answers['fd_mode'] == 'random'
    cfg['kgrid'] = [answers['kgrid_axis'] // n] * 3  # the k-grid of the supercell
    cfg['qgrid'] = [1, 1, 1]
    cfg['finite_differences'] = dict(
        supercell=(n, n, n),
        mode=answers['fd_mode'],
        distance=answers['fd_distance'],
        nconfig=answers['fd_nconfig'] if random else None,
        seed=answers['fd_seed'] if random else 0,
        nbnd=answers['nbnd'] * n**3 if answers['nbnd'] > 0 else None,
        primitive_save='qe/phonon/%s.save' % answers['prefix'],
        pseudo_dir=answers['pseudo_dir'],
        prefix=answers['prefix'],
    )
    cutoff = answers.get('fd_cutoff') if random else None
    cfg['fd_options'] = {'locality_cutoff': cutoff} if cutoff is not None else {}


def assemble_elphon(answers: Answers, context: Context) -> dict:
    """Configuration of the system folder (:func:`PAOFLOW.gen.elphon_system.write_system_folder`).

    Parameters
    ----------
    answers : dict
        Answers of :data:`ELPHON_ITEMS`.
    context : Context
        The session context (``out`` override: the system folder).

    Returns
    -------
    dict
    """
    cfg: dict = {'workdir': context.workdir}
    cfg['pw_template'] = answers['pw_template']
    cfg['pw_template_text'] = template_text(answers, context)
    cfg['prefix'] = answers['prefix']
    cfg['name'] = answers['name']
    cfg['out'] = output_folder(context)
    cfg['pseudo_dir'] = answers['pseudo_dir']
    cfg['coupling_source'] = answers['coupling_source']
    if answers['coupling_source'] == 'fd':
        _finite_differences(cfg, answers)
    else:
        kg, qg = answers['kgrid_axis'], answers['qgrid_axis']
        cfg['kgrid'], cfg['qgrid'] = [kg] * 3, [qg] * 3
    cfg['irreducible'] = answers['irreducible'] if answers['coupling_source'] == 'dfpt' else False
    cfg['nbnd'] = answers['nbnd']
    cfg['exclude_bands'] = answers['exclude_bands'] if answers['coupling_source'] == 'epw' else []
    cfg['masses_amu'] = answers['masses_amu']
    cfg['nelec'] = answers['nelec']
    cfg['metal'] = answers['metal']
    cfg['basisdir'] = answers['basisdir']
    cfg['configuration'] = 'standard'
    cfg['pthr'] = answers['pthr']
    cfg['spin_degeneracy'] = None
    cfg['long_range'] = True
    cfg['quadrupoles'] = None
    cfg['cache_dir'] = 'cache'
    cfg['workflows'] = list(answers['workflows'])
    cfg['optics_nscf'] = False
    if 'superconductivity' in cfg['workflows']:
        cfg['superconductivity'] = _superconductivity(answers)
    if 'optics' in cfg['workflows']:
        if _emissivity(answers):
            cfg['optics_nscf'] = answers['optics_direct']
        cfg['optics'] = _optics(answers)
    if 'transport' in cfg['workflows']:
        if cfg['metal']:
            _resistivity(cfg, answers)
        else:
            cfg['transport'] = _mobility(answers)
    if 'arpes' in cfg['workflows']:
        cfg['arpes'] = _arpes(cfg, answers)
    if 'thermal' in cfg['workflows']:
        cfg['thermal'] = _thermal(cfg, answers)
    if 'phonons' in cfg['workflows']:
        cfg['phonons'] = dict(units='cm-1', mesh=(24, 24, 24), sigma=3.0)
    if 'coupling' in cfg['workflows']:
        cfg['coupling'] = dict(nk=24, direction=(1.0, 0.0, 0.0))
    if cfg['optics_nscf']:
        cfg['optics_kgrid'] = [answers['optics_kgrid_axis']] * 3
        cfg['optics_nbnd'] = answers['optics_nbnd']
    if 'tr2_ph' in answers:  # GUI-only, when changed
        cfg['tr2_ph'] = answers['tr2_ph']
    return cfg


ELPHON = Workflow(
    name='elphon',
    title='Electron–phonon system',
    description='A system folder: system.py, the QE inputs (scf, ph.x, nscf) and the drivers of '
    'the selected workflows (superconductivity, optics, transport, ARPES, thermal, ...).',
    items=ELPHON_ITEMS,
    assemble=assemble_elphon,
    overrides=('prefix', 'out'),
)
