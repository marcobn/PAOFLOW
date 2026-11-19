"""Questions of the Landauer workflow: coherent transport with :class:`PAOFLOW.Transport.Transport`.

From the QE save of a crystal or a supercell, ``main.landauer.py`` builds the PAO Hamiltonian and computes the
transmission and the density of states of the conductor:

- ``bulk``: the cell is the principal layer of a perfect crystal along the transport direction; its transmission is
  the number of channels;
- ``conductor``: the supercell is the conductor, and its first and last atomic layers are the principal layers of
  the two leads (they must be bulk-like).

Optionally the current from the zero-bias transmission (rigid shift of the leads) and, for a conductor, the
self-consistent finite-bias current (non-equilibrium density and Poisson equation, planar or three-dimensional;
:meth:`PAOFLOW.Transport.Transport.compute_current_self_consistent`). The script builder is
:func:`PAOFLOW.gen.landauer_driver.build_landauer_script`.
"""

from __future__ import annotations

from .common import COMMON_ITEMS, COMMON_OVERRIDES, assemble_common
from .model import Answers, Context, Note, Question, Workflow, optional_number

CONDUCTOR = 'Conductor'
GRID = 'Energy grid'
CURRENT = 'Current'
FINITE_BIAS = 'Finite bias'
TRANSPORT = 'PAOFLOW.Transport.Transport.'


def parse_transverse_mesh(text: str) -> tuple[int, int]:
    """The transverse k mesh: two integers, blank or ``0 0`` for the DFT mesh.

    Parameters
    ----------
    text : str

    Returns
    -------
    tuple of int

    Raises
    ------
    ValueError
        Unless two non-negative integers are given.
    """
    items = [item for item in str(text).replace(',', ' ').split() if item]
    if not items:
        return (0, 0)
    if len(items) != 2:
        raise ValueError('give two integers (the transverse mesh), or leave blank for the DFT mesh')
    mesh = tuple(int(item) for item in items)
    if min(mesh) < 0:
        raise ValueError('the mesh must be non-negative')
    return mesh


def parse_biases(text: str) -> list[float]:
    """Biases of the self-consistent scan, separated by commas or spaces (V).

    Parameters
    ----------
    text : str

    Returns
    -------
    list of float

    Raises
    ------
    ValueError
        Without a bias.
    """
    values = [float(item) for item in str(text).replace(',', ' ').split() if item]
    if not values:
        raise ValueError('give at least one bias, e.g. 0 0.1 0.25 0.5')
    return values


def parse_charge_widths(text: str) -> dict[str, float] | None:
    """Gaussian widths per species, ``'Au 1.9, S 1.1'`` (bohr); blank for the pseudopotential values.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict or None

    Raises
    ------
    ValueError
        For an entry that is not ``<species> <width>`` or a non-positive width.
    """
    entries = [entry.split() for entry in str(text).split(',') if entry.strip()]
    if not entries:
        return None
    widths = {}
    for entry in entries:
        if len(entry) != 2:
            raise ValueError('write <species> <width in bohr>, separated by commas: Au 1.9, S 1.1')
        widths[entry[0]] = float(entry[1])
        if widths[entry[0]] <= 0.0:
            raise ValueError('the widths must be positive')
    return widths


def _conductor(answers: Answers) -> bool:
    return answers.get('calculation') == 'conductor'


def _finite_bias(answers: Answers) -> bool:
    return _conductor(answers) and bool(answers.get('finite_bias'))


def _three_d(answers: Answers) -> bool:
    return _finite_bias(answers) and answers.get('electrostatics') == '3d'


LANDAUER_ITEMS: tuple = (
    Question(
        'calculation',
        'Calculation (bulk: transmission of the perfect crystal; conductor: the supercell is the '
        'conductor, its two ends are the leads)',
        kind='choice',
        choices=lambda a, c: ['bulk', 'conductor'],
        default='bulk',
        label='Calculation',
        help='bulk: the cell is the principal layer of the crystal along the transport direction, and '
        'T(E) counts its channels. conductor: interfaces, junctions, defects, molecules; the first and '
        'last atomic layers of the supercell are the principal layers of the two leads and must be '
        'bulk-like (both leads take the block of the first ones).',
        doc=TRANSPORT + 'build_hamiltonian_blocks:calculation_type',
        section=CONDUCTOR,
    ),
    Question(
        'transport_direction',
        'Transport direction',
        kind='choice',
        choices=lambda a, c: ['x', 'y', 'z'],
        default='z',
        doc=TRANSPORT + 'define_partition:transport_direction',
        section=CONDUCTOR,
    ),
    Question(
        'lead_layers',
        'Atomic layers of each lead at the ends of the supercell',
        kind='int',
        default=4,
        when=lambda a, c: _conductor(a),
        validate=lambda v, a, c: None if v >= 1 else '  give at least one layer.',
        doc=TRANSPORT + 'define_partition:left_lead_layers',
        section=CONDUCTOR,
    ),
    Question(
        'basis',
        'Basis of the internal projections (minimal: the orbitals of the pseudopotentials)',
        kind='choice',
        choices=lambda a, c: ['minimal', 'standard', 'extended'],
        default='standard',
        when=lambda a, c: a.get('projector', 'internal') != 'projwfc',
        label='Basis',
        help='The Green functions scale as the cube of the number of orbitals: minimal is the '
        'cheapest, extended the most accurate above the Fermi level.',
        section=CONDUCTOR,
        level='advanced',
    ),
    Question(
        'principal_layers',
        'Atomic layers of the principal layer from the start of the cell (0: the whole cell)',
        kind='int',
        default=0,
        when=lambda a, c: not _conductor(a),
        validate=lambda v, a, c: None if v >= 0 else '  0 or more.',
        label='Principal layer',
        help='A cell longer than the principal layer (a lead cut from a junction): the first N atomic '
        'layers along the transport direction are the repeated unit.',
        doc=TRANSPORT + 'define_partition:central_layers',
        section=CONDUCTOR,
        level='advanced',
    ),
    Question(
        'doublings',
        'Doublings of the cell along the transport direction (0: none)',
        kind='int',
        default=0,
        validate=lambda v, a, c: None if v >= 0 else '  0 or more.',
        help='A principal layer must contain every hopping to the next one: for diffuse orbitals or a '
        'short cell, double it (doubling_Hamiltonian along the transport direction).',
        section=CONDUCTOR,
        level='advanced',
    ),
    Question(
        'transverse_nk',
        'Transverse k mesh (two integers; blank: the DFT mesh)',
        default='',
        parse=parse_transverse_mesh,
        label='Transverse k mesh',
        help='k-points across the transport direction; 1 1 for a wire or a molecule in vacuum.',
        doc=TRANSPORT + 'configure_energy_grid:nk',
        section=CONDUCTOR,
    ),
    Question(
        'emin',
        'Lowest energy',
        kind='float',
        default=-3.0,
        unit='eV',
        help='Energies relative to the Fermi level.',
        doc=TRANSPORT + 'configure_energy_grid:emin',
        section=GRID,
    ),
    Question(
        'emax',
        'Highest energy',
        kind='float',
        default=3.0,
        unit='eV',
        doc=TRANSPORT + 'configure_energy_grid:emax',
        section=GRID,
    ),
    Question(
        'ne',
        'Number of energies',
        kind='int',
        default=601,
        doc=TRANSPORT + 'configure_energy_grid:ne',
        section=GRID,
    ),
    Question(
        'delta',
        'Broadening of the Green functions',
        kind='float',
        default=1.0e-3,
        unit='eV',
        doc=TRANSPORT + 'configure_energy_grid:delta',
        section=GRID,
        level='advanced',
    ),
    Question(
        'eigenchannels',
        'Transmission eigenchannels kept (0: none)',
        kind='int',
        default=0,
        when=lambda a, c: _conductor(a),
        validate=lambda v, a, c: None if v >= 0 else '  0 or more.',
        label='Eigenchannels',
        help='The eigenvalues of t†t at each energy: how many channels carry the transmission.',
        doc=TRANSPORT + 'configure_eigenchannels:neigchnx',
        section=GRID,
        level='advanced',
    ),
    Question(
        'eigenchannel_energy',
        'Energy of the eigenchannel wavefunctions (blank: none)',
        default='',
        parse=optional_number,
        when=lambda a, c: _conductor(a) and bool(a.get('eigenchannels')),
        unit='eV',
        label='Eigenchannel plot energy',
        help='Writes the eigenchannels at the grid energy closest to this one (first transverse k).',
        doc=TRANSPORT + 'configure_eigenchannels:ie_eigplot',
        section=GRID,
        level='advanced',
    ),
    Question(
        'current',
        'Current from the zero-bias transmission (rigid shift of the leads)?',
        kind='bool',
        default=True,
        label='Current (rigid shift)',
        help='I(V) = ∫T(E)[f_L − f_R] in units of 2e/h·eV per transverse cell, positive for μ_L > μ_R; '
        'exact for the linear conductance.',
        section=CURRENT,
    ),
    Question(
        'bias_max',
        'Largest bias of the I(V) curve (symmetric scan)',
        kind='float',
        default=1.0,
        unit='V',
        when=lambda a, c: bool(a.get('current')),
        section=CURRENT,
    ),
    Question(
        'nbias',
        'Number of biases',
        kind='int',
        default=41,
        when=lambda a, c: bool(a.get('current')),
        section=CURRENT,
        level='advanced',
    ),
    Question(
        'temperature',
        'Electron temperature of the leads',
        kind='float',
        default=300.0,
        unit='K',
        when=lambda a, c: bool(a.get('current')) or _finite_bias(a),
        doc=TRANSPORT + 'configure_bias_self_consistency:temperature',
        section=CURRENT,
    ),
    Question(
        'finite_bias',
        'Self-consistent finite-bias current (non-equilibrium density + Poisson)?',
        kind='bool',
        default=False,
        when=lambda a, c: _conductor(a),
        label='Finite bias, self-consistent',
        help='At every bias the leads shift to ±V/2 and the conductor gets the Hartree potential of the '
        'bias-induced charge, iterated to self-consistency; T(E; V) gives the current.',
        section=FINITE_BIAS,
    ),
    Question(
        'bias_values',
        'Biases of the self-consistent scan',
        default='0, 0.1, 0.25, 0.5',
        parse=parse_biases,
        when=lambda a, c: _finite_bias(a),
        unit='V',
        help='Solved in order, each from the previous one.',
        doc=TRANSPORT + 'compute_current_self_consistent:bias',
        section=FINITE_BIAS,
    ),
    Question(
        'electrostatics',
        'Electrostatics (planar: junctions uniform across the cell; 3d: molecules, wires)',
        kind='choice',
        choices=lambda a, c: ['planar', '3d'],
        default='planar',
        when=lambda a, c: _finite_bias(a),
        doc=TRANSPORT + 'configure_bias_self_consistency:electrostatics',
        section=FINITE_BIAS,
    ),
    Question(
        'charge_widths',
        'Gaussian widths of the atomic charges (e.g. Au 1.9, S 1.1; blank: from the pseudopotentials)',
        default='',
        parse=parse_charge_widths,
        when=lambda a, c: _three_d(a),
        unit='bohr',
        doc=TRANSPORT + 'configure_bias_self_consistency:charge_widths',
        section=FINITE_BIAS,
        level='advanced',
    ),
    Question(
        'potential_grid',
        'Write U(r) on a grid (XSF) at the largest bias?',
        kind='bool',
        default=True,
        when=lambda a, c: _three_d(a),
        section=FINITE_BIAS,
    ),
    Question(
        'linear_ramp',
        'Also the linear ramp between the electrodes (no self-consistency), for comparison?',
        kind='bool',
        default=False,
        when=lambda a, c: _finite_bias(a),
        section=FINITE_BIAS,
        level='advanced',
    ),
    Question(
        'window_spacing',
        'Energy step in the bias window',
        kind='float',
        default=0.005,
        unit='eV',
        when=lambda a, c: _finite_bias(a),
        doc=TRANSPORT + 'configure_bias_self_consistency:window_spacing',
        section=FINITE_BIAS,
        level='advanced',
    ),
    Question(
        'mixing',
        'Anderson mixing',
        kind='float',
        default=0.5,
        when=lambda a, c: _finite_bias(a),
        validate=lambda v, a, c: None if 0 < v <= 1 else '  between 0 and 1.',
        doc=TRANSPORT + 'configure_bias_self_consistency:mixing',
        section=FINITE_BIAS,
        level='advanced',
    ),
    Question(
        'tolerance',
        'Convergence threshold on the potential',
        kind='float',
        default=1.0e-3,
        unit='eV',
        when=lambda a, c: _finite_bias(a),
        doc=TRANSPORT + 'configure_bias_self_consistency:tolerance',
        section=FINITE_BIAS,
        level='advanced',
    ),
    Question(
        'max_iterations',
        'Iteration limit per bias',
        kind='int',
        default=80,
        when=lambda a, c: _finite_bias(a),
        doc=TRANSPORT + 'configure_bias_self_consistency:max_iterations',
        section=FINITE_BIAS,
        level='advanced',
    ),
    Note(
        '  The zero-bias PAO Hamiltonian is the reference: only the bias-induced charge acts (Hartree; '
        'no exchange-correlation response).',
        when=lambda a, c: _finite_bias(a),
        section=FINITE_BIAS,
    ),
    Question(
        'plot',
        'Also generate a plotting script (plot.landauer.py)?',
        kind='bool',
        default=True,
        section='Output',
        gui=False,  # the Studio: "Write the plot scripts" of its Output box (options.plot)
    ),
)


def assemble_landauer(answers: Answers, context: Context) -> dict:
    """Configuration of the Landauer workflow (:func:`PAOFLOW.gen.landauer_driver.build_landauer_script`).

    Parameters
    ----------
    answers : dict
        Answers of :data:`COMMON_ITEMS` and :data:`LANDAUER_ITEMS`.
    context : Context
        The session context.

    Returns
    -------
    dict
    """
    cfg = assemble_common(answers, context)
    conductor = answers['calculation'] == 'conductor'
    cfg['calculation'] = answers['calculation']
    cfg['transport_direction'] = answers['transport_direction']
    cfg['lead_layers'] = int(answers['lead_layers']) if conductor else 0
    cfg['basis'] = answers.get('basis', 'standard')
    cfg['principal_layers'] = 0 if conductor else int(answers.get('principal_layers', 0))
    cfg['doublings'] = int(answers.get('doublings', 0))
    cfg['transverse_nk'] = tuple(answers.get('transverse_nk') or (0, 0))
    for key in ('emin', 'emax', 'delta'):
        cfg[key] = float(answers[key])
    cfg['ne'] = int(answers['ne'])
    cfg['eigenchannels'] = int(answers.get('eigenchannels', 0)) if conductor else 0
    cfg['eigenchannel_energy'] = (
        answers.get('eigenchannel_energy') if cfg['eigenchannels'] else None
    )
    cfg['current'] = bool(answers['current'])
    cfg['bias_max'] = float(answers.get('bias_max', 1.0))
    cfg['nbias'] = int(answers.get('nbias', 41))
    cfg['temperature'] = float(answers.get('temperature', 300.0))
    finite_bias = conductor and bool(answers.get('finite_bias'))
    cfg['finite_bias'] = finite_bias
    cfg['bias_values'] = list(answers.get('bias_values') or [0.0, 0.1, 0.25, 0.5])
    cfg['electrostatics'] = answers.get('electrostatics', 'planar') if finite_bias else 'planar'
    three_d = cfg['electrostatics'] == '3d'
    cfg['charge_widths'] = answers.get('charge_widths') if three_d else None
    cfg['potential_grid'] = bool(answers.get('potential_grid', True)) and three_d
    cfg['linear_ramp'] = bool(answers.get('linear_ramp', False)) and finite_bias
    cfg['window_spacing'] = float(answers.get('window_spacing', 0.005))
    cfg['mixing'] = float(answers.get('mixing', 0.5))
    cfg['tolerance'] = float(answers.get('tolerance', 1.0e-3))
    cfg['max_iterations'] = int(answers.get('max_iterations', 80))
    cfg['plot'] = bool(answers.get('plot', True))
    return cfg


LANDAUER = Workflow(
    name='landauer',
    title='Landauer transport',
    description='Transmission, conductor DOS and current of a crystal or a supercell junction, with the '
    'self-consistent finite-bias current (main.landauer.py).',
    items=COMMON_ITEMS + LANDAUER_ITEMS,
    assemble=assemble_landauer,
    overrides=COMMON_OVERRIDES,
)
