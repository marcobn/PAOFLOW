"""Questions of the ``acbn0`` workflow: self-consistent Hubbard U and intersite V (ACBN0 / eACBN0)."""

from __future__ import annotations

from .common import COMMON_ITEMS, COMMON_OVERRIDES, assemble_common
from .model import Answers, Context, Note, Question, Workflow, number

HUBBARD = 'Hubbard manifolds'
SCF = 'Self-consistency'
LAUNCH = 'Launch commands'


def manifold_row(line: str) -> tuple[str, float]:
    """One Hubbard manifold ``'<species>-<nl> <initU>'`` (initial U 0.5 eV when missing).

    Parameters
    ----------
    line : str
        The entered line, e.g. ``'Si-3p 0.5'``.

    Returns
    -------
    tuple of (str, float)
        The manifold label and the initial U (eV).
    """
    parts = line.split()
    orbital = parts[0].strip().strip('\'"')
    try:
        value = float(parts[1]) if len(parts) > 1 else 0.5
    except ValueError:
        value = 0.5
    return (orbital, value)


def _intersite(answers: Answers) -> bool:
    return bool(answers.get('use_intersite_v'))


def _default_v_cutoff(answers: Answers, context: Context) -> str:
    from ..paoflow_driver import detect_v_cutoff

    cutoff = context.cached('v_cutoff', lambda: detect_v_cutoff(context.workdir))
    return repr(cutoff or 2.6)


ACBN0_ITEMS = (
    Question(
        'use_intersite_v',
        'Include intersite V (eACBN0)? (No = on-site U only, ACBN0)',
        kind='bool',
        default=False,
        label='Intersite V (eACBN0)',
        section=SCF,
    ),
    Note(
        '\nNote: intersite V (eACBN0) requires the true non-orthogonal atomic\n'
        '  overlap S(k), which only the projwfc.x projector provides. The\n'
        '  internal ortho-atomic projection dumps S(k)=I and gives wrong V.\n'
        '  Switching the projector to projwfc for this workflow.',
        when=lambda a, c: _intersite(a) and a.get('projector', 'internal') != 'projwfc',
        section=SCF,
    ),
    Question('projection', 'Projection scheme', default='ortho-atomic', section=SCF),
    Question(
        'conv_thr',
        'U/V convergence threshold (eV)',
        default='0.05',
        parse=number,
        unit='eV',
        section=SCF,
    ),
    Question('nk', 'Band path k-points (NK)', kind='int', default=400, section=SCF),
    Question(
        'gaussian_threshold',
        'Gaussian-fit tolerance (loosen for ONCV/SOC pseudos)',
        default='0.05',
        parse=number,
        section=SCF,
    ),
    Note(
        "\nEnter the Hubbard manifolds, one per line, as '<species>-<nl> <initU>'\n"
        "  e.g. 'Si-3p 0.5'.  Leave blank to finish.",
        section=HUBBARD,
    ),
    Question(
        'hubbard',
        'Hubbard manifolds',
        kind='rows',
        row_prompt='  manifold: ',
        parse=manifold_row,
        abort_if_empty='No Hubbard manifolds entered; nothing to generate.',
        columns=('manifold (e.g. Si-3p)', 'initial U (eV)'),
        section=HUBBARD,
    ),
    Question(
        'v_cutoff',
        'Intersite V neighbour cutoff (Angstrom)',
        default=_default_v_cutoff,
        parse=number,
        when=lambda a, c: _intersite(a),
        unit='Å',
        section=HUBBARD,
    ),
    Question(
        'v_init',
        'Initial intersite V (eV)',
        default='0.5',
        parse=number,
        when=lambda a, c: _intersite(a),
        unit='eV',
        section=HUBBARD,
    ),
    Note('\nParallel launch commands (edit later in the script if unsure):', section=LAUNCH),
    Question('mpi_qe', 'MPI command for pw.x', default='mpirun -np 4', section=LAUNCH),
    Question('mpi_py', 'MPI command for python', default='mpirun -np 1', section=LAUNCH),
    Question(
        'mpi_hartree', 'MPI command for the Hartree step', default='mpirun -np 4', section=LAUNCH
    ),
    Question('qe_path', 'Quantum ESPRESSO bin path', default='', section=LAUNCH),
    Question('py_path', 'Python bin path', default='', section=LAUNCH),
    Question(
        'emin',
        'Lowest energy of the band plots (eV, relative to E_F)',
        kind='float',
        default=-8.0,
        unit='eV',
        section='Plots',
        level='advanced',
        cli=False,
    ),
    Question(
        'emax',
        'Highest energy of the band plots (eV, relative to E_F)',
        kind='float',
        default=4.0,
        unit='eV',
        section='Plots',
        level='advanced',
        cli=False,
    ),
)


def assemble_acbn0_from_common(common: dict, answers: Answers) -> dict:
    """Configuration of the ACBN0 driver on top of the common configuration.

    Parameters
    ----------
    common : dict
        The common configuration.
    answers : dict
        Answers of :data:`ACBN0_ITEMS`.

    Returns
    -------
    dict
    """
    cfg = dict(common)
    cfg['use_intersite_v'] = answers['use_intersite_v']
    if cfg['use_intersite_v'] and cfg.get('projector', 'internal') != 'projwfc':
        cfg['projector'] = 'projwfc'
    cfg['projection'] = answers['projection']
    cfg['conv_thr'] = answers['conv_thr']
    cfg['nk'] = answers['nk']
    cfg['gaussian_threshold'] = answers['gaussian_threshold']
    cfg['hubbard'] = [tuple(row) for row in answers['hubbard']]
    if cfg['use_intersite_v']:
        cfg['v_cutoff'] = answers['v_cutoff']
        cfg['v_init'] = answers['v_init']
    for key in ('mpi_qe', 'mpi_py', 'mpi_hartree', 'qe_path', 'py_path'):
        cfg[key] = answers[key]
    for key in ('emin', 'emax'):  # GUI-only, when changed
        if key in answers:
            cfg[key] = answers[key]
    return cfg


def assemble_acbn0(answers: Answers, context: Context) -> dict:
    """Configuration of the ACBN0 driver (:func:`PAOFLOW.gen.paoflow_driver.build_acbn0_script`).

    Parameters
    ----------
    answers : dict
        Answers of :data:`COMMON_ITEMS` and :data:`ACBN0_ITEMS`.
    context : Context
        The session context.

    Returns
    -------
    dict
    """
    return assemble_acbn0_from_common(assemble_common(answers, context), answers)


ACBN0 = Workflow(
    name='acbn0',
    title='ACBN0 / eACBN0',
    description='Self-consistent Hubbard U (and intersite V) with pw.x in the loop '
    '(main.acbn0.py and plot.acbn0.py).',
    items=COMMON_ITEMS + ACBN0_ITEMS,
    assemble=assemble_acbn0,
    overrides=COMMON_OVERRIDES,
)
