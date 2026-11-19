"""Questions of the spin-susceptibility property: the uniform spin response and the Stoner instability.

``run_spin_response()`` of ``main.py`` builds the PAO Hamiltonian, computes the Stoner integrals of its
shells (:meth:`PAOFLOW.PAOFLOW.PAOFLOW.stoner_parameters`) and the uniform static spin susceptibility
(:meth:`PAOFLOW.PAOFLOW.PAOFLOW.spin_susceptibility`) with up to three methods:

=========  =================================================================================
method     what it is
=========  =================================================================================
bare       the frozen bands: Pauli (and Van Vleck with spin-orbit), 2 N(E_F) per cell
stoner     the on-site LSDA mean field of the PAO shells, chi0 / (1 - I chi0 / 2)
alda       the LSDA spin kernel on the real-space grid of the crystal (spin-degenerate runs
           with the internal projections)
=========  =================================================================================

A negative susceptibility is a Stoner instability; ``spin_ordered`` then seeds the self-consistent
mean field (:meth:`PAOFLOW.PAOFLOW.PAOFLOW.self_consistent_spin`) and prints the ordered moment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .model import Answers, Context, Note, Question

SPIN_RESPONSE = 'Spin susceptibility'

#: The methods of ``spin_susceptibility`` (``alda`` only on spin-degenerate internal-basis runs).
SPIN_METHODS: tuple[str, ...] = ('bare', 'stoner', 'alda')


def spin_methods(text: str) -> tuple[str, ...]:
    """``'bare, stoner'`` -> ``('bare', 'stoner')``.

    Raises
    ------
    ValueError
        For an unknown method or an empty list.
    """
    names = tuple(name.strip() for name in str(text).split(',') if name.strip())
    unknown = [name for name in names if name not in SPIN_METHODS]
    if unknown or not names:
        raise ValueError(f'methods are a comma-separated subset of {", ".join(SPIN_METHODS)}')
    return names


def _default_methods(answers: Answers, context: Context) -> str:
    internal = answers.get('projector', 'internal') != 'projwfc'
    if not internal:
        return 'bare'
    return 'bare, stoner' if answers.get('spin_orbit') else 'bare, stoner, alda'


def spin_response_items(when: Callable[[Answers, Context], bool]) -> tuple:
    """The questions of the property, shown when ``when(answers, context)``."""
    return (
        Note(
            '\nSpin susceptibility (bare, on-site Stoner, LSDA on the grid) and Stoner instability:',
            when=when,
            section=SPIN_RESPONSE,
        ),
        Question(
            'spin_methods',
            '  methods (comma separated: bare, stoner, alda)',
            default=_default_methods,
            parse=spin_methods,
            when=when,
            label='Methods',
            help='bare: frozen bands (Pauli); stoner: on-site LSDA mean field of the PAO shells; '
            'alda: LSDA spin kernel on the crystal grid (spin-degenerate runs, internal '
            'projections). stoner and alda need the internal projections.',
            doc='PAOFLOW.PAOFLOW.PAOFLOW.spin_susceptibility:methods',
            section=SPIN_RESPONSE,
        ),
        Question(
            'spin_nk',
            '  k-grid per direction',
            kind='int',
            default=16,
            when=when,
            label='k-grid per direction',
            help='Uniform grid of the self-consistency; converge it (16-24 for transition metals). '
            'The grid method stores the periodic parts of every k-point: about 10 MB per k-point '
            'for a 4d metal.',
            section=SPIN_RESPONSE,
        ),
        Question(
            'spin_degauss',
            '  smearing width (eV, Marzari-Vanderbilt)',
            kind='float',
            default=0.272,
            unit='eV',
            when=when,
            label='Smearing width',
            help="QE's degauss in Ry times 13.606 (0.02 Ry = 0.272 eV): use the width of the scf "
            'run to compare with a fixed-spin-moment calculation.',
            section=SPIN_RESPONSE,
        ),
        Question(
            'spin_field',
            '  field of the finite differences (T)',
            kind='float',
            default=10.0,
            unit='T',
            when=when,
            label='Field',
            help='Symmetric fields +-B; the response is linear well beyond laboratory fields.',
            section=SPIN_RESPONSE,
        ),
        Question(
            'spin_ordered',
            '  also look for an ordered moment (seeded self-consistent Stoner mean field)?',
            kind='bool',
            default=False,
            when=lambda a, c: when(a, c) and a.get('projector', 'internal') != 'projwfc',
            label='Ordered moment',
            help='For a ferromagnet computed non-magnetic (e.g. bcc Fe): the seeded mean field '
            'finds the moment the Stoner instability leads to.',
            doc='PAOFLOW.PAOFLOW.PAOFLOW.self_consistent_spin:seed',
            section=SPIN_RESPONSE,
        ),
    )


def assemble_spin_response(cfg: dict, answers: Answers) -> dict:
    """Add the ``spin_*`` keys of the property to ``cfg`` (modified in place and returned).

    The on-site and grid methods need the internal projections; ``alda`` also a spin-degenerate
    run: they are dropped otherwise.
    """
    methods = answers.get('spin_methods') or ('bare',)
    if isinstance(methods, str):
        methods = spin_methods(methods)
    internal = cfg.get('projector', 'internal') != 'projwfc'
    keep = [m for m in methods if internal or m == 'bare']
    if cfg.get('spin_orbit'):
        keep = [m for m in keep if m != 'alda']
    cfg['spin_methods'] = tuple(keep) or ('bare',)
    cfg['spin_nk'] = int(answers.get('spin_nk', 16))
    cfg['spin_degauss'] = float(answers.get('spin_degauss', 0.272))
    cfg['spin_field'] = float(answers.get('spin_field', 10.0))
    cfg['spin_ordered'] = bool(answers.get('spin_ordered', False)) and internal
    return cfg


def spin_response_lines(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Constants and ``run_spin_response()`` of ``main.py``.

    Returns
    -------
    tuple of (list of str, list of str)
        The constant lines and the function lines.
    """
    constants = [
        '# Spin susceptibility dM/dh (mu_B per eV per cell, h = g_s mu_B B / 2): bare (Pauli),',
        '# on-site Stoner mean field and the LSDA kernel on the crystal grid. Negative = Stoner',
        '# instability. Converge SPIN_NK; SPIN_DEGAUSS = the scf width to compare with QE',
        '# fixed-spin-moment runs (QE degauss in Ry x 13.606).',
        'SPIN_METHODS = {!r}'.format(tuple(cfg['spin_methods'])),
        'SPIN_NK = ({n}, {n}, {n})'.format(n=int(cfg['spin_nk'])),
        'SPIN_DEGAUSS = {}   # eV, Marzari-Vanderbilt'.format(float(cfg['spin_degauss'])),
        'SPIN_FIELD = {}   # T'.format(float(cfg['spin_field'])),
        'SPIN_ORDERED = {}   # seeded mean field: the ordered moment of a Stoner instability'.format(
            bool(cfg['spin_ordered'])
        ),
        '',
    ]
    stoner = any(m != 'bare' for m in cfg['spin_methods']) or cfg['spin_ordered']
    body = [
        '',
        'def run_spin_response():',
        '    """Uniform spin susceptibility and Stoner instability (STD_BASIS basis).',
        '',
        '    spin_susceptibility() rewrites the Hamiltonian in spinors, so it runs on its own',
        '    PAOFLOW instance, on HRs (before interpolated_hamiltonian).',
        '    """',
        *(['    import numpy as np', ''] if cfg['spin_ordered'] else []),
        '    p = PAOFLOW.PAOFLOW(',
        '        workpath=HERE,',
        '        outputdir=OUTPUTDIR,',
        '        savedir=SAVEDIR,',
        '        smearing=SMEARING,',
        '        npool=NPOOL,',
        '        save_overlaps=SAVE_OVERLAPS,',
        '        verbose=False,',
        '    )',
        '',
    ]
    return constants, body + [
        '__PROJECTIONS__',
        '    p.projectability(pthr=PTHR)',
        '    p.pao_hamiltonian()',
        *(
            [
                '    shells, _ = p.stoner_parameters()  # eV per mu_B, printed'
                if cfg['spin_ordered']
                else '    p.stoner_parameters()  # eV per mu_B, printed'
            ]
            if stoner
            else []
        ),
        '    chi = p.spin_susceptibility(',
        "        field=SPIN_FIELD, nk=SPIN_NK, smearing='marzari-vanderbilt', degauss=SPIN_DEGAUSS,",
        "        methods=SPIN_METHODS, alda_options={'valence_density': 'qe'},",
        '    )',
        '    if RANK == 0:',
        '        for method in SPIN_METHODS:',
        "            print('spin susceptibility {:7s} {: .4f} mu_B/eV'.format(method, chi[method]))",
        *(
            [
                '    if SPIN_ORDERED:',
                '        seed = np.zeros((len(shells), 3))',
                "        seed[[i for i, g in enumerate(shells) if g['l'] == 2], 2] = 0.5",
                '        res = p.self_consistent_spin(',
                "            nk=SPIN_NK, smearing='marzari-vanderbilt', degauss=SPIN_DEGAUSS, seed=seed,",
                '            update_hamiltonian=False, max_iter=600,',
                '        )',
                '        if RANK == 0:',
                "            print('ordered moment {:.4f} mu_B (converged {}, stable {})'.format(",
                "                abs(res['total_moment'][2]), res['converged'], res['stable']))",
            ]
            if cfg['spin_ordered']
            else []
        ),
        '',
        '    p.finish_execution()',
        '',
    ]
