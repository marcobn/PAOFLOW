"""Questions of the ``run`` workflow: the property menu, the grids and the non-linear optics block."""

from __future__ import annotations

from .common import COMMON_ITEMS, COMMON_OVERRIDES, assemble_common, from_save
from .model import Answers, Context, Note, Question, Workflow, gated

PROPERTIES = 'Properties'
GRIDS = 'Grids and run settings'
BAND_PATH = 'Band path'
NONLINEAR = 'Non-linear optics'
COUPLED = 'Coupled optical response'


def _properties(answers: Answers) -> list[str]:
    return answers.get('properties') or []


def _needs_energy_window(answers: Answers, context: Context) -> bool:
    from ..paoflow_driver import ENERGY_WINDOW_PROPERTIES

    return any(p in ENERGY_WINDOW_PROPERTIES for p in _properties(answers))


def _property_choices(answers: Answers, context: Context) -> list[tuple[str, str]]:
    from ..paoflow_driver import PROPERTY_MENU

    return list(PROPERTY_MENU)


def _path_defaults(answers: Answers, context: Context) -> tuple[str, list]:
    """Default band path of the answered save, ibrav and dimensionality (cached)."""
    from ..band_paths import default_band_path

    savedir = answers.get('savedir')
    ibrav = int(answers.get('ibrav') or 0)
    is_2d = bool(answers.get('is_2d'))
    return context.cached(
        'band-path:%s:%d:%s' % (savedir, ibrav, is_2d),
        lambda: default_band_path(context.workdir, savedir, ibrav, is_2d),
    )


def point_row(line: str) -> tuple[str, float, float, float]:
    """One high-symmetry point: ``'X 0 0.5 0.5'`` -> ``('X', 0.0, 0.5, 0.5)``.

    Parameters
    ----------
    line : str
        Label and three crystal coordinates.

    Returns
    -------
    tuple of (str, float, float, float)

    Raises
    ------
    ValueError
        Without a label and three numbers.
    """
    parts = line.split()
    if len(parts) != 4:
        raise ValueError('a point needs a label and three crystal coordinates')
    try:
        return (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        raise ValueError('coordinates of %s are not numbers' % parts[0]) from None


def _validate_path(value: str, answers: Answers, context: Context) -> str | None:
    from ..band_paths import check_band_path

    points = answers.get('high_sym') or _path_defaults(answers, context)[1]
    return check_band_path(value, points)


def _internal(answers: Answers) -> bool:
    return answers.get('projector', 'internal') != 'projwfc'


def _molecule(answers: Answers) -> bool:
    return bool(answers.get('nlo_molecule', False))


def _components(text: str) -> list[str] | str:
    return [c.strip() for c in text.split(',')] if ',' in text else text


def _tuple_of_names(text: str) -> tuple[str, ...]:
    return tuple(r.strip() for r in text.split(',') if r.strip())


def _energies(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(',') if x.strip()]
    except ValueError:
        raise ValueError('not a comma-separated list of energies: %r' % text) from None


#: The non-linear optics block (``collect_nonlinear``); it reads ``is_2d`` and ``projector``.
NONLINEAR_ITEMS = (
    Note('\nNon-linear optics (shift / injection currents, SHG):', section=NONLINEAR),
    Question(
        'nlo_molecule',
        '  isolated molecule in a box (Gamma point, beta tensor)?',
        kind='bool',
        default=False,
        when=lambda a, c: not a.get('is_2d'),
        label='Isolated molecule in a box (Γ point, β tensor)',
        section=NONLINEAR,
    ),
    Question(
        'nlo_nk',
        '  dense k-grid per direction',
        kind='int',
        default=40,
        when=lambda a, c: not _molecule(a),
        section=NONLINEAR,
    ),
    Question(
        'nlo_emin',
        '  smallest photon energy (eV)',
        kind='float',
        default=0.0,
        unit='eV',
        section=NONLINEAR,
    ),
    Question(
        'nlo_emax',
        '  largest photon energy (eV)',
        kind='float',
        default=4.0,
        unit='eV',
        section=NONLINEAR,
    ),
    Question('nlo_ne', '  number of photon energies', kind='int', default=401, section=NONLINEAR),
    Question(
        'nlo_eta',
        '  broadening eta (eV)',
        kind='float',
        default=0.05,
        unit='eV',
        section=NONLINEAR,
    ),
    Question(
        'nlo_scissor',
        '  scissor shift of the conduction bands (eV)',
        kind='float',
        default=0.0,
        unit='eV',
        section=NONLINEAR,
    ),
    Question(
        'nlo_components',
        "  tensor components ('all', 'diag' or e.g. xyz,yyy)",
        default='all',
        parse=_components,
        when=lambda a, c: not _molecule(a),
        section=NONLINEAR,
    ),
    Question(
        'nlo_responses',
        '  responses (any of shift,injection,shg)',
        default='shift,injection,shg',
        parse=_tuple_of_names,
        section=NONLINEAR,
    ),
    Question(
        'nlo_dipoles',
        '  Löwdin position dipoles (recommended: basis-independent results)?',
        kind='bool',
        default=True,
        when=lambda a, c: _internal(a),
        section=NONLINEAR,
    ),
    Question(
        'nlo_shg_terms',
        '  split the SHG into interband / intraband terms at w and 2w?',
        kind='bool',
        default=False,
        section=NONLINEAR,
    ),
    Question(
        'nlo_coupling',
        '  coupled response of the molecule (none, monopole, hartree, hartree+alda)',
        default='hartree+alda',
        when=lambda a, c: _molecule(a),
        section=NONLINEAR,
    ),
    Question(
        'nlo_coupled',
        '  coupled optical response (local fields, ALDA, long-range kernels)?',
        kind='bool',
        default=False,
        when=lambda a, c: _internal(a) and not _molecule(a) and not a.get('is_2d'),
        section=COUPLED,
    ),
    *gated(
        (
            Question(
                'coupled_nk',
                '    coupled-response k-grid per direction',
                kind='int',
                default=16,
                section=COUPLED,
            ),
            Question(
                'coupled_kernels',
                '    kernel levels (none,hartree,hartree+alda,hartree+alda+lrc,hartree+alda+bootstrap)',
                default='none,hartree,hartree+alda,hartree+alda+bootstrap',
                parse=_tuple_of_names,
                section=COUPLED,
            ),
            Question(
                'coupled_emax',
                '    largest photon energy of the dielectric function (eV)',
                kind='float',
                default=5.0,
                unit='eV',
                section=COUPLED,
            ),
            Question(
                'coupled_ne',
                '    number of photon energies',
                kind='int',
                default=51,
                section=COUPLED,
            ),
            Question(
                'coupled_shg_energies',
                '    SHG photon energies (eV, comma separated; blank skips the coupled SHG)',
                default='0.1',
                parse=_energies,
                unit='eV',
                section=COUPLED,
            ),
            Question(
                'coupled_eta',
                '    broadening eta (eV)',
                kind='float',
                default=0.1,
                unit='eV',
                section=COUPLED,
            ),
            Question(
                'coupled_emin',
                'Lowest photon energy of the coupled dielectric function (eV)',
                kind='float',
                default=0.0,
                unit='eV',
                section=COUPLED,
                level='advanced',
                cli=False,
            ),
            Question(
                'coupled_components',
                'Tensor components of the coupled SHG (comma separated)',
                default='xyz',
                parse=lambda text: [c.strip() for c in text.split(',') if c.strip()],
                section=COUPLED,
                level='advanced',
                cli=False,
            ),
        ),
        lambda a, c: bool(a.get('nlo_coupled')),
    ),
)

RUN_ITEMS = (
    Question(
        'properties',
        'Available properties:',
        kind='menu',
        choices=_property_choices,
        abort_if_empty='No valid properties selected; nothing to generate.',
        label='Properties',
        section=PROPERTIES,
    ),
    Question(
        'std_basis',
        '\nBasis configuration for the standard properties:',
        kind='choice',
        choices=['minimal', 'standard', 'extended'],
        default='standard',
        when=lambda a, c: any(p != 'optical' for p in _properties(a)),
        label='Basis of the standard properties',
        help='The optical and non-linear runs always use the extended basis.',
        section=PROPERTIES,
    ),
    Question('npool', 'npool', kind='int', default=1, section=GRIDS),
    Question('smearing', 'Smearing (gauss / m-p / None)', default='gauss', section=GRIDS),
    Question(
        'spin_orbit',
        'Spin-orbit (noncollinear) calculation?',
        kind='bool',
        default=from_save('spinorbit', False),
        help='Default from the save (lspinorb).',
        section=GRIDS,
    ),
    Question(
        'nk',
        'Band path k-points (NK)',
        kind='int',
        default=400,
        when=lambda a, c: 'bands' in _properties(a),
        section=GRIDS,
        doc='PAOFLOW.PAOFLOW.PAOFLOW.bands:nk',
    ),
    Question(
        'high_sym',
        'High-symmetry points (label and crystal coordinates)',
        kind='rows',
        default=lambda a, c: _path_defaults(a, c)[1],
        parse=point_row,
        when=lambda a, c: 'bands' in _properties(a),
        columns=('label', 'k1', 'k2', 'k3'),
        label='High-symmetry points',
        help='Crystal coordinates (units of the reciprocal vectors). The defaults are the points '
        'PAOFLOW uses for this ibrav; edit, add or remove rows for a custom path.',
        section=BAND_PATH,
        level='advanced',
        cli=False,
    ),
    Question(
        'band_path',
        'Band path (labels joined by "-", segments split by "|")',
        default=lambda a, c: _path_defaults(a, c)[0],
        when=lambda a, c: 'bands' in _properties(a),
        validate=_validate_path,
        label='Band path',
        help='e.g. gG-X-W-K-gG-L|U-X; gG is Γ. The points are in the advanced table below.',
        section=BAND_PATH,
        widget='bandpath',
        cli=False,
    ),
    Note(
        '\nEnergy grid for DOS / transport / Hall properties (eV, relative to E_F).\n'
        '  The full PAO band range is computed at run time and printed as a suggestion;\n'
        '  the values below are what the calculation actually uses.',
        when=_needs_energy_window,
        section=GRIDS,
    ),
    Question(
        'emin',
        '  EMIN',
        kind='float',
        default=-8.0,
        when=_needs_energy_window,
        unit='eV',
        section=GRIDS,
    ),
    Question(
        'emax',
        '  EMAX',
        kind='float',
        default=4.0,
        when=_needs_energy_window,
        unit='eV',
        section=GRIDS,
    ),
    Question(
        'ne',
        '  NE (number of energy points)',
        kind='int',
        default=1000,
        when=_needs_energy_window,
        section=GRIDS,
    ),
    Question(
        'do_pdos',
        'Projected DOS as well?',
        kind='bool',
        default=True,
        when=lambda a, c: 'dos' in _properties(a),
        section=GRIDS,
    ),
    Question(
        'interpolate',
        'Use double-grid interpolation (denser FFT)?',
        kind='bool',
        default=lambda a, c: 'optical' in _properties(a),
        section=GRIDS,
    ),
    Question(
        'nfft',
        'FFT grid size per direction',
        kind='int',
        default=24,
        when=lambda a, c: bool(a.get('interpolate')),
        section=GRIDS,
    ),
    *gated(NONLINEAR_ITEMS, lambda a, c: 'nonlinear' in _properties(a)),
    Question(
        'plot',
        'Also generate a plotting script (plot.py)?',
        kind='bool',
        default=True,
        section='Output',
    ),
)


def assemble_nonlinear(cfg: dict, answers: Answers) -> dict:
    """Add the ``nlo_*`` and ``coupled_*`` keys of the non-linear block to ``cfg``.

    Parameters
    ----------
    cfg : dict
        Run configuration (``is_2d`` and ``projector`` set); modified in place.
    answers : dict
        Answers of :data:`NONLINEAR_ITEMS`.

    Returns
    -------
    dict
        ``cfg``.
    """
    molecule = _molecule(answers)
    cfg['nlo_molecule'] = molecule
    if not molecule:
        cfg['nlo_nk'] = answers['nlo_nk']
    for key in ('nlo_emin', 'nlo_emax', 'nlo_ne', 'nlo_eta', 'nlo_scissor'):
        cfg[key] = answers[key]
    if not molecule:
        cfg['nlo_components'] = answers['nlo_components']
    cfg['nlo_responses'] = answers['nlo_responses']
    cfg['nlo_dipoles'] = answers['nlo_dipoles'] if _internal(cfg) else False
    cfg['nlo_shg_terms'] = answers['nlo_shg_terms']
    if molecule:
        cfg['nlo_coupling'] = answers['nlo_coupling']
    cfg['nlo_coupled'] = bool(answers.get('nlo_coupled', False))
    if cfg['nlo_coupled']:
        for key in (
            'coupled_nk',
            'coupled_kernels',
            'coupled_emax',
            'coupled_ne',
            'coupled_shg_energies',
            'coupled_eta',
        ):
            cfg[key] = answers[key]
        for key in ('coupled_emin', 'coupled_components'):  # GUI-only, when changed
            if key in answers:
                cfg[key] = answers[key]
    return cfg


def assemble_run(answers: Answers, context: Context) -> dict:
    """Configuration of the property run (:func:`PAOFLOW.gen.paoflow_driver.build_run_script`).

    Parameters
    ----------
    answers : dict
        Answers of :data:`COMMON_ITEMS` and :data:`RUN_ITEMS`.
    context : Context
        The session context.

    Returns
    -------
    dict
    """
    cfg = assemble_common(answers, context)
    return assemble_run_from_common(cfg, answers)


def assemble_run_from_common(common: dict, answers: Answers) -> dict:
    """Configuration of the property run on top of the common configuration.

    Parameters
    ----------
    common : dict
        The common configuration.
    answers : dict
        Answers of :data:`RUN_ITEMS`.

    Returns
    -------
    dict
    """
    props = list(answers['properties'])
    cfg = dict(common)
    cfg['properties'] = props
    cfg['std_basis'] = answers.get('std_basis', 'standard')
    cfg['npool'] = answers['npool']
    smearing = answers['smearing']
    cfg['smearing'] = None if smearing.lower() == 'none' else smearing
    cfg['spin_orbit'] = answers['spin_orbit']
    cfg['nk'] = answers['nk'] if 'bands' in props else 400
    if 'bands' in props and ('band_path' in answers or 'high_sym' in answers):
        # GUI-only custom path: the side the user left untouched keeps the default.
        from ..band_paths import default_band_path

        default_path, default_points = default_band_path(
            common['workdir'], common['savedir'], int(common['ibrav']), bool(common['is_2d'])
        )
        points = answers.get('high_sym') or default_points
        cfg['band_path'] = answers.get('band_path') or default_path
        cfg['high_sym'] = {point[0]: tuple(point[1:]) for point in points}
    if _needs_energy_window(answers, None):
        cfg['emin'], cfg['emax'], cfg['ne'] = answers['emin'], answers['emax'], answers['ne']
    else:
        cfg['emin'], cfg['emax'], cfg['ne'] = -8.0, 4.0, 1000
    cfg['do_pdos'] = answers['do_pdos'] if 'dos' in props else False
    cfg['interpolate'] = answers['interpolate']
    cfg['nfft'] = answers['nfft'] if cfg['interpolate'] else 0
    if 'nonlinear' in props:
        assemble_nonlinear(cfg, answers)
    cfg['plot'] = answers['plot']
    return cfg


RUN = Workflow(
    name='run',
    title='Property run',
    description='Bands, DOS, transport, Hall, topology, optics and non-linear optics from a QE save '
    '(main.py and plot.py).',
    items=COMMON_ITEMS + RUN_ITEMS,
    assemble=assemble_run,
    overrides=COMMON_OVERRIDES,
)
