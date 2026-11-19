"""Questions of the ``run`` workflow: the property menu, the grids and the non-linear optics block."""

from __future__ import annotations

from .common import COMMON_ITEMS, COMMON_OVERRIDES, assemble_common, from_save
from .linear_response import assemble_linear_response, linear_response_items
from .model import Answers, Context, Note, Question, Workflow, gated
from .run_extras import RUN_EXTRA_ITEMS, SHC_ATOMS, assemble_run_extras, sparse_conflicts
from .spin_response import assemble_spin_response, spin_response_items

PROPERTIES = 'Properties'
GRIDS = 'Grids and run settings'
BAND_PATH = 'Band path'
NONLINEAR = 'Non-linear optics'
COUPLED = 'Coupled optical response'


def _properties(answers: Answers) -> list[str]:
    return answers.get('properties') or []


def _band_path(answers: Answers) -> bool:
    """The run computes a band path (the bands, or the site-projected bands that need them)."""
    props = _properties(answers)
    return 'bands' in props or 'site_projected_bands' in props


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
        'nlo_metal',
        '  metal or semimetal (Fermi-surface terms: Drude, Berry-curvature dipole)?',
        kind='bool',
        default=False,
        when=lambda a, c: not _molecule(a),
        label='Metal or semimetal (Fermi-surface terms)',
        help='Keeps the Fermi-surface terms of the SHG and of the zero-frequency photocurrent '
        '(nonlinear Drude, Berry-curvature dipole) and the Drude term of the coupled dielectric '
        'function; needs an electronic temperature. Metallic channels are also detected.',
        section=NONLINEAR,
    ),
    Question(
        'nlo_temperature',
        '    electronic temperature (K)',
        kind='float',
        default=300.0,
        unit='K',
        when=lambda a, c: bool(a.get('nlo_metal')),
        section=NONLINEAR,
    ),
    Question(
        'nlo_tau',
        '    intraband relaxation time tau (fs)',
        kind='float',
        default=20.0,
        unit='fs',
        when=lambda a, c: bool(a.get('nlo_metal')),
        section=NONLINEAR,
    ),
    Question(
        'nlo_coherence_damping',
        '    coherence damping in units of eta (1: one dephasing rate, 2: adiabatic / Kubo)',
        kind='float',
        default=1.0,
        when=lambda a, c: bool(a.get('nlo_metal')),
        label='Coherence damping (units of eta)',
        help='Damping of the second-order interband coherences of the metal engine. 1: every '
        'coherence decays at eta, two-photon resonances have width eta as for insulators; 2: '
        'adiabatic switching, equal to the velocity-gauge Kubo response. They agree as eta -> 0.',
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
                default=lambda a, c: (
                    'none,hartree,hartree+alda'
                    if a.get('nlo_metal')
                    else 'none,hartree,hartree+alda,hartree+alda+bootstrap'
                ),
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

#: Fourier-interpolation routes: answer -> argument of ``bands`` and ``interpolated_hamiltonian``
#: (``standard`` keeps their defaults, ``fft_grid`` and ``zero_padding``).
INTERPOLATION_ROUTES: tuple[str, ...] = ('standard', 'wigner_seitz', 'wigner_seitz_pairs')

INTERPOLATION_PROMPT = (
    'Fourier interpolation (standard: zero padding; wigner_seitz; wigner_seitz_pairs: '
    'bond-resolved, keeps the space group)'
)
HALL_PROMPT = 'Hall coefficient and Nernst tensor with the Boltzmann transport (do_hall)?'


def interpolation_question(when, section: str) -> Question:
    """The Fourier-interpolation route of the band path and of the dense grid.

    Parameters
    ----------
    when : callable
        ``when(answers, context)``: the run interpolates (a band path or a dense grid).
    section : str
        GUI section.

    Returns
    -------
    Question
        Key ``interpolation``, one of :data:`INTERPOLATION_ROUTES`.
    """
    return Question(
        'interpolation',
        INTERPOLATION_PROMPT,
        kind='choice',
        choices=lambda a, c: list(INTERPOLATION_ROUTES),
        default='standard',
        when=when,
        label='Fourier interpolation',
        help='standard: the simple sum of the band path and zero padding on the dense grid. '
        'wigner_seitz: every H(R) on the Wigner-Seitz cell of the supercell lattice, as Wannier '
        'interpolation does; wigner_seitz_pairs: the bond vectors of every orbital pair, which keeps '
        'the full space group. Both reproduce the DFT mesh exactly, at the same cost.',
        doc='PAOFLOW.PAOFLOW.PAOFLOW.interpolated_hamiltonian:interpolation',
        section=section,
        level='advanced',
    )


def hall_question(when, section: str) -> Question:
    """Hall coefficient and Nernst tensor of the Boltzmann transport.

    Parameters
    ----------
    when : callable
        ``when(answers, context)``: Boltzmann transport is computed.
    section : str
        GUI section.

    Returns
    -------
    Question
        Key ``do_hall``.
    """
    return Question(
        'do_hall',
        HALL_PROMPT,
        kind='bool',
        default=False,
        when=when,
        label='Hall and Nernst',
        help='hall_trace, hall and nernst files: the Hall coefficient R_H (m³/C, negative for '
        'electrons) and the Nernst tensor over the relaxation time; they need the second '
        'derivatives of the bands.',
        doc='PAOFLOW.PAOFLOW.PAOFLOW.transport:do_hall',
        section=section,
    )


RUN_ITEMS = (
    Question(
        'properties',
        'Available properties:',
        kind='menu',
        choices=_property_choices,
        validate=sparse_conflicts,
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
        when=lambda a, c: _band_path(a),
        section=GRIDS,
        doc='PAOFLOW.PAOFLOW.PAOFLOW.bands:nk',
    ),
    Question(
        'high_sym',
        'High-symmetry points (label and crystal coordinates)',
        kind='rows',
        default=lambda a, c: _path_defaults(a, c)[1],
        parse=point_row,
        when=lambda a, c: _band_path(a),
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
        when=lambda a, c: _band_path(a),
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
    hall_question(lambda a, c: 'transport' in _properties(a), GRIDS),
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
    interpolation_question(
        # the sparse bond list is the standard sum; the Wigner-Seitz routes are dense-only
        lambda a, c: (
            (_band_path(a) or bool(a.get('interpolate'))) and 'sparse' not in _properties(a)
        ),
        GRIDS,
    ),
    *gated(NONLINEAR_ITEMS, lambda a, c: 'nonlinear' in _properties(a)),
    *linear_response_items(lambda a, c: bool(a.get('spin_orbit'))),
    *spin_response_items(lambda a, c: 'spin_susceptibility' in _properties(a)),
    SHC_ATOMS,
    *RUN_EXTRA_ITEMS,
    Question(
        'plot',
        'Also generate a plotting script (plot.py)?',
        kind='bool',
        default=True,
        section='Output',
        gui=False,  # the Studio: "Write the plot scripts" of its Output box (options.plot)
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
    cfg['nlo_metal'] = bool(answers.get('nlo_metal', False)) and not molecule
    if cfg['nlo_metal']:
        cfg['nlo_temperature'] = answers['nlo_temperature']
        cfg['nlo_tau'] = answers['nlo_tau']
        cfg['nlo_coherence_damping'] = answers['nlo_coherence_damping']
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
    if 'site_projected_bands' in props and 'bands' not in props:
        props.insert(0, 'bands')  # the projection reads the eigenvectors of the band path
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
    cfg['do_hall'] = bool(answers.get('do_hall', False)) if 'transport' in props else False
    cfg['interpolate'] = answers['interpolate']
    cfg['nfft'] = answers['nfft'] if cfg['interpolate'] else 0
    interpolates = 'bands' in props or cfg['interpolate']
    cfg['interpolation'] = answers.get('interpolation', 'standard') if interpolates else 'standard'
    if 'nonlinear' in props:
        assemble_nonlinear(cfg, answers)
    if 'linear_response' in props:
        assemble_linear_response(cfg, answers, bool(answers['spin_orbit']))
    if 'spin_susceptibility' in props:
        assemble_spin_response(cfg, answers)
    assemble_run_extras(cfg, answers)
    cfg['plot'] = answers['plot']
    return cfg


RUN = Workflow(
    name='run',
    title='Property run',
    description='Bands, DOS, transport, Hall, topology, optics, non-linear optics and the spin '
    'susceptibility from a QE save (main.py and plot.py).',
    items=COMMON_ITEMS + RUN_ITEMS,
    assemble=assemble_run,
    overrides=COMMON_OVERRIDES,
)
