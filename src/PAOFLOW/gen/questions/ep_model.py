"""Questions of the ``ep_model`` workflow: electron-phonon coupling of a model crystal, no DFT run.

The crystal is one of :data:`PAOFLOW.models.electron_phonon.MODEL_CRYSTALS`: tight-binding
electrons whose hoppings depend on the bond length, springs between the same atoms, and so the
electron-phonon vertex of the same model.  ``main.ep_model.py`` builds it with its function and
calls the property steps of :mod:`PAOFLOW.models.electron_phonon.workflows`: superconductivity,
Migdal-Eliashberg, resistivity and self-energy for the metals, mobility and phonon-assisted
absorption for the semiconductors.  The parameter questions are made from the signature and the
docstring of each model function, as in the ``tb_model`` workflow.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

from .model import Answers, Context, Note, Question, Workflow, float_list
from .run import point_row

if TYPE_CHECKING:
    from ...models.electron_phonon import ElectronPhononModel

MODEL = 'Model crystal'
PROPERTIES = 'Properties'
BAND_PATH = 'Band path'
COUPLING = 'Electron–phonon coupling (metals)'
CARRIERS = 'Carriers and absorption (semiconductors)'

#: Model crystals offered: name -> (function of PAOFLOW.models.electron_phonon, title).
EP_MODELS: dict[str, tuple[str, str]] = {
    'cubic_metal': (
        'cubic_metal_model',
        'One-band simple-cubic metal (3D): λ ≈ 0.76, Tc ≈ 11 K with the defaults',
    ),
    'light_atom_metal': (
        'light_atom_metal_model',
        'Heavy and light atom metal (2D or 3D): optical phonons, two-peak α²F, Tc ≈ 26 K',
    ),
    'indirect_semiconductor': (
        'indirect_semiconductor_model',
        'Indirect-gap s + p semiconductor, 2 to 4 bands (1D to 3D): mobility, absorption',
    ),
}

#: Parameters that are not plain numbers: (model, parameter) -> question settings.
SPECIAL_PARAMETERS: dict[tuple[str, str], dict[str, Any]] = {
    ('indirect_semiconductor', 'n_bands'): {'kind': 'choice', 'choices': ['2', '3', '4']},
    ('light_atom_metal', 'dim'): {'kind': 'choice', 'choices': ['2', '3']},
}

#: Properties: (key, label, 'metal' / 'semiconductor' / None for both).
EP_PROPERTIES = (
    ('bands', 'Electron band structure', None),
    ('phonons', 'Phonon dispersion and density of states', None),
    (
        'superconductivity',
        'Coupling λ, Eliashberg function α²F, Tc (McMillan, Allen–Dynes)',
        'metal',
    ),
    ('migdal_eliashberg', 'Migdal–Eliashberg gap, renormalisation and Tc (isotropic)', 'metal'),
    ('resistivity', 'Resistivity ρ(T) from α²F_tr (and the Drude emissivity)', 'metal'),
    ('self_energy', 'Electron self-energy Σ(ω) and mass enhancement', 'metal'),
    ('mobility', 'Carrier mobility (SERTA and MRTA)', 'semiconductor'),
    ('absorption', 'Phonon-assisted absorption across the indirect gap', 'semiconductor'),
)
#: Properties that read ``eliashberg.npz`` of ``superconductivity`` (added when they are chosen).
NEEDS_ELIASHBERG = {'migdal_eliashberg', 'resistivity', 'self_energy'}
METAL_PROPERTIES = {key for key, _label, kind in EP_PROPERTIES if kind == 'metal'}
COUPLING_GRID = {1: (200, 20), 2: (48, 12), 3: (40, 8)}
MOBILITY_GRID = {1: 1200, 2: 120, 3: 40}
ABSORPTION_GRID = {1: 400, 2: 60, 3: 28}


# ------------------------------------------------------------------------------- the models
def model_function(name: str) -> Callable[..., 'ElectronPhononModel']:
    """The function of :mod:`PAOFLOW.models.electron_phonon` that builds a model of :data:`EP_MODELS`.

    Parameters
    ----------
    name : str
        A key of :data:`EP_MODELS`.

    Returns
    -------
    callable
    """
    from ...models import electron_phonon

    return getattr(electron_phonon, EP_MODELS[name][0])


def model_parameters(name: str) -> list[inspect.Parameter]:
    """The keyword parameters (with defaults) of a model function.

    Parameters
    ----------
    name : str
        A key of :data:`EP_MODELS`.

    Returns
    -------
    list of inspect.Parameter
    """
    return [
        parameter
        for parameter in inspect.signature(model_function(name)).parameters.values()
        if parameter.default is not inspect.Parameter.empty
    ]


def model_arguments(answers: Answers) -> dict[str, Any]:
    """Keyword arguments of the chosen model function from the answers.

    Parameters
    ----------
    answers : dict
        Answers (``ep_model`` and ``<model>_<parameter>``).

    Returns
    -------
    dict
    """
    name = answers['ep_model']
    arguments = {}
    for parameter in model_parameters(name):
        key = '%s_%s' % (name, parameter.name)
        if key not in answers:
            continue
        value = answers[key]
        if (name, parameter.name) in SPECIAL_PARAMETERS:
            value = int(value)
        arguments[parameter.name] = value
    return arguments


def build_model(answers: Answers, context: Context | None = None) -> 'ElectronPhononModel':
    """The model crystal of the answers.

    Parameters
    ----------
    answers : dict
        Answers.
    context : Context, optional
        Caches the model of equal answers.

    Returns
    -------
    ElectronPhononModel

    Raises
    ------
    ValueError
        When the model function refuses the parameters.
    """
    arguments = model_arguments(answers)

    def build():
        try:
            return model_function(answers['ep_model'])(**arguments)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(str(exc)) from None

    if context is None:
        return build()
    key = 'ep-model:%s:%r' % (answers['ep_model'], sorted(arguments.items()))
    return context.cached(key, build)


def model_facts(model: 'ElectronPhononModel', context: Context | None = None) -> dict:
    """Metal or insulator, gap and phonon range of a model (cached with the context).

    Parameters
    ----------
    model : ElectronPhononModel
    context : Context, optional

    Returns
    -------
    dict
        :func:`PAOFLOW.models.electron_phonon.workflows.energy_reference` plus ``phonon_max_mev``
        and ``unstable`` (imaginary modes on a coarse grid).
    """

    def compute():
        import numpy as np

        from ...models.electron_phonon import properties, workflows

        reference = workflows.energy_reference(model, {1: 200, 2: 32, 3: 12}[model.dimension])
        energies = model.phonons(
            properties.crystal_grid(model, {1: 64, 2: 16, 3: 8}[model.dimension])
        )[0]
        return {
            **reference,
            'phonon_max_mev': float(1e3 * energies.max()),
            'unstable': bool(energies.min() < -1e-3 * max(float(np.abs(energies).max()), 1e-9)),
        }

    if context is None:
        return compute()
    signature = (
        model.name,
        model.nelec,
        model.onsite.tolist(),
        model.masses.tolist(),
        repr(model.bonds),
        repr(model.springs),
    )
    return context.cached('ep-facts:%r' % (signature,), compute)


def model_summary(model: 'ElectronPhononModel', context: Context | None = None) -> str:
    """One line on a model: atoms, orbitals, dimension, filling, gap, phonons."""
    facts = model_facts(model, context)
    electrons = 'metal' if facts['metal'] else 'insulator, gap %.3f eV' % facts['gap']
    return (
        '  (%d atom%s, %d orbitals, %dD, %g electrons per cell: %s; phonons up to %.1f meV%s)'
        % (
            model.number_of_atoms,
            '' if model.number_of_atoms == 1 else 's',
            model.number_of_orbitals,
            model.dimension,
            model.nelec,
            electrons,
            facts['phonon_max_mev'],
            '; UNSTABLE: imaginary modes' if facts['unstable'] else '',
        )
    )


# ------------------------------------------------------------------------------ helpers
def _chosen(answers: Answers) -> str | None:
    return answers.get('ep_model')


def _model_or_none(answers: Answers, context: Context) -> 'ElectronPhononModel | None':
    try:
        return build_model(answers, context)
    except (ValueError, KeyError):
        return None


def _dimension(answers: Answers, context: Context) -> int:
    model = _model_or_none(answers, context)
    return model.dimension if model is not None else 3


def _is_metal(answers: Answers, context: Context) -> bool:
    model = _model_or_none(answers, context)
    if model is None:
        return True
    return bool(model_facts(model, context)['metal'])


def _properties(answers: Answers) -> list[str]:
    return answers.get('properties') or []


def _wants(*names: str) -> Callable[[Answers, Context], bool]:
    def when(answers: Answers, context: Context) -> bool:
        return bool(set(_properties(answers)) & set(names))

    return when


def _property_choices(answers: Answers, context: Context) -> list[tuple[str, str]]:
    metal = _is_metal(answers, context)
    return [
        (name, label)
        for name, label, kind in EP_PROPERTIES
        if kind is None or kind == ('metal' if metal else 'semiconductor')
    ]


def _path_defaults(answers: Answers, context: Context) -> tuple[str, list]:
    from .tb_model import model_band_path

    model = _model_or_none(answers, context)
    if model is None:
        return 'gG-X', [('gG', 0.0, 0.0, 0.0), ('X', 0.5, 0.0, 0.0)]
    return model_band_path(model.to_hopping_model())


def _validate_path(value: str, answers: Answers, context: Context) -> str | None:
    from ..band_paths import check_band_path

    points = answers.get('high_sym') or _path_defaults(answers, context)[1]
    return check_band_path(value, points)


def _validate_nq(value: int, answers: Answers, context: Context) -> str | None:
    nk = int(answers.get('ep_nk') or 0)
    if value > 0 and nk % value == 0:
        return None
    return '  NQ must divide NK = %d (k + q stays on the k-grid).' % nk


def _summary_note(answers: Answers, context: Context) -> str:
    title = EP_MODELS.get(answers.get('ep_model'), ('', ''))[1]
    try:
        return '  %s.\n%s' % (title, model_summary(build_model(answers, context), context))
    except ValueError as exc:
        return '  The model cannot be built with these parameters: %s' % exc


def _parameter_question(name: str, function: str, parameter: inspect.Parameter) -> Question:
    """The question of one parameter of a model function (kind, default and help from it)."""
    from ...models import electron_phonon
    from .docs import docstring_parameters

    doc = docstring_parameters(getattr(electron_phonon, function)).get(parameter.name, '')
    settings = dict(SPECIAL_PARAMETERS.get((name, parameter.name), {}))
    default = settings.pop('default', parameter.default)
    if 'kind' not in settings:
        if isinstance(parameter.default, bool):
            settings['kind'] = 'bool'
        elif isinstance(parameter.default, int):
            settings['kind'] = 'int'
        else:
            settings['kind'] = 'float'
    if settings['kind'] == 'choice':
        default = str(default)
    first = doc.split('.')[0] if doc else ''
    unit = ''
    for text, symbol in (
        ('(eV/bohr^2)', 'eV/bohr²'),
        ('(eV)', 'eV'),
        ('(bohr)', 'bohr'),
        ('(amu)', 'amu'),
    ):
        if text in first:
            unit = symbol
            break
    return Question(
        '%s_%s' % (name, parameter.name),
        '%s (%s)' % (parameter.name, first.strip() or 'see the model function'),
        default=default,
        unit=unit,
        label=parameter.name,
        doc='PAOFLOW.models.electron_phonon.crystals.%s:%s' % (function, parameter.name),
        when=lambda a, c, name=name: _chosen(a) == name,
        section=MODEL,
        **settings,
    )


def _parameter_items() -> tuple[Question, ...]:
    return tuple(
        _parameter_question(name, function, parameter)
        for name, (function, _title) in EP_MODELS.items()
        for parameter in model_parameters(name)
    )


def _temperatures(
    prompt: str,
    default: str,
    when: Callable[[Answers, Context], bool],
    section: str,
    key: str,
    help_text: str = '',
) -> Question:
    """A list of temperatures (K), given as text."""
    return Question(
        key,
        prompt,
        default=default,
        parse=float_list,
        when=when,
        unit='K',
        help=help_text,
        section=section,
    )


EP_MODEL_ITEMS = (
    Question(
        'ep_model',
        'Model crystal (%s)' % ', '.join(EP_MODELS),
        kind='choice',
        choices=list(EP_MODELS),
        default='cubic_metal',
        label='Model crystal',
        help='The crystals of PAOFLOW.models.electron_phonon: tight-binding electrons with '
        'bond-length dependent hoppings, springs as force constants, and the vertex dH/du of the '
        'same model (the electron-phonon lecture notes of the PAOFLOW wiki).',
        section=MODEL,
    ),
    *_parameter_items(),
    Note(_summary_note, section=MODEL),
    Question(
        'properties',
        'Available properties:',
        kind='menu',
        choices=_property_choices,
        abort_if_empty='No valid properties selected; nothing to generate.',
        label='Properties',
        help='Migdal–Eliashberg, resistivity and self-energy run on the Eliashberg function of the '
        'coupling step, which is added when one of them is chosen.',
        section=PROPERTIES,
    ),
    Question(
        'nk',
        'Points along the band path (electrons and phonons)',
        kind='int',
        default=200,
        when=_wants('bands', 'phonons'),
        section=BAND_PATH,
    ),
    Question(
        'high_sym',
        'High-symmetry points (label and crystal coordinates)',
        kind='rows',
        default=lambda a, c: _path_defaults(a, c)[1],
        parse=point_row,
        when=_wants('bands', 'phonons'),
        columns=('label', 'k1', 'k2', 'k3'),
        label='High-symmetry points',
        help='Crystal coordinates of the reciprocal vectors of the model; the defaults follow its '
        'lattice (chain, square or cubic).',
        section=BAND_PATH,
        level='advanced',
    ),
    Question(
        'band_path',
        'Band path (labels joined by "-", segments split by "|")',
        default=lambda a, c: _path_defaults(a, c)[0],
        when=_wants('bands', 'phonons'),
        validate=_validate_path,
        label='Band path',
        help='gG is Γ; the points are in the advanced table below.',
        section=BAND_PATH,
        widget='bandpath',
    ),
    Question(
        'ep_nk',
        'k-points per periodic direction of the double delta (NK)',
        kind='int',
        default=lambda a, c: COUPLING_GRID[_dimension(a, c)][0],
        when=_wants(*METAL_PROPERTIES),
        help='The Fermi-surface sums of λ; with adaptive smearing λ is converged to a few per cent '
        'at 32-40 (3D).',
        section=COUPLING,
    ),
    Question(
        'ep_nq',
        'q-points per periodic direction (NQ, divides NK; irreducible wedge in 3D)',
        kind='int',
        default=lambda a, c: COUPLING_GRID[_dimension(a, c)][1],
        when=_wants(*METAL_PROPERTIES),
        validate=_validate_nq,
        help='The low-temperature resistivity (T⁵) needs a fine q-grid (16 in 3D).',
        section=COUPLING,
    ),
    Question(
        'adaptive_smearing',
        'Adaptive-smearing prefactor a (widths a |v| Δk; 0: the fixed width below)',
        kind='float',
        default=0.7,
        when=_wants(*METAL_PROPERTIES),
        doc='PAOFLOW.models.electron_phonon.workflows.superconductivity:adaptive_smearing',
        section=COUPLING,
    ),
    Question(
        'sigma_ev',
        'Gaussian width of the double delta, or the floor of the adaptive widths',
        kind='float',
        default=0.02,
        unit='eV',
        when=_wants(*METAL_PROPERTIES),
        section=COUPLING,
    ),
    Question(
        'mu_star',
        'Coulomb pseudopotential μ*',
        kind='float',
        default=0.1,
        when=_wants(*METAL_PROPERTIES),
        section=COUPLING,
    ),
    Question(
        'me_temperatures',
        'Migdal–Eliashberg temperatures (blank: from 1 K to 1.6 Tc of Allen–Dynes)',
        default='',
        parse=lambda text: float_list(text) if text.strip() else None,
        when=_wants('migdal_eliashberg'),
        unit='K',
        section=COUPLING,
    ),
    _temperatures(
        'Resistivity temperatures',
        '10 20 50 100 200 300 400 500 600',
        _wants('resistivity'),
        COUPLING,
        'rho_temperatures',
    ),
    _temperatures(
        'Self-energy temperatures', '0 100', _wants('self_energy'), COUPLING, 'se_temperatures'
    ),
    Question(
        'mobility_nk',
        'k-points (= q-points) per periodic direction of the mobility',
        kind='int',
        default=lambda a, c: MOBILITY_GRID[_dimension(a, c)],
        when=_wants('mobility'),
        section=CARRIERS,
    ),
    Question(
        'mobility_carriers',
        'Carriers (electrons / holes / both)',
        kind='choice',
        choices=['electrons', 'holes', 'both'],
        default='electrons',
        when=_wants('mobility'),
        help='The heavy p holes of the 3D model fill ten times more of the energy window than the '
        'electrons: about a minute and a half at 40³ against a second for the electrons.',
        section=CARRIERS,
    ),
    _temperatures(
        'Mobility temperatures',
        '100 150 200 300 400 500',
        _wants('mobility'),
        CARRIERS,
        'mobility_temperatures',
    ),
    Question(
        'doping_cm3',
        'Net doping n_e - n_h (0: intrinsic)',
        kind='float',
        default=0.0,
        unit='cm⁻³',
        when=_wants('mobility'),
        section=CARRIERS,
        level='advanced',
    ),
    Question(
        'absorption_nk',
        'k-points (= q-points) per periodic direction of the absorption',
        kind='int',
        default=lambda a, c: ABSORPTION_GRID[_dimension(a, c)],
        when=_wants('absorption'),
        section=CARRIERS,
    ),
    _temperatures(
        'Absorption temperatures',
        '10 300 600',
        _wants('absorption'),
        CARRIERS,
        'absorption_temperatures',
    ),
    Question(
        'refractive_index',
        'Refractive index of the absorption coefficient',
        kind='float',
        default=3.4,
        when=_wants('absorption'),
        section=CARRIERS,
        level='advanced',
    ),
    Question(
        'plot',
        'Also generate a plotting script (plot.ep_model.py)?',
        kind='bool',
        default=True,
        section='Output',
        gui=False,  # the Studio: "Write the plot scripts" of its Output box (options.plot)
    ),
)


def assemble_ep_model(answers: Answers, context: Context) -> dict:
    """Configuration of the model run (:func:`PAOFLOW.gen.ep_model_driver.build_ep_model_script`).

    Parameters
    ----------
    answers : dict
        Answers of :data:`EP_MODEL_ITEMS`.
    context : Context
        The session context.

    Returns
    -------
    dict
        ``ep_model``, ``model_function``, ``model_arguments``, ``metal``, ``dimension``,
        ``properties`` (with ``superconductivity`` when a property reads its output), the path
        (``nk``, ``band_path``, ``high_sym``), the coupling settings (``ep_nk``, ``ep_nq``,
        ``adaptive_smearing``, ``sigma_ev``, ``mu_star``, ``me_temperatures``,
        ``rho_temperatures``, ``se_temperatures``), the semiconductor settings (``mobility_nk``,
        ``mobility_carriers``, ``mobility_temperatures``, ``doping_cm3``, ``absorption_nk``,
        ``absorption_temperatures``, ``refractive_index``), ``outputdir`` and ``plot``.

    Raises
    ------
    ValueError
        When the model cannot be built with the parameters.
    """
    model = build_model(answers, context)
    facts = model_facts(model, context)
    props = list(answers['properties'])
    if set(props) & NEEDS_ELIASHBERG and 'superconductivity' not in props:
        props.insert(
            next((i for i, p in enumerate(props) if p in NEEDS_ELIASHBERG), len(props)),
            'superconductivity',
        )
    nk_coupling, nq_coupling = COUPLING_GRID[model.dimension]
    cfg = {
        'workdir': context.workdir,
        'outputdir': 'output',
        'ep_model': answers['ep_model'],
        'model_function': EP_MODELS[answers['ep_model']][0],
        'model_arguments': model_arguments(answers),
        'metal': bool(facts['metal']),
        'dimension': int(model.dimension),
        'properties': props,
        'plot': answers['plot'],
    }
    if set(props) & {'bands', 'phonons'}:
        path, points = _path_defaults(answers, context)
        rows = answers.get('high_sym') or points
        cfg['nk'] = answers['nk']
        cfg['band_path'] = answers.get('band_path') or path
        cfg['high_sym'] = {row[0]: tuple(float(v) for v in row[1:]) for row in rows}
    if set(props) & METAL_PROPERTIES:
        cfg['ep_nk'] = answers.get('ep_nk', nk_coupling)
        cfg['ep_nq'] = answers.get('ep_nq', nq_coupling)
        cfg['adaptive_smearing'] = answers.get('adaptive_smearing', 0.7)
        cfg['sigma_ev'] = answers.get('sigma_ev', 0.02)
        cfg['mu_star'] = answers.get('mu_star', 0.1)
    if 'migdal_eliashberg' in props:
        cfg['me_temperatures'] = answers.get('me_temperatures')
    if 'resistivity' in props:
        cfg['rho_temperatures'] = answers['rho_temperatures']
    if 'self_energy' in props:
        cfg['se_temperatures'] = answers['se_temperatures']
    if 'mobility' in props:
        cfg['mobility_nk'] = answers['mobility_nk']
        cfg['mobility_carriers'] = answers['mobility_carriers']
        cfg['mobility_temperatures'] = answers['mobility_temperatures']
        cfg['doping_cm3'] = answers.get('doping_cm3', 0.0)
    if 'absorption' in props:
        cfg['absorption_nk'] = answers['absorption_nk']
        cfg['absorption_temperatures'] = answers['absorption_temperatures']
        cfg['refractive_index'] = answers.get('refractive_index', 3.4)
    return cfg


EP_MODEL = Workflow(
    name='ep_model',
    title='Electron–phonon run from a model crystal',
    description='Electron-phonon coupling of a model crystal of PAOFLOW.models.electron_phonon '
    '(no DFT run): λ, α²F, Tc and the Migdal–Eliashberg gap, resistivity and self-energy of a '
    'metal; mobility and phonon-assisted absorption of an indirect-gap semiconductor '
    '(main.ep_model.py and plot.ep_model.py, serial).',
    items=EP_MODEL_ITEMS,
    assemble=assemble_ep_model,
)
