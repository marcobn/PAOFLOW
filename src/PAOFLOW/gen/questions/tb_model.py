"""Questions of the ``tb_model`` workflow: a property run on a tight-binding model, no DFT run.

The model is one of the hopping-list models of :mod:`PAOFLOW.models.models` (:data:`TB_MODELS`);
``main.py`` builds it with its own function (``PAOFLOW.PAOFLOW(model=honeycomb_model(...))``) and
runs the chosen properties on it, so there is no QE save, no projection and no basis.  The
parameter questions are made from the signature and the docstring of each model function, so a
new parameter in ``models.py`` appears here by itself.  Slater-Koster models come later.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from .model import Answers, Context, Note, Question, Workflow, optional_number
from .run import COUPLED, NONLINEAR_ITEMS, assemble_nonlinear, point_row

MODEL = 'Tight-binding model'
PROPERTIES = 'Properties'
GRIDS = 'Grids and run settings'
BAND_PATH = 'Band path'

#: Models offered: name -> (function of PAOFLOW.models.models, title).
TB_MODELS: dict[str, tuple[str, str]] = {
    'honeycomb': ('honeycomb_model', 'Honeycomb lattice: gapped graphene, Haldane (2D)'),
    'kane_mele': (
        'kane_mele_model',
        'Kane–Mele: honeycomb with spin–orbit and Rashba coupling (2D, spinful)',
    ),
    'tmd': ('tmd_model', 'Transition-metal dichalcogenide MX₂, three d bands (2D)'),
    'rice_mele': ('rice_mele_model', 'Rice–Mele chain (1D)'),
    'weyl': ('weyl_model', 'Weyl semimetal, two bands (3D, spinful)'),
    'parabolic': ('parabolic_model', 'Parabolic-band semiconductor, 2 to 4 bands (1D to 3D)'),
}

#: Parameters that are not plain numbers: (model, parameter) -> question settings.
SPECIAL_PARAMETERS: dict[tuple[str, str], dict[str, Any]] = {
    ('tmd', 'material'): {
        'kind': 'choice',
        'choices': ['MoS2', 'WS2', 'MoSe2', 'WSe2', 'MoTe2', 'WTe2'],
    },
    ('tmd', 'functional'): {'kind': 'choice', 'choices': ['GGA', 'LDA']},
    ('tmd', 'thickness'): {'kind': 'text', 'default': '', 'parse': optional_number},
    ('parabolic', 'n_bands'): {'kind': 'choice', 'choices': ['2', '3', '4']},
}

#: Properties of a model run (PROPERTY_MENU of the QE run without the ones that need a PAO basis).
MODEL_PROPERTIES = (
    ('bands', 'Band structure'),
    ('dos', 'DOS / projected DOS (on the orbitals of the model)'),
    ('transport', 'Boltzmann transport (conductivity, Seebeck): 3D models'),
    ('fermi_surface', 'Fermi surface'),
    ('spin_texture', 'Spin texture: spinful models'),
    ('spin_Hall', 'Spin Hall conductivity: spinful models'),
    ('anomalous_Hall', 'Anomalous Hall / Berry curvature: spinful models'),
    ('topology', 'Band topology (Berry, effective mass)'),
    ('optical', 'Optical / dielectric tensor'),
    ('nonlinear', 'Non-linear optics (shift / injection currents, SHG)'),
)
SPIN_ONLY = {'spin_texture', 'spin_Hall', 'anomalous_Hall'}
THREE_D_ONLY = {'transport'}  # Boltzmann transport inverts the 3x3 conductivity tensor
DENSE_GRID = {1: 400, 2: 96, 3: 32}
NLO_GRID = {1: 2000, 2: 200, 3: 40}


# ------------------------------------------------------------------------------- the models
def model_function(name: str):
    """The function of :mod:`PAOFLOW.models.models` that builds a model of :data:`TB_MODELS`.

    Parameters
    ----------
    name : str
        A key of :data:`TB_MODELS`.

    Returns
    -------
    callable
    """
    from ...models import models

    return getattr(models, TB_MODELS[name][0])


def model_parameters(name: str) -> list[inspect.Parameter]:
    """The keyword parameters (with defaults) of a model function.

    Parameters
    ----------
    name : str
        A key of :data:`TB_MODELS`.

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
        Answers (``tb_model`` and ``<model>_<parameter>``).

    Returns
    -------
    dict
    """
    name = answers['tb_model']
    arguments = {}
    for parameter in model_parameters(name):
        key = '%s_%s' % (name, parameter.name)
        if key not in answers:
            continue
        value = answers[key]
        if (name, parameter.name) == ('parabolic', 'n_bands'):
            value = int(value)
        arguments[parameter.name] = value
    return arguments


def build_model(answers: Answers, context: Context | None = None):
    """The model of the answers (:class:`PAOFLOW.models.models.HoppingModel`).

    Parameters
    ----------
    answers : dict
        Answers.
    context : Context, optional
        Caches the model of equal answers.

    Returns
    -------
    HoppingModel

    Raises
    ------
    ValueError
        When the model function refuses the parameters.
    """
    arguments = model_arguments(answers)

    def build():
        try:
            return model_function(answers['tb_model'])(**arguments)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(str(exc)) from None

    if context is None:
        return build()
    key = 'tb-model:%s:%r' % (answers['tb_model'], sorted(arguments.items()))
    return context.cached(key, build)


def periodic_axes(model) -> tuple[bool, bool, bool]:
    """Which lattice directions of a model are periodic (more than one cell of hoppings).

    Parameters
    ----------
    model : HoppingModel

    Returns
    -------
    tuple of bool
    """
    return tuple(int(n) > 1 for n in model.real_space_grid)


def model_dimension(model) -> int:
    """1, 2 or 3: the periodic directions of a model."""
    return max(1, sum(periodic_axes(model)))


def model_band_path(model) -> tuple[str, list[tuple[str, float, float, float]]]:
    """Default band path of a model, from its lattice.

    1D: Γ-X; hexagonal 2D: Γ-M-K-Γ (K the corner of the zone of this lattice); other 2D: Γ-X-M-Γ;
    3D: Γ-X-M-Γ-R-X|M-R (simple cubic labels).

    Parameters
    ----------
    model : HoppingModel

    Returns
    -------
    tuple
        ``(band_path, points)``, the points as ``(label, k1, k2, k3)`` in crystal coordinates.
    """
    periodic = periodic_axes(model)
    axes = [i for i, p in enumerate(periodic) if p] or [0]
    vectors = np.asarray(model.lattice_vectors, dtype=float)

    def point(label: str, *values: float) -> tuple[str, float, float, float]:
        crystal = [0.0, 0.0, 0.0]
        for axis, value in zip(axes, values):
            crystal[axis] = round(float(value), 10)
        return (label, *crystal)

    if len(axes) == 1:
        return 'gG-X', [point('gG'), point('X', 0.5)]
    if len(axes) == 2:
        a1, a2 = vectors[axes[0]], vectors[axes[1]]
        cosine = float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)))
        if abs(np.linalg.norm(a1) - np.linalg.norm(a2)) < 1e-6 and abs(abs(cosine) - 0.5) < 1e-6:
            reciprocal = np.linalg.inv(vectors).T
            b1, b2 = reciprocal[axes[0]], reciprocal[axes[1]]
            corner = None
            for i in range(1, 3):
                for j in range(-2, 3):
                    k = (i * b1 + j * b2) / 3.0
                    distances = [np.linalg.norm(k - g) for g in (0 * b1, b1, b2, b1 + b2, b1 - b2)]
                    nearest = sorted(distances)
                    if abs(nearest[0] - nearest[2]) < 1e-9 and np.isclose(nearest[0], distances[0]):
                        corner = (i / 3.0, j / 3.0)
                        break
                if corner:
                    break
            corner = corner or (1 / 3.0, 1 / 3.0)
            return 'gG-M-K-gG', [point('gG'), point('M', 0.5, 0.0), point('K', *corner)]
        return 'gG-X-M-gG', [point('gG'), point('X', 0.5, 0.0), point('M', 0.5, 0.5)]
    return 'gG-X-M-gG-R-X|M-R', [
        point('gG'),
        point('X', 0.0, 0.5, 0.0),
        point('M', 0.5, 0.5, 0.0),
        point('R', 0.5, 0.5, 0.5),
    ]


def model_energy_range(model) -> tuple[float, float]:
    """Lowest and highest band energy of a model on the grid of its hoppings (eV).

    Parameters
    ----------
    model : HoppingModel

    Returns
    -------
    tuple of float
    """
    hoppings = model.real_space_hamiltonian()[..., 0]
    hk = np.fft.fftn(hoppings, axes=(2, 3, 4))
    hk = np.moveaxis(hk.reshape(hk.shape[0], hk.shape[1], -1), 2, 0)
    energies = np.linalg.eigvalsh(0.5 * (hk + np.conj(np.swapaxes(hk, 1, 2))))
    return float(energies.min()), float(energies.max())


def model_summary(model) -> str:
    """One line on a model: orbitals, spin, filling, dimension, lattice parameter."""
    dimension = model_dimension(model)
    filling = '%g electrons per cell' % model.nelec if model.nelec is not None else 'no filling'
    low, high = model_energy_range(model)
    return '  (%d orbitals, %s, %s, %dD, alat %.3f bohr; bands from %.2f to %.2f eV)' % (
        model.number_of_orbitals,
        'spinful' if model.spinful else 'spin-degenerate',
        filling,
        dimension,
        float(model.alat),
        low,
        high,
    )


# ------------------------------------------------------------------------------ helpers
def _chosen(answers: Answers) -> str | None:
    return answers.get('tb_model')


def _model_or_none(answers: Answers, context: Context):
    try:
        return build_model(answers, context)
    except (ValueError, KeyError):
        return None


def _dimension(answers: Answers, context: Context) -> int:
    model = _model_or_none(answers, context)
    return model_dimension(model) if model is not None else 3


def _properties(answers: Answers) -> list[str]:
    return answers.get('properties') or []


def _property_choices(answers: Answers, context: Context) -> list[tuple[str, str]]:
    model = _model_or_none(answers, context)
    spinful = bool(model is not None and model.spinful)
    three_d = model is not None and model_dimension(model) == 3
    return [
        (name, label)
        for name, label in MODEL_PROPERTIES
        if (spinful or name not in SPIN_ONLY) and (three_d or name not in THREE_D_ONLY)
    ]


def _path_defaults(answers: Answers, context: Context) -> tuple[str, list]:
    model = _model_or_none(answers, context)
    if model is None:
        return 'gG-X', [('gG', 0.0, 0.0, 0.0), ('X', 0.5, 0.0, 0.0)]
    return model_band_path(model)


def _energy_default(answers: Answers, context: Context, index: int) -> float:
    model = _model_or_none(answers, context)
    if model is None:
        return (-4.0, 4.0)[index]
    low, high = model_energy_range(model)
    return float(np.floor(low - 0.5)) if index == 0 else float(np.ceil(high + 0.5))


def _validate_path(value: str, answers: Answers, context: Context) -> str | None:
    from ..band_paths import check_band_path

    points = answers.get('high_sym') or _path_defaults(answers, context)[1]
    return check_band_path(value, points)


def _summary_note(answers: Answers, context: Context) -> str:
    title = TB_MODELS.get(answers.get('tb_model'), ('', ''))[1]
    try:
        return '  %s.\n%s' % (title, model_summary(build_model(answers, context)))
    except ValueError as exc:
        return '  The model cannot be built with these parameters: %s' % exc


def _parameter_question(name: str, function: str, parameter: inspect.Parameter) -> Question:
    """The question of one parameter of a model function (kind, default and help from it)."""
    from .docs import docstring_parameters

    from ...models import models

    doc = docstring_parameters(getattr(models, function)).get(parameter.name, '')
    settings = dict(SPECIAL_PARAMETERS.get((name, parameter.name), {}))
    default = settings.pop('default', parameter.default)
    if 'kind' not in settings:
        if isinstance(parameter.default, bool):
            settings['kind'] = 'bool'
        elif isinstance(parameter.default, int):
            settings['kind'] = 'int'
        elif isinstance(parameter.default, float):
            settings['kind'] = 'float'
        else:
            settings['kind'] = 'text'
    if settings['kind'] == 'choice':
        default = str(default)
    unit = ''
    first = doc.split('.')[0]
    for text, symbol in (('(eV)', 'eV'), ('(bohr)', 'bohr'), ('(angstrom)', 'Å')):
        if text in first:
            unit = symbol
    if (name, parameter.name) == ('tmd', 'soc'):
        settings = {'kind': 'text', 'parse': _soc_value}
        default = 'True'
    return Question(
        '%s_%s' % (name, parameter.name),
        '%s (%s)' % (parameter.name, first.strip() or 'see the model function'),
        default=default,
        unit=unit,
        label=parameter.name,
        doc='PAOFLOW.models.models.%s:%s' % (function, parameter.name),
        when=lambda a, c, name=name: _chosen(a) == name,
        section=MODEL,
        **settings,
    )


def _soc_value(text: str) -> bool | float:
    """The ``soc`` of the TMD model: True (tabulated), False, or a coupling in eV."""
    lowered = text.strip().lower()
    if lowered in ('true', 'yes', 'y', ''):
        return True
    if lowered in ('false', 'no', 'n', '0'):
        return False
    try:
        return float(text)
    except ValueError:
        raise ValueError('give True, False or a spin-orbit coupling in eV') from None


def _parameter_items() -> tuple[Question, ...]:
    return tuple(
        _parameter_question(name, function, parameter)
        for name, (function, _title) in TB_MODELS.items()
        for parameter in model_parameters(name)
    )


def _nonlinear_items() -> tuple:
    """The non-linear block of the property run without what needs a PAO basis or a molecule."""
    from dataclasses import replace

    left_out = {'nlo_molecule', 'nlo_dipoles', 'nlo_coupling', 'nlo_coupled'}
    items = []
    for item in NONLINEAR_ITEMS:
        if item.section == COUPLED or getattr(item, 'key', None) in left_out:
            continue
        if getattr(item, 'key', None) == 'nlo_nk':
            item = replace(item, default=lambda a, c: NLO_GRID[_dimension(a, c)])
        own = item.when

        def when(a, c, own=own):
            return 'nonlinear' in _properties(a) and (own is None or own(a, c))

        items.append(replace(item, when=when))
    return tuple(items)


TB_MODEL_ITEMS = (
    Question(
        'tb_model',
        'Tight-binding model (%s)' % ', '.join(TB_MODELS),
        kind='choice',
        choices=list(TB_MODELS),
        default='honeycomb',
        label='Model',
        help='The hopping-list models of PAOFLOW.models.models (Slater–Koster models come later).',
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
        section=PROPERTIES,
    ),
    Question('npool', 'npool', kind='int', default=1, section=GRIDS),
    Question('smearing', 'Smearing (gauss / m-p / None)', default='gauss', section=GRIDS),
    Question(
        'nk',
        'Band path k-points (NK)',
        kind='int',
        default=200,
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
        help='Crystal coordinates of the reciprocal vectors of the model; the defaults follow its '
        'lattice (1D chain, hexagonal or square 2D, cubic 3D).',
        section=BAND_PATH,
        level='advanced',
    ),
    Question(
        'band_path',
        'Band path (labels joined by "-", segments split by "|")',
        default=lambda a, c: _path_defaults(a, c)[0],
        when=lambda a, c: 'bands' in _properties(a),
        validate=_validate_path,
        label='Band path',
        help='gG is Γ; the points are in the advanced table below.',
        section=BAND_PATH,
        widget='bandpath',
    ),
    Note(
        '\nEnergy grid for DOS / transport / Hall properties (eV; the model sets its own zero).',
        when=lambda a, c: bool(
            set(_properties(a)) & {'dos', 'transport', 'spin_Hall', 'anomalous_Hall'}
        ),
        section=GRIDS,
    ),
    Question(
        'emin',
        '  EMIN',
        kind='float',
        default=lambda a, c: _energy_default(a, c, 0),
        when=lambda a, c: bool(
            set(_properties(a)) & {'dos', 'transport', 'spin_Hall', 'anomalous_Hall'}
        ),
        unit='eV',
        help='Default: below the lowest band of the model.',
        section=GRIDS,
    ),
    Question(
        'emax',
        '  EMAX',
        kind='float',
        default=lambda a, c: _energy_default(a, c, 1),
        when=lambda a, c: bool(
            set(_properties(a)) & {'dos', 'transport', 'spin_Hall', 'anomalous_Hall'}
        ),
        unit='eV',
        help='Default: above the highest band of the model.',
        section=GRIDS,
    ),
    Question(
        'ne',
        '  NE (number of energy points)',
        kind='int',
        default=1000,
        when=lambda a, c: bool(
            set(_properties(a)) & {'dos', 'transport', 'spin_Hall', 'anomalous_Hall'}
        ),
        section=GRIDS,
    ),
    Question(
        'do_pdos',
        'Projected DOS (on the orbitals of the model) as well?',
        kind='bool',
        default=True,
        when=lambda a, c: 'dos' in _properties(a),
        section=GRIDS,
    ),
    Question(
        'nfft',
        'Dense k-grid per periodic direction (interpolated Hamiltonian)',
        kind='int',
        default=lambda a, c: DENSE_GRID[_dimension(a, c)],
        help='Points along each periodic direction of the model (1 along the others) for the '
        'DOS, transport, Hall, Fermi surface and optics.',
        section=GRIDS,
    ),
    *_nonlinear_items(),
    Question(
        'plot',
        'Also generate a plotting script (plot.py)?',
        kind='bool',
        default=True,
        section='Output',
        gui=False,  # the Studio: "Write the plot scripts" of its Output box (options.plot)
    ),
)


def assemble_tb_model(answers: Answers, context: Context) -> dict:
    """Configuration of the model run (:func:`PAOFLOW.gen.tb_model_driver.build_tb_model_script`).

    Parameters
    ----------
    answers : dict
        Answers of :data:`TB_MODEL_ITEMS`.
    context : Context
        The session context.

    Returns
    -------
    dict
        ``tb_model``, ``model_function``, ``model_arguments``, ``periodic``, ``is_2d``,
        ``spinful``, the property-run keys (``properties``, ``nk``, ``band_path``, ``high_sym``,
        ``emin``, ``emax``, ``ne``, ``do_pdos``, ``nfft``, ``npool``, ``smearing``, ``ibrav`` = 0,
        ``outputdir``) and the ``nlo_*`` keys.

    Raises
    ------
    ValueError
        When the model cannot be built with the parameters.
    """
    model = build_model(answers, context)
    periodic = periodic_axes(model)
    props = list(answers['properties'])
    path, points = _path_defaults(answers, context)
    cfg = {
        'workdir': context.workdir,
        'outputdir': 'output',
        'tb_model': answers['tb_model'],
        'model_function': TB_MODELS[answers['tb_model']][0],
        'model_arguments': model_arguments(answers),
        'periodic': periodic,
        'is_2d': model_dimension(model) == 2,
        'spinful': bool(model.spinful),
        'projector': 'model',
        'properties': props,
        'npool': answers['npool'],
        'smearing': None if answers['smearing'].lower() == 'none' else answers['smearing'],
        'ibrav': 0,
        'nk': answers['nk'] if 'bands' in props else 200,
    }
    if 'bands' in props:
        rows = answers.get('high_sym') or points
        cfg['band_path'] = answers.get('band_path') or path
        cfg['high_sym'] = {row[0]: tuple(row[1:]) for row in rows}
    energies = bool(set(props) & {'dos', 'transport', 'spin_Hall', 'anomalous_Hall'})
    cfg['emin'] = answers['emin'] if energies else _energy_default(answers, context, 0)
    cfg['emax'] = answers['emax'] if energies else _energy_default(answers, context, 1)
    cfg['ne'] = answers['ne'] if energies else 1000
    cfg['do_pdos'] = answers['do_pdos'] if 'dos' in props else False
    cfg['nfft'] = answers['nfft']
    if 'nonlinear' in props:
        assemble_nonlinear(
            cfg, {**answers, 'nlo_molecule': False, 'nlo_dipoles': False, 'nlo_coupled': False}
        )
    cfg['plot'] = answers['plot']
    return cfg


TB_MODEL = Workflow(
    name='tb_model',
    title='Property run from a TB model',
    description='The properties of the run (bands, DOS, Hall, topology, optics, non-linear '
    'optics) on a tight-binding model of PAOFLOW.models: no DFT run, the parameters of the model '
    'instead (main.py and plot.py).',
    items=TB_MODEL_ITEMS,
    assemble=assemble_tb_model,
)
