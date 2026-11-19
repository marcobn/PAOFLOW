"""Questions of the linear-response property: one Kubo-Bastin hook for every observable and term.

The response of an observable A to an electric field E_j is the Kubo-Bastin tensor at constant
broadening Γ = ħ/2τ, the sum of a Fermi-surface term (∝ τ in clean metals) and a Fermi-sea term
(its Γ → 0 limit is the Berry-curvature form). The questions ask for the observable, the term, its
variant, the components, Γ and the temperature; the script builder
(:func:`PAOFLOW.gen.paoflow_driver._linear_response_lines`) writes one
:meth:`PAOFLOW.PAOFLOW.PAOFLOW.kubo_response` call:

=================  ==========================================================================
answer             ``terms`` of ``kubo_response``
=================  ==========================================================================
fermi_sea          ``intrinsic`` (Γ → 0, Berry curvature) or ``sea`` (finite Γ, exact)
fermi_surface      ``surface`` (exact; tutorial03 Eq. 1), ``surface_intraband`` (Boltzmann-like)
                   or ``surface_interband``
both               ``total`` plus the chosen surface and sea variants
total              ``total`` (the full Kubo-Bastin response)
=================  ==========================================================================

Observables: charge current (conductivity, anomalous Hall), spin current (spin Hall), orbital current
(orbital Hall), spin density (Rashba-Edelstein), orbital density (orbital Edelstein), spin torque
(spin-orbit torkance). Until 14 November 2026 the answers were mapped onto ``spin_Hall``,
``anomalous_Hall`` and the ``linear_response`` equations; the Kubo-Bastin kernel has every term for
every observable, with the sign convention of the Kubo formula (``docs/internals/kubo-bastin.md``).
The time-reversal parity of the result is printed, not asked: a Fermi-sea spin-density response or
a Fermi-surface spin-current response is T-odd and needs a magnetic system.

With ``lr_frequency`` the same observable and components are computed as functions of the photon
energy by :meth:`PAOFLOW.PAOFLOW.PAOFLOW.kubo_optical_response` (Kubo-Greenwood: interband
transitions and the Drude term of the bands at the Fermi level, all three terms written).
"""

from __future__ import annotations

from typing import Any, Callable

from .model import Answers, Context, Note, Question

LINEAR_RESPONSE = 'Linear response (Kubo)'

#: Observables: key -> (label, needs spin-orbit, number of component letters).
OBSERVABLES: dict[str, tuple[str, bool, int]] = {
    'charge_current': ('charge current: conductivity, anomalous Hall', False, 2),
    'spin_current': ('spin current: spin Hall (spin-orbit runs)', True, 3),
    'orbital_current': ('orbital current: orbital Hall', False, 3),
    'spin_density': ('spin density: Rashba-Edelstein (spin-orbit runs)', True, 2),
    'orbital_density': ('orbital density: orbital Edelstein', False, 2),
    'spin_torque': ('spin torque: spin-orbit torkance (spin-orbit runs)', True, 2),
}

#: Observables whose operator comes from the PAO shells (not offered for models).
ORBITAL_OBSERVABLES = ('orbital_current', 'orbital_density')

#: Terms (every observable has all of them).
TERMS: tuple[str, ...] = ('fermi_sea', 'fermi_surface', 'both', 'total')

#: Fermi-sea variants (``gamma_to_zero``: Berry curvature; ``finite_gamma``: exact at Γ).
SEA_VARIANTS: tuple[str, ...] = ('gamma_to_zero', 'finite_gamma')

#: Fermi-surface variants.
SURFACE_VARIANTS: tuple[str, ...] = ('total', 'intraband', 'interband')

#: ``kubo_response`` term of each variant.
SEA_TERM = {'gamma_to_zero': 'intrinsic', 'finite_gamma': 'sea'}
SURFACE_TERM = {
    'total': 'surface',
    'intraband': 'surface_intraband',
    'interband': 'surface_interband',
}

#: Terms of the optical (frequency-dependent) response, all written.
OPTICAL_TERMS: tuple[str, ...] = ('total', 'interband', 'intraband')

#: Default components: (polarization,) response, field.
DEFAULT_COMPONENTS = {
    'charge_current': 'xy',
    'spin_current': 'zxy',
    'orbital_current': 'zxy',
    'spin_density': 'yx',
    'orbital_density': 'yx',
    'spin_torque': 'yx',
}

#: Time-reversal parity of each (observable, term): the Fermi-sea term of a T-odd observable and
#: the Fermi-surface term of a T-even one are T-odd (they need a magnetic system).
PARITY = {
    ('charge_current', 'fermi_sea'): 'T-odd (anomalous Hall: magnetic systems)',
    ('charge_current', 'fermi_surface'): 'T-even (ordinary conductivity)',
    ('spin_current', 'fermi_sea'): 'T-even (intrinsic spin Hall)',
    ('spin_current', 'fermi_surface'): 'T-odd (magnetic spin Hall: magnetic systems)',
    ('orbital_current', 'fermi_sea'): 'T-even (intrinsic orbital Hall)',
    ('orbital_current', 'fermi_surface'): 'T-odd (magnetic orbital Hall: magnetic systems)',
    ('spin_density', 'fermi_sea'): 'T-odd (magnetic Edelstein: magnetic systems)',
    ('spin_density', 'fermi_surface'): 'T-even (Rashba-Edelstein effect)',
    ('orbital_density', 'fermi_sea'): 'T-odd (magnetic systems)',
    ('orbital_density', 'fermi_surface'): 'T-even (orbital Edelstein effect)',
    ('spin_torque', 'fermi_sea'): 'T-even (intrinsic, Berry-curvature torque)',
    ('spin_torque', 'fermi_surface'): 'T-odd (∝ lifetime; field-like torque in magnets)',
}


def parse_components(text: str) -> list[str]:
    """Tensor components written with letters, e.g. ``'zxy, xyz'`` or ``'xy yx'``.

    Parameters
    ----------
    text : str
        Components separated by commas or spaces; each a string of 2 or 3 of x, y, z.

    Returns
    -------
    list of str

    Raises
    ------
    ValueError
        If a component has letters other than x, y, z, or none is given.
    """
    items = [item.strip().lower() for item in text.replace(',', ' ').split() if item.strip()]
    if not items:
        raise ValueError('give at least one component, e.g. zxy or xy')
    for item in items:
        if set(item) - set('xyz') or len(item) not in (2, 3):
            raise ValueError('%r: use 2 or 3 of the letters x, y, z' % item)
    return items


def _wants(answers: Answers) -> bool:
    return 'linear_response' in (answers.get('properties') or [])


def _static(answers: Answers) -> bool:
    return _wants(answers) and not answers.get('lr_frequency')


def _optical(answers: Answers) -> bool:
    return _wants(answers) and bool(answers.get('lr_frequency'))


def linear_response_items(
    spinful: Callable[[Answers, Context], bool],
    section: str = LINEAR_RESPONSE,
    orbital: bool = True,
) -> tuple:
    """The questions of the linear-response property.

    Parameters
    ----------
    spinful : callable
        ``spinful(answers, context)``: the run has spin-orbit (spinor) bands, so the spin and orbital
        observables are offered.
    section : str, optional
        GUI section.
    orbital : bool, optional
        Offer the orbital current (its operator comes from the PAO shells, so not for models).

    Returns
    -------
    tuple of Question and Note
        Keys ``lr_observable``, ``lr_frequency``, ``lr_term``, ``lr_sea``, ``lr_surface``,
        ``lr_components``, ``lr_gamma``, ``lr_omega_max``, ``lr_nomega``, ``lr_temperature``.
    """

    def observables(answers: Answers, context: Context) -> list[str]:
        allowed = spinful(answers, context)
        return [
            key
            for key, (_label, needs_so, _n) in OBSERVABLES.items()
            if (allowed or not needs_so) and (orbital or key not in ORBITAL_OBSERVABLES)
        ]

    def observable(answers: Answers) -> str:
        return answers.get('lr_observable') or 'charge_current'

    def term_has(answers: Answers, term: str) -> bool:
        return answers.get('lr_term') in (term, 'both')

    def describe(answers: Answers, context: Context) -> str:
        key = observable(answers)
        terms = (
            ('fermi_sea', 'fermi_surface')
            if answers.get('lr_term') in ('both', 'total')
            else (answers.get('lr_term'),)
        )
        parities = [PARITY.get((key, t)) for t in terms if PARITY.get((key, t))]
        return '  Response tensor: %s.' % '; '.join(parities) if parities else ''

    def validate_components(value: list[str], answers: Answers, context: Context) -> str | None:
        letters = OBSERVABLES[observable(answers)][2]
        wrong = [c for c in value if len(c) != letters]
        if wrong:
            shape = (
                'polarization, current, field (e.g. zxy)'
                if letters == 3
                else 'response, field (e.g. xy)'
            )
            return '  %s: give %d letters, %s.' % (', '.join(wrong), letters, shape)
        return None

    return (
        Question(
            'lr_observable',
            'Observable that responds to the field (%s)' % ', '.join(OBSERVABLES),
            kind='choice',
            choices=observables,
            default='charge_current',
            when=lambda a, c: _wants(a),
            label='Observable',
            help='charge current: conductivity and anomalous Hall; spin current: spin Hall; '
            'orbital current: orbital Hall; spin density: Rashba-Edelstein; orbital density: '
            'orbital Edelstein; spin torque: spin-orbit torkance. The spin observables need a '
            'spin-orbit (spinor) calculation.',
            section=section,
        ),
        Question(
            'lr_frequency',
            'Frequency-dependent (optical) response, Kubo-Greenwood?',
            kind='bool',
            default=False,
            when=lambda a, c: _wants(a),
            label='Optical (photon energy)',
            help='kubo_optical_response: the same observable and components as functions of the '
            'photon energy; interband transitions and the Drude term of the bands at the Fermi level '
            '(ħ/τ = 2Γ), all three written. Optical and magneto-optical conductivity, AC spin and '
            'orbital Hall, AC Edelstein.',
            section=section,
        ),
        Question(
            'lr_term',
            'Term (fermi_sea: intrinsic, Berry-curvature-like; fermi_surface: proportional to the '
            'lifetime; both: total and the two terms; total)',
            kind='choice',
            choices=lambda a, c: list(TERMS),
            default='fermi_sea',
            when=lambda a, c: _static(a),
            label='Term',
            help='Kubo-Bastin at constant broadening Γ: Fermi-surface and Fermi-sea terms '
            '(Bonbien-Manchon), their sum, or the Γ → 0 Berry-curvature form of the sea term.',
            section=section,
        ),
        Note(describe, when=lambda a, c: _static(a), section=section),
        Question(
            'lr_sea',
            'Fermi-sea term (gamma_to_zero: Berry curvature; finite_gamma: exact at Γ)',
            kind='choice',
            choices=lambda a, c: list(SEA_VARIANTS),
            default='gamma_to_zero',
            when=lambda a, c: _static(a) and term_has(a, 'fermi_sea'),
            label='Fermi-sea variant',
            section=section,
            level='advanced',
        ),
        Question(
            'lr_surface',
            'Fermi-surface term (total; intraband: Boltzmann-like; interband)',
            kind='choice',
            choices=lambda a, c: list(SURFACE_VARIANTS),
            default='total',
            when=lambda a, c: _static(a) and term_has(a, 'fermi_surface'),
            label='Fermi-surface variant',
            section=section,
            level='advanced',
        ),
        Question(
            'lr_components',
            'Components (letters; currents: polarization current field, e.g. zxy; charge current '
            'and spin density: response field, e.g. xy)',
            default=lambda a, c: DEFAULT_COMPONENTS[observable(a)],
            parse=parse_components,
            validate=validate_components,
            when=lambda a, c: _wants(a),
            label='Components',
            help='zxy is σ^z_xy (spin z, current along x, field along y); several components are '
            'separated by commas.',
            section=section,
        ),
        Question(
            'lr_gamma',
            'Spectral broadening Γ = ħ/2τ of the Γ-dependent terms',
            kind='float',
            default=0.01,
            unit='eV',
            when=lambda a, c: _optical(a)
            or (
                _static(a)
                and (
                    a.get('lr_term') == 'total'
                    or term_has(a, 'fermi_surface')
                    or (term_has(a, 'fermi_sea') and a.get('lr_sea') == 'finite_gamma')
                )
            ),
            doc='PAOFLOW.PAOFLOW.PAOFLOW.kubo_response:eta',
            section=section,
        ),
        Question(
            'lr_omega_max',
            'Highest photon energy (0: the energy range of the bands)',
            kind='float',
            default=0.0,
            unit='eV',
            when=lambda a, c: _optical(a),
            doc='PAOFLOW.PAOFLOW.PAOFLOW.kubo_optical_response:omega_max',
            section=section,
        ),
        Question(
            'lr_nomega',
            'Number of photon energies',
            kind='int',
            default=501,
            when=lambda a, c: _optical(a),
            doc='PAOFLOW.PAOFLOW.PAOFLOW.kubo_optical_response:nomega',
            section=section,
            level='advanced',
        ),
        Question(
            'lr_temperature',
            'Electronic temperature of the response',
            kind='float',
            default=0.0,
            unit='K',
            when=lambda a, c: _wants(a),
            doc='PAOFLOW.PAOFLOW.PAOFLOW.kubo_response:temperature',
            section=section,
            level='advanced',
        ),
    )


def assemble_linear_response(cfg: dict, answers: Answers, spinful: bool) -> dict:
    """Add the linear-response keys to a run configuration.

    Parameters
    ----------
    cfg : dict
        Run configuration; modified in place.
    answers : dict
        Answers of :func:`linear_response_items`.
    spinful : bool
        The run has spin-orbit (spinor) bands.

    Returns
    -------
    dict
        ``cfg`` with ``lr_observable``, ``lr_term``, ``lr_sea``, ``lr_surface``,
        ``lr_components`` (list of letter strings), ``lr_gamma``, ``lr_temperature``,
        ``lr_spinful``, ``lr_frequency``, ``lr_omega_max`` and ``lr_nomega``.
    """
    observable = answers.get('lr_observable', 'charge_current')
    term = answers.get('lr_term', 'fermi_sea')
    cfg['lr_observable'] = observable
    cfg['lr_term'] = term
    cfg['lr_sea'] = answers.get('lr_sea', 'gamma_to_zero')
    cfg['lr_surface'] = answers.get('lr_surface', 'total')
    cfg['lr_components'] = list(answers.get('lr_components') or [DEFAULT_COMPONENTS[observable]])
    cfg['lr_gamma'] = float(answers.get('lr_gamma', 0.01))
    cfg['lr_temperature'] = float(answers.get('lr_temperature', 0.0))
    cfg['lr_spinful'] = bool(spinful)
    cfg['lr_frequency'] = bool(answers.get('lr_frequency', False))
    cfg['lr_omega_max'] = float(answers.get('lr_omega_max', 0.0))
    cfg['lr_nomega'] = int(answers.get('lr_nomega', 501))
    return cfg


def kubo_terms(cfg: dict) -> tuple[str, ...]:
    """``terms`` of :meth:`PAOFLOW.PAOFLOW.PAOFLOW.kubo_response` for the answers.

    Parameters
    ----------
    cfg : dict
        Run configuration with ``lr_term``, ``lr_sea`` and ``lr_surface``.

    Returns
    -------
    tuple of str
    """
    term = cfg['lr_term']
    sea = SEA_TERM[cfg.get('lr_sea', 'gamma_to_zero')]
    surface = SURFACE_TERM[cfg.get('lr_surface', 'total')]
    if term == 'fermi_sea':
        return (sea,)
    if term == 'fermi_surface':
        return (surface,)
    if term == 'total':
        return ('total',)
    return tuple(dict.fromkeys(('total', surface, sea)))


def linear_response_calls(cfg: dict) -> list[tuple[str, dict[str, Any], str]]:
    """The PAOFLOW call of the chosen observable and terms: ``(method, keyword arguments, comment)``.

    Parameters
    ----------
    cfg : dict
        Run configuration with the ``lr_*`` keys and ``is_2d``.

    Returns
    -------
    list of tuple
        One :meth:`PAOFLOW.PAOFLOW.PAOFLOW.kubo_response` call: components as letters (the field
        direction last), the terms of :func:`kubo_terms`, Γ, the temperature and ``twoD``. With
        ``lr_frequency``, one :meth:`PAOFLOW.PAOFLOW.PAOFLOW.kubo_optical_response` call with the
        :data:`OPTICAL_TERMS`, Γ, the photon energies, the temperature and ``twoD``.
    """
    observable = cfg['lr_observable']
    if cfg.get('lr_frequency'):
        omega_max = float(cfg.get('lr_omega_max', 0.0))
        kwargs = {
            'observable': observable,
            'components': list(cfg['lr_components']),
            'terms': OPTICAL_TERMS,
            'eta': float(cfg['lr_gamma']),
            'omega_max': omega_max if omega_max > 0.0 else None,
            'nomega': int(cfg.get('lr_nomega', 501)),
            'temperature': float(cfg.get('lr_temperature', 0.0)),
            'twoD': bool(cfg.get('is_2d', False)),
        }
        comment = 'Kubo-Greenwood %s: %s' % (OBSERVABLES[observable][0], ', '.join(OPTICAL_TERMS))
        return [('kubo_optical_response', kwargs, comment)]
    terms = kubo_terms(cfg)
    kwargs = {
        'observable': observable,
        'components': list(cfg['lr_components']),
        'terms': terms,
        'eta': float(cfg['lr_gamma']),
        'temperature': float(cfg.get('lr_temperature', 0.0)),
        'twoD': bool(cfg.get('is_2d', False)),
    }
    comment = 'Kubo-Bastin %s: %s' % (OBSERVABLES[observable][0], ', '.join(terms))
    return [('kubo_response', kwargs, comment)]
