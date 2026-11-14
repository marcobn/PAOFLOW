"""Questions of the linear-response property: one Kubo hook for Hall, Edelstein and conductivity.

The response of an observable A to an electric field E_j is the Kubo-Bastin tensor of the
``linear_response`` tutorial (tutorial03), the sum of a Fermi-surface term (Eq. 1, proportional to
the lifetime) and a Fermi-sea term (Eq. 2; Eq. 3 is its Γ → 0 limit, the Berry-curvature form).
The questions ask for the observable, the term, its variant and the components. The script
builder (:func:`PAOFLOW.gen.paoflow_driver._linear_response_lines`) maps the answers onto the
PAOFLOW calls that compute that term today, without changing them:

=================  ===========================================  =================================================
observable         Fermi sea (intrinsic)                        Fermi surface (∝ τ)
=================  ===========================================  =================================================
charge current     ``anomalous_Hall`` (spin-orbit runs) or      ``linear_response('cond')``: total (Eq. 1),
                   ``linear_response('ahc')`` (Γ → 0, Eq. 3);   intraband (Eq. 4) or interband (Eq. 5)
                   ``linear_response('ahc', full_chi2)`` (Eq. 2)
spin current       ``spin_Hall`` (Γ → 0)                        ``linear_response('shc', t_odd)`` (Eq. 1)
orbital current    ``orbital_Hall`` (Γ → 0)                     --
spin density       ``linear_response('ree', t_odd)``:           ``linear_response('ree')``: total, intraband or
                   Γ → 0 (Eq. 3) or finite Γ (Eq. 2)            interband
=================  ===========================================  =================================================

Variants that the library cannot run correctly yet are not offered:
- the finite-Γ and split spin-current terms read the tensor list with another index order;
- the orbital current has no Fermi-surface term.
The time-reversal parity of the result is printed, not asked: a Fermi-sea spin-density response
or a Fermi-surface spin-current response is T-odd and needs a magnetic system.
"""

from __future__ import annotations

from typing import Any, Callable

from .model import Answers, Context, Note, Question

LINEAR_RESPONSE = 'Linear response (Kubo)'

#: Observables: key -> (label, needs spin-orbit, number of component letters).
OBSERVABLES: dict[str, tuple[str, bool, int]] = {
    'charge_current': ('charge current: conductivity, anomalous Hall', False, 2),
    'spin_current': ('spin current: spin Hall (spin-orbit runs)', True, 3),
    'orbital_current': ('orbital current: orbital Hall (spin-orbit runs)', True, 3),
    'spin_density': ('spin density: Rashba-Edelstein (spin-orbit runs)', True, 2),
}

#: Terms available per observable.
TERMS: dict[str, tuple[str, ...]] = {
    'charge_current': ('fermi_sea', 'fermi_surface', 'both'),
    'spin_current': ('fermi_sea', 'fermi_surface', 'both'),
    'orbital_current': ('fermi_sea',),
    'spin_density': ('fermi_sea', 'fermi_surface', 'both'),
}

#: Fermi-sea variants per observable (``gamma_to_zero``: Eq. 3; ``finite_gamma``: Eq. 2).
SEA_VARIANTS: dict[str, tuple[str, ...]] = {
    'charge_current': ('gamma_to_zero', 'finite_gamma'),
    'spin_current': ('gamma_to_zero',),
    'orbital_current': ('gamma_to_zero',),
    'spin_density': ('gamma_to_zero', 'finite_gamma'),
}

#: Fermi-surface variants per observable (``total``: Eq. 1; ``intraband``: Eq. 4; ``interband``: Eq. 5).
SURFACE_VARIANTS: dict[str, tuple[str, ...]] = {
    'charge_current': ('total', 'intraband', 'interband'),
    'spin_current': ('total',),
    'orbital_current': (),
    'spin_density': ('total', 'intraband', 'interband'),
}

#: Default components: (polarization,) response, field.
DEFAULT_COMPONENTS = {
    'charge_current': 'xy',
    'spin_current': 'zxy',
    'orbital_current': 'zxy',
    'spin_density': 'yx',
}

#: Time-reversal parity of each (observable, term): the Fermi-sea term of a T-odd observable and
#: the Fermi-surface term of a T-even one are T-even (allowed in non-magnetic systems).
PARITY = {
    ('charge_current', 'fermi_sea'): 'T-odd (anomalous Hall: magnetic systems)',
    ('charge_current', 'fermi_surface'): 'T-even (ordinary conductivity)',
    ('spin_current', 'fermi_sea'): 'T-even (intrinsic spin Hall)',
    ('spin_current', 'fermi_surface'): 'T-odd (magnetic spin Hall: magnetic systems)',
    ('orbital_current', 'fermi_sea'): 'T-even (intrinsic orbital Hall)',
    ('spin_density', 'fermi_sea'): 'T-odd (magnetic Edelstein: magnetic systems)',
    ('spin_density', 'fermi_surface'): 'T-even (Rashba-Edelstein effect)',
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
        Keys ``lr_observable``, ``lr_term``, ``lr_sea``, ``lr_surface``, ``lr_components``,
        ``lr_gamma``.
    """

    def observables(answers: Answers, context: Context) -> list[str]:
        allowed = spinful(answers, context)
        return [
            key
            for key, (_label, needs_so, _n) in OBSERVABLES.items()
            if (allowed or not needs_so) and (orbital or key != 'orbital_current')
        ]

    def observable(answers: Answers) -> str:
        return answers.get('lr_observable') or 'charge_current'

    def term_has(answers: Answers, term: str) -> bool:
        return answers.get('lr_term') in (term, 'both')

    def describe(answers: Answers, context: Context) -> str:
        key = observable(answers)
        terms = (
            ('fermi_sea', 'fermi_surface')
            if answers.get('lr_term') == 'both'
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
            'orbital current: orbital Hall; spin density: Rashba-Edelstein. The spin and orbital '
            'observables need a spin-orbit (spinor) calculation.',
            section=section,
        ),
        Question(
            'lr_term',
            'Term (fermi_sea: intrinsic, Berry-curvature-like; fermi_surface: proportional to the '
            'lifetime; both)',
            kind='choice',
            choices=lambda a, c: list(TERMS[observable(a)]),
            default='fermi_sea',
            when=lambda a, c: _wants(a),
            label='Term',
            help='Kubo-Bastin: Fermi-surface term (Eq. 1 of tutorial03) and Fermi-sea term (Eq. 2; '
            'Eq. 3 is its Γ → 0 limit).',
            section=section,
        ),
        Note(describe, when=lambda a, c: _wants(a), section=section),
        Question(
            'lr_sea',
            'Fermi-sea term (gamma_to_zero: Berry curvature, Eq. 3; finite_gamma: Eq. 2)',
            kind='choice',
            choices=lambda a, c: list(SEA_VARIANTS[observable(a)]),
            default='gamma_to_zero',
            when=lambda a, c: _wants(a) and term_has(a, 'fermi_sea'),
            label='Fermi-sea variant',
            section=section,
            level='advanced',
        ),
        Question(
            'lr_surface',
            'Fermi-surface term (total: Eq. 1; intraband: Eq. 4, Boltzmann-like; interband: Eq. 5)',
            kind='choice',
            choices=lambda a, c: list(SURFACE_VARIANTS[observable(a)]) or ['total'],
            default='total',
            when=lambda a, c: _wants(a) and term_has(a, 'fermi_surface'),
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
            when=lambda a, c: _wants(a)
            and (
                term_has(a, 'fermi_surface')
                or (term_has(a, 'fermi_sea') and a.get('lr_sea') == 'finite_gamma')
            ),
            doc='PAOFLOW.PAOFLOW.PAOFLOW.linear_response:gamma',
            section=section,
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
        ``lr_components`` (list of letter strings), ``lr_gamma`` and ``lr_spinful``.
    """
    observable = answers.get('lr_observable', 'charge_current')
    term = answers.get('lr_term', 'fermi_sea')
    cfg['lr_observable'] = observable
    cfg['lr_term'] = term
    cfg['lr_sea'] = answers.get('lr_sea', 'gamma_to_zero')
    cfg['lr_surface'] = answers.get('lr_surface', 'total')
    cfg['lr_components'] = list(answers.get('lr_components') or [DEFAULT_COMPONENTS[observable]])
    cfg['lr_gamma'] = float(answers.get('lr_gamma', 0.01))
    cfg['lr_spinful'] = bool(spinful)
    return cfg


def linear_response_calls(cfg: dict) -> list[tuple[str, dict[str, Any], str]]:
    """The PAOFLOW calls of the chosen term(s): ``(method, keyword arguments, comment)``.

    Parameters
    ----------
    cfg : dict
        Run configuration with the ``lr_*`` keys and ``is_2d``.

    Returns
    -------
    list of tuple
        One entry per call. Component indices follow each method's convention: ``spin_Hall`` and
        ``orbital_Hall`` take (current, field, polarization); ``linear_response`` takes
        ``s_tensor`` as (polarization, current, field) and ``a_tensor`` as (response, field).
    """
    axis = {'x': 0, 'y': 1, 'z': 2}
    observable = cfg['lr_observable']
    term = cfg['lr_term']
    components = [[axis[c] for c in comp] for comp in cfg['lr_components']]
    two_d = bool(cfg.get('is_2d', False))
    gamma = float(cfg['lr_gamma'])
    terms = ('fermi_sea', 'fermi_surface') if term == 'both' else (term,)
    calls = []

    def kubo(response: str, comment: str, **flags: Any) -> tuple[str, dict, str]:
        # linear_response reads a_tensor only when s_tensor is given: both are always passed.
        if observable in ('spin_current', 'orbital_current'):
            s_tensor = components
            a_tensor = [comp[1:] for comp in components]
        else:
            a_tensor = components
            s_tensor = [[2] + comp for comp in components]
        kwargs = {'response': response, 'gamma': gamma, 'twoD': two_d, **flags,
                  's_tensor': s_tensor, 'a_tensor': a_tensor}  # fmt: skip
        return ('linear_response', kwargs, comment)

    for part in terms:
        if part == 'fermi_sea':
            finite = cfg['lr_sea'] == 'finite_gamma'
            if observable == 'charge_current':
                if finite:
                    calls.append(
                        kubo('ahc', 'anomalous Hall, Fermi sea at finite Γ (Eq. 2)', full_chi2=True)
                    )
                elif cfg['lr_spinful']:
                    calls.append(('anomalous_Hall', {'do_ac': True, 'a_tensor': components},
                                  'anomalous Hall, Berry curvature (Eq. 3), with the magneto-optical MCD'))  # fmt: skip
                else:
                    calls.append(kubo('ahc', 'anomalous Hall, Berry curvature (Eq. 3)'))
            elif observable == 'spin_current':
                tensor = [[c[1], c[2], c[0]] for c in components]
                calls.append(('spin_Hall', {'twoD': two_d, 's_tensor': tensor},
                              'intrinsic spin Hall, spin Berry curvature (Eq. 3)'))  # fmt: skip
            elif observable == 'orbital_current':
                tensor = [[c[1], c[2], c[0]] for c in components]
                calls.append(('orbital_Hall', {'twoD': two_d, 'o_tensor': tensor},
                              'intrinsic orbital Hall, orbital Berry curvature (Eq. 3)'))  # fmt: skip
            else:
                label = 'finite Γ (Eq. 2)' if finite else 'Γ → 0 (Eq. 3)'
                calls.append(kubo('ree', 'magnetic (T-odd) Edelstein, Fermi sea, %s' % label,
                                  t_odd=True, full_chi2=finite))  # fmt: skip
        else:
            variant = cfg['lr_surface']
            flags = {'intraband': True} if variant == 'intraband' else {}
            flags.update({'interband': True} if variant == 'interband' else {})
            equation = {'total': 'Eq. 1', 'intraband': 'Eq. 4', 'interband': 'Eq. 5'}[variant]
            if observable == 'charge_current':
                calls.append(
                    kubo(
                        'cond',
                        'conductivity, Fermi surface, %s (%s)' % (variant, equation),
                        **flags,
                    )
                )
            elif observable == 'spin_current':
                calls.append(
                    kubo('shc', 'magnetic (T-odd) spin Hall, Fermi surface (Eq. 1)', t_odd=True)
                )
            elif observable == 'spin_density':
                calls.append(
                    kubo(
                        'ree',
                        'Rashba-Edelstein, Fermi surface, %s (%s)' % (variant, equation),
                        **flags,
                    )
                )
    return calls
