"""Run configuration of the sparse engine.

``PAOFLOW(..., sparse=True, sparse_config={...})`` selects the sparse engine (``sparse=False``,
the default, keeps the dense pipeline); ``None`` or ``{}`` takes every default.
:meth:`SparseConfig.parse` validates the dict into an immutable :class:`SparseConfig` (internal,
never built by users). Every sparse knob lives here, so none widens the :class:`PAOFLOW.PAOFLOW`
constructor or a method shared with the dense pipeline. The solve window has no option: it
follows the property ranges (:mod:`PAOFLOW.sparse.window`); ``nev`` fixes the band count
instead and ``interior_window`` switches to an interior solve.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from difflib import get_close_matches

from .solver import DENSE_N_MAX, DENSE_RATIO

HK_SOLVERS = ('auto', 'sparse', 'dense')

# element threshold (eV) used when neither a threshold nor a real-space cutoff is given
DEFAULT_THRESHOLD = 1.0e-3
# default distance (eV) below the PAO shift where the well-projected range
# ends: twice the spread of the shifted states measured off the DFT grid
SHIFT_MARGIN = 0.2


def resolve_threshold(
    threshold: float | None,
    rcut: float | None = None,
    bond_order: int | None = None,
    name: str = 'threshold',
) -> float:
    """The element threshold (eV) a truncation actually applies.

    Parameters
    ----------
    threshold : float or None
        Requested threshold; ``None`` picks the default for the truncation mode.
    rcut, bond_order : optional
        Real-space cutoff, as a radius (Bohr) or a neighbour-shell count.
    name : str, default ``'threshold'``
        Name used in the error (``'hopping_threshold'`` in ``sparse_config``).

    Returns
    -------
    float
        ``0.0`` with a real-space cutoff, else ``threshold`` or :data:`DEFAULT_THRESHOLD`.

    Raises
    ------
    ValueError
        If a positive ``threshold`` is combined with ``rcut`` or ``bond_order``.

    Notes
    -----
    A cutoff keeps whole atom-pair blocks by bond length, which every space-group operation
    preserves. A magnitude cut does not: equivalent bonds carry rotated orbitals, hence
    differently sized elements, so it splits protected degeneracies (Si: ``bond_order=36``
    keeps them all, adding a 1e-3 eV threshold splits them by 22 meV). Hence exclusive.
    """
    geometric = rcut is not None or bond_order is not None
    if threshold is None:
        return 0.0 if geometric else DEFAULT_THRESHOLD
    threshold = float(threshold)
    if geometric and threshold > 0.0:
        raise ValueError(
            '%s cannot be combined with rcut or bond_order: an element threshold '
            'breaks the point-group symmetry a real-space cutoff preserves. Give either a '
            '%s (eV) or a real-space cutoff, not both.' % (name, name)
        )
    return threshold


def _reject_unknown_keys(value: dict, names: list, what: str) -> None:
    """Raise ``ValueError`` naming every key of ``value`` not in ``names``."""
    unknown = [k for k in value if k not in names]
    if not unknown:
        return
    lines = []
    for key in unknown:
        close = get_close_matches(str(key), names, n=1)
        lines.append('%r%s' % (key, " (did you mean '%s'?)" % close[0] if close else ''))
    raise ValueError(
        '%s: unknown option%s %s. Valid options: %s.'
        % (what, 's' if len(unknown) > 1 else '', ', '.join(lines), ', '.join(names))
    )


@dataclass(frozen=True)
class InteriorWindow:
    """The ``'interior_window'`` option, documented under :class:`SparseConfig`.

    Attributes
    ----------
    elo, ehi : float or None
        Fixed bounds (eV), or both ``None`` for automatic bounds.
    """

    elo: float | None = None
    ehi: float | None = None

    def __post_init__(self):
        if (self.elo is None) != (self.ehi is None):
            raise ValueError(
                "sparse_config=: 'interior_window' takes both 'elo' and 'ehi' (eV), or neither "
                'for automatic bounds.'
            )
        if self.elo is None:
            return
        for name in ('elo', 'ehi'):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not self.ehi > self.elo:
            raise ValueError(
                "sparse_config=: 'interior_window' needs ehi > elo, got [%g, %g]."
                % (self.elo, self.ehi)
            )

    @property
    def auto(self) -> bool:
        """Whether the bounds come from the property ranges."""
        return self.elo is None

    @classmethod
    def parse(cls, value) -> InteriorWindow:
        """The ``'interior_window'`` value as an :class:`InteriorWindow`.

        Parameters
        ----------
        value : dict or InteriorWindow
            ``{}`` for automatic bounds, or ``{'elo': ..., 'ehi': ...}``.

        Raises
        ------
        TypeError
            If ``value`` is not a dict.
        ValueError
            On an unknown key, a single bound, or ``ehi <= elo``.
        """
        return _parse_window(cls, value, 'interior_window', ())


def _parse_window(cls, value, key: str, required: tuple):
    """Parser of the window option (a dict of the fields of ``cls``)."""
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise TypeError(
            "sparse_config=: '%s' takes a dict, {} for automatic bounds or "
            "{'elo': ..., 'ehi': ...}; got %r." % (key, type(value).__name__)
        )
    _reject_unknown_keys(value, [f.name for f in fields(cls)], "sparse_config= '%s'" % key)
    return cls(**value)


@dataclass(frozen=True)
class SparseConfig:
    """Truncation, solver and resource settings of a sparse run.

    Built by :meth:`parse` from the ``sparse_config=`` dict of
    :class:`PAOFLOW.PAOFLOW`; each attribute below is a key of that dict.

    Attributes
    ----------
    hopping_threshold : float or None
        Magnitude (eV) below which H(R) elements are dropped when the base cell becomes the
        bond list; ``None`` means 1e-3 eV without a real-space cutoff, no element cut with one.
        The conversion prints a rigorous bound on the resulting eigenvalue error at any k. Most
        compact, but it splits symmetry-protected degeneracies (see :func:`resolve_threshold`).
        A positive value excludes ``rcut`` and ``bond_order``.
    rcut : float or None
        Cutoff (Bohr) on the physical bond length at the base cell, replacing the threshold. It
        keeps whole atom-pair blocks, so it respects the space group. The radius is snapped into
        the gap above the outermost shell it reaches (a shell within 1e-3 Bohr counts), so it
        cannot keep part of a shell. Raises past the aliasing-safe radius or at a bond the grid
        stores at a non-shortest image (as does ``bond_order``). Excludes ``bond_order``.
    bond_order : int or None
        The same cutoff as a shell count: keep bonds up to the n-th distinct interatomic
        distance (1 = nearest neighbours); see :func:`PAOFLOW.sparse.shells.shell_cutoff`.
    hk_solver : {'auto', 'sparse', 'dense'}
        Per-k eigensolver only (H(R) stays a bond list): ``'dense'`` is LAPACK on one
        ``(n, n)`` scratch; ``'auto'`` dispatches on ``(nawf, nev)``, see
        :func:`PAOFLOW.sparse.solver.select_hk_solver`.
    nev : int or None
        Bands per k-point from the bottom. ``None`` (default) derives it each pass: top =
        highest property ``emax`` + margin (:mod:`PAOFLOW.sparse.window`), capped at the
        well-projected top, bands counted at a few probe k-points. Set it to skip the probe (a
        dense eigensolve per probe k on a large cell) or fix the count across runs. Excludes
        ``interior_window``.
    interior_window : dict or None
        Solve only inside a window: ``{}`` places it around each pass's property ranges,
        ``{'elo': ..., 'ehi': ...}`` fixes it (eV); ``None`` (default) solves from the bottom.
        This is where the iterative kernel pays (a narrow window is a small part of the
        spectrum, a from-the-bottom one 20-50%), but states below ``elo`` are never computed,
        losing the electron count (so carrier density, Hall coefficient), DoS/PDoS outside the
        window (absent, not zero) and band indices (the count varies with k). Properties that
        need them are skipped with a warning and listed again at ``finish_execution``, a
        deliberate exception to fail-loud since an interior run is usually a batch only partly
        supportable. Parsed to :class:`InteriorWindow`. Excludes ``nev``.
    dense_n_max : int, default :data:`~PAOFLOW.sparse.solver.DENSE_N_MAX`
        Largest ``nawf`` admitting a per-k ``(n, n)`` dense scratch (dense ``hk_solver``,
        full-spectrum properties, window probe); above it those raise ``NotImplementedError``
        rather than allocate. A policy value, not a LAPACK limit; raise it if memory allows.
    dense_ratio : float, default :data:`~PAOFLOW.sparse.solver.DENSE_RATIO`
        Spectrum fraction ``(nev + guard) / n`` above which ``'auto'`` takes the dense kernel
        instead of ARPACK; in ``(0, 1]``.
    shift_margin : float, default :data:`SHIFT_MARGIN`
        Distance (eV) below the PAO shift ``attr['shift']`` where the well-projected range
        ends. Non-projectable states sit exactly at the shift on the DFT grid but spread at
        interpolated k (example01: 0.09 eV below to 0.15 eV above), so every state at or above
        ``shift - shift_margin`` is dropped at every k, and windows and property ranges stay
        below it (``SparseEngine._projectable_top``). Real states in that stretch go too.
    """

    hopping_threshold: float | None = None
    rcut: float | None = None
    bond_order: int | None = None
    hk_solver: str = 'auto'
    nev: int | None = None
    interior_window: InteriorWindow | None = None
    dense_n_max: int = DENSE_N_MAX
    dense_ratio: float = DENSE_RATIO
    shift_margin: float = SHIFT_MARGIN

    def __post_init__(self):
        if self.rcut is not None and self.bond_order is not None:
            raise ValueError(
                'sparse_config=: give either rcut (Bohr) or bond_order (shells), not both.'
            )
        if self.hk_solver not in HK_SOLVERS:
            raise ValueError(
                'sparse_config=: hk_solver must be one of %s, got %r.'
                % (HK_SOLVERS, self.hk_solver)
            )
        object.__setattr__(self, 'dense_n_max', int(self.dense_n_max))
        object.__setattr__(self, 'dense_ratio', float(self.dense_ratio))
        object.__setattr__(self, 'shift_margin', float(self.shift_margin))
        if self.shift_margin < 0.0:
            raise ValueError(
                'sparse_config=: shift_margin must be >= 0 eV, got %g.' % self.shift_margin
            )
        if self.dense_n_max < 1:
            raise ValueError('sparse_config=: dense_n_max must be >= 1, got %d.' % self.dense_n_max)
        if not 0.0 < self.dense_ratio <= 1.0:
            raise ValueError(
                'sparse_config=: dense_ratio must lie in (0, 1], got %g.' % self.dense_ratio
            )
        # validates the threshold/cutoff combination; the field keeps the
        # user's value (None stays None) so the log shows what was asked for
        resolve_threshold(self.hopping_threshold, self.rcut, self.bond_order, 'hopping_threshold')
        # frozen: normalize types through object.__setattr__
        if self.hopping_threshold is not None:
            object.__setattr__(self, 'hopping_threshold', float(self.hopping_threshold))
        if self.rcut is not None:
            object.__setattr__(self, 'rcut', float(self.rcut))
        if self.bond_order is not None:
            object.__setattr__(self, 'bond_order', int(self.bond_order))
        if self.nev is not None:
            if isinstance(self.nev, bool) or int(self.nev) != self.nev or self.nev < 1:
                raise ValueError('sparse_config=: nev must be an integer >= 1, got %r.' % self.nev)
            object.__setattr__(self, 'nev', int(self.nev))
        if self.interior_window is not None:
            object.__setattr__(self, 'interior_window', InteriorWindow.parse(self.interior_window))
        if self.nev is not None and self.interior_window is not None:
            raise ValueError(
                "sparse_config=: give either 'nev' or 'interior_window', not both: an interior "
                'solve finds every state inside the window, however many there are at each '
                'k-point.'
            )

    @property
    def limits(self) -> dict:
        """The dispatch limits as keyword arguments of the solver functions."""
        return {'dense_ratio': self.dense_ratio, 'dense_n_max': self.dense_n_max}

    @classmethod
    def parse(cls, value) -> SparseConfig:
        """The ``sparse_config=`` constructor argument as a config.

        Parameters
        ----------
        value : dict or None
            The options; ``None`` or ``{}`` takes every default.

        Raises
        ------
        TypeError
            If ``value`` is neither ``None`` nor a dict.
        ValueError
            On an unknown key or inconsistent options.
        """
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError(
                'sparse_config= takes a dict of options (None for the defaults); got %r.'
                % type(value).__name__
            )
        _reject_unknown_keys(value, [f.name for f in fields(cls)], 'sparse_config=')
        return cls(**value)
