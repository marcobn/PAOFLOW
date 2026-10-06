"""Run configuration of the sparse engine.

Users select the sparse engine with ``PAOFLOW(..., sparse=True)`` and set
its options with a plain dict, ``sparse_config={...}`` (``None`` or ``{}``
takes every default); ``sparse=False`` (the default) keeps the dense
pipeline.  :meth:`SparseConfig.parse` turns that dict into a validated,
immutable :class:`SparseConfig`, which is what the engine holds; the class
itself is internal and never constructed by users.  Every knob of the sparse
backend is a field here, so a new one never widens the
:class:`PAOFLOW.PAOFLOW` constructor or the signature of a method shared
with the dense pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from difflib import get_close_matches

from .solver import DENSE_N_MAX, DENSE_RATIO

HK_SOLVERS = ('auto', 'sparse', 'dense')

# element threshold (eV) used when neither a threshold nor a real-space cutoff is given
DEFAULT_THRESHOLD = 1.0e-3


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
        Requested element threshold; ``None`` picks the default for the
        truncation mode.
    rcut, bond_order : optional
        Real-space cutoff, as a radius (Bohr) or a neighbour-shell count.
    name : str, default ``'threshold'``
        What the caller calls the threshold, for the error message
        (``'hopping_threshold'`` in ``sparse_config``).

    Returns
    -------
    float
        ``0.0`` (keep every element inside the cutoff) when a real-space
        cutoff is given, otherwise ``threshold`` or :data:`DEFAULT_THRESHOLD`.

    Raises
    ------
    ValueError
        If a positive ``threshold`` is combined with ``rcut`` or
        ``bond_order``.

    Notes
    -----
    A real-space cutoff keeps or drops whole atom-pair blocks by bond
    length, a quantity every space-group operation preserves.  An element
    threshold does not: symmetry operations rotate the orbitals of a bond
    into each other, so two equivalent bonds carry differently sized
    elements and a fixed magnitude cut keeps different parts of each,
    which splits symmetry-protected degeneracies (measured on Si:
    ``bond_order=36`` alone keeps every degeneracy, adding a 1e-3 eV
    threshold splits them by 22 meV).  The two are therefore exclusive.
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
class EnergyWindow:
    """The ``'energy_window'`` option: size the per-k solve from the
    property energy range instead of from ``attr['bnd']``.

    Given as a dict, ``sparse_config={'energy_window': {'emax': 2.2}}``;
    each attribute below is a key of that dict.  The
    engine applies it once, right before the first solve (``bands`` or the
    first property), so it always acts after ``doubling_Hamiltonian``.  It
    sets ``attr['bnd']``, which every downstream band-diagonal consumer
    reads, so the band path and the mesh both pick the new width up.

    Attributes
    ----------
    emax : float
        Highest energy (eV) the properties will ask for.  The window top is
        ``ehi = emax + margin``.  There is no lower bound: the solve always
        starts at the bottom of the spectrum, so every state below ``ehi``
        is computed.
    margin : float, default 1.0
        Extra range (eV) above ``emax``.  It has to cover the adaptive
        smearing tail (Yates widths here are <~ 0.22 eV, so 4 sigma is
        <~ 0.9 eV) and the transport occupation derivative at 300 K
        (~0.1 eV); 1.0 eV covers both.
    nprobe : int, default 16
        Number of k-points at which the eigenvalues below ``ehi`` are
        counted to size ``nev``: Gamma, the supercell-BZ corners, then
        strided mesh points.  The count is padded by ``max(8, 2%)``.
    nev : int or None
        Give the number of bands to solve explicitly and skip the probe.

    Notes
    -----
    This narrows the solve but does not make the workload iterative
    again: the fraction of the spectrum below a fixed ``emax`` is scale
    invariant under folding, so ``nev/nawf`` stays put as the cell grows.
    Only an *interior* window (:class:`InteriorWindow`) changes that, and
    the two are mutually exclusive.

    ``attr['bnd']`` changes meaning under a window, from "bands with
    projectability > pthr, times the cell multiplier" to "bands inside the
    property window".  The downstream normalizations are unaffected, but
    ``bands_*.dat`` gains or loses columns, so band files are not
    column-comparable across runs with and without a window.
    """

    emax: float
    margin: float = 1.0
    nprobe: int = 16
    nev: int | None = None

    def __post_init__(self):
        # frozen: normalize types through object.__setattr__
        object.__setattr__(self, 'emax', float(self.emax))
        object.__setattr__(self, 'margin', float(self.margin))
        object.__setattr__(self, 'nprobe', int(self.nprobe))
        if self.margin < 0.0:
            raise ValueError(
                "sparse_config=: 'energy_window' margin must be >= 0 eV, got %g." % self.margin
            )
        if self.nprobe < 1:
            raise ValueError(
                "sparse_config=: 'energy_window' nprobe must be >= 1, got %d." % self.nprobe
            )
        if self.nev is not None:
            object.__setattr__(self, 'nev', int(self.nev))
            if self.nev < 1:
                raise ValueError(
                    "sparse_config=: 'energy_window' nev must be >= 1, got %d." % self.nev
                )

    @property
    def ehi(self) -> float:
        """The window top, ``emax + margin`` (eV)."""
        return self.emax + self.margin

    @classmethod
    def parse(cls, value) -> EnergyWindow:
        """The ``'energy_window'`` value as an :class:`EnergyWindow`.

        Parameters
        ----------
        value : dict or EnergyWindow
            A dict with ``'emax'`` (eV) and optionally ``'margin'``,
            ``'nprobe'`` and ``'nev'``.

        Raises
        ------
        TypeError
            If ``value`` is not a dict.
        ValueError
            If a key is not an option (``'emin'`` included), ``emax`` is
            missing, or the values are inconsistent.
        """
        if isinstance(value, dict) and 'emin' in value:
            raise ValueError(
                "sparse_config=: 'energy_window' takes no 'emin'. The solve always starts at "
                "the bottom of the spectrum, so only the top matters: give {'emax': ...}."
            )
        return _parse_window(cls, value, 'energy_window', ('emax',))


@dataclass(frozen=True)
class InteriorWindow:
    """The ``'interior_window'`` option: solve only the states *inside*
    ``[elo, ehi]`` instead of from the bottom of the spectrum.

    Given as a dict, ``sparse_config={'interior_window': {'elo': -3.0,
    'ehi': 3.0}}``; each attribute below is a key of that dict.  The
    engine applies it once, right before the first solve (``bands`` or the
    first property), so it always acts after ``doubling_Hamiltonian``.
    Mutually exclusive with ``'energy_window'``.

    This is the mode that makes the iterative kernel pay: the count in a
    narrow interior window is a small fraction of the spectrum, whereas a
    from-the-bottom window must reach E_F and so is 20-50% of it for any
    real material.  The cost is that the states below ``elo`` are never
    computed, which permanently removes three things:

    - the total electron count, so anything fixed by charge neutrality
      (carrier density, the Hall coefficient) cannot be evaluated;
    - DoS/PDoS outside the window, which is *absent*, not zero;
    - band indices, since the number of states in the window varies from
      k-point to k-point.

    Properties that need any of those are **skipped with a warning** and
    the run continues to the next one; they are listed again at
    ``finish_execution``.  This is a deliberate exception to the backend's
    fail-loud rule, made because an interior run is normally a batch of
    properties of which only some are supportable.

    Attributes
    ----------
    elo, ehi : float
        The window (eV).
    kT_margin_eV : float, default 0.26
        Margin the transport occupation derivative needs on each side of
        the chemical-potential scan; 0.26 eV is 10 kT at 300 K.  Raise it
        for higher temperatures.
    smear_margin_eV : float, default 0.5
        The analogous margin for DoS/PDoS.  Adaptive smearing gives every
        state a finite width, so a state just below ``elo`` (one this mode
        never computes) would still contribute inside the window.  Plotted
        ranges are therefore clamped this far inside each edge.  0.5 eV
        covers the <~0.22 eV Yates widths of a converged mesh at 4 sigma;
        coarse meshes smear wider, so after the mesh runs the *measured*
        maximum width is checked against this value and a warning is
        issued if it was too small.
    """

    elo: float
    ehi: float
    kT_margin_eV: float = 0.26
    smear_margin_eV: float = 0.5

    def __post_init__(self):
        for name in ('elo', 'ehi', 'kT_margin_eV', 'smear_margin_eV'):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not self.ehi > self.elo:
            raise ValueError(
                "sparse_config=: 'interior_window' needs ehi > elo, got [%g, %g]."
                % (self.elo, self.ehi)
            )
        for name in ('kT_margin_eV', 'smear_margin_eV'):
            if getattr(self, name) < 0.0:
                raise ValueError(
                    "sparse_config=: 'interior_window' %s must be >= 0 eV, got %g."
                    % (name, getattr(self, name))
                )

    @classmethod
    def parse(cls, value) -> InteriorWindow:
        """The ``'interior_window'`` value as an :class:`InteriorWindow`.

        Parameters
        ----------
        value : dict or InteriorWindow
            A dict with ``'elo'`` and ``'ehi'`` (eV) and optionally
            ``'kT_margin_eV'`` and ``'smear_margin_eV'``.

        Raises
        ------
        TypeError
            If ``value`` is not a dict.
        ValueError
            If a key is not an option, ``elo``/``ehi`` are missing, or the
            values are inconsistent.
        """
        return _parse_window(cls, value, 'interior_window', ('elo', 'ehi'))


def _parse_window(cls, value, key: str, required: tuple):
    """Shared parser of the window options (a dict of the fields of ``cls``)."""
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise TypeError(
            "sparse_config=: '%s' takes a dict {%s}; got %r."
            % (key, ', '.join("'%s': ..." % k for k in required), type(value).__name__)
        )
    _reject_unknown_keys(value, [f.name for f in fields(cls)], "sparse_config= '%s'" % key)
    missing = [k for k in required if k not in value]
    if missing:
        raise ValueError(
            "sparse_config=: '%s' needs %s (eV)." % (key, ' and '.join("'%s'" % k for k in missing))
        )
    return cls(**value)


@dataclass(frozen=True)
class SparseConfig:
    """Truncation, solver and resource settings of a sparse run.

    Built by :meth:`parse` from the ``sparse_config=`` dict of
    :class:`PAOFLOW.PAOFLOW`; each attribute below is a key of that dict.

    Attributes
    ----------
    hopping_threshold : float or None
        Magnitude (eV) below which H(R) matrix elements are dropped when the
        dense base-cell Hamiltonian is converted to the sparse bond list.
        ``None`` means 1e-3 eV without a real-space cutoff and no element
        cut with one.  The conversion prints a rigorous bound on the
        eigenvalue error this truncation can cause at any k-point.  It is
        the most compact truncation, but it does **not** preserve
        symmetry: equivalent bonds lose different elements, so
        symmetry-protected degeneracies split (see
        :func:`resolve_threshold`).  A positive value is mutually
        exclusive with ``rcut`` and ``bond_order``.
    rcut : float or None
        Real-space cutoff in Bohr on the physical bond length, applied at
        the base cell in place of the element threshold.  It keeps or drops
        whole atom-pair blocks by bond length, so it respects the
        space-group symmetry of the crystal.  The radius is moved into the
        gap above the outermost neighbour shell it reaches (a shell within
        1e-3 Bohr of it counts as reached), so a value typed on a shell
        distance cannot keep part of that shell.  A radius the k-grid
        cannot represent raises: past the aliasing-safe radius, or at a bond
        the grid stores at a longer image than its shortest one (this also
        applies to ``bond_order``).  Mutually exclusive with ``bond_order``.
    bond_order : int or None
        The same cutoff as a neighbour-shell count: keep every bond up to
        and including the n-th distinct interatomic distance (1 = nearest
        neighbours).  Resolved to a radius by
        :func:`PAOFLOW.sparse.shells.shell_cutoff`.
    hk_solver : {'auto', 'sparse', 'dense'}
        Kernel that diagonalizes ``H(k)`` at one k-point.  It names the
        eigensolver only: ``'dense'`` is LAPACK on one ``(n, n)`` scratch
        matrix, and H(R) stays a bond list either way.  ``'auto'``
        dispatches on ``(nawf, nev)``; see
        :func:`PAOFLOW.sparse.solver.select_hk_solver`.
    energy_window : EnergyWindow or None
        Size the per-k solve from the property energy range, given as a
        dict ``{'emax': ..., 'margin': 1.0, 'nprobe': 16, 'nev': None}``; see :class:`EnergyWindow`.  Applied by the engine
        right before the first solve, after any doubling.  ``None`` solves
        the ``attr['bnd']`` projectable bands (times the cell multiplier).
    interior_window : InteriorWindow or None
        Solve only inside an energy window, given as a dict ``{'elo': ...,
        'ehi': ..., 'kT_margin_eV': 0.26, 'smear_margin_eV': 0.5}``; see
        :class:`InteriorWindow`.  Applied like ``energy_window``, and
        mutually exclusive with it.
    dense_n_max : int, default :data:`~PAOFLOW.sparse.solver.DENSE_N_MAX`
        Largest ``nawf`` for which a per-k ``(n, n)`` dense scratch matrix
        is admitted: the dense ``hk_solver`` branch, the full-spectrum
        properties and the ``energy_window`` probe.  Above it those refuse
        with ``NotImplementedError`` rather than allocate.  The default is
        a policy value, not a LAPACK limit; raise it to run the dense
        per-k kernel where the memory allows.
    dense_ratio : float, default :data:`~PAOFLOW.sparse.solver.DENSE_RATIO`
        Fraction of the spectrum (``(nev + guard) / n``) above which
        ``hk_solver='auto'`` stops using ARPACK and takes the dense
        kernel.  Must lie in ``(0, 1]``.
    """

    hopping_threshold: float | None = None
    rcut: float | None = None
    bond_order: int | None = None
    hk_solver: str = 'auto'
    energy_window: EnergyWindow | None = None
    interior_window: InteriorWindow | None = None
    dense_n_max: int = DENSE_N_MAX
    dense_ratio: float = DENSE_RATIO

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
        if self.energy_window is not None:
            object.__setattr__(self, 'energy_window', EnergyWindow.parse(self.energy_window))
        if self.interior_window is not None:
            object.__setattr__(self, 'interior_window', InteriorWindow.parse(self.interior_window))
        if self.energy_window is not None and self.interior_window is not None:
            raise ValueError(
                "sparse_config=: give either 'energy_window' or 'interior_window', not both. "
                'The two window modes are mutually exclusive: one sizes the solve from the '
                'bottom of the spectrum, the other solves inside a window and never computes '
                'the states below it.'
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
            If a key is not an option, or the options are inconsistent.
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
