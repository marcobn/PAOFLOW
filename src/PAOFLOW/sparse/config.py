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

HK_SOLVERS = ('auto', 'sparse', 'dense')

# element threshold (eV) used when neither a threshold nor a real-space cutoff is given
DEFAULT_THRESHOLD = 1.0e-3


def resolve_threshold(
    threshold: float | None, rcut: float | None = None, bond_order: int | None = None
) -> float:
    """The element threshold (eV) a truncation actually applies.

    Parameters
    ----------
    threshold : float or None
        Requested element threshold; ``None`` picks the default for the
        truncation mode.
    rcut, bond_order : optional
        Real-space cutoff, as a radius (Bohr) or a neighbour-shell count.

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
            'threshold cannot be combined with rcut or bond_order: an element threshold '
            'breaks the point-group symmetry a real-space cutoff preserves. Give either a '
            'threshold (eV) or a real-space cutoff, not both.'
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

    Given as a dict, ``sparse_config={'energy_window': {'emin': -12.0,
    'emax': 2.2}}``; each attribute below is a key of that dict.  The
    engine applies it once, right before the first solve (``bands`` or the
    first property), so it always acts after ``doubling_Hamiltonian``.  It
    sets ``attr['bnd']``, which every downstream band-diagonal consumer
    reads, so the band path and the mesh both pick the new width up.

    Attributes
    ----------
    emin, emax : float
        Energy range (eV) the properties will ask for.  The window top is
        ``ehi = emax + margin``; ``emin`` is recorded for the log and for
        the range checks of the properties.
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
    Only an *interior* window (``pao.sparse.interior_window``, transport
    only) changes that, and the two are mutually exclusive.

    ``attr['bnd']`` changes meaning under a window, from "bands with
    projectability > pthr, times the cell multiplier" to "bands inside the
    property window".  The downstream normalizations are unaffected, but
    ``bands_*.dat`` gains or loses columns, so band files are not
    column-comparable across runs with and without a window.
    """

    emin: float
    emax: float
    margin: float = 1.0
    nprobe: int = 16
    nev: int | None = None

    def __post_init__(self):
        # frozen: normalize types through object.__setattr__
        object.__setattr__(self, 'emin', float(self.emin))
        object.__setattr__(self, 'emax', float(self.emax))
        object.__setattr__(self, 'margin', float(self.margin))
        object.__setattr__(self, 'nprobe', int(self.nprobe))
        if not self.emax > self.emin:
            raise ValueError(
                "sparse_config=: 'energy_window' needs emax > emin, got [%g, %g]."
                % (self.emin, self.emax)
            )
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
            A dict with ``'emin'`` and ``'emax'`` (eV) and optionally
            ``'margin'``, ``'nprobe'`` and ``'nev'``.

        Raises
        ------
        TypeError
            If ``value`` is not a dict.
        ValueError
            If a key is not an option, ``emin``/``emax`` are missing, or
            the values are inconsistent.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "sparse_config=: 'energy_window' takes a dict {'emin': ..., 'emax': ...}; got %r."
                % type(value).__name__
            )
        _reject_unknown_keys(value, [f.name for f in fields(cls)], "sparse_config= 'energy_window'")
        missing = [k for k in ('emin', 'emax') if k not in value]
        if missing:
            raise ValueError(
                "sparse_config=: 'energy_window' needs %s (eV)."
                % ' and '.join("'%s'" % k for k in missing)
            )
        return cls(**value)


@dataclass(frozen=True)
class SparseConfig:
    """Truncation, solver and resource settings of a sparse run.

    Built by :meth:`parse` from the ``sparse_config=`` dict of
    :class:`PAOFLOW.PAOFLOW`; each attribute below is a key of that dict.

    Attributes
    ----------
    threshold : float or None
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
        dict ``{'emin': ..., 'emax': ..., 'margin': 1.0, 'nprobe': 16,
        'nev': None}``; see :class:`EnergyWindow`.  Applied by the engine
        right before the first solve, after any doubling.  ``None`` solves
        the ``attr['bnd']`` projectable bands (times the cell multiplier).
    """

    threshold: float | None = None
    rcut: float | None = None
    bond_order: int | None = None
    hk_solver: str = 'auto'
    energy_window: EnergyWindow | None = None

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
        # validates the threshold/cutoff combination; the field keeps the
        # user's value (None stays None) so the log shows what was asked for
        resolve_threshold(self.threshold, self.rcut, self.bond_order)
        # frozen: normalize types through object.__setattr__
        if self.threshold is not None:
            object.__setattr__(self, 'threshold', float(self.threshold))
        if self.rcut is not None:
            object.__setattr__(self, 'rcut', float(self.rcut))
        if self.bond_order is not None:
            object.__setattr__(self, 'bond_order', int(self.bond_order))
        if self.energy_window is not None:
            object.__setattr__(self, 'energy_window', EnergyWindow.parse(self.energy_window))

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
