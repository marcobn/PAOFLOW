"""Run configuration of the sparse engine.

Users select the sparse engine with a plain dict, ``PAOFLOW(..., sparse={...})``
(``{}`` takes every default); ``sparse=None`` (the default) keeps the dense
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


@dataclass(frozen=True)
class SparseConfig:
    """Truncation, solver and resource settings of a sparse run.

    Built by :meth:`parse` from the ``sparse=`` dict of
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
    mem_budget_gb : float or None
        Per-rank memory budget the doubling pre-flight checks against.
        ``None`` uses 80% of ``MemAvailable`` shared over the ranks of the
        node.
    force_doubling : bool
        Start doubling even when the pre-flight projection exceeds the
        budget.
    """

    threshold: float | None = None
    rcut: float | None = None
    bond_order: int | None = None
    hk_solver: str = 'auto'
    mem_budget_gb: float | None = None
    force_doubling: bool = False

    def __post_init__(self):
        if self.rcut is not None and self.bond_order is not None:
            raise ValueError('sparse=: give either rcut (Bohr) or bond_order (shells), not both.')
        if self.hk_solver not in HK_SOLVERS:
            raise ValueError(
                'sparse=: hk_solver must be one of %s, got %r.' % (HK_SOLVERS, self.hk_solver)
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
        if self.mem_budget_gb is not None:
            object.__setattr__(self, 'mem_budget_gb', float(self.mem_budget_gb))

    @classmethod
    def parse(cls, value) -> SparseConfig | None:
        """The ``sparse=`` constructor argument as a config, or ``None`` for dense.

        Parameters
        ----------
        value : dict or None
            ``None`` runs dense; a dict runs sparse, with its keys setting
            the attributes of this class (``{}`` takes every default).

        Raises
        ------
        TypeError
            If ``value`` is neither ``None`` nor a dict.
        ValueError
            If a key is not an option, or the options are inconsistent.
        """
        if value is None:
            return None
        if not isinstance(value, dict):
            hint = ' Use sparse={} for the sparse defaults.' if value is True else ''
            raise TypeError(
                'sparse= takes a dict of options or None (dense); got %r.%s'
                % (type(value).__name__, hint)
            )
        names = [f.name for f in fields(cls)]
        unknown = [k for k in value if k not in names]
        if unknown:
            lines = []
            for key in unknown:
                close = get_close_matches(str(key), names, n=1)
                lines.append('%r%s' % (key, " (did you mean '%s'?)" % close[0] if close else ''))
            raise ValueError(
                'sparse=: unknown option%s %s. Valid options: %s.'
                % ('s' if len(unknown) > 1 else '', ', '.join(lines), ', '.join(names))
            )
        return cls(**value)
