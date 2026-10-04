"""Run configuration of the sparse engine.

A :class:`SparseConfig` passed as ``PAOFLOW(..., sparse=...)`` selects the
sparse engine for the whole run; ``sparse=None`` (the default) keeps the
dense pipeline.  Every knob of the sparse backend is a field here, so a
new one never widens the :class:`PAOFLOW.PAOFLOW` constructor or the
signature of a method shared with the dense pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

HK_SOLVERS = ('auto', 'sparse', 'dense')


@dataclass(frozen=True)
class SparseConfig:
    """Truncation, solver and resource settings of a sparse run.

    Attributes
    ----------
    threshold : float
        Magnitude (eV) below which H(R) matrix elements are dropped when the
        dense base-cell Hamiltonian is converted to the sparse bond list.
        The conversion prints a rigorous bound on the eigenvalue error this
        truncation can cause at any k-point.
    rcut : float or None
        Real-space cutoff in Bohr on the physical bond length, applied
        together with ``threshold`` at the base cell.  A second, physically
        different truncation axis; the ``eig_bound`` printed at conversion
        covers both.  Mutually exclusive with ``bond_order``.
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

    threshold: float = 1.0e-3
    rcut: float | None = None
    bond_order: int | None = None
    hk_solver: str = 'auto'
    mem_budget_gb: float | None = None
    force_doubling: bool = False

    def __post_init__(self):
        if self.rcut is not None and self.bond_order is not None:
            raise ValueError(
                'SparseConfig: give either rcut (Bohr) or bond_order (shells), not both.'
            )
        if self.hk_solver not in HK_SOLVERS:
            raise ValueError(
                'SparseConfig: hk_solver must be one of %s, got %r.' % (HK_SOLVERS, self.hk_solver)
            )
        # frozen: normalize types through object.__setattr__
        object.__setattr__(self, 'threshold', float(self.threshold))
        if self.rcut is not None:
            object.__setattr__(self, 'rcut', float(self.rcut))
        if self.bond_order is not None:
            object.__setattr__(self, 'bond_order', int(self.bond_order))
        if self.mem_budget_gb is not None:
            object.__setattr__(self, 'mem_budget_gb', float(self.mem_budget_gb))

    @classmethod
    def coerce(cls, value) -> SparseConfig | None:
        """The ``sparse=`` constructor argument as a config, or ``None`` for dense.

        Accepts ``None``/``False`` (dense), ``True`` (all defaults), a
        ``dict`` of fields, or a ``SparseConfig``.
        """
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            'sparse= takes None, True, a dict of SparseConfig fields or a SparseConfig; got %r.'
            % type(value).__name__
        )
