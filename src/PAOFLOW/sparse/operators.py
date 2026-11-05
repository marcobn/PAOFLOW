"""PAO-basis operators (spin ``Sj``, orbital ``Lj``) in the sparse engine.

The dense ``(3, nawf, nawf)`` array is O(nawf^2) after doubling (805 MB at ``nawf = 4096``),
which the memory contract forbids, yet the operators are block-diagonal over atoms and nearly
empty. A sparse run keeps three CSR matrices, built at the base cell by the dense
``spin_operator`` / ``orbital_operator`` and extended block-diagonally by each doubling
(``hamiltonian.do_doubling.doubling_attr_arry``) as the dense ``Sj`` is. ``op[l]`` is one
Cartesian component in both forms, all the per-k kernels use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import block_diag, csr_matrix, issparse

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

OPERATOR_KEYS = ('Sj', 'Lj')
"""Data-array keys holding a three-component PAO-basis operator."""


def is_sparse_operator(op: Any) -> bool:
    """Whether ``op`` is the sparse representation (a list of CSR matrices)."""
    return isinstance(op, list) and len(op) > 0 and issparse(op[0])


def to_sparse_operator(op: np.ndarray) -> list[csr_matrix]:
    """Convert a dense ``(3, nawf, nawf)`` operator to three CSR matrices (exact)."""
    if is_sparse_operator(op):
        return op
    return [csr_matrix(np.asarray(op[l])) for l in range(3)]


def to_dense_operator(op: Any) -> np.ndarray:
    """The ``(3, nawf, nawf)`` dense form of either representation."""
    if is_sparse_operator(op):
        return np.array([component.toarray() for component in op])
    return np.asarray(op)


def double_operator(op: list[csr_matrix]) -> list[csr_matrix]:
    """Block-diagonal doubling of a sparse operator, ``diag(O, O)`` per component.

    Orbitals of the doubled cell are [original, copy], as in ``doubling_HRs``,
    ``sparse.doubling.double_axis`` and the dense ``Sj``; an ad-hoc spin-orbit basis keeps its
    up/down blocks inside each cell's block.
    """
    return [block_diag([component, component], format='csr') for component in op]


def projection_operator(data_controller: DataController, proj_array: np.ndarray) -> csr_matrix:
    """Sparse ``projection.projection_operator.do_projection_operator``.

    Returns the same diagonal ``(nawf, nawf)`` matrix, as CSR: identity on the selected atoms'
    orbitals.
    """
    from scipy.sparse import diags

    from ..projection.projection_operator import orbital_array

    arrays, attr = data_controller.data_dicts()
    if 'naw' not in arrays:
        arrays['naw'] = orbital_array(data_controller)
    diagonal = np.zeros(attr['nawf'], dtype=float)
    for atom in np.asarray(proj_array).reshape(-1):
        start = int(np.sum(arrays['naw'][0:atom]))
        diagonal[start : start + int(arrays['naw'][atom])] = 1.0
    return diags(diagonal, format='csr')
