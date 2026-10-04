"""PAO-basis operators (spin ``Sj``, orbital ``Lj``) in the sparse engine.

The dense pipeline stores an operator as one ``(3, nawf, nawf)`` complex
array.  After doubling that is the O(nawf^2) object the memory contract
forbids (805 MB at ``nawf = 4096``), while the operators themselves are
block-diagonal over atoms and nearly empty.  In a sparse run they are
therefore a list of three CSR matrices, built once at the base cell by the
dense ``spin_operator`` / ``orbital_operator`` and extended
block-diagonally by every doubling (``hamiltonian.do_doubling
.doubling_attr_arry``), exactly as the dense ``Sj`` is.

Indexing ``op[l]`` gives one Cartesian component in both representations,
which is all the per-k kernels use.
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
    """Convert a dense ``(3, nawf, nawf)`` operator to three CSR matrices.

    Parameters
    ----------
    op : np.ndarray, shape ``(3, nawf, nawf)``

    Returns
    -------
    list of scipy.sparse.csr_matrix
        Exact: only the stored zeros are dropped.
    """
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

    Notes
    -----
    The doubled cell orders its orbitals as [original cell, new copy], for
    the Hamiltonian (``doubling_HRs`` and ``sparse.doubling.double_axis``
    alike) and for the dense ``Sj`` (``scipy.linalg.block_diag``), so the
    copy goes in the second diagonal block.  Spin up/down blocks of an
    ad-hoc spin-orbit basis stay inside each cell's block.
    """
    return [block_diag([component, component], format='csr') for component in op]


def projection_operator(data_controller: DataController, proj_array: np.ndarray) -> csr_matrix:
    """Sparse form of ``projection.projection_operator.do_projection_operator``.

    Returns
    -------
    scipy.sparse.csr_matrix, shape ``(nawf, nawf)``
        The same identity blocks on the selected atoms' orbitals, as a
        diagonal matrix (the dense one is diagonal too).
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
