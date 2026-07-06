"""Sparse PAOFLOW backend.

This subpackage implements a purely sparse counterpart to the dense PAOFLOW
pipeline.  It never materialises the dense ``(nawf, nawf, nkpnts, nspin)``
tensors (``Hksp``, ``dHksp``, ``pksp``) that dominate the memory footprint of
the dense path.  Instead the PAO Hamiltonian is stored as a thresholded
real-space hopping list (:class:`~PAOFLOW.sparse.containers.SparseHamiltonian`)
from which ``H(k)`` and ``dH/dk`` are assembled matrix-free at each k-point, and
selected eigenpairs are extracted with :func:`scipy.sparse.linalg.eigsh`.

The user-facing driver is :class:`PAOFLOW.SparsePAOFLOW.SparsePAOFLOW`.
"""

from .containers import SparseEigenpairs, SparseHamiltonian

__all__ = ['SparseHamiltonian', 'SparseEigenpairs']
