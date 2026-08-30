"""Sparse backend for PAOFLOW.

This package provides a purely sparse implementation of the PAOFLOW
property pipeline (bands, DOS, PDOS, Boltzmann transport), designed for
systems where ``doubling_Hamiltonian`` makes the dense arrays
(``HRs``, ``Hksp``, ``dHksp``, ``pksp``) too large to hold in memory.

Core contract (see each module's docstring for details):

- The real-space Hamiltonian is stored as a thresholded bond list
  (:class:`~PAOFLOW.sparse.hamiltonian.SparseHamiltonian`); global dense
  tensors of shape ``(nawf, nawf, ...)`` are never materialized.
- Eigenproblems are solved with sparse iterative methods only
  (:func:`~PAOFLOW.sparse.solver.solve_lowest`); there is no
  ``.toarray()``/dense ``eigh`` fallback at any size.
- The only sanctioned dense stage is the base-cell (pre-doubling) QE
  projection input, which is thresholded into the bond list immediately
  and deleted.
- Per-k dense workspaces are limited to one ``(nawf, nev)`` eigenvector
  block, discarded before the next k-point.
"""

from .hamiltonian import SparseHamiltonian

__all__ = ['SparseHamiltonian']
