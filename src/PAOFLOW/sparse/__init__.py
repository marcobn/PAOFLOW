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

The real-space cutoff can be given as a radius or as a neighbour-shell
count (:mod:`~PAOFLOW.sparse.shells`).  The base-cell bond list can be
saved, reloaded in place of the DFT input stages, and read back as a
labelled dataset of hopping integrals (:mod:`~PAOFLOW.sparse.io`).
The archive is shared with the dense pipeline: :mod:`~PAOFLOW.sparse.bridge`
is the single dense <-> sparse conversion point.

There is no separate driver: ``PAOFLOW(..., sparse=SparseConfig(...))``
(:mod:`~PAOFLOW.sparse.config`) runs on :class:`~PAOFLOW.sparse.engine.SparseEngine`,
to which :mod:`~PAOFLOW.sparse.dispatch` routes the methods it implements.
"""

from .bridge import archive_Dnm, densify, init_restart_session, sparsify
from .config import SparseConfig
from .hamiltonian import SparseHamiltonian

__all__ = [
    'SparseConfig',
    'SparseHamiltonian',
    'archive_Dnm',
    'densify',
    'init_restart_session',
    'sparsify',
]
