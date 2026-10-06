"""Sparse backend for PAOFLOW.

This package provides a sparse implementation of the PAOFLOW property
pipeline, designed for systems where ``doubling_Hamiltonian`` makes the
dense arrays (``HRs``, ``Hksp``, ``dHksp``, ``pksp``) too large to hold in
memory.  Bands, the mesh properties (DOS/PDOS, Boltzmann transport with the
Hall term, effective masses, doping, Fermi surfaces, conductivity, spin and
orbital textures, the anomalous/spin/orbital Hall family, the dielectric
tensor, Rashba--Edelstein, linear response, IPR, density) and the band-path
properties (Berry curvature, topology, site projections, Berry phase) run
on it; the base-cell ``H(R)`` transforms run through their dense bodies;
the rest says why it is dense-only (:mod:`~PAOFLOW.sparse.dispatch`).

Core contract (see each module's docstring for details):

- The real-space Hamiltonian is stored as a thresholded bond list
  (:class:`~PAOFLOW.sparse.hamiltonian.SparseHamiltonian`); global dense
  tensors of shape ``(nawf, nawf, ...)`` indexed by k are never
  materialized.
- Each k-point is solved by :func:`~PAOFLOW.sparse.solver.solve_lowest`
  (shift-invert ARPACK, or a per-k dense ``zheevr`` scratch while
  ``nawf <= DENSE_N_MAX``).
- The only sanctioned dense stage is the base-cell (pre-doubling) QE
  projection input, which is thresholded into the bond list immediately
  and deleted (and, on request, base-cell ``H(R)`` transforms).
- Per-k dense workspace lives for one k-point: the eigenvector block
  ``(nawf, nev)``, or ``(nawf, nawf)`` with the matrices built from it when
  a property needs interband sums over the full spectrum.
- Properties are one protocol (:mod:`~PAOFLOW.sparse.properties`): a
  per-k ``on_k`` on a :class:`~PAOFLOW.sparse.kpoint.KPoint` that calls the
  per-k body extracted from the dense kernel, and a ``finalize`` that calls
  the dense reduce/write.  ``with pao.sparse.fused():`` runs several of
  them in one pass.

The real-space cutoff can be given as a radius or as a neighbour-shell
count (:mod:`~PAOFLOW.sparse.shells`).  The base-cell bond list can be
saved, reloaded in place of the DFT input stages, and read back as a
labelled dataset of hopping integrals (:mod:`~PAOFLOW.sparse.io`).
The archive is shared with the dense pipeline: :mod:`~PAOFLOW.sparse.bridge`
is the single dense <-> sparse conversion point.

There is no separate driver: ``PAOFLOW(..., sparse={...})``, a dict of
options (:mod:`~PAOFLOW.sparse.config`), runs on :class:`~PAOFLOW.sparse.engine.SparseEngine`,
to which :mod:`~PAOFLOW.sparse.dispatch` routes the methods it implements.
"""

from .bridge import archive_Dnm, densify, init_restart_session, sparsify
from .hamiltonian import SparseHamiltonian

__all__ = [
    'SparseHamiltonian',
    'archive_Dnm',
    'densify',
    'init_restart_session',
    'sparsify',
]
