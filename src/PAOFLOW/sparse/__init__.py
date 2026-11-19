"""Sparse backend for PAOFLOW.

For systems where ``doubling_Hamiltonian`` makes the dense ``HRs``, ``Hksp``, ``dHksp``,
``pksp`` too large. ``PAOFLOW(..., sparse=True, sparse_config={...})``
(:mod:`~PAOFLOW.sparse.config`) runs on :class:`~PAOFLOW.sparse.engine.SparseEngine`, to which
:mod:`~PAOFLOW.sparse.dispatch` routes: bands, mesh properties (DOS/PDOS, Boltzmann transport
with Hall, effective masses, doping, Fermi surfaces, conductivity, spin/orbital textures, the
anomalous/spin/orbital Hall family, dielectric tensor, Rashba--Edelstein, linear response,
IPR, density) and path properties (Berry curvature, topology, site projections, Berry phase).
Base-cell ``H(R)`` transforms run their dense bodies; the rest says why it is dense-only.

Core contract:

- ``H(R)`` is a thresholded bond list (:class:`~PAOFLOW.sparse.hamiltonian.SparseHamiltonian`);
  no k-indexed ``(nawf, nawf, ...)`` tensor is ever formed.
- Each k-point is solved by :func:`~PAOFLOW.sparse.solver.solve_lowest` (shift-invert ARPACK,
  or per-k dense ``zheevr`` while ``nawf <= DENSE_N_MAX``).
- The only dense stage is the base-cell QE projection, thresholded into the bond list and
  deleted at once (plus base-cell ``H(R)`` transforms on request).
- Per-k dense workspace lives one k-point: ``(nawf, nev)`` eigenvectors, or ``(nawf, nawf)``
  and its products for full-spectrum interband sums.
- A property is a :mod:`~PAOFLOW.sparse.properties` class: ``on_k`` on a
  :class:`~PAOFLOW.sparse.kpoint.KPoint` calls the per-k body of the dense kernel, ``finalize``
  the dense reduce/write; ``with pao.sparse.fused():`` runs several in one pass.

Cutoffs may be a radius or a shell count (:mod:`~PAOFLOW.sparse.shells`). The base-cell bond
list can be saved, reloaded in place of the DFT stages, and read as a labelled hopping dataset
(:mod:`~PAOFLOW.sparse.io`); the archive is shared with the dense pipeline via
:mod:`~PAOFLOW.sparse.bridge`, the single dense <-> sparse conversion point.
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
