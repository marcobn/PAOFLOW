"""Band structure along a high-symmetry path, sparse iterative version.

Same k-path and MPI scatter as ``spectrum.do_bands``, but each point is assembled from the bond
list (``sign=+1``, Cartesian k) and only the lowest ``nsel`` bands are solved; eigenvectors
only warm-start the next point. The dense path diagonalizes the upper triangle of a slightly
non-Hermitian H(k) (``eigh(lower=False)``), the sparse one the exactly Hermitian Nyquist-split
interpolant (``zero_pad``); they differ by O(|H(R)| at the Nyquist shell), below plot
resolution on converged grids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from .solver import DENSE_N_MAX, DENSE_RATIO

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from .hamiltonian import SparseHamiltonian

comm = MPI.COMM_WORLD


def do_bands_sparse(
    data_controller: DataController,
    sparse_h: SparseHamiltonian,
    nsel: int,
    verbose: bool = False,
    hk_solver: str = 'auto',
    ehi: float | None = None,
    interior: tuple[float, float] | None = None,
    ptop: float | None = None,
    dense_ratio: float = DENSE_RATIO,
    dense_n_max: int = DENSE_N_MAX,
) -> None:
    """Compute the band structure along the interpolation path.

    Parameters
    ----------
    data_controller : DataController
        Supplies the k-path (see :func:`prepare_path`); receives ``arrays['E_k']``, this
        rank's ``(nkpi_local, nsel, nspin)`` slice, as in the dense pipeline.
    sparse_h : SparseHamiltonian
        Bond list H(k) is assembled from.
    nsel : int
        Number of lowest bands; ignored under ``interior``.
    verbose : bool, optional
        Log progress every few percent of this rank's k-points.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        Per-k kernel, one for the whole path (:func:`~PAOFLOW.sparse.solver.select_hk_solver`).
    ehi : float or None, optional
        Window top (eV) ``nsel`` was sized for; raises after the loop if any point's highest
        band is below it (the mesh-pass coverage guard).
    interior : (float, float) or None, optional
        Window ``(elo, ehi)`` in eV to solve inside instead of from the bottom.
    ptop : float or None, optional
        Top of the well-projected range (eV); states above it (PAO-shift placeholders) -> NaN.
    dense_ratio, dense_n_max : optional
        Per-k dispatch limits, see :func:`~PAOFLOW.sparse.solver.select_hk_solver`.

    Notes
    -----
    The previous point's lowest eigenvector warm-starts the next solve (shorter Krylov
    iteration, same result). Under ``interior`` the state count varies along the path, so
    rows are NaN-padded to the global maximum and columns are not band indices; the mesh pass
    pads with a far energy instead, but ``bands_*.dat`` is plotted, so missing must read
    missing.
    """
    from .log import get_sparse_log
    from .mesh import check_window_coverage
    from .solver import creep_warning, hk_solver_fields, solve_k

    arrays, attr = data_controller.data_dicts()

    kq_aux, nkpi = prepare_path(data_controller)
    nk_local = kq_aux.shape[0]
    nspin = sparse_h.nspin

    log = get_sparse_log(data_controller)
    log.section('Path pass (bands)')
    log.field('k-points on path', nkpi)
    limits = {'dense_ratio': dense_ratio, 'dense_n_max': dense_n_max}
    if interior is None:
        log.field('bands per k-point', nsel)
        log.rows(hk_solver_fields(sparse_h.nawf, nsel, hk_solver=hk_solver, **limits))
        warning = creep_warning(sparse_h.nawf, nsel)
        if warning:
            log.write('  WARNING: ' + warning)
    else:
        log.field('bands per k-point', 'varies with k (interior solve); missing states as NaN')

    if interior is None:
        E_k = np.zeros((nk_local, nsel, nspin), dtype=float)
    else:
        acc = {}
    deficit = 0
    step = max(1, min(100, nk_local // 10))
    for ispin in range(nspin):
        warm = None
        for ik in range(nk_local):
            hk = sparse_h.assemble_hk(kq_aux[ik], ispin=ispin, sign=+1, cart=True)
            E, _, warm = solve_k(hk, nsel, interior, warm, hk_solver, **limits)
            if interior is None:
                E_k[ik, :, ispin] = E
                if ehi is not None and E[-1] < ehi:
                    deficit += 1
            else:
                acc[(ispin, ik)] = E
            if verbose and (ik + 1) % step == 0:
                log.field('progress', f'{ik + 1}/{nk_local} local k-points')

    if interior is not None:
        local_max = max((len(E) for E in acc.values()), default=0)
        m = int(comm.allreduce(int(local_max), op=MPI.MAX))
        log.field(
            'interior states per k-point', f'up to {local_max} on this rank; padded to {m} with NaN'
        )
        E_k = np.full((nk_local, max(m, 1), nspin), np.nan, dtype=float)
        for (ispin, ik), E in acc.items():
            E_k[ik, : len(E), ispin] = E
    if ptop is not None:
        placeholders = E_k >= ptop
        count = comm.allreduce(int(placeholders.sum()), op=MPI.SUM)
        if count:
            log.field('states dropped at the PAO shift', f'{count} (written as NaN)')
        E_k[placeholders] = np.nan

    arrays['E_k'] = E_k
    if interior is None:
        check_window_coverage(deficit, nsel, ehi, 'bands')


def prepare_path(data_controller: DataController) -> tuple[np.ndarray, int]:
    """This rank's share of the band path, Cartesian in units of ``2 pi / alat``.

    The path is rebuilt from ``ibrav`` (``alat`` in Angstrom around the generator, as in the
    dense ``do_bands``) or taken from an external crystal ``kq``, then rotated to Cartesian
    in place in ``arrays['kq']`` with ``b_vectors``. ``attr['sparse_kq_cartesian']`` records
    the rotation so an external path is rotated once across passes.

    Returns
    -------
    (kq_aux, nkpi) : tuple
        Local path points ``(nkpi_local, 3)`` and the total path length.
    """
    from ..spectrum.kpnts_interpolation_mesh import kpnts_interpolation_mesh
    from ..utils.communication import scatter_full
    from ..utils.constants import ANGSTROM_AU

    arrays, attr = data_controller.data_dicts()

    attr['alat'] /= ANGSTROM_AU
    if 'ibrav' in attr:
        kpnts_interpolation_mesh(data_controller)
        attr['sparse_kq_cartesian'] = False
    attr['alat'] *= ANGSTROM_AU
    if 'kq' not in arrays:
        raise RuntimeError('sparse bands: need external kq for bands')

    nkpi = arrays['kq'].shape[1]
    if not attr.get('sparse_kq_cartesian', False):
        for n in range(nkpi):
            arrays['kq'][:, n] = np.dot(arrays['kq'][:, n], arrays['b_vectors'])
        attr['sparse_kq_cartesian'] = True

    return scatter_full(arrays['kq'].T.copy(), attr['npool']), nkpi


def run_path(
    data_controller: DataController,
    sparse_h: SparseHamiltonian,
    consumers: list,
    nsolve: int,
    with_dnm: bool = True,
    hk_solver: str = 'auto',
    verbose: bool = False,
    ptop: float | None = None,
    dense_ratio: float = DENSE_RATIO,
    dense_n_max: int = DENSE_N_MAX,
) -> None:
    """Stream the band path to path properties, one :class:`KPoint` at a time.

    Parameters
    ----------
    consumers : list of MeshProperty
        Path properties fed every local point via ``on_k``; the engine finalizes them.
    nsolve : int
        States per point: ``attr['bnd']``, or ``nawf`` for a full-spectrum consumer.
    with_dnm : bool, optional
        Whether ``dH/dk`` carries the intra-cell ``Dnm`` terms (the dense path kernels differ
        here; see ``SparseHamiltonian.assemble_derivatives``).
    ptop : float or None, optional
        Top of the well-projected range (eV); ``KPoint.bnd`` counts only states below it.
    dense_ratio, dense_n_max : optional
        Per-k dispatch limits, see :func:`~PAOFLOW.sparse.solver.select_hk_solver`.

    Notes
    -----
    Path convention ``sign=+1``, ``cart=True``. No adaptive smearing on a path, so
    ``KPoint.vel`` and ``KPoint.delta`` are ``None``; ``KPoint.kglobal`` is the path index.
    """
    from ..utils.communication import scatter_full
    from .kpoint import KPoint
    from .log import get_sparse_log
    from .solver import hk_solver_fields, solve_k

    attr = data_controller.data_attributes
    kq_aux, nkpi = prepare_path(data_controller)
    kglobal = scatter_full(np.arange(nkpi), attr['npool'])
    nawf = sparse_h.nawf
    solver = 'dense' if nsolve >= nawf else hk_solver

    log = get_sparse_log(data_controller)
    log.section('Path pass (%s)' % ', '.join(sorted({c.method for c in consumers})))
    log.field('k-points on path', nkpi)
    log.field('states per k-point', '%d of %d' % (nsolve, nawf))
    limits = {'dense_ratio': dense_ratio, 'dense_n_max': dense_n_max}
    log.rows(hk_solver_fields(nawf, nsolve, hk_solver=solver, **limits))

    bnd = min(int(attr['bnd']), nsolve)
    step = max(1, min(100, kq_aux.shape[0] // 10))
    for ispin in range(sparse_h.nspin):
        warm = None
        for ik in range(kq_aux.shape[0]):
            hk, dhk = sparse_h.assemble_derivatives(
                kq_aux[ik], ispin=ispin, sign=+1, cart=True, order=1, with_dnm=with_dnm
            )
            E, V, warm = solve_k(hk, nsolve, None, warm, solver, **limits)
            nreal = bnd if ptop is None else min(bnd, int(np.searchsorted(E, ptop)))
            kp = KPoint(
                sparse_h,
                ik,
                ispin,
                kq_aux[ik],
                E,
                V,
                hk,
                dhk,
                None,
                None,
                nreal,
                None,
                None,
                sign=+1,
                cart=True,
                kglobal=int(kglobal[ik]),
                with_dnm=with_dnm,
            )
            for c in consumers:
                c.activate()
                c.on_k(kp)
            kp = None
            if verbose and (ik + 1) % step == 0:
                log.field('progress', f'{ik + 1}/{kq_aux.shape[0]} local path points')
