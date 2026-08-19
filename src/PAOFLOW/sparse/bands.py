"""Band structure along a high-symmetry path, sparse iterative version.

Mirrors ``spectrum.do_bands`` scaffolding exactly — same k-path generation
(``kpnts_interpolation_mesh``, including the Angstrom/Bohr ``alat`` dance
and the in-place rotation of ``kq`` to Cartesian by ``b_vectors``), same
MPI k-scatter — but replaces the dense Fourier sum + ``eigh`` with the
fixed-pattern sparse assembly (``sign=+1``, the band-path phase
convention) and the iterative ``solve_lowest``, computing only the lowest
``nsel`` bands.  Eigenvectors are used solely to warm-start the next
k-point and are never stored.

Convention note: the dense path diagonalizes the upper triangle of a
slightly non-Hermitian interpolated H(k) (``eigh(lower=False)``); the
sparse path solves the exactly Hermitian interpolant (Nyquist-split, the
``zero_pad`` convention).  The two differ by O(|H(R)| at the Nyquist
shell), far below plotting resolution for converged R grids.
"""

import numpy as np


def do_bands_sparse(data_controller, sparse_h, nsel, verbose=False, hk_solver='auto', ehi=None):
    """Compute ``arrays['E_k']`` (local slice, ``(nkpi_local, nsel, nspin)``)
    along the interpolation path.  Returns nothing; mirrors dense layout.

    ``hk_solver`` picks the per-k kernel once for the whole path;
    ``ehi`` enables the same window-coverage guard the mesh pass uses."""
    from ..spectrum.kpnts_interpolation_mesh import kpnts_interpolation_mesh
    from ..utils.communication import scatter_full
    from ..utils.constants import ANGSTROM_AU
    from .log import get_sparse_log
    from .mesh import check_window_coverage
    from .solver import describe_hk_solver, solve_lowest

    arrays, attr = data_controller.data_dicts()

    # Bohr to Angstrom (dense do_bands does the same around path generation)
    attr['alat'] /= ANGSTROM_AU

    if 'ibrav' in attr:
        kpnts_interpolation_mesh(data_controller)
    if 'kq' not in arrays:
        raise RuntimeError('sparse bands: need external kq for bands')

    # rotate kq to Cartesian in place, replicating dense do_bands (the
    # sparse assembler consumes Cartesian k with cart=True)
    nkpi = arrays['kq'].shape[1]
    for n in range(nkpi):
        arrays['kq'][:, n] = np.dot(arrays['kq'][:, n], arrays['b_vectors'])

    attr['alat'] *= ANGSTROM_AU

    kq_aux = scatter_full(arrays['kq'].T.copy(), attr['npool'])  # (nk_local, 3)
    nk_local = kq_aux.shape[0]
    nspin = sparse_h.nspin

    log = get_sparse_log(data_controller)
    log.section('Bands (path solve)')
    log.field('k-points on path', nkpi)
    log.field('bands requested', nsel)
    log.field('window top ehi (eV)', 'none' if ehi is None else '%.3f' % ehi)
    log.write(describe_hk_solver(sparse_h.nawf, nsel, hk_solver=hk_solver))

    E_k = np.zeros((nk_local, nsel, nspin), dtype=float)
    deficit = 0
    step = max(1, min(100, nk_local // 10))
    for ispin in range(nspin):
        v0 = None
        for ik in range(nk_local):
            hk = sparse_h.assemble_hk(kq_aux[ik], ispin=ispin, sign=+1, cart=True)
            E, V = solve_lowest(hk, nsel, v0=v0, hk_solver=hk_solver)
            E_k[ik, :, ispin] = E
            v0 = np.ascontiguousarray(V[:, 0])  # warm start; V is discarded
            if ehi is not None and E[-1] < ehi:
                deficit += 1
            if verbose and (ik + 1) % step == 0:
                log.write('  progress: %d/%d local k-points' % (ik + 1, nk_local))

    arrays['E_k'] = E_k
    check_window_coverage(deficit, nsel, ehi, 'bands')
