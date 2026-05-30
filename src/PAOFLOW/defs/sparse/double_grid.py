from __future__ import annotations

import numpy as np
from mpi4py import MPI
from scipy import sparse

from ..get_K_grid_fft import get_K_grid_fft_crystal
from .communication import load_balancing_kpoints


def do_double_grid(data_controller) -> None:
    """Interpolate the Hamiltonian onto the enlarged FFT k-grid in sparse form.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding the sparse real-space Hamiltonian and the FFT
        grid information for the interpolation step.

    Returns
    -------
    None
        Populates ``arrays['Hks_sparse']`` with the local sparse Hamiltonian
        blocks on the enlarged grid.

    Notes
    -----
    The interpolated Hamiltonian is the Fourier sum

    ``H(k) = sum_R exp(-2 pi i k . R) H(R)``.

    In dense mode this stage can create a large global tensor ``Hksp`` over the
    enlarged grid. The sparse version avoids that allocation by evaluating only
    the k-points owned by the current rank and storing each result directly as a
    sparse block.

    Parallelization strategy:
        The enlarged ``(k, s)`` grid is divided into the same contiguous
        k-point windows used elsewhere in the sparse workflow. Rank 0 broadcasts
        only the active real-space blocks needed to evaluate the Fourier sum, so
        each rank can assemble its own portion independently without a dense
        global ``H(k)`` tensor.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    arrays, attr = data_controller.data_dicts()

    nk1p = int(attr['nfft1'])
    nk2p = int(attr['nfft2'])
    nk3p = int(attr['nfft3'])

    sparse_hrs = arrays.get('SparseHRs') if rank == 0 else None
    if rank == 0 and sparse_hrs is None:
        raise KeyError('SparseHRs')

    a_vectors = arrays.get('a_vectors') if rank == 0 else None
    b_vectors = arrays.get('b_vectors') if rank == 0 else None
    a_vectors = comm.bcast(a_vectors, root=0)
    b_vectors = comm.bcast(b_vectors, root=0)
    if a_vectors is None:
        raise KeyError('a_vectors')
    if b_vectors is None:
        raise KeyError('b_vectors')

    nkpnts = nk1p * nk2p * nk3p
    sparse_meta = None
    r_cart = None
    if rank == 0:
        assert sparse_hrs is not None
        sparse_meta = (int(sparse_hrs.nawf), int(sparse_hrs.nspin))
        r_cart = sparse_hrs.compute_R_cart(a_vectors)

    sparse_meta = comm.bcast(sparse_meta, root=0)
    nawf, nspin = sparse_meta

    start_block, stop_block = load_balancing_kpoints(nkpnts, nspin, rank, size)
    start_kpoint = int(start_block) // nspin
    stop_kpoint = int(stop_block) // nspin

    kgrid_crystal = get_K_grid_fft_crystal(nk1p, nk2p, nk3p)
    kgrid_cart = np.dot(kgrid_crystal, b_vectors)

    threshold = float(attr['sparse_threshold'])
    local_kgrid = np.asarray(kgrid_cart[start_kpoint:stop_kpoint, :], dtype=float)
    local_blocks: dict[int, sparse.csr_matrix] = {}
    batch_size = 256

    for ispin in range(nspin):
        if rank == 0:
            assert sparse_hrs is not None and r_cart is not None
            active_ir, hr_dense = sparse_hrs._collect_spin_hr_blocks(ispin, threshold)
            r_active = r_cart[active_ir, :] if active_ir.size else np.zeros((0, 3), dtype=float)
        else:
            hr_dense = None
            r_active = None

        hr_dense = comm.bcast(hr_dense, root=0)
        r_active = comm.bcast(r_active, root=0)

        if hr_dense is None or r_active is None or hr_dense.shape[0] == 0:
            for ik_local in range(local_kgrid.shape[0]):
                ik = start_kpoint + ik_local
                local_blocks[ik * nspin + ispin] = sparse.csr_matrix(
                    (nawf, nawf), dtype=np.complex128
                )
            continue

        for batch_start in range(0, local_kgrid.shape[0], batch_size):
            batch_stop = min(batch_start + batch_size, local_kgrid.shape[0])
            k_batch = local_kgrid[batch_start:batch_stop, :]
            phase_batch = np.exp(-2.0j * np.pi * np.dot(k_batch, r_active.T))
            hk_batch = np.einsum('kr,rij->kij', phase_batch, hr_dense, optimize=True)
            hk_batch = 0.5 * (hk_batch + np.swapaxes(np.conj(hk_batch), -1, -2))

            for ib in range(batch_stop - batch_start):
                hk_dense = hk_batch[ib]
                if threshold > 0.0:
                    hk_dense = np.where(np.abs(hk_dense) >= threshold, hk_dense, 0.0)

                hk_sparse = sparse.csr_matrix(hk_dense, dtype=np.complex128)
                hk_sparse.eliminate_zeros()
                ik = start_kpoint + batch_start + ib
                local_blocks[ik * nspin + ispin] = hk_sparse

    arrays['Hks_sparse'] = local_blocks

    # Sparse interpolated path now stores distributed sparse blocks rather than Hksp.
    if 'Hksp' in arrays:
        del arrays['Hksp']

    attr['sparse_kpoint_block_range'] = (start_block, stop_block)
    attr['sparse_interpolated_storage'] = 'Hks_sparse_blocks'

    attr['nk1'] = nk1p
    attr['nk2'] = nk2p
    attr['nk3'] = nk3p
    attr['nkpnts'] = nk1p * nk2p * nk3p
