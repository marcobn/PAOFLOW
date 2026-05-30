from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from scipy import sparse

from .utils import report_hk_stats

if TYPE_CHECKING:
    from ..DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def do_build_pao_hamiltonian(data_controller: DataController) -> None:
    """Construct the sparse k-space Hamiltonian for the PAO basis.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding the projected eigenvectors, eigenvalues, and
        the scalar parameters needed to build ``H(k)``.

    Returns
    -------
    None
        Stores the distributed sparse Hamiltonian blocks in
        ``arrays['Hks_sparse']``.

    Notes
    -----
    This is the sparse analogue of the standard PAO Hamiltonian builder. The
    Hamiltonian is assembled directly as one sparse block per ``(k, s)`` pair,
    instead of first forming a dense global tensor and sparsifying it later.
    That is the key memory-saving step for large calculations.
    """
    arry, attr = data_controller.data_dicts()

    arry['Hks_sparse'] = build_Hks(data_controller)
    sparse_threshold = attr['sparse_threshold']
    min_sparsity_warn_pct = float(attr.get('sparse_min_sparsity_warn_pct', 15.0))
    aggressive_threshold_warn = float(attr.get('sparse_aggressive_threshold_warn', 1.0e-2))
    report_hk_stats(
        sparse_blocks=arry['Hks_sparse'],
        nawf=int(attr['nawf']),
        nkpnts=int(attr['nkpnts']),
        nspin=int(attr['nspin']),
        sparse_threshold=sparse_threshold,
        min_sparsity_warn_pct=min_sparsity_warn_pct,
        aggressive_threshold_warn=aggressive_threshold_warn,
    )


def build_hk(
    u_transposed: np.ndarray,
    eigvals: np.ndarray,
    bnd: int,
    eta: float,
    shift_type: int,
    threshold: float,
) -> sparse.csr_matrix:
    """Assemble one Hamiltonian block in the PAO basis.

    Parameters
    ----------
    u_transposed : numpy.ndarray
        Transposed projected eigenvector matrix with shape ``(nawf, nbands)``.
        Its columns define the PAO representation of the retained Bloch states
        at one k-point and one spin channel.
    eigvals : numpy.ndarray
        Band energies for the same k-point and spin channel, with shape
        ``(nbands,)``.
    bnd : int
        Number of bands considered when selecting states below the energy
        cutoff.
    eta : float
        Energy shift entering the same PAO Hamiltonian construction used by the
        dense workflow.
    shift_type : int
        Choice of shift prescription used in the original dense code.
    threshold : float
        Magnitude below which matrix elements are dropped from the sparse block.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse Hamiltonian block ``H(k)`` with shape ``(nawf, nawf)``.

    Notes
    -----
    The matrix elements are obtained from the projected states and their
    energies,

    ``H_ij(k) = sum_n A_in(k) E_n(k) A_jn*(k)``

    together with the same shift correction used by the dense builder. The
    sparse implementation differs only in how the result is stored: entries
    smaller than ``threshold`` are discarded during assembly, so the block is
    created directly in sparse form rather than converted from a dense matrix.
    """
    nawf = u_transposed.shape[0]

    bnd_ik = int(np.count_nonzero(eigvals[:bnd] <= eta))
    if bnd_ik == 0:
        if rank == 0:
            raise ValueError('No eigenvalues in the selected energy range for sparse H(k) build.')

    ac = u_transposed[:, :bnd_ik]
    ee = np.asarray(eigvals[:bnd_ik], dtype=np.complex128)

    aux_p = None
    if shift_type == 1:
        aux_p = np.linalg.inv(np.conj(ac).T @ ac)
    elif shift_type not in (0, 2):
        if rank == 0:
            raise ValueError(f"'shift_type' not recognized: {shift_type}")

    rows: list[int] = []
    cols: list[int] = []
    vals: list[np.complex128] = []
    ac_conj_t = np.conj(ac.T)

    for i in range(nawf):
        ai = ac[i, :]
        row_vals = (ai * ee) @ ac_conj_t

        if shift_type == 0:
            proj_row = ai @ ac_conj_t
            row_vals = row_vals - eta * proj_row
            row_vals[i] += eta
        elif shift_type == 1:
            damp_row = (ai @ aux_p) @ ac_conj_t
            row_vals = row_vals - eta * damp_row
            row_vals[i] += eta

        nz_cols = np.flatnonzero(np.abs(row_vals) >= threshold)
        if nz_cols.size == 0:
            continue

        rows.extend([i] * int(nz_cols.size))
        cols.extend(nz_cols.tolist())
        vals.extend(row_vals[nz_cols].tolist())

    h_k = sparse.csr_matrix((vals, (rows, cols)), shape=(nawf, nawf), dtype=np.complex128)
    h_k = 0.5 * (h_k + h_k.getH())
    h_k.eliminate_zeros()
    return h_k


def build_Hks(data_controller: DataController) -> dict[int, sparse.csr_matrix]:
    """Assemble the distributed sparse Hamiltonian over the full k-mesh.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding the projected eigenvectors ``U``, the band
        energies, and the mesh metadata.

    Returns
    -------
    dict[int, scipy.sparse.csr_matrix]
        Local sparse blocks keyed by ``block_idx = ik * nspin + ispin``. Each
        value is a sparse matrix of shape ``(nawf, nawf)``.

    Notes
    -----
    The Hamiltonian is built block by block for the portion of the k-mesh owned
    by the current MPI rank. Keeping the distribution in contiguous k-point
    windows lets later sparse eigensolve and post-processing stages reuse the
    same ownership pattern without gathering a dense global tensor.
    """
    from .communication import load_balancing_kpoints

    arrays, attributes = data_controller.data_dicts()

    bnd = attributes['bnd']
    nawf = attributes['nawf']
    eta = attributes['shift']
    nspin = attributes['nspin']
    nkpnts = attributes['nkpnts']
    shift_type = attributes['shift_type']
    threshold = attributes['sparse_threshold']

    U = arrays['U']
    my_eigsmat = arrays['my_eigsmat']

    start_block, stop_block = load_balancing_kpoints(nkpnts, nspin, rank, size)
    attributes['sparse_kpoint_block_range'] = (start_block, stop_block)

    hks_sparse_local: dict[int, sparse.csr_matrix] = {}

    for block_idx in range(start_block, stop_block):
        ik = block_idx // nspin
        ispin = block_idx % nspin
        my_eigs = my_eigsmat[:, ik, ispin]
        UU = np.transpose(U[:, :, ik, ispin]).copy()
        norms = 1.0 / np.sqrt(np.real(np.sum(np.conj(UU) * UU, axis=0)))
        UU[:, :nawf] = UU[:, :nawf] * norms[:nawf]

        hks_sparse_local[block_idx] = build_hk(
            u_transposed=UU,
            eigvals=my_eigs,
            bnd=bnd,
            eta=eta,
            shift_type=shift_type,
            threshold=threshold,
        )

    return hks_sparse_local
