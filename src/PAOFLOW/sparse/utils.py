from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mpi4py import MPI
from scipy import sparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def fmt_bytes(n: int) -> str:
    """Format a storage size in human-readable units."""
    for unit, thresh in (('GB', 1024**3), ('MB', 1024**2), ('KB', 1024)):
        if abs(n) >= thresh:
            return f'{n / thresh:.2f} {unit}'
    return f'{n} B'


def report_sparse_hr_stats(arrays: dict[str, Any], attr: dict[str, Any]) -> None:
    """Report sparsity and memory estimates for sparse real-space Hamiltonian blocks."""
    if 'SparseHRs' not in arrays:
        return

    from PAOFLOW.sparse.doubling import _as_hr_dict

    blocks = _as_hr_dict(arrays['SparseHRs'], nspin=int(attr['nspin']))
    nk1 = int(attr['nk1'])
    nk2 = int(attr['nk2'])
    nk3 = int(attr['nk3'])
    nspin = int(attr['nspin'])
    nawf = int(attr['nawf'])

    total_possible_blocks = nk1 * nk2 * nk3 * nspin
    report_sparse_block_stats(
        title='Sparsity statistics for H(R) blocks after doubling:',
        sparse_blocks=blocks,
        nawf=nawf,
        total_possible_blocks=total_possible_blocks,
        show_present_blocks=True,
    )

    if bool(attr.get('sparse_doubling_block_pruning', True)):
        rel_tol = float(attr.get('sparse_doubling_block_rel_tol', 1.0e-7))
        abs_tol = float(attr.get('sparse_doubling_block_abs_tol', 1e-5))
        print(f'Doubling block pruning: on (rel_tol={rel_tol:g}, abs_tol={abs_tol:g})')


def report_sparse_block_stats(
    *,
    title: str,
    sparse_blocks: Mapping[Any, sparse.spmatrix],
    nawf: int,
    total_possible_blocks: int,
    show_present_blocks: bool = False,
    comm: MPI.Comm | None = None,
) -> dict[str, float | int] | None:
    """Print dense-vs-sparse memory statistics for block-sparse matrices.

    Parameters
    ----------
    title : str
        Header line printed before the statistics.
    sparse_blocks : mapping
        Sparse matrix blocks for a block-structured Hamiltonian representation.
    nawf : int
        Number of basis functions per block dimension.
    total_possible_blocks : int
        Number of blocks in the corresponding dense representation.
    show_present_blocks : bool, optional
        If True, also print how many sparse blocks are currently present.
    comm : mpi4py.MPI.Comm or None, optional
        Communicator used to reduce distributed local block statistics. If None,
        the provided mapping is treated as complete on the current rank.

    Returns
    -------
    dict or None
        A dictionary of computed statistics on rank 0, otherwise None when
        MPI reduction is active and this rank is non-root.
    """
    local_nnz = sum(int(block.nnz) for block in sparse_blocks.values())
    local_sparse_bytes = 0
    for block in sparse_blocks.values():
        csr = block.tocsr(copy=False)
        local_sparse_bytes += csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
    local_present_blocks = int(len(sparse_blocks))

    if comm is not None and comm.Get_size() > 1:
        total_nnz = comm.reduce(local_nnz, op=MPI.SUM, root=0)
        sparse_bytes = comm.reduce(local_sparse_bytes, op=MPI.SUM, root=0)
        present_blocks = comm.reduce(local_present_blocks, op=MPI.SUM, root=0)
        rank = comm.Get_rank()
    else:
        total_nnz = local_nnz
        sparse_bytes = local_sparse_bytes
        present_blocks = local_present_blocks
        rank = 0

    if rank != 0:
        return None

    total_elements = int(total_possible_blocks) * int(nawf) * int(nawf)
    sparsity = 1.0 - total_nnz / total_elements if total_elements > 0 else 0.0

    dense_bytes = total_elements * 16  # complex128
    saved_bytes = dense_bytes - sparse_bytes
    saving_pct = 100.0 * saved_bytes / dense_bytes if dense_bytes > 0 else 0.0

    print(title)
    print(
        f'Total elements: {total_elements:,}  Non-zero elements: {total_nnz:,}  '
        f'Sparsity: {sparsity * 100:.1f}%'
    )
    if show_present_blocks:
        present_pct = (
            100.0 * present_blocks / total_possible_blocks if total_possible_blocks else 0.0
        )
        print(
            f'Stored sparse blocks: {present_blocks:,} / {total_possible_blocks:,} '
            f'({present_pct:.1f}% present)'
        )
    print(f'Dense storage: {fmt_bytes(dense_bytes)}')
    print(f'Sparse storage:   {fmt_bytes(sparse_bytes)}  ({saving_pct:.1f}% memory saved vs dense)')

    return {
        'total_elements': total_elements,
        'total_nnz': int(total_nnz),
        'sparsity_pct': sparsity * 100.0,
        'dense_bytes': int(dense_bytes),
        'sparse_bytes': int(sparse_bytes),
        'saving_pct': saving_pct,
        'present_blocks': int(present_blocks),
        'total_possible_blocks': int(total_possible_blocks),
    }


def report_hk_stats(
    sparse_blocks: dict[int, sparse.csr_matrix],
    nawf: int,
    nkpnts: int,
    nspin: int,
    sparse_threshold: float,
    min_sparsity_warn_pct: float = 15.0,
    aggressive_threshold_warn: float = 1.0e-2,
) -> None:
    """Report how much memory the sparse ``H(k)`` representation saves.

    Parameters
    ----------
    sparse_blocks : dict[int, scipy.sparse.csr_matrix]
        Distributed sparse Hamiltonian blocks for the current calculation.
    nawf : int
        Number of atomic-like basis functions, so each block has shape
        ``(nawf, nawf)``.
    nkpnts : int
        Number of k-points in the mesh.
    nspin : int
        Number of spin channels.
    sparse_threshold : float
        Magnitude threshold used when pruning small Hamiltonian elements.
    min_sparsity_warn_pct : float, optional
        Warning threshold percent below which the sparse representation may no longer be
        worthwhile.
    aggressive_threshold_warn : float, optional
        Threshold above which pruning may noticeably alter the physics.

    Returns
    -------
    None
        Prints a rank-0 summary of sparsity and estimated storage.

    Notes
    -----
    The comparison is between the full dense tensor
    ``H(k, i, j, s)`` and the blockwise sparse representation actually stored in
    memory. The goal is not to change the Hamiltonian, but to make explicit when
    the sparse approximation is providing a real storage benefit and when the
    chosen threshold may be too aggressive.
    """
    stats = report_sparse_block_stats(
        title=f'Sparsity statistics for H(k) blocks (threshold={sparse_threshold:g}):',
        sparse_blocks=sparse_blocks,
        nawf=nawf,
        total_possible_blocks=nkpnts * nspin,
        show_present_blocks=False,
        comm=comm,
    )
    if stats is None:
        return

    sparsity_pct = float(stats['sparsity_pct'])
    if sparsity_pct < min_sparsity_warn_pct:
        print(
            'WARNING: low sparsity '
            f'({sparsity_pct:.1f}% < {min_sparsity_warn_pct:.1f}%). '
            'Dense mode may be faster and use comparable or less memory. '
            'Consider running dense mode or increasing sparse_threshold cautiously.'
        )

    if sparse_threshold >= aggressive_threshold_warn:
        print(
            'WARNING: sparse_threshold='
            f'{sparse_threshold:g} is aggressive (>= {aggressive_threshold_warn:g}). '
            'This can over-prune Hamiltonian entries and may produce zero adaptive-smearing '
            'widths (deltakp), leading to NaN values in dosdk/sigmadk.'
        )
