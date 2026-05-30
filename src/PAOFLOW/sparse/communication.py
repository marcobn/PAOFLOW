from mpi4py import MPI

from PAOFLOW.defs.communication import load_balancing

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_balancing_kpoints(nkpnts: int, nspin: int, rank: int, size: int) -> tuple[int, int]:
    """Split the sparse ``(k, s)`` blocks across MPI ranks.

    Parameters
    ----------
    nkpnts : int
        Total number of k-points in the mesh.
    nspin : int
        Number of spin channels.
    rank : int
        Index of the current MPI rank.
    size : int
        Total number of MPI ranks.

    Returns
    -------
    tuple[int, int]
        ``(start_block, stop_block)`` for the global block numbering
        ``block_idx = ik * nspin + ispin``. Each rank receives all spin
        channels for a contiguous window of k-points.

    Notes
    -----
    The sparse workflow still follows the physical organization of the problem:
    each k-point carries all of its spin channels together. Grouping the work in
    contiguous k-point windows preserves the same layout used by the dense path
    for ``E_k`` and ``v_k``, which avoids unnecessary gather-and-redistribute
    steps.

    Example: For 4 k-points and 2 spins (8 blocks) across 2 ranks:
        Rank 0: blocks [0, 1, 2, 3] (k=0-1, all spins)
        Rank 1: blocks [4, 5, 6, 7] (k=2-3, all spins)
    """
    start_k, stop_k = load_balancing(size, rank, nkpnts)
    return (int(start_k) * int(nspin), int(stop_k) * int(nspin))
