from __future__ import annotations

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#: Largest element count one MPI call takes: a C ``int`` (MPI libraries without the MPI-4
#: large-count interface, e.g. Open MPI 5.0, reject more with ``MPI_ERR_ARG``).
MAX_MPI_COUNT = 2**31 - 1

#: Bytes one collective call moves at most; larger transfers run in rounds (also a guard for
#: MPI libraries that mishandle single messages over 2 GiB).
MAX_MESSAGE_BYTES = 2**30


def _row_datatype(mpidtype: MPI.Datatype, row_len: int) -> MPI.Datatype:
    """Committed contiguous MPI datatype of one row of ``row_len`` elements (free it after use).

    Parameters
    ----------
    mpidtype : MPI.Datatype
        Datatype of one element.
    row_len : int
        Elements per row (the product of the trailing dimensions of the array).

    Returns
    -------
    MPI.Datatype
        ``mpidtype.Create_contiguous(row_len)``, committed.

    Raises
    ------
    ValueError
        If a single row exceeds :data:`MAX_MPI_COUNT` elements.
    """
    if row_len > MAX_MPI_COUNT:
        raise ValueError(
            f'one row of {row_len} elements exceeds the MPI count limit ({MAX_MPI_COUNT})'
        )
    rowtype = mpidtype.Create_contiguous(max(1, int(row_len)))
    rowtype.Commit()
    return rowtype


def _row_rounds(counts: NDArray[np.int64], row_bytes: int, nranks: int) -> tuple[int, int]:
    """Rows per rank per round and number of rounds of a row-wise scatter or gather.

    Parameters
    ----------
    counts : ndarray of int, shape ``(nranks,)``
        Rows of every rank.
    row_bytes : int
        Bytes of one row.
    nranks : int
        Number of ranks taking part.

    Returns
    -------
    rows_per_round : int
        At most this many rows per rank in one call, so that one call moves at most
        :data:`MAX_MESSAGE_BYTES` (at least one row).
    nrounds : int
        Number of calls (``0`` when no rank holds a row).

    Notes
    -----
    Every rank computes the same pair from the same broadcast layout, so all ranks make the same
    number of collective calls.
    """
    rows_per_round = max(1, MAX_MESSAGE_BYTES // max(1, int(nranks) * int(row_bytes)))
    most = int(np.max(counts)) if len(counts) else 0
    return rows_per_round, -(-most // rows_per_round)


def scatter_full_indices(
    nsize: int, npool: int, nranks: int | None = None, rank_index: int | None = None
) -> NDArray[np.int64]:
    """Global first-axis indices that :func:`scatter_full` gives one rank, in order.

    Parameters
    ----------
    nsize : int
        Length of the scattered axis.
    npool : int
        Pools of :func:`scatter_full`.
    nranks, rank_index : int, optional
        Number of ranks and the rank (default: ``MPI.COMM_WORLD``).

    Returns
    -------
    ndarray of int, shape ``(n_local,)``
        ``scatter_full(arr, npool)`` on rank ``rank_index`` equals ``arr[indices]``.

    Notes
    -----
    Lets every rank compute its own rows of a k-distributed array directly, instead of computing
    another partition, gathering it on one rank and scattering it again.
    """
    nranks = size if nranks is None else int(nranks)
    rank_index = rank if rank_index is None else int(rank_index)
    indices = []
    nchunks = nsize // nranks
    chunk_e = 0
    if nchunks != 0:
        for pool in range(npool):
            chunk_s, chunk_e = load_balancing(npool, pool, nchunks)
            sizes = load_sizes(nranks, (chunk_e - chunk_s) * nranks, 1)
            first = chunk_s * nranks + sizes[rank_index, 1]
            indices.extend(range(first, first + sizes[rank_index, 2]))
    if nsize % nranks != 0:
        sizes = load_sizes(nranks, nsize - chunk_e * nranks, 1)
        first = chunk_e * nranks + sizes[rank_index, 1]
        indices.extend(range(first, first + sizes[rank_index, 2]))
    return np.array(indices, dtype=np.int64)


def load_balancing(size, rank, n):
    """Compute start and stop indices for a simple 1-D load-balanced partition.

    Parameters
    ----------
    size : int
        Total number of MPI ranks.
    rank : int
        Current MPI rank.
    n : int
        Total number of items to distribute.

    Returns
    -------
    start : int
        First item index assigned to this rank.
    stop : int
        One past the last item index assigned to this rank.
    """
    # Load balancing
    splitsize = float(n) / float(size)
    start = int(np.around(rank * splitsize, decimals=2))
    stop = int(np.around((rank + 1) * splitsize, decimals=2))
    return (start, stop)


# For each processor calculate 3 values:
# 0 - Total number of items to be scattered/gathered on this processor
# 1 - Index in complete array where the subarray begins
# 2 - Dimension of the subarray on this processor
def load_sizes(size, n, dim):
    """Compute scatter/gather layout for all MPI ranks.

    Parameters
    ----------
    size : int
        Total number of MPI ranks.
    n : int
        Number of elements in the first dimension to distribute.
    dim : int
        Product of all remaining dimensions (i.e. the stride per element).

    Returns
    -------
    np.ndarray, shape ``(size, 3)``, int
        Row ``i`` contains:

        - ``sizes[i, 0]`` — number of scalar elements on rank ``i``
          (first-dimension count times ``dim``).
        - ``sizes[i, 1]`` — scalar offset in the flat array where rank
          ``i``\'s sub-array begins.
        - ``sizes[i, 2]`` — first-dimension count on rank ``i``.
    """
    sizes = np.empty((size, 3), dtype=int)
    splitsize = float(n) / float(size)
    for i in range(size):
        start = int(np.around(i * splitsize, decimals=2))
        stop = int(np.around((i + 1) * splitsize, decimals=2))
        sizes[i][0] = dim * (stop - start)
        sizes[i][1] = dim * start
        sizes[i][2] = stop - start
    return sizes


# Scatters first dimension of an array of arbitrary length
def scatter_array(arr: np.ndarray | None, sroot: int = 0) -> np.ndarray:
    """Scatter the first dimension of an array to all MPI ranks.

    Parameters
    ----------
    arr : np.ndarray or None
        Array to scatter (only significant on ``sroot``).  The first
        dimension is distributed; all remaining dimensions are replicated.
    sroot : int, optional
        Rank of the process that holds the data to scatter (default ``0``).

    Returns
    -------
    np.ndarray
        Local sub-array for this rank.  The first dimension has size
        determined by :func:`load_sizes`; remaining dimensions are
        identical to those of ``arr``.

    Notes
    -----
    Uses ``MPI.Scatterv`` for variable-count scatter so that the array
    does not need to be exactly divisible by the number of ranks.  Shape
    and dtype are broadcast from ``sroot`` before the transfer.

    Counts and offsets are in rows (a contiguous MPI datatype of the trailing dimensions) and the
    transfer runs in rounds of at most :data:`MAX_MESSAGE_BYTES`. When even the row offsets would
    exceed the C ``int`` of the MPI interface (more than :data:`MAX_MPI_COUNT` rows), every round
    is staged in a buffer of at most that size on ``sroot``. The data layout is that of an
    element-wise ``Scatterv``: the result is the same, bit for bit.
    """
    layout = None
    if rank == sroot:
        sizes = load_sizes(size, arr.shape[0], 1)
        layout = (sizes[:, 2].copy(), sizes[:, 1].copy(), tuple(arr.shape[1:]), arr.dtype.str)
    counts, starts, tail, dtype_str = comm.bcast(layout, root=sroot)
    pydtype = np.dtype(dtype_str)
    row_len = max(1, int(np.prod(tail, dtype=np.int64)))

    arraux = np.empty((int(counts[rank]),) + tuple(tail), dtype=pydtype)
    recv = arraux.reshape(-1, row_len)
    send = np.ascontiguousarray(arr).reshape(-1, row_len) if rank == sroot else None
    # offsets in rows fit a C int: send straight from the array; otherwise stage every round
    staged = int(np.sum(counts)) > MAX_MPI_COUNT

    rowtype = _row_datatype(MPI._typedict[pydtype.char], row_len)
    try:
        per_round, nrounds = _row_rounds(counts, row_len * pydtype.itemsize, size)
        for step in range(nrounds):
            offset = step * per_round
            round_counts = np.clip(counts - offset, 0, per_round).astype(np.int64)
            mine = int(round_counts[rank])
            sendbuf = None
            if staged:
                displs = np.concatenate(([0], np.cumsum(round_counts)[:-1])).astype(np.int64)
                if rank == sroot:
                    stage = np.empty((int(round_counts.sum()), row_len), dtype=pydtype)
                    for r in range(size):
                        if round_counts[r]:
                            first = int(starts[r]) + offset
                            stage[displs[r] : displs[r] + round_counts[r]] = send[
                                first : first + round_counts[r]
                            ]
                    sendbuf = [stage, round_counts, displs, rowtype]
            elif rank == sroot:
                displs = (starts + np.minimum(offset, counts)).astype(np.int64)
                sendbuf = [send, round_counts, displs, rowtype]
            comm.Scatterv(sendbuf, [recv[offset : offset + mine], rowtype], root=sroot)
    finally:
        rowtype.Free()

    return arraux


# Gathers first dimension of an array of arbitrary length
def gather_array(arr: np.ndarray | None, arraux: np.ndarray, sroot: int = 0) -> None:
    """Gather local sub-arrays into a full array on ``sroot``.

    Parameters
    ----------
    arr : np.ndarray or None
        Destination array on rank ``sroot`` (only written on ``sroot``).
        Its shape must match the result of gathering all ``arraux``
        sub-arrays along the first dimension.
    arraux : np.ndarray
        Local sub-array on this rank.
    sroot : int, optional
        Rank that collects the gathered result (default ``0``).

    Returns
    -------
    None
        The result is written directly into ``arr`` on ``sroot``.

    Raises
    ------
    ValueError
        If ``arr`` on ``sroot`` is not C-contiguous (the rows are written in place).

    Notes
    -----
    Uses ``MPI.Gatherv`` with offsets computed from :func:`load_sizes`.
    The layout is consistent with :func:`scatter_array`.

    Counts and offsets are in rows (a contiguous MPI datatype of the trailing dimensions) and the
    transfer runs in rounds of at most :data:`MAX_MESSAGE_BYTES`, so arrays of more than
    :data:`MAX_MPI_COUNT` elements gather without overflowing the C ``int`` of the MPI interface;
    with more rows than that, every round is staged in a buffer on ``sroot``. The result is the
    same, bit for bit.
    """
    layout = None
    if rank == sroot:
        if not arr.flags.c_contiguous:
            raise ValueError('gather_array writes into arr in place: it must be C-contiguous')
        sizes = load_sizes(size, arr.shape[0], 1)
        layout = (
            sizes[:, 2].copy(),
            sizes[:, 1].copy(),
            int(np.prod(arr.shape[1:], dtype=np.int64)),
        )
    counts, starts, row_len = comm.bcast(layout, root=sroot)
    row_len = max(1, row_len)

    pydtype = np.dtype(arraux.dtype)
    send = np.ascontiguousarray(arraux).reshape(int(counts[rank]), row_len)
    recv = arr.reshape(-1, row_len) if rank == sroot else None
    # offsets in rows fit a C int: receive straight into the array; otherwise stage every round
    staged = int(np.sum(counts)) > MAX_MPI_COUNT

    rowtype = _row_datatype(MPI._typedict[pydtype.char], row_len)
    try:
        per_round, nrounds = _row_rounds(counts, row_len * pydtype.itemsize, size)
        for step in range(nrounds):
            offset = step * per_round
            round_counts = np.clip(counts - offset, 0, per_round).astype(np.int64)
            mine = int(round_counts[rank])
            recvbuf = None
            if staged:
                displs = np.concatenate(([0], np.cumsum(round_counts)[:-1])).astype(np.int64)
                if rank == sroot:
                    stage = np.empty((int(round_counts.sum()), row_len), dtype=pydtype)
                    recvbuf = [stage, round_counts, displs, rowtype]
            elif rank == sroot:
                displs = (starts + np.minimum(offset, counts)).astype(np.int64)
                recvbuf = [recv, round_counts, displs, rowtype]
            comm.Gatherv([send[offset : offset + mine], rowtype], recvbuf, root=sroot)
            if staged and rank == sroot:
                for r in range(size):
                    if round_counts[r]:
                        first = int(starts[r]) + offset
                        recv[first : first + round_counts[r]] = stage[
                            displs[r] : displs[r] + round_counts[r]
                        ]
    finally:
        rowtype.Free()


def scatter_full(arr, npool, sroot=0):
    """Scatter the first dimension of an array using a chunked pool strategy.

    Parameters
    ----------
    arr : np.ndarray or None
        Array to scatter (only significant on ``sroot``).
    npool : int
        Number of scatter pools; the array is chunked into ``npool``
        groups to avoid large single MPI messages.
    sroot : int, optional
        Root rank (default ``0``).

    Returns
    -------
    np.ndarray
        Local sub-array on this rank, sized according to
        :func:`load_balancing`.

    Notes
    -----
    Internally delegates to :func:`scatter_array` for each pool chunk to
    keep individual MPI message sizes manageable.
    """
    if rank == sroot:
        nsizes = comm.bcast(arr.shape)
    else:
        nsizes = comm.bcast(None)

    comm.Barrier()

    nsize = nsizes[0]
    full = int(nsize / size)

    ts, te = load_balancing(size, rank, nsize % size)
    full += te - ts

    if len(nsizes) > 1:
        per_proc_shape = np.concatenate((np.array([full]), nsizes[1:]))
    else:
        per_proc_shape = np.array([full])

    pydtype = None
    if rank == sroot:
        pydtype = arr.dtype
    pydtype = comm.bcast(pydtype)

    temp = np.zeros(per_proc_shape, order='C', dtype=pydtype)

    nchunks = nsize // size

    if nchunks != 0:
        for pool in range(npool):
            chunk_s, chunk_e = load_balancing(npool, pool, nchunks)

            if rank == sroot:
                temp[chunk_s:chunk_e] = scatter_array(
                    np.ascontiguousarray(arr[(chunk_s * size) : (chunk_e * size)])
                )
            else:
                temp[chunk_s:chunk_e] = scatter_array(None)
    else:
        chunk_e = 0

    if nsize % size != 0:
        if rank == sroot:
            temp[chunk_e:] = scatter_array(np.ascontiguousarray(arr[(chunk_e * size) :]))
        else:
            temp[chunk_e:] = scatter_array(None)

    return temp


def gather_full(arr, npool, sroot=0):
    """Gather local sub-arrays from all ranks using a chunked pool strategy.

    Parameters
    ----------
    arr : np.ndarray
        Local sub-array on this rank.
    npool : int
        Number of gather pools.
    sroot : int, optional
        Rank that collects the result (default ``0``).

    Returns
    -------
    np.ndarray or None
        Assembled array on ``sroot``; ``None`` on all other ranks.

    Notes
    -----
    Internally delegates to :func:`gather_array` for each pool chunk.
    The inverse operation of :func:`scatter_full`.
    """
    first_ind_per_proc = np.array([arr.shape[0]])
    nsize = np.zeros_like(first_ind_per_proc)

    comm.Barrier()
    comm.Allreduce(first_ind_per_proc, nsize)

    if len(arr.shape) > 1:
        per_proc_shape = np.concatenate((nsize, arr.shape[1:]))
    else:
        per_proc_shape = np.array([arr.shape[0]])

    nsize = nsize[0]

    if rank == sroot:
        temp = np.zeros(per_proc_shape, order='C', dtype=arr.dtype)
    else:
        temp = None

    nchunks = nsize // size

    if nchunks != 0:
        for pool in range(npool):
            chunk_s, chunk_e = load_balancing(npool, pool, nchunks)

            if rank == sroot:
                gather_array(
                    temp[(chunk_s * size) : (chunk_e * size)], arr[chunk_s:chunk_e], sroot=sroot
                )
            else:
                gather_array(None, arr[chunk_s:chunk_e], sroot=sroot)
    else:
        chunk_e = 0

    if nsize % size != 0:
        if rank == sroot:
            gather_array(temp[(chunk_e * size) :], arr[chunk_e:], sroot=sroot)
        else:
            gather_array(None, arr[chunk_e:], sroot=sroot)

    if rank == sroot:
        return temp


def gather_scatter(arr, scatter_axis, npool):
    """Redistribute an array so that ``scatter_axis`` is the distributed axis.

    Parameters
    ----------
    arr : np.ndarray
        Array to redistribute.  All ranks hold an identical copy on entry.
    scatter_axis : int
        Axis along which the output will be scattered across ranks.
    npool : int
        Number of pools passed to the underlying :func:`scatter_full` /
        :func:`gather_full` calls.

    Returns
    -------
    np.ndarray
        Sub-array for this rank, with ``scatter_axis`` distributed.

    Notes
    -----
    This is useful when a computation requires a different distribution
    axis than the one provided by an earlier :func:`scatter_full` call.
    Each rank first gathers the slices it needs, then the process with
    ``rank == r`` calls :func:`gather_full` with ``sroot=r`` in a loop
    over all ranks.
    """
    # scatter indices for scatter_axis to each proc
    axis_ind = np.array(list(range(arr.shape[scatter_axis])), dtype=int)
    axis_ind = scatter_full(axis_ind, npool)

    # broadcast indices that for scattered array to proc with rank 'r'
    size_r = np.zeros((size), dtype=int, order='C')
    scatter_ind = np.zeros((arr.shape[scatter_axis]), dtype=int, order='C')

    if rank == 0:
        gather_array(size_r, np.array(axis_ind.size, dtype=int))
        gather_array(scatter_ind, np.array(axis_ind, dtype=int))
    else:
        gather_array(None, np.array(axis_ind.size, dtype=int))
        gather_array(None, np.array(axis_ind, dtype=int))

    axis_ind = None

    comm.Bcast(size_r)
    comm.Bcast(scatter_ind)

    # start and end points of indices of scatter axis for each proc
    end = np.cumsum(size_r)
    start = end - size_r
    size_r = None

    for r in range(size):
        comm.Barrier()
        # gather array from each proc with indices for each proc on scatter_axis
        if r == rank:
            temp = gather_full(
                np.take(arr, scatter_ind[start[r] : end[r]], axis=scatter_axis), npool, sroot=r
            )
        else:
            gather_full(
                np.take(arr, scatter_ind[start[r] : end[r]], axis=scatter_axis), npool, sroot=r
            )

    start = end = scatter_ind = None

    return temp


# ================================================================
# NEW FUNCTION
# ================================================================
def _flat_view(arr: np.ndarray) -> np.ndarray:
    """One-dimensional view of a C- or Fortran-contiguous array (writes reach ``arr``).

    Parameters
    ----------
    arr : np.ndarray
        Contiguous array.

    Returns
    -------
    np.ndarray
        ``arr`` as a flat view in its memory order.

    Raises
    ------
    ValueError
        If ``arr`` is not contiguous (no flat view exists).
    """
    if not (arr.flags.c_contiguous or arr.flags.f_contiguous):
        raise ValueError('the array must be contiguous for an in-place MPI operation')
    return arr.reshape(-1, order='A')


def _steps(nelements: int, step: int) -> list[tuple[int, int]]:
    """``(first, last)`` slices of ``nelements`` elements in pieces of at most ``step``.

    Parameters
    ----------
    nelements : int
        Elements to cover.
    step : int
        Largest piece.

    Returns
    -------
    list of tuple of int
        ``[(0, step), (step, 2 step), ...]``, the last one ending at ``nelements``; one piece for an
        empty buffer, so that every rank makes at least one call.
    """
    nelements, step = int(nelements), max(1, int(step))
    if nelements == 0:
        return [(0, 0)]
    return [(first, min(first + step, nelements)) for first in range(0, nelements, step)]


def reduce_full(arr: np.ndarray, sroot: int = 0) -> np.ndarray | None:
    """
    - Sum an array over all MPI ranks and return the result on sroot.
    Parameters
    ----------
    arr : numpy.ndarray
        Local array on each MPI rank. All ranks must have arrays
        with the same shape and dtype.

    sroot : int
        MPI rank that receives the reduced result.

    Returns
    -------
    numpy.ndarray or None
        Sum over all MPI ranks on sroot.
        None on all other ranks.

    Notes
    -----
    Arrays of more than :data:`MAX_MPI_COUNT` elements are reduced in slices of that size; smaller
    ones in one call, as before (the summation order of MPI, hence the last bits, can depend on the
    message size).
    """
    source = np.ascontiguousarray(arr)
    arr_full = np.empty_like(source) if rank == sroot else None
    flat_in = source.reshape(-1)
    flat_out = arr_full.reshape(-1) if arr_full is not None else None
    for first, last in _steps(flat_in.size, MAX_MPI_COUNT):
        out = flat_out[first:last] if flat_out is not None else None
        comm.Reduce(flat_in[first:last], out, op=MPI.SUM, root=sroot)

    return arr_full


def bcast_array(arr: np.ndarray, root: int = 0, communicator: MPI.Comm | None = None) -> np.ndarray:
    """Broadcast a NumPy array in place, in slices that fit the MPI count.

    Parameters
    ----------
    arr : np.ndarray
        Contiguous array of the same shape and dtype on every rank; ``root`` holds the data.
    root : int, optional
        Rank that holds the data (default ``0``).
    communicator : MPI.Comm, optional
        Default ``MPI.COMM_WORLD``.

    Returns
    -------
    np.ndarray
        ``arr``, overwritten on every rank with the data of ``root``.

    Raises
    ------
    ValueError
        If ``arr`` is not contiguous.

    Notes
    -----
    Slices of at most :data:`MAX_MPI_COUNT` elements and :data:`MAX_MESSAGE_BYTES`; a broadcast
    only copies, so the slicing does not change any value.
    """
    world = comm if communicator is None else communicator
    flat = _flat_view(arr)
    step = min(MAX_MPI_COUNT, MAX_MESSAGE_BYTES // max(1, arr.itemsize))
    for first, last in _steps(flat.size, step):
        world.Bcast(flat[first:last], root=root)
    return arr


def allreduce_inplace(
    arr: np.ndarray, op: MPI.Op = MPI.SUM, communicator: MPI.Comm | None = None
) -> np.ndarray:
    """``Allreduce`` of a NumPy array in place (``MPI.IN_PLACE``), in slices that fit the MPI count.

    Parameters
    ----------
    arr : np.ndarray
        Contiguous local array; the same shape and dtype on every rank.
    op : MPI.Op, optional
        Reduction (default ``MPI.SUM``).
    communicator : MPI.Comm, optional
        Default ``MPI.COMM_WORLD``.

    Returns
    -------
    np.ndarray
        ``arr``, overwritten with the reduction on every rank.

    Raises
    ------
    ValueError
        If ``arr`` is not contiguous.

    Notes
    -----
    Arrays of more than :data:`MAX_MPI_COUNT` elements are reduced in slices of that size; smaller
    ones in one call, exactly as ``comm.Allreduce(MPI.IN_PLACE, arr, op=op)``.
    """
    world = comm if communicator is None else communicator
    flat = _flat_view(arr)
    for first, last in _steps(flat.size, MAX_MPI_COUNT):
        world.Allreduce(MPI.IN_PLACE, flat[first:last], op=op)
    return arr


def gen_window(array, root=0):
    """Create a shared-memory window containing a copy of ``array``.

    Parameters
    ----------
    array : np.ndarray or None
        Array to place in shared memory (only significant on ``root``).
    root : int, optional
        Rank that owns the source data (default ``0``).

    Returns
    -------
    np.ndarray
        View into the shared-memory buffer.  All ranks on the same
        compute node can read the data without additional communication
        after the initial barrier.

    Notes
    -----
    Uses ``MPI.Win.Allocate_shared`` to create a one-sided memory window.
    Non-root ranks allocate zero bytes; the data pointer is obtained on
    all ranks via ``Shared_query(0)``.
    """
    # creates a shared memory copy of array on
    # rank == root that all procs can access

    if rank == root:
        array_shape = array.shape
        pydtype = array.dtype
    else:
        array_shape = None
        pydtype = None

    array_shape = comm.bcast(array_shape)
    pydtype = comm.bcast(pydtype)

    size = np.prod(array_shape)

    itemsize = MPI._typedict[np.dtype(pydtype).char].Get_size()
    if rank == root:
        nbytes = size * itemsize
    else:
        nbytes = 0

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
    buf, itemsize = win.Shared_query(0)
    win_array = np.ndarray(
        buffer=buf,
        dtype=pydtype,
        shape=array_shape,
    )

    if rank == root:
        win_array[:] = array

    comm.Barrier()

    return win_array
