import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
def scatter_array(arr, sroot=0):
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
    """
    # Compute data type and shape of the scattered array on this process
    pydtype = None
    auxlen = None

    # An array to store the size and dimensions of scattered arrays
    lsizes = np.empty((size, 3), dtype=int)
    if rank == sroot:
        pydtype = arr.dtype
        auxshape = np.array(list(arr.shape))
        auxlen = len(auxshape)
        lsizes = load_sizes(size, arr.shape[0], np.prod(arr.shape[1:]))

    # Broadcast the data type and dimension of the scattered array
    pydtype = comm.bcast(pydtype, root=sroot)
    auxlen = comm.bcast(auxlen, root=sroot)

    # An array to store the shape of array's dimensions
    if rank != sroot:
        auxshape = np.zeros((auxlen,), dtype=int)

    # Broadcast the shape of each dimension
    for i in np.arange(auxlen):
        auxshape[i] = comm.bcast(auxshape[i], root=sroot)

    comm.Bcast([auxshape, MPI.INT], root=sroot)
    comm.Bcast([lsizes, MPI.INT], root=sroot)

    # Change the first dimension of auxshape to the correct size for scatter
    auxshape[0] = lsizes[rank][2]

    # Initialize aux array
    arraux = np.empty(auxshape, dtype=pydtype)

    # Get the datatype for the MPI transfer
    mpidtype = MPI._typedict[np.dtype(pydtype).char]

    # Scatter the data according to load_sizes
    comm.Scatterv([arr, lsizes[:, 0], lsizes[:, 1], mpidtype], [arraux, mpidtype], root=sroot)

    return arraux


# Gathers first dimension of an array of arbitrary length
def gather_array(arr, arraux, sroot=0):
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

    Notes
    -----
    Uses ``MPI.Gatherv`` with offsets computed from :func:`load_sizes`.
    The layout is consistent with :func:`scatter_array`.
    """
    # An array to store the size and dimensions of gathered arrays
    lsizes = np.empty((size, 3), dtype=int)
    if rank == sroot:
        lsizes = load_sizes(size, arr.shape[0], np.prod(arr.shape[1:]))

    # Broadcast the data offsets
    comm.Bcast([lsizes, MPI.INT], root=sroot)

    # Get the datatype for the MPI transfer
    mpidtype = MPI._typedict[np.dtype(arraux.dtype).char]

    # Gather the data according to load_sizes
    comm.Gatherv([arraux, mpidtype], [arr, lsizes[:, 0], lsizes[:, 1], mpidtype], root=sroot)


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
def reduce_full(arr, sroot=0):
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
    """
    if rank == sroot:
        arr_full = np.empty_like(arr)
    else:
        arr_full = None
    comm.Reduce(arr, arr_full, op=MPI.SUM, root=sroot)

    return arr_full


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
