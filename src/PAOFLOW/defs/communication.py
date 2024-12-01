#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def load_balancing ( size, rank, n ):
    # Load balancing
    splitsize = float(n)/float(size)
    start = int(np.around(rank*splitsize,decimals=2))
    stop = int(np.around((rank+1)*splitsize,decimals=2))
    return(start, stop)

# For each processor calculate 3 values:
# 0 - Total number of items to be scattered/gathered on this processor
# 1 - Index in complete array where the subarray begins
# 2 - Dimension of the subarray on this processor
def load_sizes ( size, n, dim):
    sizes = np.empty((size,3), dtype=int)
    splitsize = float(n)/float(size)
    for i in range(size):
        start = int(np.around(i*splitsize,decimals=2))
        stop = int(np.around((i+1)*splitsize,decimals=2))
        sizes[i][0] = dim*(stop-start)
        sizes[i][1] = dim*start
        sizes[i][2] = stop-start
    return sizes



# Scatters first dimension of an array of arbitrary length
def scatter_array ( arr, sroot=0 ):

    # Compute data type and shape of the scattered array on this process
    pydtype = None
    auxlen = None

    # An array to store the size and dimensions of scattered arrays
    lsizes = np.empty((size,3), dtype=int)
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
    comm.Scatterv([arr, lsizes[:,0], lsizes[:,1], mpidtype], [arraux, mpidtype], root=sroot)

    return arraux

# Gathers first dimension of an array of arbitrary length
def gather_array ( arr, arraux, sroot=0 ):

    # An array to store the size and dimensions of gathered arrays
    lsizes = np.empty((size,3), dtype=int)
    if rank == sroot:
        lsizes = load_sizes(size, arr.shape[0], np.prod(arr.shape[1:]))

    # Broadcast the data offsets
    comm.Bcast([lsizes, MPI.INT], root=sroot)

    # Get the datatype for the MPI transfer
    mpidtype = MPI._typedict[np.dtype(arraux.dtype).char]

    # Gather the data according to load_sizes
    comm.Gatherv([arraux, mpidtype], [arr, lsizes[:,0], lsizes[:,1], mpidtype], root=sroot)


def scatter_full(arr,npool,sroot=0):

    if rank==sroot:
        nsizes = comm.bcast(arr.shape)
    else:
        nsizes = comm.bcast(None)

    comm.Barrier()

    nsize=nsizes[0]
    full = int(nsize/size)

    ts,te = load_balancing(size,rank,nsize%size)
    full+=te-ts

    if len(nsizes)>1:
        per_proc_shape = np.concatenate((np.array([full]),
                                         nsizes[1:]))
    else: per_proc_shape = np.array([full])

    pydtype=None
    if rank==sroot:
        pydtype=arr.dtype
    pydtype = comm.bcast(pydtype)

    temp = np.zeros(per_proc_shape,order="C",dtype=pydtype)

    nchunks = nsize//size
    
    if nchunks!=0:
        for pool in range(npool):
            chunk_s,chunk_e = load_balancing(npool,pool,nchunks)

            if rank==sroot:
                temp[chunk_s:chunk_e] = scatter_array(np.ascontiguousarray(arr[(chunk_s*size):(chunk_e*size)]))
            else:
                temp[chunk_s:chunk_e] = scatter_array(None)
    else:
        chunk_e=0

    if nsize%size!=0:
        if rank==sroot:
            temp[chunk_e:] = scatter_array(np.ascontiguousarray(arr[(chunk_e*size):]))
        else:
            temp[chunk_e:] = scatter_array(None)

    return temp


def gather_full(arr,npool,sroot=0):

    first_ind_per_proc = np.array([arr.shape[0]])
    nsize              = np.zeros_like(first_ind_per_proc)

    comm.Barrier()
    comm.Allreduce(first_ind_per_proc,nsize)

    if len(arr.shape)>1:
        per_proc_shape = np.concatenate((nsize,arr.shape[1:]))
    else: per_proc_shape = np.array([arr.shape[0]])

    nsize=nsize[0]

    if rank==sroot:
        temp = np.zeros(per_proc_shape,order="C",dtype=arr.dtype)
    else: temp = None

    nchunks = nsize//size
    
    if nchunks!=0:
        for pool in range(npool):
            chunk_s,chunk_e = load_balancing(npool,pool,nchunks)

            if rank==sroot:
                gather_array(temp[(chunk_s*size):(chunk_e*size)],arr[chunk_s:chunk_e],sroot=sroot)
            else:
                gather_array(None,arr[chunk_s:chunk_e],sroot=sroot)
    else:
        chunk_e=0

    if nsize%size!=0:
        if rank==sroot:
            gather_array(temp[(chunk_e*size):],arr[chunk_e:],sroot=sroot)
        else:
            gather_array(None,arr[chunk_e:],sroot=sroot)
        
    if rank == sroot:
        return temp


def gather_scatter(arr,scatter_axis,npool):
    #scatter indices for scatter_axis to each proc
    axis_ind = np.array(list(range(arr.shape[scatter_axis])),dtype=int)
    axis_ind = scatter_full(axis_ind,npool)

    #broadcast indices that for scattered array to proc with rank 'r'
    size_r = np.zeros((size),dtype=int,order='C')
    scatter_ind = np.zeros((arr.shape[scatter_axis]),dtype=int,order='C')
    
    if rank==0:
        gather_array(size_r,np.array(axis_ind.size,dtype=int))
        gather_array(scatter_ind,np.array(axis_ind,dtype=int))
    else:
        gather_array(None,np.array(axis_ind.size,dtype=int))
        gather_array(None,np.array(axis_ind,dtype=int))

    axis_ind = None

    comm.Bcast(size_r)
    comm.Bcast(scatter_ind)

    #start and end points of indices of scatter axis for each proc
    end   = np.cumsum(size_r)
    start = end - size_r
    size_r = None
    
    for r in range(size):
        comm.Barrier()
        #gather array from each proc with indices for each proc on scatter_axis
        if r == rank:
            temp = gather_full(np.take(arr,scatter_ind[start[r]:end[r]],axis=scatter_axis),npool,sroot=r)
        else:
            gather_full(np.take(arr,scatter_ind[start[r]:end[r]],axis=scatter_axis),npool,sroot=r)

    start = end = scatter_ind = None

    return temp


def gen_window(array,root=0):
    # creates a shared memory copy of array on
    # rank == root that all procs can access

    if rank==root:
        array_shape=array.shape
        pydtype = array.dtype
    else:
        array_shape=None
        pydtype=None

    array_shape=comm.bcast(array_shape)
    pydtype=comm.bcast(pydtype)

    size=np.prod(array_shape)

    itemsize = MPI._typedict[np.dtype(pydtype).char].Get_size()
    if rank == root:
        nbytes = size * itemsize
    else:
        nbytes = 0

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
    buf, itemsize = win.Shared_query(0)
    win_array = np.ndarray(buffer=buf, dtype=pydtype, shape=array_shape,)

    if rank==root:
        win_array[:]=array

    comm.Barrier()
    
    return win_array
