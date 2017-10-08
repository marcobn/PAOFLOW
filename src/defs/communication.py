# 
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
import time
from mpi4py import MPI
from load_balancing import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
    # Data type of the scattered array on this process
    pydtype = None

    # An array to store the size and dimensions of gathered arrays
    lsizes = np.empty((size,3), dtype=int)
    if rank == sroot:
        pydtype = arr.dtype
        lsizes = load_sizes(size, arr.shape[0], np.prod(arr.shape[1:]))
    # Broadcast the data type and offsets
    pydtype = comm.bcast(pydtype, root=sroot)
    comm.Bcast([lsizes, MPI.INT], root=sroot)

    # Get the datatype for the MPI transfer
    mpidtype = MPI._typedict[np.dtype(pydtype).char]

    # Gather the data according to load_sizes
    comm.Gatherv([arraux, mpidtype], [arr, lsizes[:,0], lsizes[:,1], mpidtype], root=sroot)





def scatter_full(arr,npool):

    if rank==0:
        nsizes = comm.bcast(arr.shape)
    else:
        nsizes = comm.bcast(None)

    comm.Barrier()

    nsize=nsizes[0]
    start_tot,end_tot   = load_balancing(size,rank,nsize)

    if nsizes.size>1:
        per_proc_shape = np.concatenate((np.array([end_tot-start_tot]),
                                         nsizes[1:]))
    else: per_proc_shape = np.array([end_tot-start_tot])

    pydtype=None
    if rank==0:
        pydtype=arr.dtype
    pydtype = comm.bcast(pydtype)

    temp = np.zeros(per_proc_shape,order="C",dtype=pydtype)

    nchunks = nsize/size
    
    if nchunks!=0:
        for pool in xrange(npool):
            chunk_s,chunk_e = load_balancing(npool,pool,nchunks)

            if rank==0:
                temp[chunk_s:chunk_e] = scatter_array(np.ascontiguousarray(arr[(chunk_s*size):(chunk_e*size)]))
            else:
                temp[chunk_s:chunk_e] = scatter_array(None)
    else:
        chunk_e=0

    if nsize%size!=0:
        if rank==0:
            temp[chunk_e:] = scatter_array(np.ascontiguousarray(arr[(chunk_e*size):]))
        else:
            temp[chunk_e:] = scatter_array(None)

    return temp


def gather_full(arr,npool):

    first_ind_per_proc = np.array([arr.shape[0]])
    nsize              = np.zeros_like(first_ind_per_proc)

    comm.Barrier()
    comm.Allreduce(first_ind_per_proc,nsize)

    if nsizes.size>1:
        per_proc_shape = np.concatenate((nsize,arr.shape[1:]))
    else: per_proc_shape = np.array([arr.shape[0]])

    nsize=nsize[0]

    if rank==0:
        temp = np.zeros(per_proc_shape,order="C",dtype=arr.dtype)
    else: temp = None

    nchunks = nsize/size
    

    if nchunks!=0:
        for pool in xrange(npool):
            chunk_s,chunk_e = load_balancing(npool,pool,nchunks)
            if rank==0:
                gather_array(temp[(chunk_s*size):(chunk_e*size)],arr[chunk_s:chunk_e])
            else:
                gather_array(None,arr[chunk_s:chunk_e])
    else:
        chunk_e=0

    if nsize%size!=0:
        if rank==0:
            gather_array(temp[(chunk_e*size):],arr[chunk_e:])
        else:
            gather_array(None,arr[chunk_e:])
        

    return temp
