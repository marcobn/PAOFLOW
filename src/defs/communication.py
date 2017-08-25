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
  try:
    # Compute data type and shape of the scattered array on this process
    pydtype = None
    auxlen = None

    # An array to store the size and dimensions of scattered arrays
    lsizes = np.empty((size,3), dtype=int)
    if rank == sroot:
        pydtype = arr.dtype
        auxshape = np.array(list(arr.shape))
        auxlen = len(auxshape)
        lsizes = load_sizes(size, arr.shape[0], arr[0].size)

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
  except Exception as e:
    raise e

# Gathers first dimension of an array of arbitrary length
def gather_array ( arr, arraux, sroot=0 ):
  try:
    # Data type of the scattered array on this process
    pydtype = None

    # An array to store the size and dimensions of gathered arrays
    lsizes = np.empty((size,3), dtype=int)
    if rank == sroot:
        pydtype = arr.dtype
        lsizes = load_sizes(size, arr.shape[0], arr[0].size)

    # Broadcast the data type and offsets
    pydtype = comm.bcast(pydtype, root=sroot)
    comm.Bcast([lsizes, MPI.INT], root=sroot)

    # Get the datatype for the MPI transfer
    mpidtype = MPI._typedict[np.dtype(pydtype).char]

    # Gather the data according to load_sizes
    comm.Gatherv([arraux, mpidtype], [arr, lsizes[:,0], lsizes[:,1], mpidtype], root=sroot)
  except Exception as e:
    raise e
