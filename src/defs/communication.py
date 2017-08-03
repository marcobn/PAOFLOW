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

from mpi4py import MPI
from load_balancing import *

comm = MPI.COMM_WORLD

# Scatters first dimension of an array of arbitrary length
def scatter_array ( arr, auxshape, pydtype, sroot, gcomm=comm ):
    rank = gcomm.Get_rank()
    size = gcomm.Get_size() 

    # An array to store the size and dimensions of scattered arrays
    lsizes = np.empty((size,3), dtype=int)
    if rank == sroot:
        lsizes = load_sizes(size, arr.shape[0], arr[0].size)
    gcomm.Bcast([lsizes, MPI.INT], root=sroot)

    # Change the first dimension of auxshape to the correct size for scatter
    auxshape = list(auxshape)
    auxshape[0] = lsizes[rank][2]

    # Initialize aux array
    arraux = np.empty(auxshape, dtype=pydtype)

    # Get the datatype for the MPI transfer
    mpidtype = MPI._typedict[np.dtype(pydtype).char]

    # Scatter the data according to load_sizes
    gcomm.Scatterv([arr, lsizes[:,0], lsizes[:,1], mpidtype], [arraux, mpidtype], root=sroot)

    return arraux

# Scatters first dimension of an array of arbitrary length
def gather_array ( arr, arraux, pydtype, sroot, gcomm=comm ):
    rank = gcomm.Get_rank()
    size = gcomm.Get_size()

    # An array to store the size and dimensions of gathered arrays
    lsizes = np.empty((size,3), dtype=int)
    if rank == sroot:
        lsizes = load_sizes(size, arr.shape[0], arr[0].size)
    gcomm.Bcast([lsizes, MPI.INT], root=sroot)

    # Get the datatype for the MPI transfer
    mpidtype = MPI._typedict[np.dtype(pydtype).char]


    # Gather the data according to load_sizes
    gcomm.Gatherv([arraux, mpidtype], [arr, lsizes[:,0], lsizes[:,1], mpidtype], root=sroot)
