
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
import sys, time
from mpi4py import MPI
from communication import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
import skcuda.fft as skfft
driver.init()
nGPU = driver.Device.count()

def cuda_ifftn ( Hk, axes=[0,1,2], sroot=0 ):
    return cuda_efftn(Hk, axes, False, comm, sroot)

def cuda_fftn ( Hr, axes=[0,1,2], sroot=0 ):
    return cuda_efftn(Hr, axes, True, sroot)

# Perform an inverse FFT on 'axes' of 'Hk'
# Restriction: len(Hk) >= len(axes)
# Restriction: 'axes' must be a list of unique, monotonically increasing integers
#@profile
def cuda_efftn ( Haux, axes, forward, sroot ): 
    group = comm.Get_group()
    cgroup = group.Incl(np.arange(0,nGPU,1))
    gcomm = comm.Create(cgroup)

    if rank < nGPU:

        if rank == sroot:
            hShape = Haux.shape
        else:
            hShape = None
        hShape = gcomm.bcast(hShape, root=sroot)

        H = scatter_array(Haux, hShape, complex, sroot, gcomm=gcomm)

        hShape = H.shape
        hDim = len(hShape)
        fftDim = len(axes)

        # Reshape 'axes' to be the array's end dimensions and ensure contiguity
        H = np.ascontiguousarray(np.moveaxis(H, axes, np.arange(hDim-fftDim, hDim, 1)))
        newShape = H.shape

        # Calculate number of batches
        batchSize = 1
        for i in range(hDim-fftDim):
            batchSize *= H.shape[i]

        # Reshape to accomodate batching
        batchShape = [None for _ in np.arange(fftDim+1)]
        batchShape[0] = batchSize
        for i in np.arange(0, fftDim, 1):
            batchShape[i+1] = newShape[hDim-(fftDim-i)]

        H = np.reshape(H, batchShape)

        # Create Device and Context
        context = driver.Device(rank).make_context()

        # Pass array to the GPU and perform iFFT on each batch
        H_gpu = gpuarray.to_gpu(H)
        plan = skfft.Plan(H_gpu.shape[1:fftDim+1], H.dtype, H.dtype, H_gpu.shape[0])

        if forward:
            skfft.fft(H_gpu, H_gpu, plan)
        else:
            skfft.ifft(H_gpu, H_gpu, plan, True)

        # Reshape to original dimensions
        H = np.reshape(H_gpu.get(), newShape)
        H = np.ascontiguousarray(np.moveaxis(H, np.arange(hDim-fftDim, hDim, 1), axes))

        # Pop Context
        context.pop()

        gather_array(Haux, H, complex, sroot, gcomm=gcomm)

        H = None

    comm.Barrier()
    cgroup.Free()

    return Haux
