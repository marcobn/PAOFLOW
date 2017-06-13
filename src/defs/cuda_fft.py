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
import sys, time
import multiprocessing

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as skfft


# Perform an inverse FFT on 'axes' of 'Hk'
# Restriction: len(Hk) >= len(axes)
# Restriction: 'axes' must be a list of unique, monotonically increasing, integers
#               beginning with 0.
def cuda_ifftn ( Hk, axes=[0,1,2] ):

    hkShape = Hk.shape
    hkDim = len(hkShape)
    fftDim = len(axes)

    # Reshape 'axes' to be the array's end dimensions and ensure contiguity
    Hk = np.ascontiguousarray(np.moveaxis(Hk, axes, np.arange(hkDim-fftDim, hkDim, 1)))

    # Calculate number of batches
    batchSize = 1
    for i in range(hkDim-fftDim):
        batchSize *= Hk.shape[i]

    # Reshape to accomodate batching
    Hk = np.reshape(Hk, (batchSize, Hk.shape[hkDim-3], Hk.shape[hkDim-2], Hk.shape[hkDim-1]))

    # Pass array to the GPU and perform iFFT on each batch
    Hk_gpu = gpuarray.to_gpu(Hk)
    plan = skfft.Plan(Hk_gpu.shape[1:fftDim+1], Hk.dtype, Hk.dtype, Hk_gpu.shape[0])
    skfft.ifft(Hk_gpu, Hk_gpu, plan, True)

    # Reshape to original dimensions
    Hk = np.moveaxis(Hk_gpu.get(), 0, fftDim)
    Hk = np.reshape(Hk, hkShape)

    return Hk


# Perform a FFT on 'axes' of 'Hk'
# Restriction: len(Hk) >= len(axes)
# Restriction: 'axes' must be a list of unique, monotonically increasing, integers
#               beginning with 0.
def cuda_fftn ( Hr, axes=[0,1,2] ):

    hrShape = Hr.shape
    hrDim = len(hrShape)
    fftDim = len(axes)

    # Reshape 'axes' to be the array's end dimensions and ensure contiguity
    Hr = np.ascontiguousarray(np.moveaxis(Hr, axes, np.arange(hrDim-fftDim, hrDim, 1)))

    # Calculate number of batches
    batchSize = 1
    for i in range(hrDim-fftDim):
        batchSize *= Hr.shape[i]

    # Reshape to accomodate batching
    Hr = np.reshape(Hr, (batchSize, Hr.shape[hrDim-3], Hr.shape[hrDim-2], Hr.shape[hrDim-1]))

    # Pass array to the GPU and perform FFT on each batch
    Hr_gpu = gpuarray.to_gpu(Hr)
    plan = skfft.Plan(Hr_gpu.shape[1:fftDim+1], Hr.dtype, Hr.dtype, Hr_gpu.shape[0])
    skfft.fft(Hr_gpu, Hr_gpu, plan)

    # Reshape to original dimensions
    Hr = np.moveaxis(Hr_gpu.get(), 0, fftDim)
    Hr = np.reshape(Hr, hrShape)

    return Hr
