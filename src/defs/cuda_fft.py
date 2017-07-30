
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
# Restriction: 'axes' must be a list of unique, monotonically increasing integers
def cuda_ifftn ( Hk, axes=[0,1,2] ):

    hkShape = Hk.shape
    hkDim = len(hkShape)
    fftDim = len(axes)

    # Reshape 'axes' to be the array's end dimensions and ensure contiguity
    Hk = np.ascontiguousarray(np.moveaxis(Hk, axes, np.arange(hkDim-fftDim, hkDim, 1)))
    newShape = Hk.shape

    # Calculate number of batches
    batchSize = 1
    for i in range(hkDim-fftDim):
        batchSize *= Hk.shape[i]

    # Reshape to accomodate batching
    batchShape = [None for _ in np.arange(fftDim+1)]
    batchShape[0] = batchSize
    for i in np.arange(0, fftDim, 1):
        batchShape[i+1] = newShape[hkDim-(fftDim-i)]

    Hk = np.reshape(Hk, batchShape)

    # Pass array to the GPU and perform iFFT on each batch
    Hk_gpu = gpuarray.to_gpu(Hk)
    plan = skfft.Plan(Hk_gpu.shape[1:fftDim+1], Hk.dtype, Hk.dtype, Hk_gpu.shape[0])
    skfft.ifft(Hk_gpu, Hk_gpu, plan, True)

    # Reshape to original dimensions
    #Hk = np.moveaxis(Hk_gpu.get(), 0, fftDim)
    Hk = np.reshape(Hk_gpu.get(), newShape)
    Hk = np.moveaxis(Hk, np.arange(hkDim-fftDim, hkDim, 1), axes)

    return Hk


# Perform a FFT on 'axes' of 'Hr'
# Restriction: len(Hr) >= len(axes)
# Restriction: 'axes' must be a list of unique, monotonically increasing integers
def cuda_fftn ( Hr, axes=[0,1,2] ):

    hrShape = Hr.shape
    hrDim = len(hrShape)
    fftDim = len(axes)

    # Reshape 'axes' to be the array's end dimensions and ensure contiguity
    Hr = np.ascontiguousarray(np.moveaxis(Hr, axes, np.arange(hrDim-fftDim, hrDim, 1)))
    newShape = Hr.shape

    # Calculate number of batches
    batchSize = 1
    for i in range(hrDim-fftDim):
        batchSize *= Hr.shape[i]

    # Reshape to accomodate batching
    batchShape = [None for _ in np.arange(fftDim+1)]
    batchShape[0] = batchSize
    for i in np.arange(0, fftDim, 1):
        batchShape[i+1] = newShape[hrDim-(fftDim-i)]

    Hr = np.reshape(Hr, batchShape)

    # Pass array to the GPU and perform FFT on each batch
    Hr_gpu = gpuarray.to_gpu(Hr)
    plan = skfft.Plan(Hr_gpu.shape[1:fftDim+1], Hr.dtype, Hr.dtype, Hr_gpu.shape[0])
    skfft.fft(Hr_gpu, Hr_gpu, plan)

    # Reshape to original dimensions
    #Hr = np.moveaxis(Hr_gpu.get(), 0, fftDim)
    Hr = np.reshape(Hr_gpu.get(), newShape)
    Hr = np.moveaxis(Hr, np.arange(hrDim-fftDim, hrDim, 1), axes)


    return Hr
