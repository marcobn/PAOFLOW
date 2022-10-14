#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
#F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli, Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
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
    return cuda_efftn(Hk, axes, False)

# Perform an FFT on 'axes' of 'Hr'
# Restriction: len(Hr) >= len(axes)
# Restriction: 'axes' must be a list of unique, monotonically increasing, integers
#               beginning with 0.
def cuda_fftn ( Hr, axes=[0,1,2] ):
    return cuda_efftn(Hr, axes, True)


def cuda_efftn(H, axes, forward):
    hShape = H.shape
    hDim = len(hShape)
    fftDim = len(axes)

    # Reshape 'axes' to be the array's end dimensions and ensure contiguity
    H = np.ascontiguousarray(np.moveaxis(H, axes, np.arange(hDim-fftDim, hDim, 1)))

    # Calculate number of batches
    batchSize = 1
    for i in range(hDim-fftDim):
        batchSize *= H.shape[i]

    # Reshape to accomodate batching
    H = np.reshape(H, (batchSize, H.shape[hDim-3], H.shape[hDim-2], H.shape[hDim-1]))

    # Pass array to the GPU and perform iFFT on each batch
    H_gpu = gpuarray.to_gpu(H)
    plan = skfft.Plan(H_gpu.shape[1:fftDim+1], H.dtype, H.dtype, H_gpu.shape[0])

    if forward:
        skfft.fft(H_gpu, H_gpu, plan)
    else:
        skfft.ifft(H_gpu, H_gpu, plan, True)

    # Reshape to original dimensions
    H = np.moveaxis(H_gpu.get(), 0, fftDim)
    H = np.reshape(H, hShape)

    return H
