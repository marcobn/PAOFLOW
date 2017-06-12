import numpy as np
import sys, time
import multiprocessing

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as skfft

def gpu_ifftn ( Hk, axes=[0,1,2] ):

    hkShape = Hk.shape
    hkDim = len(hkShape)
    fftDim = len(axes)

    Hk = np.ascontiguousarray(np.moveaxis(Hk, axes, np.arange(hkDim-fftDim, hkDim, 1)))

    batchSize = 1
    for i in range(hkDim-fftDim):
        batchSize *= Hk.shape[i]

    Hk = np.reshape(Hk, (batchSize, Hk.shape[hkDim-3], Hk.shape[hkDim-2], Hk.shape[hkDim-1]))

    Hk_gpu = gpuarray.to_gpu(Hk)
    plan = skfft.Plan(Hk_gpu.shape[1:fftDim+1], Hk.dtype, Hk.dtype, Hk_gpu.shape[0])
    skfft.ifft(Hk_gpu, Hk_gpu, plan, True)

    Hk = np.moveaxis(Hk_gpu.get(), 0, fftDim)
    Hk = np.reshape(Hk, hkShape)

    return Hk


def gpu_fftn ( Hr, axes=[0,1,2] ):

    hrShape = Hr.shape
    hrDim = len(hrShape)
    fftDim = len(axes)

    Hr = np.ascontiguousarray(np.moveaxis(Hr, axes, np.arange(hrDim-fftDim, hrDim, 1)))

    batchSize = 1
    for i in range(hrDim-fftDim):
        batchSize *= Hr.shape[i]

    Hr = np.reshape(Hr, (batchSize, Hr.shape[hrDim-3], Hr.shape[hrDim-2], Hr.shape[hrDim-1]))

    Hr_gpu = gpuarray.to_gpu(Hr)
    plan = skfft.Plan(Hr_gpu.shape[1:fftDim+1], Hr.dtype, Hr.dtype, Hr_gpu.shape[0])
    skfft.fft(Hr_gpu, Hr_gpu, plan)

    Hr = np.moveaxis(Hr_gpu.get(), 0, fftDim)
    Hr = np.reshape(Hr, hrShape)

    return Hr
