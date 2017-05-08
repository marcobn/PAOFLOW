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
#from numpy import fft as NFFT
import numpy as np
import cmath
import sys, time
from mpi4py import MPI
import multiprocessing

try:
    import pyfftw
except:
    from scipy import fftpack as FFT

from zero_pad import *

comm=MPI.COMM_WORLD
size = comm.Get_size()

nthread = size

def do_double_grid(nfft1,nfft2,nfft3,HRaux,nthread,scipyfft):
    # Fourier interpolation on extended grid (zero padding)
    if HRaux.shape[0] != 3 and HRaux.shape[1] == HRaux.shape[0]:
        nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape
        nk1p = nfft1
        nk2p = nfft2
        nk3p = nfft3
        nfft1 = nfft1-nk1
        nfft2 = nfft2-nk2
        nfft3 = nfft3-nk3
        nktotp= nk1p*nk2p*nk3p

        # Extended R to k (with zero padding)
        Hksp  = np.zeros((nk1p,nk2p,nk3p,nawf,nawf,nspin),dtype=complex)
        aux = np.zeros((nk1,nk2,nk3),dtype=complex)

        for ispin in xrange(nspin):
            if not scipyfft:
                for i in xrange(nawf):
                    for j in xrange(nawf):
                        aux = zero_pad(HRaux[i,j,:,:,:,ispin],nk1,nk2,nk3,nfft1,nfft2,nfft3)
                        fft = pyfftw.FFTW(aux,Hksp[:,:,:,i,j,ispin], axes=(0,1,2), direction='FFTW_FORWARD',\
                                flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                        Hksp[:,:,:,i,j,ispin] = fft()
            else:
                for i in xrange(nawf):
                    for j in xrange(nawf):
                        aux = HRaux[i,j,:,:,:,ispin]
                        Hksp[:,:,:,i,j,ispin] = FFT.fftn(zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3))

    else:
        sys.exit('wrong dimensions in input array')

    nk1 = nk1p
    nk2 = nk2p
    nk3 = nk3p
    aux = None
    return(Hksp,nk1,nk2,nk3)

