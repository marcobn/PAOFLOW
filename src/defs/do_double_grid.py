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

from zero_pad import *

scipyfft = False
try:
    import pyfftw
except:
    from scipy import fftpack as FFT
    scipyfft = True


from communication import *
from load_balancing import *

comm=MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

nthread = size

def do_double_grid(nfft1,nfft2,nfft3,HRaux,nthread,npool):
    # Fourier interpolation on extended grid (zero padding)
    index = None
    if rank==0:
        nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape
        nktot = nk1*nk2*nk3
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin,'nk1':nk1,'nk2':nk2,'nk3':nk3}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf  = index['nawf']
    nspin = index['nspin']
    nk1   = index['nk1']
    nk2   = index['nk2']
    nk3   = index['nk3']

    nk1p = nfft1
    nk2p = nfft2
    nk3p = nfft3
    nfft1 = nfft1-nk1
    nfft2 = nfft2-nk2
    nfft3 = nfft3-nk3
    nktotp= nk1p*nk2p*nk3p

    # Extended R to k (with zero padding)
    if rank==0:
        HRaux = np.reshape(HRaux,(nawf**2,nk1,nk2,nk3,nspin))
    else:
        Hksp = None
    if rank==0:
        HR_aux = scatter_full(HRaux,npool)
    else:
        HR_aux = scatter_full(None,npool)

    Hk_aux  = np.zeros((HR_aux.shape[0],nk1p,nk2p,nk3p,nspin),dtype=complex)

    for ispin in xrange(nspin):
        if not scipyfft:
            for i in xrange(nawf):
                for j in xrange(nawf):
                    aux = zero_pad(H_aux[i,j,:,:,:,ispin],nk1,nk2,nk3,nfft1,nfft2,nfft3)
                    fft = pyfftw.FFTW(aux,Hksp[:,:,:,i,j,ispin], axes=(0,1,2), direction='FFTW_FORWARD',\
                        flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                    Hk_aux[:,:,:,i,j,ispin] = fft()
        else:
            for n in xrange(HR_aux.shape[0]):
                Hk_aux[n,:,:,:,ispin] = FFT.fftn(zero_pad(HR_aux[n,:,:,:,ispin],
                                                          nk1,nk2,nk3,nfft1,nfft2,nfft3))

    nk1 = nk1p
    nk2 = nk2p
    nk3 = nk3p
    HR_aux = None
    aux    = None

    return(Hk_aux,nk1,nk2,nk3)

