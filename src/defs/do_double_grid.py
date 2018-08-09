#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
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

    for ispin in range(nspin):
        for n in range(HR_aux.shape[0]):
            Hk_aux[n,:,:,:,ispin] = FFT.fftn(zero_pad(HR_aux[n,:,:,:,ispin],
                                                      nk1,nk2,nk3,nfft1,nfft2,nfft3))


    Hk_aux=FFT.fftshift(Hk_aux,axes=(1,2,3))
    nk1 = nk1p
    nk2 = nk2p
    nk3 = nk3p
    HR_aux = None
    aux    = None

    return(Hk_aux,nk1,nk2,nk3)

