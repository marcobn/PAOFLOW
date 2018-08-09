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
from scipy import linalg as LA
from numpy import linalg as LAN
import numpy as np
import os, sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *
from communication import *
import scipy.fftpack as FFT

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def calc_PAO_eigs_vecs(Hksp,bnd,npool):

    _,nk1,nk2,nk3,nspin = Hksp.shape

    Hksp = np.reshape(Hksp,(Hksp.shape[0],nk1*nk2*nk3,nspin))

    aux = gather_scatter(Hksp,1,npool)
    
    nawf=int(np.sqrt(aux.shape[0]))
    aux = np.rollaxis(aux,0,2)
    aux = np.reshape(aux,(aux.shape[0],nawf,nawf,nspin),order="C")

    E_kaux = np.zeros((aux.shape[0],nawf,nspin),dtype=float)
    v_kaux = np.zeros((aux.shape[0],nawf,nawf,nspin),dtype=complex)



    for ispin in range(nspin):
        E_kaux[:,:,ispin], v_kaux[:,:,:,ispin] = diago(aux.shape[0],aux[:,:,:,ispin])

    aux = None

    return(E_kaux,v_kaux)

def diago(nsize,aux):
    nawf = aux.shape[1]
    ekp = np.zeros((nsize,nawf),dtype=float)
    ekv = np.zeros((nsize,nawf,nawf),dtype=complex)

    for n in range(nsize):
        eigval,eigvec = LAN.eigh(aux[n,:,:],UPLO='U')
        ekp[n,:] = np.real(eigval)
        ekv[n,:,:] = eigvec

    return(ekp,ekv)
