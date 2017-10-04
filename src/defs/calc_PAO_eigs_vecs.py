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
from scipy import linalg as LA
from numpy import linalg as LAN
import numpy as np
import os, sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def calc_PAO_eigs_vecs(Hksp,bnd,npool):
    index = None

    if rank == 0:
        nktot,nawf,nawf,nspin = Hksp.shape
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    

    aux = scatter_full(Hksp,npool)

    E_kaux = np.zeros((aux.shape[0],nawf,nspin),dtype=float)
    v_kaux = np.zeros((aux.shape[0],nawf,nawf,nspin),dtype=complex)



    for ispin in xrange(nspin):
        E_kaux[:,:,ispin], v_kaux[:,:,:,ispin] = diago(aux.shape[0],aux[:,:,:,ispin])


    v_k = gather_full(v_kaux,npool)
    v_kaux = None
    E_k = gather_full(E_kaux,npool)
    v_kaux = None


    if rank == 0:
        eall = np.reshape(np.delete(E_k,np.s_[bnd:],axis=1),(nktot*bnd,nspin),order='C')
    else: 
        eall = None
        v_k = None
        E_k = None

    return(eall,E_k,v_k)

def diago(nsize,aux):
    nawf = aux.shape[1]
    ekp = np.zeros((nsize,nawf),dtype=float)
    ekv = np.zeros((nsize,nawf,nawf),dtype=complex)

    for n in xrange(nsize):
        eigval,eigvec = LAN.eigh(aux[n,:,:],UPLO='U')
        ekp[n,:] = np.real(eigval)
        ekv[n,:,:] = eigvec

    return(ekp,ekv)
