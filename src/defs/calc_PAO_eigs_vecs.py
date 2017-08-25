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
  try:
    index = None

    if rank == 0:
        nktot,nawf,nawf,nspin = Hksp.shape
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    if rank == 0:
        eall = np.zeros((bnd*nktot,nspin),dtype=float)
        E_k = np.zeros((nktot,nawf,nspin),dtype=float)
        v_k = np.zeros((nktot,nawf,nawf,nspin),dtype=complex)
    else:
        eall = None
        E_k = None
        v_k = None
        Hks_split = None
        E_k_split = None
        v_k_split = None

    for pool in xrange (npool):
        ini_ip, end_ip = load_balancing(npool,pool,nktot)
        nkpool = end_ip - ini_ip

        if rank == 0:
            Hks_split = Hksp[ini_ip:end_ip]
            E_k_split = E_k[ini_ip:end_ip]
            v_k_split = v_k[ini_ip:end_ip]

        ini_ik, end_ik = load_balancing(size, rank, nkpool)
        nsize = end_ik - ini_ik

        E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)
        v_kaux = np.zeros((nsize,nawf,nawf,nspin),dtype=complex)

        aux = scatter_array(Hks_split)

        for ispin in xrange(nspin):
            E_kaux[:,:,ispin], v_kaux[:,:,:,ispin] = diago(nsize,aux[:,:,:,ispin])

        gather_array(E_k_split, E_kaux)
        gather_array(v_k_split, v_kaux)

        if rank == 0:
            E_k[ini_ip:end_ip,:,:] = E_k_split[:,:,:]
            v_k[ini_ip:end_ip,:,:,:] = v_k_split[:,:,:,:]

    if rank == 0:
        eall = np.reshape(np.delete(E_k,np.s_[bnd:],axis=1),(nktot*bnd,nspin),order='C')

    return(eall,E_k,v_k)
  except Exception as e:
    raise e

def diago(nsize,aux):
  try:
    nawf = aux.shape[1]
    ekp = np.zeros((nsize,nawf),dtype=float)
    ekv = np.zeros((nsize,nawf,nawf),dtype=complex)

    for n in xrange(nsize):
        eigval,eigvec = LAN.eigh(aux[n,:,:],UPLO='U')
        ekp[n,:] = np.real(eigval)
        ekv[n,:,:] = eigvec

    return(ekp,ekv)
  except Exception as e:
    raise e
