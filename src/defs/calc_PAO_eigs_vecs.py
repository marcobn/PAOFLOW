#
# PAOpy
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

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def calc_PAO_eigs_vecs(Hksp,npool):

    index = None

    if rank == 0:
        nktot,nawf,nawf,nspin = Hksp.shape
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    if rank == 0:
        eall = np.zeros((nawf*nktot,nspin),dtype=float)
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
        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh - calc_PAO_eigs_vecs')
        nkpool = nktot/npool

        if rank == 0:
            Hks_split = np.array_split(Hksp,npool,axis=0)[pool]
            E_k_split = np.array_split(E_k,npool,axis=0)[pool]
            v_k_split = np.array_split(v_k,npool,axis=0)[pool]

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)

        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)
        v_kaux = np.zeros((nsize,nawf,nawf,nspin),dtype=complex)
        aux = np.zeros((nsize,nawf,nawf,nspin),dtype=complex)

        comm.barrier()
        comm.Scatter(Hks_split,aux,root=0)

        for ispin in xrange(nspin):
            E_kaux[:,:,ispin], v_kaux[:,:,:,ispin] = diago(nsize,aux[:,:,:,ispin])

        comm.barrier()
        comm.Gather(E_kaux,E_k_split,root=0)
        comm.Gather(v_kaux,v_k_split,root=0)

        if rank == 0:
            E_k[pool*nkpool:(pool+1)*nkpool,:,:] = E_k_split[:,:,:]
            v_k[pool*nkpool:(pool+1)*nkpool,:,:,:] = v_k_split[:,:,:,:]

    if rank == 0:
        eall = np.reshape(E_k,(nktot*nawf,nspin),order='C')

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
