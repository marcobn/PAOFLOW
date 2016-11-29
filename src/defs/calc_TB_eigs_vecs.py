#
# AFLOWpi_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
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

def calc_TB_eigs_vecs(Hks,ispin,npool):

    index = None

    if rank == 0:
        nawf,nawf,nk1,nk2,nk3,nspin = Hks.shape
        index = {'nawf':nawf,'nk1':nk1,'nk2':nk2,'nk3':nk3,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nk1 = index['nk1']
    nk2 = index['nk2']
    nk3 = index['nk3']
    nawf = index['nawf']
    nspin = index['nspin']

    if rank == 0:
        eall = np.zeros((nawf*nk1*nk2*nk3,nspin),dtype=float)
        E_k = np.zeros((nk1*nk2*nk3,nawf,nspin),dtype=float)
        v_k = np.zeros((nk1*nk2*nk3,nawf,nawf,nspin),dtype=complex)
        Hks = np.reshape(Hks[:,:,:,:,:,ispin],(nawf,nawf,nk1*nk2*nk3),order='C')
        Hksaux = np.zeros((nk1*nk2*nk3,nawf,nawf),dtype=complex)
        for n in range(nawf):
            for m in range(nawf):
                Hksaux[:,n,m] = Hks[n,m,:]
        Hks = None
    else:
        eall = None
        E_k = None
        v_k = None
        Hksaux = None
        Hks_split = None
        E_k_split = None
        v_k_split = None

    for pool in range (npool):
        if nk1*nk2*nk3%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nk1*nk2*nk3/npool
        #if rank == 0: print('running on ',npool,' pools for nkpool = ',nkpool)

        if rank == 0:
            Hks_split = np.array_split(Hksaux,npool,axis=0)[pool]
            E_k_split = np.array_split(E_k,npool,axis=0)[pool]
            v_k_split = np.array_split(v_k,npool,axis=0)[pool]

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)

        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)
        v_kaux = np.zeros((nsize,nawf,nawf,nspin),dtype=complex)
        aux = np.zeros((nsize,nawf,nawf),dtype=complex)

        comm.barrier()
        comm.Scatter(Hks_split,aux,root=0)

        E_kaux, v_kaux = diago(nsize,aux) 

        comm.barrier()
        comm.Gather(E_kaux,E_k_split,root=0)
        comm.Gather(v_kaux,v_k_split,root=0)

        if rank == 0:
            E_k[pool*nkpool:(pool+1)*nkpool,:] = E_k_split[:,:]
            v_k[pool*nkpool:(pool+1)*nkpool,:,:] = v_k_split[:,:,:]

    if rank == 0:
        #f=open('eig_'+str(ispin)+'.dat','w')
        nall=0
        for n in range(nk1*nk2*nk3):
            for m in range(nawf):
                eall[nall,ispin]=E_k[n,m,ispin]
                #f.write('%.5f  %.5f \n' %(n,E_k[n,m,ispin]))
                nall += 1
        #f.close()

    return(eall,E_k,v_k)

def diago(nsize,aux):

    nawf = aux.shape[1]
    ekp = np.zeros((nsize,nawf),dtype=float)
    ekv = np.zeros((nsize,nawf,nawf),dtype=complex)

    for n in range(nsize):
        eigval,eigvec = LAN.eigh(aux[n,:,:],UPLO='U')
        ekp[n,:] = np.real(eigval)
        ekv[n,:,:] = eigvec

    return(ekp,ekv)
