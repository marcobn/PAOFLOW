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
import os

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def calc_TB_eigs(Hks,Sks,read_S,ispin):

    nawf,nawf,nk1,nk2,nk3,nspin = Hks.shape
    eall = np.zeros((nawf*nk1*nk2*nk3))

    aux = np.zeros((nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
    saux = np.zeros((nawf,nawf,nk1*nk2*nk3),dtype=complex)

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j*nk3 + i*nk2*nk3
                aux[:,:,n,ispin] = Hks[:,:,i,j,k,ispin]
                if read_S:
                    saux[:,:,n] = Sks[:,:,i,j,k]

    E_k = np.zeros((nawf,nk1*nk2*nk3,nspin),dtype=float)
    E_kaux = np.zeros((nawf,nk1*nk2*nk3,nspin,1),dtype=float)
    E_kaux1 = np.zeros((nawf,nk1*nk2*nk3,nspin,1),dtype=float)

    # Load balancing
    ini_i = np.zeros((size),dtype=int)
    end_i = np.zeros((size),dtype=int)
    splitsize = 1.0/size*(nk1*nk2*nk3)
    for i in range(size):
        ini_i[i] = int(round(i*splitsize))
        end_i[i] = int(round((i+1)*splitsize))
    ini_ik = ini_i[rank]
    end_ik = end_i[rank]

    E_kaux[:,:,:,0] = diago(ini_ik,end_ik,aux,saux,read_S,ispin)

    if rank == 0:
        E_k[:,:,:]=E_kaux[:,:,:,0]
        for i in range(1,size):
            comm.Recv(E_kaux1,ANY_SOURCE)
            E_k[:,:,:] += E_kaux1[:,:,:,0]
    else:
        comm.Send(E_kaux,0)
    E_k = comm.bcast(E_k)

    nall=0
    for n in range(nk1*nk2*nk3):
        for m in range(nawf):
            eall[nall]=E_k[m,n,ispin]
            nall += 1

    return(eall)

def diago(ini_ik,end_ik,aux,saux,read_S,ispin):

    nawf = aux.shape[0]
    nk = aux.shape[2]
    nspin = aux.shape[3]
    ekp = np.zeros((nawf,nk,nspin))

    for n in range(ini_ik,end_ik):
        if read_S:
            eigval,_ = LA.eigh(aux[:,:,n,ispin],saux[:,:,n])
        else:
            eigval,_ = LAN.eigh(aux[:,:,n,ispin],UPLO='U')
        ekp[:,n,ispin] = np.sort(np.real(eigval))

    return(ekp)
