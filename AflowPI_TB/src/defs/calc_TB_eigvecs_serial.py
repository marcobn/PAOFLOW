#
# AflowPI_TB
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

def calc_TB_eigvecs(Hks,Sks,read_S):

    nawf,nawf,nk1,nk2,nk3,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nbnds_tb,nk1*nk2*nk3,nspin))
    V_k_unsorted = np.zeros((nawf,nawf,nk1*nk2*nk3,nspin),dtype=np.complex_)
    E_k_unsorted = np.zeros((nbnds_tb,nk1*nk2*nk3,nspin))
    eall = np.zeros((nbnds_tb*nk1*nk2*nk3))
    vall = np.zeros((nbnds_tb*nk1*nk2*nk3,nbnds_tb),dtype=np.complex_)

    ispin = 0 #plots only 1 spin channel
    #for ispin in range(nspin):
    nk=0
    for ik1 in range(nk1):
        for ik2 in range(nk2):
            for ik3 in range(nk3):
                if read_S:
                    eigval,eigvec = LA.eigh(Hks[:,:,ik1,ik2,ik3,ispin],Sks[:,:,ik1,ik2,ik3])
                else:
                    eigval,eigvec = LAN.eigh(Hks[:,:,ik1,ik2,ik3,ispin],UPLO='U')
                E_k[:,nk,ispin] = np.sort(np.real(eigval))
                E_k_unsorted[:,nk,ispin] = np.real(eigval)
                V_k_unsorted[:,:,nk,ispin] = eigvec
                nk += 1
    nall=0
    for n in range(nk):
        for m in range(nawf):
            eall[nall]=E_k_unsorted[m,n,ispin]
            vall[nall,:]=V_k_unsorted[:,m,n,ispin]
            nall += 1

    return(eall,vall,nall,nawf) 

