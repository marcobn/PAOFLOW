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
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

#import matplotlib.pyplot as plt

from kpnts_interpolation_mesh import *
#from new_kpoint_interpolation import *
from do_non_ortho import *
from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_eigh_calc(HRaux,SRaux,kq,R_wght,R,idx,read_S):
    # Compute bands on a selected mesh in the BZ

    nkpi=kq.shape[0]

    nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape
    Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex) # final data arrays

    Hks_int[:,:,:,:] = band_loop_H(nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx)

    Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
    if read_S:
        Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
        Sks_int[:,:,:] = band_loop_S(nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx)

    E_kp = np.zeros((nkpi,nawf,nspin),dtype=float)
    v_kp = np.zeros((nkpi,nawf,nawf,nspin),dtype=complex)

    for ispin in range(nspin):
        for ik in range(nkpi):
            if read_S:
                E_kp[ik,:,ispin],v_kp[ik,:,:,ispin] = LA.eigh(Hks_int[:,:,ik,ispin],Sks_int[:,:,ik])
            else:
                E_kp[ik,:,ispin],v_kp[ik,:,:,ispin] = LAN.eigh(Hks_int[:,:,ik,ispin],UPLO='U')


#    if rank == 0:
#        plt.matshow(abs(Hks_int[:,:,1445,0]))
#        plt.colorbar()
#        plt.show()
#
#        np.save('Hks_noSO0',Hks_int[:,:,0,0])

    return(E_kp,v_kp)

def band_loop_H(nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx):

    auxh = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)
    HRaux = np.reshape(HRaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')

    for ik in range(nkpi):
        for ispin in range(nspin):
             auxh[:,:,ik,ispin] = np.sum(HRaux[:,:,:,ispin]*np.exp(2.0*np.pi*kq[ik,:].dot(R[:,:].T)*1j),axis=2)

    return(auxh)

def band_loop_S(nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx):

    auxs = np.zeros((nawf,nawf,nkpi),dtype=complex)

    for ik in range(nkpi):
        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[ik,:].dot(R[idx[i,j,k],:])*1j)
                    auxs[:,:,ik] += SRaux[:,:,i,j,k]*phase

    return(auxs)
