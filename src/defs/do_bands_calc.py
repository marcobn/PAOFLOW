#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital basis (PAO)
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
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from write_TB_eigs import *
#from kpnts_interpolation_mesh import *
from kpnts_interpolation_mesh import *
from do_non_ortho import *
from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_bands_calc(HRaux,SRaux,R_wght,R,idx,read_S,ibrav,alat,a_vectors,b_vectors,dkres):
    # Compute bands on a selected path in the BZ
    # Define k-point mesh for bands interpolation
    kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
    nkpi=kq.shape[1]
    for n in xrange(nkpi):
        kq [:,n]=kq[:,n].dot(b_vectors)

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nkpi)

    nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape
    Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex) # final data arrays
    Hks_aux  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex) # read data arrays from tasks

    Hks_aux[:,:,:,:] = band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx)

    comm.Allreduce(Hks_aux,Hks_int,op=MPI.SUM)

    Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
    if read_S:
        Sks_aux  = np.zeros((nawf,nawf,nkpi,1),dtype=complex)
        Sks_aux[:,:,:,0] = band_loop_S(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx)

        comm.Allreduce(Sks_aux,Sks_int,op=MPI.SUM)

    E_kp = np.zeros((nkpi,nawf,nspin),dtype=float)
    v_kp = np.zeros((nkpi,nawf,nawf,nspin),dtype=complex)

    for ispin in xrange(nspin):
        E_kp[:,:,ispin],v_kp[:,:,:,ispin] = write_TB_eigs(Hks_int,Sks_int,read_S,ispin)

    return(E_kp,v_kp)

def band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx):

    auxh = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)
    HRaux = np.reshape(HRaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')

    for ik in xrange(ini_ik,end_ik):
        for ispin in xrange(nspin):
             auxh[:,:,ik,ispin] = np.sum(HRaux[:,:,:,ispin]*np.exp(2.0*np.pi*kq[:,ik].dot(R[:,:].T)*1j),axis=2)

    return(auxh)

def band_loop_S(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx):

    auxs = np.zeros((nawf,nawf,nkpi),dtype=complex)

    for ik in xrange(ini_ik,end_ik):
        for i in xrange(nk1):
            for j in xrange(nk2):
                for k in xrange(nk3):
                    phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                    auxs[:,:,ik] += SRaux[:,:,i,j,k]*phase

    return(auxs)
