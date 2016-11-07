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
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from write_TB_eigs import write_TB_eigs
#from kpnts_interpolation_mesh import *
from kpnts_interpolation_mesh import *
from do_non_ortho import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_bands_calc(HRaux,SRaux,R_wght,R,idx,read_S,ibrav,alat,a_vectors,b_vectors,dkres):
    # Compute bands on a selected path in the BZ
    # Define k-point mesh for bands interpolation
    kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
    nkpi=kq.shape[1]
    for n in range(nkpi):
        kq [:,n]=kq[:,n].dot(b_vectors)

    nawf = HRaux.shape[0]
    nk1 = HRaux.shape[2]
    nk2 = HRaux.shape[3]
    nk3 = HRaux.shape[4]
    nspin = HRaux.shape[5]
    Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex) # final data arrays
    Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
    Hks_aux  = np.zeros((nawf,nawf,nkpi,nspin,1),dtype=complex) # read data arrays from tasks
    Sks_aux  = np.zeros((nawf,nawf,nkpi,1),dtype=complex)
    Hks_aux1  = np.zeros((nawf,nawf,nkpi,nspin,1),dtype=complex) # receiving data arrays
    Sks_aux1  = np.zeros((nawf,nawf,nkpi,1),dtype=complex)

    # Load balancing
    ini_i = np.zeros((size),dtype=int)
    end_i = np.zeros((size),dtype=int)

    splitsize = 1.0/size*nkpi
    for i in range(size):
        ini_i[i] = int(round(i*splitsize))
        end_i[i] = int(round((i+1)*splitsize))

    ini_ik = ini_i[rank]
    end_ik = end_i[rank]

    Hks_aux[:,:,:,:,0] = band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx)

    if rank == 0:
        Hks_int[:,:,:,:]=Hks_aux[:,:,:,:,0]
        for i in range(1,size):
            comm.Recv(Hks_aux1,ANY_SOURCE)
            Hks_int[:,:,:,:] += Hks_aux1[:,:,:,:,0]
    else:
        comm.Send(Hks_aux,0)
    Hks_int = comm.bcast(Hks_int)

    if read_S:
        Sks_aux[:,:,:,0] = band_loop_S(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx)

        if rank == 0:
            Sks_int[:,:,:]=Sks_aux[:,:,:,0]
            for i in range(1,size):
                comm.Recv(Sks_aux1,ANY_SOURCE)
                Sks_int[:,:,:] += Sks_aux1[:,:,:,0]
        else:
            comm.Send(Sks_aux,0)
        Sks_int = comm.bcast(Sks_int)
        Hks_int = do_non_ortho(Hks_int,Sks_int)

    if rank ==0:
        for ispin in range(nspin):
            write_TB_eigs(Hks_int,Sks_int,read_S,ispin)
    return()

def band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx):

    auxh = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)

    for ik in range(ini_ik,end_ik):
        for ispin in range(nspin):
            for i in range(nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                        auxh[:,:,ik,ispin] += HRaux[:,:,i,j,k,ispin]*phase

    return(auxh)

def band_loop_S(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx):

    auxs = np.zeros((nawf,nawf,nkpi),dtype=complex)

    for ik in range(ini_ik,end_ik):
        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                    auxs[:,:,ik] += SRaux[:,:,i,j,k]*phase

    return(auxs)
