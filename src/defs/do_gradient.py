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
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 

from scipy import fftpack as FFT
import numpy as np
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from write_TB_eigs import write_TB_eigs
#from kpnts_interpolation_mesh import *
from get_K_grid_fft  import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_gradient(HRaux,R_wght,R,idx,b_vectors):

    nawf = HRaux.shape[0]
    nk1 = HRaux.shape[2]
    nk2 = HRaux.shape[3]
    nk3 = HRaux.shape[4]
    nspin = HRaux.shape[5]

    kq,_,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)

    Hks_int  = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex) # final data arrays
    Hks_aux  = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin,1),dtype=complex) # read data arrays from tasks
    Hks_aux1  = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin,1),dtype=complex) # receiving data arrays

    # Load balancing
    ini_i = np.zeros((size),dtype=int)
    end_i = np.zeros((size),dtype=int)

    splitsize = 1.0/size*nk1*nk2*nk3
    for i in range(size):
        ini_i[i] = int(round(i*splitsize))
        end_i[i] = int(round((i+1)*splitsize))

    ini_ik = ini_i[rank]
    end_ik = end_i[rank]

    Hks_aux[:,:,:,:,:,0] = grad_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,HRaux,R_wght,kq,R,idx)

    if rank == 0:
        Hks_int[::,:,:,:]=Hks_aux[:,:,:,:,:,0]
        for i in range(1,size):
            comm.Recv(Hks_aux1,ANY_SOURCE)
            Hks_int[:,:,:,:,:] += Hks_aux1[:,:,:,:,:,0]
    else:
        comm.Send(Hks_aux,0)
    Hks_int = comm.bcast(Hks_int)

    dHks = np.zeros((3,nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                dHks[:,:,:,i,j,k,:]=Hks_int[:,:,:,idk[i,j,k],:]

    return(dHks)

def grad_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,HRaux,R_wght,kq,R,idx):

    auxh = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)

    for ik in range(ini_ik,end_ik):
        for ispin in range(nspin):
            for i in range(nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        phase=1.0j*R[idx[i,j,k],:]*R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[ik,:].dot(R[idx[i,j,k],:])*1j)
                        auxh[0,:,:,ik,ispin] += HRaux[:,:,i,j,k,ispin]*phase[0]
                        auxh[1,:,:,ik,ispin] += HRaux[:,:,i,j,k,ispin]*phase[1]
                        auxh[2,:,:,ik,ispin] += HRaux[:,:,i,j,k,ispin]*phase[2]

    return(auxh)

