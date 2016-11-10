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

def do_gradient(Hks_long,R_wght,R,b_vectors,nk1,nk2,nk3,alat):

    nawf = Hks_long.shape[0]
    nktot = Hks_long.shape[2]
    nspin = Hks_long.shape[3]

    nrtot = R_wght.size

    kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)

    # ---------------------------------
    # Compute HRs on the regular R grid
    # ---------------------------------

    HRs  = np.zeros((nawf,nawf,nrtot,nspin),dtype=complex)
    HRsaux  = np.zeros((nawf,nawf,nrtot,nspin,1),dtype=complex)
    HRsaux1 = np.zeros((nawf,nawf,nrtot,nspin,1),dtype=complex)

    # Load balancing
    ini_r = np.zeros((size),dtype=int)
    end_r = np.zeros((size),dtype=int)

    splitsize = 1.0/size*nrtot
    for i in range(size):
        ini_r[i] = int(round(i*splitsize))
        end_r[i] = int(round((i+1)*splitsize))

    ini_nr = ini_r[rank]
    end_nr = end_r[rank]

    HRsaux[:,:,:,:,0] = HR_regular_loop(ini_nr,end_nr,Hks_long,kq_wght,kq,R,nspin)

    if rank == 0:
        HRs[:,:,:,:]=HRsaux[:,:,:,:,0]
        for i in range(1,size):
            comm.Recv(HRsaux1,ANY_SOURCE)
            HRs[:,:,:,:] += HRsaux1[:,:,:,:,0]
    else:
        comm.Send(HRsaux,0)
    HRs = comm.bcast(HRs)

    # ---------------------------------
    # Compute the gradient of Hks
    # ---------------------------------

    dHksi  = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex) # final data arrays
    dHksaux  = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin,1),dtype=complex) # read data arrays from tasks
    dHksaux1  = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin,1),dtype=complex) # receiving data arrays

    # Load balancing
    ini_i = np.zeros((size),dtype=int)
    end_i = np.zeros((size),dtype=int)

    splitsize = 1.0/size*nk1*nk2*nk3
    for i in range(size):
        ini_i[i] = int(round(i*splitsize))
        end_i[i] = int(round((i+1)*splitsize))

    ini_ik = ini_i[rank]
    end_ik = end_i[rank]

    dHksaux[:,:,:,:,:,0] = grad_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,HRs,R_wght,kq,R,alat)

    if rank == 0:
        dHksi[:,:,:,:]=dHksaux[:,:,:,:,:,0]
        for i in range(1,size):
            comm.Recv(dHksaux1,ANY_SOURCE)
            dHksi[:,:,:,:,:] += dHksaux1[:,:,:,:,:,0]
    else:
        comm.Send(dHksaux,0)
    dHksi = comm.bcast(dHksi)

    dHks = np.zeros((3,nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                dHks[:,:,:,i,j,k,:]=dHksi[:,:,:,idk[i,j,k],:]

    return(dHks)

def grad_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,HRs,R_wght,kq,R,alat):

    auxh = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
    phase = np.zeros((3),dtype=complex)
    nrtot = R_wght.size

    for ik in range(ini_ik,end_ik):
        for ispin in range(nspin):
            for nr in range(nrtot):
                phase=1.0j*R[nr,:]*alat*R_wght[nr]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[nr,:])*1j)
                auxh[0,:,:,ik,ispin] += HRs[:,:,nr,ispin]*phase[0]
                auxh[1,:,:,ik,ispin] += HRs[:,:,nr,ispin]*phase[1]
                auxh[2,:,:,ik,ispin] += HRs[:,:,nr,ispin]*phase[2]

    return(auxh)

def HR_regular_loop(ini_nr,end_nr,Hks_long,kq_wght,kq,R,nspin):

    nawf = Hks_long.shape[0]
    nrtot = R.shape[0]
    HRs  = np.zeros((nawf,nawf,nrtot,nspin),dtype=complex)

    for nr in range(ini_nr,end_nr):
        for ik in range(kq_wght.size):
            for ispin in range(nspin):
                HRs[:,:,nr,ispin] += Hks_long[:,:,ik,ispin]*kq_wght[ik]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[nr,:])*(-1j))

    return(HRs) 
