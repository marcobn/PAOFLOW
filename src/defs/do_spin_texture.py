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
import numpy as np
import cmath
import sys, time
import os
from write2bxsf import *
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from write3Ddatagrid import *
from clebsch_gordan import *
from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_spin_texture(fermi_dw,fermi_up,E_k,vec,sh,nl,nk1,nk2,nk3,nawf,nspin,spin_orbit,npool,inputpath):
    nktot = nk1*nk2*nk3
    ind_plot = np.zeros(nawf,dtype=int)

    E_k_full = gather_full(E_k,npool)


    icount = None
    if rank == 0:
        icount = 0
        for ib in range(nawf):
            if ((np.amin(E_k_full[:,ib]) < fermi_up and np.amax(E_k_full[:,ib]) > fermi_up) or \
                (np.amin(E_k_full[:,ib]) < fermi_dw and np.amax(E_k_full[:,ib]) > fermi_dw) or \
                (np.amin(E_k_full[:,ib]) > fermi_dw and np.amax(E_k_full[:,ib]) < fermi_up)):
                ind_plot[icount] = ib
                icount +=1

    E_k_full = None

    icount = comm.bcast(icount)



    # Compute spin operators
    # Pauli matrices (x,y,z)
    sP=0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
    if spin_orbit:
        # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
        Sj = np.zeros((3,nawf,nawf),dtype=complex)
        for spol in range(3):
            for i in range(nawf/2):
                Sj[spol,i,i] = sP[spol][0,0]
                Sj[spol,i,i+1] = sP[spol][0,1]
            for i in range(nawf/2,nawf):
                Sj[spol,i,i-1] = sP[spol][1,0]
                Sj[spol,i,i] = sP[spol][1,1]
    else:
        # Spin operator matrix  in the basis of |j,m_j,l,s> (full SO)
        Sj = np.zeros((3,nawf,nawf),dtype=complex)
        for spol in range(3):
            Sj[spol,:,:] = clebsch_gordan(nawf,sh,nl,spol)

    # Compute matrix elements of the spin operator




    sktxtaux = np.zeros((vec.shape[0],3,nawf,nawf),dtype=complex)

    for ik in range(vec.shape[0]):
        for ispin in range(nspin):
            for l in range(3):
                sktxtaux[ik,l,:,:] = np.conj(vec[ik,:,:,ispin].T).dot \
                            (Sj[l,:,:]).dot(vec[ik,:,:,ispin])

    sktxt = gather_full(sktxtaux,npool)
    sktxtaux = None

    if rank == 0:
        sktxt = np.reshape(sktxt,(nk1,nk2,nk3,3,nawf,nawf),order='C')

        for ib in range(icount):
            np.savez(os.path.join(inputpath,'spin_text_band_'+str(ib)), spinband = sktxt[:,:,:,:,ind_plot[ib],ind_plot[ib]])

    sktxt = None

    return()
