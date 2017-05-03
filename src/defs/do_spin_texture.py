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
import numpy as np
import cmath
import sys, time

from write2bxsf import *
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from write3Ddatagrid import *
from clebsch_gordan import *
from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_spin_texture(fermi_dw,fermi_up,E_k,vec,sh,nl,nk1,nk2,nk3,nawf,nspin,spin_orbit,npool):
    nktot = nk1*nk2*nk3
    ind_plot = np.zeros(nawf,dtype=int)

    icount = None
    if rank == 0:
        icount = 0
        for ib in range(nawf):
            if ((np.amin(E_k[:,ib]) < fermi_up and np.amax(E_k[:,ib]) > fermi_up) or \
                (np.amin(E_k[:,ib]) < fermi_dw and np.amax(E_k[:,ib]) > fermi_dw) or \
                (np.amin(E_k[:,ib]) > fermi_dw and np.amax(E_k[:,ib]) < fermi_up)):
                ind_plot[icount] = ib
                icount +=1
    icount = comm.bcast(icount,root=0)

    # Compute spin operators
    # Pauli matrices (x,y,z)
    sP=0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
    if spin_orbit:
        # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
        Sj = np.zeros((3,nawf,nawf),dtype=complex)
        for spol in xrange(3):
            for i in xrange(nawf/2):
                Sj[spol,i,i] = sP[spol][0,0]
                Sj[spol,i,i+1] = sP[spol][0,1]
            for i in xrange(nawf/2,nawf):
                Sj[spol,i,i-1] = sP[spol][1,0]
                Sj[spol,i,i] = sP[spol][1,1]
    else:
        # Spin operator matrix  in the basis of |j,m_j,l,s> (full SO)
        Sj = np.zeros((3,nawf,nawf),dtype=complex)
        for spol in xrange(3):
            Sj[spol,:,:] = clebsch_gordan(nawf,sh,nl,spol)

    # Compute matrix elements of the spin operator
    if rank == 0:
        sktxt = np.zeros((nktot,3,nawf,nawf),dtype=complex)
    else:
        sktxt = None

    for pool in xrange(npool):
        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nktot/npool

        if rank == 0:
            sktxt_split = np.array_split(sktxt,npool,axis=0)[pool]
            vec_split = np.array_split(vec,npool,axis=0)[pool]
        else:
            sktxt_split = None
            vec_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        sktxtaux = np.zeros((nsize,3,nawf,nawf),dtype = complex)
        vecaux = np.zeros((nsize,nawf,nawf,nspin),dtype = complex)

        comm.Barrier()
        comm.Scatter(sktxt_split,sktxtaux,root=0)
        comm.Scatter(vec_split,vecaux,root=0)

        for ik in xrange(nsize):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    sktxtaux[ik,l,:,:] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (Sj[l,:,:]).dot(vecaux[ik,:,:,ispin])

        comm.Barrier()
        comm.Gather(sktxtaux,sktxt_split,root=0)

        if rank == 0:
            sktxt[pool*nkpool:(pool+1)*nkpool,:,:,:] = sktxt_split[:,:,:,:]

    if rank == 0:
        sktxt = np.reshape(sktxt,(nk1,nk2,nk3,3,nawf,nawf),order='C')
        for ib in xrange(icount):
            np.savez('spin_text_band_'+str(ib), spinband = sktxt[:,:,:,:,ind_plot[ib],ind_plot[ib]])

    return()
