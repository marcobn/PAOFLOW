# 
# PAOFLOW
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
import os, sys
import scipy.linalg.lapack as lapack

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from clebsch_gordan import *

from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_spin_current(vec,dHksp,spol,npool,spin_orbit,sh,nl):
    # calculate spin_current operator

    index = None

    if rank == 0:
        nktot,_,nawf,nawf,nspin = dHksp.shape
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    # Compute spin current matrix elements
    # Pauli matrices (x,y,z)
    sP=0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
    if spin_orbit:
        # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
        Sj = np.zeros((nawf,nawf),dtype=complex)
        for i in xrange(nawf/2):
            Sj[i,i] = sP[spol][0,0]
            Sj[i,i+1] = sP[spol][0,1]
        for i in xrange(nawf/2,nawf):
            Sj[i,i-1] = sP[spol][1,0]
            Sj[i,i] = sP[spol][1,1]
    else:
        # Spin operator matrix  in the basis of |j,m_j,l,s> (full SO)
        Sj = clebsch_gordan(nawf,sh,nl,spol)

    if rank == 0:
        jksp = np.zeros((nktot,3,nawf,nawf,nspin),dtype=complex)
        jdHksp = np.zeros((nktot,3,nawf,nawf,nspin),dtype=complex)
    else:
        dHksp = None
        jdHksp = None
        jksp = None

    for pool in xrange(npool):
        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nktot/npool

        if rank == 0:
            dHksp_split = np.array_split(dHksp,npool,axis=0)[pool]
            jdHksp_split = np.array_split(jdHksp,npool,axis=0)[pool]
            vec_split = np.array_split(vec,npool,axis=0)[pool]
        else:
            dHksp_split = None
            jdHksp_split = None
            vec_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        comm.Barrier()
        dHkaux = scatter_array(dHksp_split, (nktot,3,nawf,nawf,nspin), complex, 0)
        jdHkaux = scatter_array(jdHksp_split, (nktot,3,nawf,nawf,nspin), complex, 0)
        vecaux = scatter_array(vec_split, (nktot,nawf,nawf,nspin), complex, 0)

        for ik in xrange(nsize):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    jdHkaux[ik,l,:,:,ispin] = \
                        0.5*(np.dot(Sj,dHkaux[ik,l,:,:,ispin])+ \
                        np.dot(dHkaux[ik,l,:,:,ispin],Sj))

        comm.Barrier()
        gather_array(jdHksp_split, jdHkaux, complex, 0)

        if rank == 0:
            jdHksp[pool*nkpool:(pool+1)*nkpool,:,:,:,:] = jdHksp_split[:,:,:,:,:,]

    comm.Barrier()

    for pool in xrange(npool):
        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nktot/npool

        if rank == 0:
            jdHksp_split = np.array_split(jdHksp,npool,axis=0)[pool]
            jks_split = np.array_split(jksp,npool,axis=0)[pool]
            vec_split = np.array_split(vec,npool,axis=0)[pool]
        else:
            jdHksp_split = None
            jks_split = None
            vec_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        comm.Barrier()
        jdHkaux = scatter_array(jdHksp_split, (nktot,3,nawf,nawf,nspin), complex, 0)
        jksaux = scatter_array(jks_split, (nktot,3,nawf,nawf,nspin), complex, 0)
        vecaux = scatter_array(vec_split, (nktot,nawf,nawf,nspin), complex, 0)

        for ik in xrange(nsize):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    jksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (jdHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

        comm.Barrier()
        gather_array(jks_split, jksaux, complex, 0)

        if rank == 0:
            jksp[pool*nkpool:(pool+1)*nkpool,:,:,:,:] = jks_split[:,:,:,:,:]

    return(jksp)
