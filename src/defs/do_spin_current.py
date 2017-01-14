#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
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

import numpy as np
import cmath
import os, sys
import scipy.linalg.lapack as lapack

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from clebsch_gordan import *

from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_spin_current(vec,dHksp,spol,npool,spin_orbit):
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
        Sj = clebsch_gordan()

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
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        dHkaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        jdHkaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        vecaux = np.zeros((nsize,nawf,nawf,nspin),dtype = complex)

        comm.Barrier()
        comm.Scatter(dHksp_split,dHkaux,root=0)
        comm.Scatter(jdHksp_split,jdHkaux,root=0)
        comm.Scatter(vec_split,vecaux,root=0)

        for ik in xrange(nsize):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    jdHkaux[ik,l,:,:,ispin] = \
                        0.5*(np.dot(Sj,dHkaux[ik,l,:,:,ispin])+ \
                        np.dot(dHkaux[ik,l,:,:,ispin],Sj))

        comm.Barrier()
        comm.Gather(jdHkaux,jdHksp_split,root=0)

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
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        jdHkaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        jksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        vecaux = np.zeros((nsize,nawf,nawf,nspin),dtype = complex)

        comm.Barrier()
        comm.Scatter(jdHksp_split,jdHkaux,root=0)
        comm.Scatter(jks_split,jksaux,root=0)
        comm.Scatter(vec_split,vecaux,root=0)

        for ik in xrange(nsize):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    jksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (jdHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

        comm.Barrier()
        comm.Gather(jksaux,jks_split,root=0)

        if rank == 0:
            jksp[pool*nkpool:(pool+1)*nkpool,:,:,:,:] = jks_split[:,:,:,:,:]

    return(jksp)
