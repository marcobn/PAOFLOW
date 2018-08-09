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
import os, sys
import scipy.linalg.lapack as lapack

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from clebsch_gordan import *

from load_balancing import *
from communication import *
from do_perturb_split import * 

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_spin_current(vec,dHksp,spol,npool,spin_orbit,sh,nl,bnd,degen):
    # calculate spin_current operator
    _,_,nawf,nawf,nspin = dHksp.shape

    # Compute spin current matrix elements
    # Pauli matrices (x,y,z)
    sP=0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
    if spin_orbit:
        # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
        Sj = np.zeros((nawf,nawf),dtype=complex)
        for i in range(nawf/2):
            Sj[i,i] = sP[spol][0,0]
            Sj[i,i+1] = sP[spol][0,1]
        for i in range(nawf/2,nawf):
            Sj[i,i-1] = sP[spol][1,0]
            Sj[i,i] = sP[spol][1,1]
    else:
        # Spin operator matrix  in the basis of |j,m_j,l,s> (full SO)
        Sj = clebsch_gordan(nawf,sh,nl,spol)

    jdHksp = np.zeros((dHksp.shape[0],3,nawf,nawf,nspin),dtype=complex)

    for ik in range(dHksp.shape[0]):
        for l in range(3):
            for ispin in range(nspin):
                jdHksp[ik,l,:,:,ispin] = \
                    0.5*(np.dot(Sj,dHksp[ik,l,:,:,ispin])+np.dot(dHksp[ik,l,:,:,ispin],Sj))
                

    jksp = np.zeros((dHksp.shape[0],3,nawf,nawf,nspin),dtype=complex)

    for ik in range(dHksp.shape[0]):
        for l in range(3):            
            for ispin in range(nspin): 
                jksp[ik,l,:,:,ispin] = np.conj(vec[ik,:,:,ispin].T).dot \
                            (jdHksp[ik,l,:,:,ispin]).dot(vec[ik,:,:,ispin])

    for ik in range(dHksp.shape[0]):
        for l in range(3):            
            for ispin in range(nspin):
                jksp[ik,l,:,:,ispin] = do_perturb_split(jksp[ik,l,:,:,ispin],
                                                        jdHksp[ik,l,:,:,ispin],
                                                        vec[ik,:,:,ispin],
                                                        degen[ispin][ik])


    jdHksp = None

    comm.Barrier()

    return(jksp)
