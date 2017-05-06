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
import os, sys
import scipy.linalg.lapack as lapack

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from Gatherv_Scatterv_wrappers import *
from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_momentum(vec,dHksp):
    # calculate momentum vector
    index = None
    if rank == 0:
        nktot,_,nawf,nawf,nspin = dHksp.shape
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin,}
    index = comm.bcast(index,root=0)
    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    if rank==0:
        #get number of them for load balancing
        num_entries=nktot
    else:
        Hksp=None
        vec=None
    
    
    #scatter dHksp and eigenvecs by k
    dHkaux = Scatterv_wrap(dHksp)
    vecaux = Scatterv_wrap(vec)

    pksaux = np.zeros_like(dHkaux,order='C')
    #precompute complex conjugate transpose of eigenvec
    vecaux_conj = np.zeros_like(vecaux,order='C')
    vecaux_conj = np.transpose(np.conj(vecaux),axes=(0,2,1,3))
    #perform dot products
    for ispin in xrange(nspin):
        for l in xrange(3):
            for ik in xrange(pksaux.shape[0]):
                pksaux[ik,l,:,:,ispin] = vecaux_conj[ik,:,:,ispin]\
                                         .dot(dHkaux[ik,l,:,:,ispin])\
                                         .dot(vecaux[ik,:,:,ispin])
                                         

    dHkaux      = None
    vecaux      = None
    vecaux_conj = None

    #gather pksp
    pksp = Gatherv_wrap(pksaux)
    pksaux = None

    if rank==0:
        return pksp

