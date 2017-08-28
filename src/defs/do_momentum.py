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
from collections import deque

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_momentum(vec,dHksp,npool):
    # calculate momentum vector

    index = None

    if rank == 0:
        nktot,_,nawf,nawf,nspin = dHksp.shape
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    if rank == 0:
        dHksp_split = deque(np.array_split(dHksp,npool,axis=0))
        vec_split   = deque(np.array_split(vec,npool,axis=0))

        ini_ik = 0
        end_ik = 0
        
        pksp = np.zeros((nktot,3,nawf,nawf,nspin),order="C",dtype=complex)

    dHksp = None
    vec   = None



    for pool in xrange(npool):

        # Load balancing
        if rank==0:
            nentry = dHksp_split[0].shape[0]
            dHkaux = scatter_array(dHksp_split.popleft())
        else:
            dHkaux = scatter_array(None)

        if rank==0:
            vecaux = scatter_array(vec_split.popleft())
        else:
            vecaux = scatter_array(None)
        
        pksaux = np.zeros((vecaux.shape[0],3,nawf,nawf,nspin),order="C",dtype=complex)

        for ik in xrange(pksaux.shape[0]):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    pksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (dHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

        comm.Barrier()
        if rank==0:
            pks_split = np.zeros((nentry,3,nawf,nawf,nspin),order="C",dtype=complex)
            gather_array(pks_split, pksaux)

            end_ik += nentry
            pksp[ini_ik:end_ik]  = np.copy(pks_split)
            ini_ik += nentry

            pks_split=None
        else:
            gather_array(None, pksaux)

        vecaux=None
        dHkaux=None
        pksaux=None

    if rank == 0:
        return pksp
