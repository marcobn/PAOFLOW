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

from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_momentum(vec,dHksp,npool):
#  try:
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
        pksp = np.zeros((nktot,3,nawf,nawf,nspin),dtype=complex)
    else:
        dHksp = None
        pksp = None

    for pool in xrange(npool):

        # Load balancing
        ini_ik, end_ik = load_balancing(npool,pool,nktot)
        
        if rank==0:
          dHkaux = scatter_array(dHksp[ini_ik:end_ik])
        else:
          dHkaux = scatter_array(None)
        comm.Barrier()

        if rank==0:
          vecaux = scatter_array(vec[ini_ik:end_ik])
        else:
          vecaux = scatter_array(None)
        comm.Barrier()

        ####################
        ### dummy dHkaux ###
        ####################
        for ik in xrange(dHkaux.shape[0]):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    dHkaux[ik,l,:,:,ispin] = dHkaux[ik,l,:,:,ispin].dot(vecaux[ik,:,:,ispin])

        ####################
        ### dummy vecaux ###
        ####################                
        vecaux = np.ascontiguousarray(np.conj(np.swapaxes(vecaux,1,2)))

        for ik in xrange(dHkaux.shape[0]):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    dHkaux[ik,l,:,:,ispin] = vecaux[ik,:,:,ispin].dot(dHkaux[ik,l,:,:,ispin])


        if rank == 0:
            gather_array(pksp[ini_ik:end_ik], dHkaux)
        else:
            gather_array(None, dHkaux)

        comm.Barrier()
        vec_dagger=None
        pks_split=None
        vecaux=None
        pksaux=None
        dHkaux=None

    return(pksp)
#  except Exception as e:
#    raise e
