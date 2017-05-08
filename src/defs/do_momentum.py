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
        pksp = np.zeros((nktot,3,nawf,nawf,nspin),dtype=complex)
    else:
        dHksp = None
        pksp = None

    for pool in xrange(npool):
        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh - do_momentum')
        nkpool = nktot/npool

        if rank == 0:
            dHksp_split = np.array_split(dHksp,npool,axis=0)[pool]
            pks_split = np.array_split(pksp,npool,axis=0)[pool]
            vec_split = np.array_split(vec,npool,axis=0)[pool]
        else:
            dHksp_split = None
            pks_split = None
            vec_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        dHkaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        pksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        vecaux = np.zeros((nsize,nawf,nawf,nspin),dtype = complex)

        comm.Barrier()
        comm.Scatter(dHksp_split,dHkaux,root=0)
        comm.Scatter(pks_split,pksaux,root=0)
        comm.Scatter(vec_split,vecaux,root=0)

        for ik in xrange(nsize):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    pksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (dHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

        comm.Barrier()
        comm.Gather(pksaux,pks_split,root=0)

        if rank == 0:
            pksp[pool*nkpool:(pool+1)*nkpool,:,:,:,:] = pks_split[:,:,:,:,:,]

    return(pksp)
