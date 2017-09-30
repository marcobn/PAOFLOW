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

    # calculate momentum vector
    nktot,_,nawf,nawf,nspin = dHksp.shape

    pksp = np.zeros_like(dHksp)



    for ik in xrange(dHksp.shape[0]):
        for ispin in xrange(nspin):
            for l in xrange(3):
                pksp[ik,l,:,:,ispin] = dHksp[ik,l,:,:,ispin].dot(vec[ik,:,:,ispin])


    vec_cross = np.ascontiguousarray(np.conj(np.swapaxes(vec,1,2)))

    for ik in xrange(dHksp.shape[0]):
        for ispin in xrange(nspin):
            for l in xrange(3):
                pksp[ik,l,:,:,ispin] = vec_cross[ik,:,:,ispin].dot(pksp[ik,l,:,:,ispin])

    comm.Barrier()


    return(pksp)
#  except Exception as e:
#    raise e
