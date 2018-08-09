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
import scipy.linalg as LAN

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
np.set_printoptions(suppress=True,linewidth=140,precision=7)
def do_momentum(vec,dHksp):

    # calculate momentum vector
    nktot,_,nawf,nawf,nspin = dHksp.shape

    pksp = np.zeros_like(dHksp)

    for ik in range(dHksp.shape[0]):
        for ispin in range(nspin):
            for l in range(3):
                pksp[ik,l,:,:,ispin] = np.dot(np.conj(vec[ik,:,:,ispin].T),
                                              np.dot(dHksp[ik,l,:,:,ispin],
                                                     vec[ik,:,:,ispin]))

    comm.Barrier()

    return(pksp)
#  except Exception as e:
#    raise e
