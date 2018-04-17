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

from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_momentum(vec,dHksp,d2Hksp,npool):
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
        tksp = np.zeros((nktot,3,3,nawf,nawf,nspin),dtype=complex)
    else:
        dHksp = None
        pksp = None
        tksp = None

    for pool in range(npool):
        ini_ip, end_ip = load_balancing(npool, pool, nktot)
        nkpool = end_ip - ini_ip 

        if rank == 0:
            dHksp_split = dHksp[ini_ip:end_ip]
            d2Hksp_split = d2Hksp[ini_ip:end_ip]
            pks_split = pksp[ini_ip:end_ip]
            tks_split = tksp[ini_ip:end_ip]
            vec_split = vec[ini_ip:end_ip]
        else:
            dHksp_split = None
            d2Hksp_split = None
            pks_split = None
            tks_split = None
            vec_split = None

        dHkaux = scatter_array(dHksp_split)
        d2Hkaux = scatter_array(d2Hksp_split)
        pksaux = scatter_array(pks_split)
        tksaux = scatter_array(tks_split)
        vecaux = scatter_array(vec_split)

        for ik in range(nsize):
            for ispin in range(nspin):
                for l in range(3):
                    pksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (dHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])
                    for lp in range(3):
                        tksaux[ik,l,lp,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                    (d2Hkaux[ik,l,lp,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

        gather_array(pks_split, pksaux)
        gather_array(tks_split, tksaux)

        if rank == 0:
            pksp[ini_ip:end_ip,:,:,:,:] = pks_split[:,:,:,:,:,]
            tksp[ini_ip:end_ip,:,:,:,:,:] = tks_split[:,:,:,:,:,:]

    return(pksp,tksp)
