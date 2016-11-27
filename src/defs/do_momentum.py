#
# AFLOWpi_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
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
import sys, time
import scipy.linalg.lapack as lapack

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_momentum(vec,dHksp):
    # calculate momentum vector

    index = None

    if rank == 0:
        nk1,nk2,nk3,_,nawf,nawf,nspin = dHksp.shape
        index = {'nawf':nawf,'nk1':nk1,'nk2':nk2,'nk3':nk3,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nk1 = index['nk1']
    nk2 = index['nk2']
    nk3 = index['nk3']
    nawf = index['nawf']
    nspin = index['nspin']

    if rank == 0:
        dHksp = np.reshape(dHksp,(nk1*nk2*nk3,3,nawf,nawf,nspin),order='C')
        pks = np.zeros((nk1*nk2*nk3,3,nawf,nawf,nspin),dtype=complex)
    else:
        dHksp = None
        pks = None

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nk1*nk2*nk3)
    nsize = end_ik-ini_ik

    dHkaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
    pksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
    vecaux = np.zeros((nsize,nawf,nawf,nspin),dtype = complex)

    comm.Barrier()
    comm.Scatter(dHksp,dHkaux,root=0)
    comm.Scatter(pks,pksaux,root=0)
    comm.Scatter(vec,vecaux,root=0)

    for ik in range(nsize):
        for ispin in range(nspin):
            for l in range(3):
                pksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                            (dHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

    comm.Barrier()
    comm.Gather(pksaux,pks,root=0)

    pksrep = None
    if rank == 0:
        pks = np.reshape(pks,(nk1,nk2,nk3,3,nawf,nawf,nspin),order='C')
        pksrep = np.zeros((3,nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    pksrep[:,:,:,i,j,k,:] = pks[i,j,k,:,:,:,:]

    return(pksrep)
