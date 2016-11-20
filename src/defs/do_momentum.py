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

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_momentum(vec,dHksp):
    # calculate momentum vector

    nawf = dHksp.shape[1]
    nk1 = dHksp.shape[3]
    nk2 = dHksp.shape[4]
    nk3 = dHksp.shape[5]
    nspin = dHksp.shape[6]

    pks = np.zeros((3,nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)

    for ispin in range(nspin):

        for l in range(3):
            for i in range(nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        n = k + j*nk3 + i*nk2*nk3
                        pks[l,:,:,i,j,k,ispin] = np.conj(vec[:,:,n,ispin].T).dot(dHksp[l,:,:,i,j,k,ispin]).dot(vec[:,:,n,ispin])

    return(pks)
