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

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from do_non_ortho import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_momentum(Hksp,dHksp):
    # calculate momentum vector

    nawf = Hksp.shape[0]
    nk1 = Hksp.shape[2]
    nk2 = Hksp.shape[3]
    nk3 = Hksp.shape[4]
    nspin = Hksp.shape[5]

    pks = np.zeros((3,nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)

    for ispin in range(nspin):
        for l in range(3):
            for i in range(nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        eig, vec = LAN.eigh(Hksp[:,:,i,j,k,ispin],UPLO='U')
                        pks[l,:,:,i,j,k,ispin] = np.conj(vec.T).dot(dHksp[l,:,:,i,j,k,ispin]).dot(vec)

    return(pks)
