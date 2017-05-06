#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
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

def do_momentum(vec,dHksp,npool):
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

    #perform dot products
    for ispin in xrange(nspin):
        for l in xrange(3):
            for ik in xrange(pksaux.shape[0]):
                pksaux[ik,l,:,:,ispin] = (np.conj(vecaux[ik,:,:,ispin]).T)\
                                         .dot(dHkaux[ik,l,:,:,ispin])\
                                         .dot(vecaux[ik,:,:,ispin])
                                         
    dHkaux      = None
    vecaux      = None

    #gather pksp
    pksp = Gatherv_wrap(pksaux)
    pksaux = None

    if rank==0:
        return pksp

