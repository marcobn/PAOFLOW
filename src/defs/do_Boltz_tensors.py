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
from math import cosh
import sys, time

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin):
    # Compute the L_alpha tensors for Boltzmann transport

    emin = -2.0 # To be read in input
    emax = 2.0
    de = (emax-emin)/500
    ene = np.arange(emin,emax,de,dtype=float)

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,velkp.shape[2])

    L0 = np.zeros((3,3,ene.size),dtype=float)
    L0aux = np.zeros((3,3,ene.size,1),dtype=float)
    L0aux1 = np.zeros((3,3,ene.size,1),dtype=float)

    L0aux[:,:,:,0] = L_loop(ini_ik,end_ik,ene,E_k,velkp,kq_wght,temp,ispin,0)

    if rank == 0:
        L0[:,:,:]=L0aux[:,:,:,0]
        for i in range(1,size):
            comm.Recv(L0aux1,ANY_SOURCE)
            L0[:,:,:] += L0aux1[:,:,:,0]
    else:
        comm.Send(L0aux,0)
    L0 = comm.bcast(L0)

    L1 = np.zeros((3,3,ene.size),dtype=float)
    L1aux = np.zeros((3,3,ene.size,1),dtype=float)
    L1aux1 = np.zeros((3,3,ene.size,1),dtype=float)

    L1aux[:,:,:,0] = L_loop(ini_ik,end_ik,ene,E_k,velkp,kq_wght,temp,ispin,1)

    if rank == 0:
        L1[:,:,:]=L1aux[:,:,:,0]
        for i in range(1,size):
            comm.Recv(L1aux1,ANY_SOURCE)
            L1[:,:,:] += L1aux1[:,:,:,0]
    else:
        comm.Send(L1aux,0)
    L1 = comm.bcast(L1)

    L2 = np.zeros((3,3,ene.size),dtype=float)
    L2aux = np.zeros((3,3,ene.size,1),dtype=float)
    L2aux1 = np.zeros((3,3,ene.size,1),dtype=float)

    L2aux[:,:,:,0] = L_loop(ini_ik,end_ik,ene,E_k,velkp,kq_wght,temp,ispin,2)

    if rank == 0:
        L2[:,:,:]=L2aux[:,:,:,0]
        for i in range(1,size):
            comm.Recv(L2aux1,ANY_SOURCE)
            L2[:,:,:] += L2aux1[:,:,:,0]
    else:
        comm.Send(L2aux,0)
    L2 = comm.bcast(L2)

    return(ene,L0,L1,L2)

def L_loop(ini_ik,end_ik,ene,E_k,velkp,kq_wght,temp,ispin,alpha):

    # We assume tau=1 in the constant relaxation time approximation

    L = np.zeros((3,3,ene.size),dtype=float)

    for nk in range(ini_ik,end_ik):
        for n in range(velkp.shape[1]):
            for i in range(3):
                for j in range(3):
                    L[i,j,:] += 1.0/temp * kq_wght[nk]*velkp[i,n,nk,ispin]*velkp[j,n,nk,ispin] * \
                    1.0/2.0 * 1.0/(1.0+np.cosh((E_k[n,nk,ispin]-ene[:])/temp)) * pow((E_k[n,nk,ispin]-ene[:]),alpha)

    return(L)
