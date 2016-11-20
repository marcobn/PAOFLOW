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
import scipy.integrate

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import load_balancing

from calc_TB_eigs import calc_TB_eigs
from constants import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_epsilon(E_k,pksp,kq_wght,omega,delta,temp,ispin):
    # Compute the dielectric tensor

    emin = 0.1 # To be read in input
    emax = 10.0
    de = (emax-emin)/500
    ene = np.arange(emin,emax,de,dtype=float)

    #=======================
    # Im
    #=======================

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,kq_wght.size)

    epsi = np.zeros((3,3,ene.size),dtype=float)
    epsi_aux = np.zeros((3,3,ene.size,1),dtype=float)
    epsi_aux1 = np.zeros((3,3,ene.size,1),dtype=float)

    epsi_aux[:,:,:,0] = epsi_loop(ini_ik,end_ik,ene,E_k,pksp,kq_wght,omega,delta,temp,ispin)

    if rank == 0:
        epsi[:,:,:]=epsi_aux[:,:,:,0]
        for i in range(1,size):
            comm.Recv(epsi_aux1,ANY_SOURCE)
            epsi[:,:,:] += epsi_aux1[:,:,:,0]
    else:
        comm.Send(epsi_aux,0)
    epsi = comm.bcast(epsi)

    #=======================
    # Re
    #=======================

    # Load balancing
    ini_ie, end_ie = load_balancing(size,rank,ene.size)

    epsr = np.zeros((3,3,ene.size),dtype=float)
    epsr_aux = np.zeros((3,3,ene.size,1),dtype=float)
    epsr_aux1 = np.zeros((3,3,ene.size,1),dtype=float)

    epsr_aux[:,:,:,0] = epsr_kramkron(ini_ie,end_ie,ene,epsi)

    if rank == 0:
        epsr[:,:,:]=epsr_aux[:,:,:,0]
        for i in range(1,size):
            comm.Recv(epsr_aux1,ANY_SOURCE)
            epsr[:,:,:] += epsr_aux1[:,:,:,0]
    else:
        comm.Send(epsr_aux,0)
    epsr = comm.bcast(epsr)

    epsr += 1.0

    return(ene,epsi,epsr)

def epsi_loop(ini_ik,end_ik,ene,E_k,pksp,kq_wght,omega,delta,temp,ispin):

    epsi = np.zeros((3,3,ene.size),dtype=float)

    arg = np.zeros((ene.size),dtype=float)
    raux = np.zeros((ene.size),dtype=float)

    for nk in range(ini_ik,end_ik):
        for n in range(pksp.shape[1]):
            arg2 = E_k[n,nk,ispin]/temp
            raux2 = 1.0/(np.exp(arg2)+1)
            for m in range(pksp.shape[1]):
                arg3 = E_k[m,nk,ispin]/temp
                raux3 = 1.0/(np.exp(arg3)+1)
                arg[:] = (ene[:] - ((E_k[m,nk,ispin]-E_k[n,nk,ispin])))/delta
                raux[:] = 1.0/np.sqrt(np.pi)*np.exp(-arg[:]**2)
                if n != m:
                    for i in range(3):
                        for j in range(3):
                            epsi[i,j,:] += 1.0/(ene[:]**2+delta**2) * \
                                    kq_wght[nk] /delta * raux[:] * (raux2 - raux3) * \
                                    abs(pksp[i,n,m,nk,ispin] * pksp[j,m,n,nk,ispin])
                else:
                    for i in range(3):
                        for j in range(3):
                            epsi[i,j,:] += 1.0/ene[:] * kq_wght[nk] * raux[:]/delta *  \
                                    1.0/2.0 * 1.0/(1.0+np.cosh((arg2)))/temp *    \
                                    abs(pksp[i,n,m,nk,ispin] * pksp[j,m,n,nk,ispin])

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)

    return(epsi)

def epsr_kramkron(ini_ie,end_ie,ene,epsi):

    epsr = np.zeros((3,3,ene.size),dtype=float)
    de = ene[1]-ene[0]

    for ie in range(ini_ie,end_ie):
        for i in range(3):
            for j in range(3):
                epsr[i,j,ie] = 2.0/np.pi * ( np.sum(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
                               np.sum(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )

    return(epsr)

