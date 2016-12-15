#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital basis (PAO)
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

    index = None

    if rank == 0:
        nktot,_,nawf,_,nspin = pksp.shape
        index = {'nktot':nktot,'nawf':nawf,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nktot)
    nsize = end_ik-ini_ik

    kq_wghtaux = np.zeros(nsize,dtype=float)
    pkspaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype=complex)
    E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)

    comm.Barrier()
    comm.Scatter(pksp,pkspaux,root=0)
    comm.Scatter(E_k,E_kaux,root=0)
    comm.Scatter(kq_wght,kq_wghtaux,root=0)

    #=======================
    # Im
    #=======================
    epsi = np.zeros((3,3,ene.size),dtype=float)
    epsi_aux = np.zeros((3,3,ene.size),dtype=float)

    epsi_aux[:,:,:] = epsi_loop(ini_ik,end_ik,ene,E_kaux,pkspaux,kq_wghtaux,nawf,omega,delta,temp,ispin)

    comm.Allreduce(epsi_aux,epsi,op=MPI.SUM)

    #=======================
    # Re
    #=======================

    # Load balancing
    ini_ie, end_ie = load_balancing(size,rank,ene.size)

    epsr = np.zeros((3,3,ene.size),dtype=float)
    epsr_aux = np.zeros((3,3,ene.size,1),dtype=float)

    epsr_aux[:,:,:,0] = epsr_kramkron(ini_ie,end_ie,ene,epsi)

    comm.Allreduce(epsr_aux,epsr,op=MPI.SUM)

    epsr += 1.0

    return(ene,epsi,epsr)

def epsi_loop(ini_ik,end_ik,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin):

    epsi = np.zeros((3,3,ene.size),dtype=float)

    dfunc = np.zeros((end_ik-ini_ik,ene.size),dtype=float)

    # Collapsing the sum over k points - FASTEST
    for n in xrange(nawf):
        fn = 1.0/(np.exp(E_k[:,n,ispin]/temp)+1)
        for m in xrange(nawf):
            fm = 1.0/(np.exp(E_k[:,m,ispin]/temp)+1)
            dfunc[:,:] = 1.0/np.sqrt(np.pi)* \
                np.exp(-((((E_k[:,n,ispin]-E_k[:,m,ispin])*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T + ene)/delta)**2)
            for j in xrange(3):
                for i in xrange(3):
                    epsi[i,j,:] += np.sum(((1.0/(ene**2+delta**2) * \
                                   kq_wght[0] /delta * dfunc * ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                                   abs(pksp[:,i,n,m,ispin] * pksp[:,j,m,n,ispin])),axis=1)

#   # Collapsing the sums over n,m and k points - MUCH SLOWER!!!
#   fermi = (1.0/(np.exp(E_k[:,:,ispin]/temp)+1))*np.transpose(np.ones((end_ik-ini_ik,nawf,nawf),dtype=float),axes=(1,0,2))
#   dfermi = fermi - np.transpose(fermi,axes=(2,1,0))
#   eigen = E_k[:,:,ispin]*np.transpose(np.ones((end_ik-ini_ik,nawf,nawf),dtype=float),axes=(1,0,2))
#   deigen = eigen - np.transpose(eigen,axes=(2,1,0))
#   dfunc = 1.0/np.sqrt(np.pi)* \
#       np.exp(-(((deigen*np.ones((nawf,end_ik-ini_ik,nawf,ene.size),dtype=float).T).T + ene)/delta)**2)
#   for j in xrange(3):
#       for i in xrange(3):
#           epsi[i,j,:] = np.sum(((1.0/(ene**2+delta**2) * \
#                          kq_wght[0] /delta * dfunc * (dfermi*np.ones((nawf,end_ik-ini_ik,nawf,ene.size),dtype=float).T).T).T* \
#                          np.transpose(abs(pksp[:,i,:,:,ispin] * np.transpose(pksp[:,j,:,:,ispin],axes=(0,2,1))),axes=(1,0,2))),axis=(1,2,3))

#   # Old way to do loops - AS IN WanT, SLOOOOOOW
#   for nk in xrange(end_ik-ini_ik):
#       for n in xrange(nawf):
#           arg2 = E_k[nk,n,ispin]/temp
#           raux2 = 1.0/(np.exp(arg2)+1)
#           for m in xrange(nawf):
#               arg3 = E_k[nk,m,ispin]/temp
#               raux3 = 1.0/(np.exp(arg3)+1)
#               arg[:] = (ene[:] - ((E_k[nk,m,ispin]-E_k[nk,n,ispin])))/delta
#               raux[:] = 1.0/np.sqrt(np.pi)*np.exp(-arg[:]**2)
#               for j in xrange(3):
#                   for i in xrange(3):
#                       epsi[i,j,:] += 1.0/(ene[:]**2+delta**2) * \
#                               kq_wght[nk] /delta * raux[:] * (raux2 - raux3) * \
#                               abs(pksp[nk,i,n,m,ispin] * pksp[nk,j,m,n,ispin])

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)

    return(epsi)

def epsr_kramkron(ini_ie,end_ie,ene,epsi):

    epsr = np.zeros((3,3,ene.size),dtype=float)
    de = ene[1]-ene[0]

    for ie in xrange(ini_ie,end_ie):
        for i in xrange(3):
            for j in xrange(3):
                epsr[i,j,ie] = 2.0/np.pi * ( np.sum(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
                               np.sum(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )

    return(epsr)

