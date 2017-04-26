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
from math import cosh
import sys, time

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *
from smearing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin,deltak,smearing,t_tensor):
    # Compute the L_alpha tensors for Boltzmann transport

    emin = -2.0 # To be read in input
    emax = 2.0
    de = (emax-emin)/500
    ene = np.arange(emin,emax,de,dtype=float)

    index = None

    if rank == 0:
        nktot,_,nawf,nspin = velkp.shape
        index = {'nktot':nktot,'nawf':nawf,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nktot)
    nsize = end_ik-ini_ik

    kq_wghtaux = np.zeros(nsize,dtype=float)
    velkpaux = np.zeros((nsize,3,nawf,nspin),dtype=float)
    E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)
    deltakaux = np.zeros((nsize,nawf,nspin),dtype = float)


    comm.Barrier()
    comm.Scatter(velkp,velkpaux,root=0)
    comm.Scatter(E_k,E_kaux,root=0)
    comm.Scatter(kq_wght,kq_wghtaux,root=0)
    if smearing != None:
        comm.Scatter(deltak,deltakaux,root=0)

    L0 = np.zeros((3,3,ene.size),dtype=float)
    L0aux = np.zeros((3,3,ene.size),dtype=float)

    L0aux[:,:,:] = L_loop(ini_ik,end_ik,ene,E_kaux,velkpaux,kq_wghtaux,temp,ispin,0,deltakaux,smearing,t_tensor)

    comm.Allreduce(L0aux,L0,op=MPI.SUM)

    if smearing == None:

        L1 = np.zeros((3,3,ene.size),dtype=float)
        L1aux = np.zeros((3,3,ene.size),dtype=float)

        L1aux[:,:,:] = L_loop(ini_ik,end_ik,ene,E_kaux,velkpaux,kq_wghtaux,temp,ispin,1,deltakaux,smearing,t_tensor)

        comm.Allreduce(L1aux,L1,op=MPI.SUM)

        L2 = np.zeros((3,3,ene.size),dtype=float)
        L2aux = np.zeros((3,3,ene.size),dtype=float)

        L2aux[:,:,:] = L_loop(ini_ik,end_ik,ene,E_kaux,velkpaux,kq_wghtaux,temp,ispin,2,deltakaux,smearing,t_tensor)

        comm.Allreduce(L2aux,L2,op=MPI.SUM)

        return(ene,L0,L1,L2)

    else:

        return(ene,L0)

def L_loop(ini_ik,end_ik,ene,E_k,velkp,kq_wght,temp,ispin,alpha,deltak,smearing,t_tensor):

    # We assume tau=1 in the constant relaxation time approximation

    L = np.zeros((3,3,ene.size),dtype=float)

    for n in xrange(velkp.shape[2]):
        Eaux = (E_k[:,n,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T - ene
        for l in xrange(t_tensor.shape[0]):
            i = t_tensor[l][0]
            j = t_tensor[l][1]
            if smearing == None:
                L[i,j,:] += np.sum((1.0/temp * kq_wght[0]*velkp[:,i,n,ispin]*velkp[:,j,n,ispin] * \
                            1.0/2.0 * (1.0/(1.0+np.cosh(Eaux[:,:]/temp)) * np.power(Eaux[:,:],alpha)).T),axis=1)
            if smearing == 'gauss':
                eig = (E_k[:,n,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                om = ((ene*np.ones((end_ik-ini_ik,ene.size),dtype=float)).T).T
                delk = (deltak[:,n,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                L[i,j,:] += np.sum((kq_wght[0]*velkp[:,i,n,ispin]*velkp[:,j,n,ispin] * \
                            (gaussian(eig,om,delk) * np.power(eig-om,alpha)).T),axis=1)

            if smearing == 'm-p': 
                eig = (E_k[:,n,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                om = ((ene*np.ones((end_ik-ini_ik,ene.size),dtype=float)).T).T
                delk = (deltak[:,n,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                L[i,j,:] += np.sum((kq_wght[0]*velkp[:,i,n,ispin]*velkp[:,j,n,ispin] * \
                                (metpax(eig,om,delk) * np.power(eig-om,alpha)).T),axis=1)
            if smearing != None and smearing != 'gauss' and smearing != 'm-p':
                sys.exit('smearing not implemented')

    return(L)
