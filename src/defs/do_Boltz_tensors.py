# 
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
import cmath
from math import cosh
import sys, time

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *
from communication import scatter_array

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

    L0 = np.zeros((3,3,ene.size),dtype=float)
    L0aux = np.zeros((3,3,ene.size),dtype=float)

    if smearing == None:
        t_tensor = np.array([[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]],dtype=int)

    L0aux[:,:,:] = L_loop(ene,E_k,velkp,kq_wght,temp,ispin,0,deltak,smearing,t_tensor)

    comm.Reduce(L0aux,L0,op=MPI.SUM)

    if smearing == None:
        L1 = np.zeros((3,3,ene.size),dtype=float)
        L1aux = np.zeros((3,3,ene.size),dtype=float)

        L1aux[:,:,:] = L_loop(ene,E_k,velkp,kq_wght,temp,ispin,1,deltak,smearing,t_tensor)

        comm.Reduce(L1aux,L1,op=MPI.SUM)

        L2 = np.zeros((3,3,ene.size),dtype=float)
        L2aux = np.zeros((3,3,ene.size),dtype=float)

        L2aux[:,:,:] = L_loop(ene,E_k,velkp,kq_wght,temp,ispin,2,deltak,smearing,t_tensor)

        comm.Reduce(L2aux,L2,op=MPI.SUM)

        L0[1,0] = L0[0,1]
        L0[2,0] = L0[2,0]
        L0[2,1] = L0[1,2]

        L1[1,0] = L1[0,1]
        L1[2,0] = L1[2,0]
        L1[2,1] = L1[1,2]

        L2[1,0] = L2[0,1]
        L2[2,0] = L2[2,0]
        L2[2,1] = L2[1,2]

        return(ene,L0,L1,L2)
    else:
        return(ene,L0)

def L_loop(ene,E_k,velkp,kq_wght,temp,ispin,alpha,deltak,smearing,t_tensor):
    # We assume tau=1 in the constant relaxation time approximation

    L = np.zeros((3,3,ene.size),dtype=float)



    if smearing == None:
        for n in xrange(velkp.shape[2]):
            Eaux = (E_k[:,n,ispin]*np.ones((E_k.shape[0],ene.size),dtype=float).T).T - ene
            for l in xrange(t_tensor.shape[0]):
                i = t_tensor[l][0]
                j = t_tensor[l][1]
                if smearing == None:
                    L[i,j,:] += np.sum((1.0/temp * kq_wght[0]*velkp[:,i,n,ispin]*velkp[:,j,n,ispin] * \
                                1.0/2.0 * (1.0/(1.0+0.5*(np.exp(Eaux[:,:]/temp)+np.exp(-Eaux[:,:]/temp))) * np.power(Eaux[:,:],alpha)).T),axis=1)


    if smearing == 'gauss':
        om = ((ene*np.ones((E_k.shape[0],ene.size),dtype=float)).T).T
        for n in xrange(velkp.shape[2]):
            eig = (E_k[:,n,ispin]*np.ones((E_k.shape[0],ene.size),dtype=float).T).T
            delk = (deltak[:,n,ispin]*np.ones((E_k.shape[0],ene.size),dtype=float).T).T
            for l in xrange(t_tensor.shape[0]):
                i = t_tensor[l][0]
                j = t_tensor[l][1]
                L[i,j,:] += np.sum((kq_wght[0]*velkp[:,i,n,ispin]*velkp[:,j,n,ispin] * \
                                        (gaussian(eig,om,delk) * np.power(eig-om,alpha)).T),axis=1)


    if smearing == 'm-p': 
        om = ((ene*np.ones((E_k.shape[0],ene.size),dtype=float)).T).T
        for n in xrange(velkp.shape[2]):
            eig = (E_k[:,n,ispin]*np.ones((E_k.shape[0],ene.size),dtype=float).T).T
            delk = (deltak[:,n,ispin]*np.ones((E_k.shape[0],ene.size),dtype=float).T).T
            for l in xrange(t_tensor.shape[0]):
                i = t_tensor[l][0]
                j = t_tensor[l][1]
                L[i,j,:] += np.sum((kq_wght[0]*velkp[:,i,n,ispin]*velkp[:,j,n,ispin] * \
                                        (metpax(eig,om,delk) * np.power(eig-om,alpha)).T),axis=1)
                

    if smearing != None and smearing != 'gauss' and smearing != 'm-p':
        sys.exit('smearing not implemented')

    return(L)
