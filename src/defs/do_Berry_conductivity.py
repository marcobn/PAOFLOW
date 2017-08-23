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
import scipy.integrate

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import load_balancing
from communication import scatter_array

from constants import *
from smearing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_Berry_conductivity(E_k,pksp,temp,ispin,npool,ipol,jpol,shift,deltak,deltak2,smearing):
    # Compute the optical conductivity tensor sigma_xy(ene)

    emin = 0.0 
    emax = shift
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

    sigxy = np.zeros((ene.size),dtype=complex)
    sigxy_sum = np.zeros((ene.size),dtype=complex)

    for pool in xrange(npool):

        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nktot/npool

        if rank == 0:
            pksp_long = np.array_split(pksp,npool,axis=0)[pool]
            E_k_long= np.array_split(E_k,npool,axis=0)[pool]
            deltak_long= np.array_split(deltak,npool,axis=0)[pool]
            deltak2_long= np.array_split(deltak2,npool,axis=0)[pool]
        else:
            pksp_long = None
            E_k_long = None
            deltak_long= None
            deltak2_long= None

        comm.Barrier()
        pkspaux = scatter_array(pksp_long)
        E_kaux = scatter_array(E_k_long)
        deltakaux = scatter_array(deltak_long)
        deltak2aux = scatter_array(deltak2_long)

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        sigxy_aux = np.zeros((ene.size),dtype=complex)

        if smearing != None:
            sigxy_aux[:] = smear_sigma_loop2(ini_ik,end_ik,ene,E_kaux,pkspaux,nawf,temp,ispin,ipol,jpol,smearing,deltakaux,deltak2aux)
        else:
            sigxy_aux[:] = sigma_loop(ini_ik,end_ik,ene,E_kaux,pkspaux,nawf,temp,ispin,ipol,jpol,smearing,deltakaux,deltak2aux)

        comm.Allreduce(sigxy_aux,sigxy_sum,op=MPI.SUM)
        sigxy += sigxy_sum

    sigxy /= float(nktot)

    return(ene,sigxy)

def sigma_loop(ini_ik,end_ik,ene,E_k,pksp,nawf,temp,ispin,ipol,jpol,smearing,deltak,deltak2):

    sigxy = np.zeros((ene.size),dtype=complex)
    func = np.zeros((end_ik-ini_ik,ene.size),dtype=complex)
    delta = 0.05
    Ef = 0.0

    # Collapsing the sum over k points
    for n in xrange(nawf):
        if smearing == None:
            fn = 1.0/(np.exp(E_k[:,n,ispin]/temp)+1)
        elif smearing == 'gauss':
            fn = intgaussian(E_k[:,n,0],Ef,deltak[:,n,0])
        elif smearing == 'm-p':
            fn = intmetpax(E_k[:,n,0],Ef,deltak[:,n,0])
        for m in xrange(nawf):
            if smearing == None:
                fm = 1.0/(np.exp(E_k[:,m,ispin]/temp)+1)
            elif smearing == 'gauss':
                fm = intgaussian(E_k[:,m,0],Ef,deltak[:,m,0])
            elif smearing == 'm-p':
                fm = intmetpax(E_k[:,m,0],Ef,deltak[:,m,0])
            func[:,:] = ((E_k[:,n,ispin]-E_k[:,m,ispin])**2*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T - (ene+1.0j*delta)**2
            sigxy[:] += np.sum(((1.0/func * \
                        ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                        np.imag(pksp[:,jpol,n,m,0]*pksp[:,ipol,m,n,0])
                        ),axis=1)

    return(sigxy)

def smear_sigma_loop2(ini_ik,end_ik,ene,E_k,pksp,nawf,temp,ispin,ipol,jpol,smearing,deltak,deltak2):

    sigxy = np.zeros((ene.size),dtype=complex)
    func = np.zeros((end_ik-ini_ik,ene.size),dtype=complex)
    delta = 0.05
    Ef = 0.0

    # Collapsing the sum over k points
    for n in xrange(nawf):
        if smearing == None:
            fn = 1.0/(np.exp(E_k[:,n,ispin]/temp)+1)
        elif smearing == 'gauss':
            fn = intgaussian(E_k[:,n,0],Ef,deltak[:,n,0])
        elif smearing == 'm-p':
            fn = intmetpax(E_k[:,n,0],Ef,deltak[:,n,0])
        for m in xrange(nawf):
            if smearing == None:
                fm = 1.0/(np.exp(E_k[:,m,ispin]/temp)+1)
            elif smearing == 'gauss':
                fm = intgaussian(E_k[:,m,0],Ef,deltak[:,m,0])
            elif smearing == 'm-p':
                fm = intmetpax(E_k[:,m,0],Ef,deltak[:,m,0])
            if m != n:
                func[:,:] = ((E_k[:,n,ispin]-E_k[:,m,ispin])**2*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T - \
                            (ene+1.0j*(deltak2[:,n,m,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T)**2
                sigxy[:] += np.sum(((1.0/func * \
                            ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                            np.imag(pksp[:,jpol,n,m,0]*pksp[:,ipol,m,n,0])
                            ),axis=1)

    return(sigxy)

