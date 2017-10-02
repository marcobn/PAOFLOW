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

    nktot,_,nawf,_,nspin = pksp.shape

    sigxy = np.zeros((ene.size),dtype=complex)
    sigxy_aux = np.zeros((ene.size),dtype=complex)


    if smearing != None:
        sigxy_aux[:] = smear_sigma_loop2(ene,E_k,pksp,nawf,temp,ispin,ipol,jpol,smearing,deltak,deltak2)
    else:
        sigxy_aux[:] = sigma_loop(ene,E_k,pksp,nawf,temp,ispin,ipol,jpol,smearing,deltak,deltak2)

    comm.Allreduce(sigxy_aux,sigxy,op=MPI.SUM)

    nk_tot = np.array([nktot],dtype=int)
    nktot = np.zeros((1),dtype=int)
    comm.Allreduce(nk_tot,nktot)

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

def smear_sigma_loop2(ene,E_k,pksp,nawf,temp,ispin,ipol,jpol,smearing,deltak,deltak2):

    sigxy = np.zeros((ene.size),dtype=complex)
    func = np.zeros((pksp.shape[0],ene.size),dtype=complex)
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
                func[:,:] = ((E_k[:,n,ispin]-E_k[:,m,ispin])**2*np.ones((pksp.shape[0],ene.size),dtype=float).T).T - \
                            (ene+1.0j*(deltak2[:,n,m,ispin]*np.ones((pksp.shape[0],ene.size),dtype=float).T).T)**2
                sigxy[:] += np.sum(((1.0/func * \
                            ((fn - fm)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T* \
                            np.imag(pksp[:,jpol,n,m,0]*pksp[:,ipol,m,n,0])
                            ),axis=1)

    return(sigxy)

