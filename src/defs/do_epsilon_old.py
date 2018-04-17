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
import scipy.integrate as tgr

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

def do_epsilon(E_k,pksp,kq_wght,omega,delta,temp,ipol,jpol,ispin,metal,ne,emin,emax,deltak,deltak2,smearing):
    # Compute the dielectric tensor

    de = (emax-emin)/float(ne)
    ene = np.arange(emin,emax,de,dtype=float)
    if ene[0]==0.0: ene[0]=0.00001

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

    comm.Barrier()
    pkspaux = scatter_array(pksp)
    E_kaux = scatter_array(E_k)
    kq_wghtaux = scatter_array(kq_wght)
    if smearing != None:
        deltakaux = scatter_array(deltak)
        deltak2aux = scatter_array(deltak2)

    #=======================
    # Im
    #=======================
    epsi = np.zeros((3,3,ene.size),dtype=float)
    epsi_aux = np.zeros((3,3,ene.size),dtype=float)

    if smearing == None:
        epsi_aux[:,:,:] = epsi_loop(ipol,jpol,ini_ik,end_ik,ene,E_kaux,pkspaux,kq_wghtaux,nawf,omega,delta,temp,ispin,metal)
    else:
        epsi_aux[:,:,:] = smear_epsi_loop(ipol,jpol,ini_ik,end_ik,ene,E_kaux,pkspaux,kq_wghtaux,nawf,omega,delta,temp,ispin,metal,deltakaux,deltak2aux,smearing)

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

def epsi_loop(ipol,jpol,ini_ik,end_ik,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal):

    epsi = np.zeros((3,3,ene.size),dtype=float)

    dfunc = np.zeros((end_ik-ini_ik,ene.size),dtype=float)

    for n in range(nawf):
        fn = 1.0/(np.exp(E_k[:,n,ispin]/temp)+1)
        fnF = 1.0/2.0 * 1.0/(1.0+np.cosh(E_k[:,n,ispin]/temp))
        for m in range(nawf):
            fm = 1.0/(np.exp(E_k[:,m,ispin]/temp)+1)
            dfunc[:,:] = 1.0/np.sqrt(np.pi)* \
            np.exp(-((((E_k[:,n,ispin]-E_k[:,m,ispin])*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T + ene)/delta)**2)
            epsi[ipol,jpol,:] += np.sum(((1.0/(ene**2+delta**2) * \
                           kq_wght[0] /delta * dfunc * ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                           abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)
            if metal and n == m:
                epsi[ipol,jpol,:] += np.sum(((1.0/ene * \
                               kq_wght[0] /delta * dfunc * ((fnF/temp)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                               abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)

    return(epsi)

def smear_epsi_loop(ipol,jpol,ini_ik,end_ik,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal,deltak,deltak2,smearing):

    epsi = np.zeros((3,3,ene.size),dtype=float)

    dfunc = np.zeros((end_ik-ini_ik,ene.size),dtype=float)
    Ef = 0.0

    for n in range(nawf):
        if smearing == 'gauss':
            fn = intgaussian(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            fnF = gaussian(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
        elif smearing == 'm-p':
            fn = intmetpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            fnF = metpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
        else:
            sys.exit('smearing not implemented')
        for m in range(nawf):
            if smearing == 'gauss':
                fm = intgaussian(E_k[:,m,ispin],Ef,deltak[:,n,ispin])
            elif smearing == 'm-p':
                fm = intmetpax(E_k[:,m,ispin],Ef,deltak[:,n,ispin])
            else:
                sys.exit('smearing not implemented')
            if m != n:
                eig = ((E_k[:,m,ispin]-E_k[:,n,ispin])*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                om = ((ene*np.ones((end_ik-ini_ik,ene.size),dtype=float)).T).T
                # the factor afac is an adjustment of the factor in the adaptive smearing: afac > 1 improves convergence in metals
                if metal:
                    afac = 2.2
                else:
                    afac = 1.0
                del2 = (afac*deltak2[:,n,m,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                if smearing == 'gauss':
                    dfunc[:,:] = gaussian(eig,om,del2)
                elif smearing == 'm-p':
                    dfunc[:,:] = metpax(eig,om,del2)
                else:
                    sys.exit('smearing not implemented')
                epsi[ipol,jpol,:] += np.sum(((1.0/(ene**2+delta**2) * \
                               kq_wght[0] * dfunc * ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                               abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)
            if metal and n == m:
                eig = (np.zeros((end_ik-ini_ik,ene.size),dtype=float).T).T
                om = ((ene*np.ones((end_ik-ini_ik,ene.size),dtype=float)).T).T
                del2 = (deltak[:,n,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                if smearing == 'gauss':
                    dfunc[:,:] = gaussian(eig,om,del2)
                elif smearing == 'm-p':
                    dfunc[:,:] = metpax(eig,om,del2)
                else:
                    sys.exit('smearing not implemented')
                epsi[ipol,jpol,:] += np.sum(((1.0/ene * \
                               kq_wght[0] * dfunc * (fnF*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                               abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)

    return(epsi)

def epsr_kramkron(ini_ie,end_ie,ene,epsi):

    epsr = np.zeros((3,3,ene.size),dtype=float)
    de = ene[1]-ene[0]

    if ini_ie == 0: ini_ie = 3
    if end_ie == ene.size: end_ie = ene.size-1
    for ie in range(ini_ie,end_ie):
        for i in range(3):
            for j in range(3):
                #epsr[i,j,ie] = 2.0/np.pi * ( np.sum(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
                #               np.sum(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )
                epsr[i,j,ie] = 2.0/np.pi * ( tgr.simps(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
                               tgr.simps(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )

    return(epsr)

