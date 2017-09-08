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

def do_epsilon(E_k,pksp,tksp,kq_wght,omega,shift,delta,temp,ipol,jpol,ispin,metal,ne,emin,emax,bnd,deltak,deltak2,smearing,kramerskronig):
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
    jdos = np.zeros((ene.size),dtype=float)
    epsi_aux = np.zeros((3,3,ene.size),dtype=float)
    jdos_aux = np.zeros((ene.size),dtype=float)

    if smearing == None:
        epsi_aux[:,:,:],jdos_aux[:] = epsi_loop(ipol,jpol,ini_ik,end_ik,ene,E_kaux,pkspaux,kq_wghtaux,bnd,omega,delta,temp,ispin,metal)
    else:
        epsi_aux[:,:,:],jdos_aux[:] = smear_epsi_loop(ipol,jpol,ini_ik,end_ik,ene,E_kaux,pkspaux,kq_wghtaux,bnd,omega,delta,temp,\
                          ispin,metal,deltakaux,deltak2aux,smearing)

    comm.Allreduce(epsi_aux,epsi,op=MPI.SUM)
    comm.Allreduce(jdos_aux,jdos,op=MPI.SUM)

    #=======================
    # Re
    #=======================

    if kramerskronig == True:
        # Load balancing
        ini_ie, end_ie = load_balancing(size,rank,ene.size)

        epsr = np.zeros((3,3,ene.size),dtype=float)
        epsr_aux = np.zeros((3,3,ene.size),dtype=float)

        epsr_aux[:,:,:] = epsr_kramkron(ini_ie,end_ie,ene,epsi,shift,ipol,jpol)

        comm.Allreduce(epsr_aux,epsr,op=MPI.SUM)

    else:
        if metal: print('CAUTION: direct calculation of epsr in metals is not working!!!!!')

        epsr = np.zeros((3,3,ene.size),dtype=float)
        epsr_aux = np.zeros((3,3,ene.size),dtype=float)

        if smearing == None:
            sys.exit('fixed smearing not implemented')
        else:
            epsr_aux[:,:,:] = smear_epsr_loop(ipol,jpol,ini_ik,end_ik,ene,E_kaux,pkspaux,tkspaux,kq_wghtaux,bnd,omega,delta,temp,\
                              ispin,metal,deltakaux,deltak2aux,smearing)

        comm.Allreduce(epsr_aux,epsr,op=MPI.SUM)

    epsr += 1.0

    return(ene,epsi,epsr,jdos)

def epsi_loop(ipol,jpol,ini_ik,end_ik,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal):

    epsi = np.zeros((3,3,ene.size),dtype=float)
    jdos = np.zeros((ene.size),dtype=float)

    dfunc = np.zeros((end_ik-ini_ik,ene.size),dtype=float)

    for n in xrange(nawf):
        fn = 1.0/(np.exp(E_k[:,n,ispin]/temp)+1)
        try:
            fnF = 1.0/2.0 * 1.0/(1.0+np.cosh(E_k[:,n,ispin]/temp))
        except:
            fnF = 1.0e8*np.ones(end_ik-ini_ik,dtype=float)
        for m in xrange(nawf):
            fm = 1.0/(np.exp(E_k[:,m,ispin]/temp)+1)
            dfunc[:,:] = 1.0/np.sqrt(np.pi)* \
                np.exp(-((((E_k[:,n,ispin]-E_k[:,m,ispin])*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T + ene)/delta)**2)
            epsi[ipol,jpol,:] += np.sum(((1.0/(ene**2+delta**2) * \
                           kq_wght[0] /delta * dfunc * ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                           abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)
            jdos[:] += np.sum((( \
                           kq_wght[0] /delta * dfunc * ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                           1.0),axis=1)
            if metal and n == m:
                epsi[ipol,jpol,:] += np.sum(((1.0/ene * \
                               kq_wght[0] /delta * dfunc * ((fnF/temp)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                               abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)

    return(epsi,jdos)

def smear_epsr_loop(ipol,jpol,ini_ik,end_ik,ene,E_k,pksp,tksp,kq_wght,nawf,omega,delta,temp,ispin,metal,deltak,deltak2,smearing):

    epsr = np.zeros((3,3,ene.size),dtype=float)

    dfunc = np.zeros((end_ik-ini_ik,ene.size),dtype=float)
    effterm = np.zeros((end_ik-ini_ik,nawf),dtype=complex)
    Ef = 0.0

    for n in xrange(nawf):
        if smearing == 'gauss':
            fn = intgaussian(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            fnF = gaussian(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
        elif smearing == 'm-p':
            fn = intmetpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            fnF = metpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
        else:
            sys.exit('smearing not implemented')
        for m in xrange(nawf):
            if smearing == 'gauss':
                fm = intgaussian(E_k[:,m,ispin],Ef,deltak[:,m,ispin])
            elif smearing == 'm-p':
                fm = intmetpax(E_k[:,m,ispin],Ef,deltak[:,m,ispin])
            else:
                sys.exit('smearing not implemented')
            eig = ((E_k[:,m,ispin]-E_k[:,n,ispin])*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
            om = ((ene*np.ones((end_ik-ini_ik,ene.size),dtype=float)).T).T
            del2 = (deltak2[:,n,m,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
            dfunc = 1.0/(eig - om + 1.0j*del2)
            if n != m:
                epsr[ipol,jpol,:] += np.real(np.sum((1.0/(eig**2+1.0j*delta**2) * \
                               kq_wght[0] * dfunc * ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                               (pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin]),axis=1))
                if metal:
                    effterm[:,n] += 1.0/(E_k[:,m,ispin]-E_k[:,n,ispin] + 1.0*deltak2[:,m,n,ispin]) * \
                                    (pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin] + pksp[:,jpol,n,m,ispin] * pksp[:,ipol,m,n,ispin])
            if metal and n == m:
                epsr[ipol,jpol,:] += np.real(np.sum((1.0/(ene**2+1.0j*delta**2) * \
                               kq_wght[0] * dfunc * (fnF*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                               (pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin]),axis=1))
    if metal:
        sum_rule = 0.0
        for n in xrange(nawf):
            if smearing == 'gauss':
                fn = intgaussian(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            elif smearing == 'm-p':
                fn = intmetpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            else:
                sys.exit('smearing not implemented')
            epsr[ipol,jpol,:] += np.real(np.sum((-1.0/(ene**2+1.0j*delta**2) * \
                           kq_wght[0] * (fn*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                           (tksp[:,ipol,jpol,n,n,ispin]),axis=1)) + \
                           np.real(np.sum((1.0/(ene**2+1.0j*delta**2) * \
                           kq_wght[0] * (fn*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                           (effterm[:,n]),axis=1))
            sum_rule += np.sum(fn*tksp[:,ipol,jpol,n,n,ispin])
        if rank == 0: print('f-sum rule = ',sum_rule)

    epsr *= 4.0/(EPS0 * EVTORY * omega)

    return(epsr)

def smear_epsi_loop(ipol,jpol,ini_ik,end_ik,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal,deltak,deltak2,smearing):

    epsi = np.zeros((3,3,ene.size),dtype=float)
    jdos = np.zeros((ene.size),dtype=float)

    dfunc = np.zeros((end_ik-ini_ik,ene.size),dtype=float)
    Ef = 0.0
    deltat = 0.1

    for n in xrange(nawf):
        if smearing == 'gauss':
            fn = intgaussian(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            if metal: fnF = gaussian(E_k[:,n,ispin],Ef,0.03*deltak[:,n,ispin])
        elif smearing == 'm-p':
            fn = intmetpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            if metal: fnF = metpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
        else:
            sys.exit('smearing not implemented')
        for m in xrange(nawf):
            if smearing == 'gauss':
                fm = intgaussian(E_k[:,m,ispin],Ef,deltak[:,m,ispin])
            elif smearing == 'm-p':
                fm = intmetpax(E_k[:,m,ispin],Ef,deltak[:,m,ispin])
            else:
                sys.exit('smearing not implemented')
            if m > n:
                eig = ((E_k[:,m,ispin]-E_k[:,n,ispin])*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                om = ((ene*np.ones((end_ik-ini_ik,ene.size),dtype=float)).T).T
                del2 = (deltak2[:,m,n,ispin]*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T
                if smearing == 'gauss':
                    dfunc[:,:] = gaussian(eig,om,del2)
                elif smearing == 'm-p':
                    dfunc[:,:] = metpax(eig,om,del2)
                else:
                    sys.exit('smearing not implemented')
                epsi[ipol,jpol,:] += np.sum(((1.0/(ene**2+delta**2) * \
                               kq_wght[0] * dfunc * ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                               np.real(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)
                jdos[:] += np.sum(((\
                           kq_wght[0] * dfunc * ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                           1.0 ),axis=1)
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
                epsi[ipol,jpol,:] += np.sum((1.0/ene * \
                               kq_wght[0] * dfunc * (fnF*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T * \
                               np.real(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin]),axis=1)

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)

    return(epsi,jdos)

def epsr_kramkron(ini_ie,end_ie,ene,epsi,shift,i,j):

    epsr = np.zeros((3,3,ene.size),dtype=float)
    de = ene[1]-ene[0]

    if ini_ie == 0: ini_ie = 3
    if end_ie == ene.size: end_ie = ene.size-1
    f_ene = intmetpax(ene,shift,1.0)
    for ie in xrange(ini_ie,end_ie):
        #epsr[i,j,ie] = 2.0/np.pi * ( np.sum(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
        #               np.sum(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )
        epsr[i,j,ie] = 2.0/np.pi * ( tgr.simps(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]*f_ene[1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
                       tgr.simps(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]*f_ene[(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )

    return(epsr)

