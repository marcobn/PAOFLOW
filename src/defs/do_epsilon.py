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
from load_balancing import *
from communication import scatter_array

from constants import *
from smearing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_epsilon(E_k,pksp,kq_wght,omega,shift,delta,temp,ipol,jpol,ispin,metal,ne,emin,emax,deltak,deltak2,smearing,kramerskronig):
    # Compute the dielectric tensor

    de = (emax-emin)/float(ne)
    ene = np.arange(emin,emax,de,dtype=float)
    if ene[0]==0.0: ene[0]=0.00001

    _,_,nawf,_,nspin = pksp.shape
    
    #=======================
    # Im
    #=======================
    epsi = np.zeros((3,3,ene.size),dtype=float)
    jdos = np.zeros((ene.size),dtype=float)
    epsi_aux = np.zeros((3,3,ene.size),dtype=float)
    jdos_aux = np.zeros((ene.size),dtype=float)

    if smearing == None:
        epsi_aux[:,:,:],jdos_aux[:] = epsi_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal)
    else:
        epsi_aux[:,:,:],jdos_aux[:] = smear_epsi_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,\
                          ispin,metal,deltak,deltak2,smearing)

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
            epsr_aux[:,:,:] = smear_epsr_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,\
                              ispin,metal,deltak,deltak2,smearing)

        comm.Allreduce(epsr_aux,epsr,op=MPI.SUM)

    epsr += 1.0




    return(ene,epsi,epsr,jdos)

def epsi_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal):
    orig_over_err = np.geterr()['over']
    np.seterr(over='raise')

    epsi = np.zeros((3,3,ene.size),dtype=float)
    jdos = np.zeros((ene.size),dtype=float)

    dfunc = np.zeros((pksp.shape[0],ene.size),dtype=float)
    eps=1.e-8
    fnF = np.zeros((pksp.shape[0]),dtype=float)

    for n in xrange(nawf):
        fn = 1.0/(np.exp(E_k[:,n,ispin]/temp)+1)
        for m in xrange(nawf):
            fm = 1.0/(np.exp(E_k[:,m,ispin]/temp)+1)
            dfunc[:,:] = 1.0/np.sqrt(np.pi)* \
                np.exp(-((((E_k[:,n,ispin]-E_k[:,m,ispin])*np.ones((pksp.shape[0],ene.size),dtype=float).T).T + ene)/delta)**2)
            epsi[ipol,jpol,:] += np.sum(((1.0/(ene**2+delta**2) * \
                           kq_wght[0] /delta * dfunc * ((fn - fm)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T* \
                           abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)
            jdos[:] += np.sum((( \
                           kq_wght[0] /delta * dfunc * ((fn - fm)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T* \
                           1.0),axis=1)
            if metal and n == m:
                for ik in xrange(pksp.shape[0]):
                    try:
                        fnF[ik] = 1.0/2.0 * 1.0/(1.0+np.cosh(E_k[ik,n,ispin]/temp))
                    except:
                        fnF[ik] = 1.0e8
                epsi[ipol,jpol,:] += np.sum(((1.0/ene * \
                               kq_wght[0] /delta * dfunc * ((fnF/temp)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T* \
                               abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)


    np.seterr(over=orig_over_err)
    return(epsi,jdos)

def smear_epsr_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal,deltak,deltak2,smearing):

    epsr = np.zeros((3,3,ene.size),dtype=float)

    dfunc = np.zeros((pksp.shape[0],ene.size),dtype=float)
    effterm = np.zeros((pksp.shape[0],nawf),dtype=complex)
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
            eig = ((E_k[:,m,ispin]-E_k[:,n,ispin])*np.ones((pksp.shape[0],ene.size),dtype=float).T).T
            om = ((ene*np.ones((pksp.shape[0],ene.size),dtype=float)).T).T
            del2 = (deltak2[:,n,m,ispin]*np.ones((pksp.shape[0],ene.size),dtype=float).T).T
            dfunc = 1.0/(eig - om + 1.0j*del2)
            if n != m:
                epsr[ipol,jpol,:] += np.real(np.sum((1.0/(eig**2+1.0j*delta**2) * \
                               kq_wght[0] * dfunc * ((fn - fm)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T * \
                               (pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin]),axis=1))

    epsr *= 4.0/(EPS0 * EVTORY * omega)

    return(epsr)


def smear_epsi_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal,deltak,deltak2,smearing):

    epsi = np.zeros((3,3,ene.size),dtype=float)
    jdos = np.zeros((ene.size),dtype=float)
    Ef = 0.0
    deltat = 0.1

    if smearing == 'gauss':
        fn = intgaussian(E_k[:,:,ispin],Ef,deltak[:,:,ispin])
    elif smearing == 'm-p':
        fn = intmetpax(E_k[:,:,ispin],Ef,deltak[:,:,ispin])
    else:
        sys.exit('smearing not implemented')

    '''upper triangle indices'''
    uind = np.triu_indices(nawf,k=1)
    nk=pksp.shape[0]

    E_diff_nm=np.zeros((nk,len(uind[0])),order='C')
    E_nm_pksp2=np.zeros((nk,len(uind[0])),order='C')
    f_nm=np.zeros((nk,len(uind[0])),order='C')

    E_diff_nm = np.ascontiguousarray((np.reshape(E_k[:,:,ispin],(nk,1,nawf))\
                     -np.reshape(E_k[:,:,ispin],(nk,nawf,1)))[:,uind[0],uind[1]])

    f_nm=np.ascontiguousarray((np.reshape(fn,(nk,nawf,1))-np.reshape(fn,(nk,1,nawf)))[:,uind[0],uind[1]])
#    f_nm_pksp2=f_nm*np.real(pksp[:,ipol,[:,uind[0],uind[1]],ispin]*\
#                                np.transpose(pksp[:,jpol,:,:,ispin],(0,2,1))[:,uind[0],uind[1]])
    f_nm_pksp2=np.ascontiguousarray(f_nm*np.real(pksp[:,ipol,uind[0],uind[1],ispin]*\
                                pksp[:,jpol,uind[1],uind[0],ispin]))




    fn = None

    dk2_nm = np.ascontiguousarray(deltak2[:,uind[0],uind[1],ispin])
    dfunc=np.zeros_like(f_nm_pksp2)
    sq2_dk2 = 1.0/(np.sqrt(np.pi)*deltak2[:,uind[0],uind[1],ispin])




    # gaussian smearing


    for e in xrange(ene.size):
        if smearing=='gauss':
            np.exp(-((ene[e]-E_diff_nm)/dk2_nm)**2,out=dfunc)
            dfunc *= sq2_dk2
        if smearing=='m-p':
            dfunc = metpax(E_diff_nm,ene[e],deltak2[:,uind[0],uind[1],ispin])

        epsi[ipol,jpol,e] = np.sum(1.0/(ene[e]**2+delta**2)*dfunc*f_nm_pksp2)
        jdos[e] = np.sum(f_nm*dfunc)

    sq2_dk2    = None
    dfunc      = None
    f_nm       = None
    f_nm_pksp2 = None
    E_diff_nm  = None
    dk2_nm     = None

    if metal:
        dk_cont = np.ascontiguousarray(deltak[:,:,ispin])
        if smearing == 'gauss':
            fnF = gaussian(E_k[:,:,ispin],Ef,0.03*deltak[:,:,ispin])
        elif smearing == 'm-p':
            fnF = metpax(E_k[:,:,ispin],Ef,deltak[:,:,ispin])

        dfunc=np.zeros_like(fnF)
        diag_ind = np.diag_indices(nawf)
        fnF *= np.ascontiguousarray(np.real(pksp[:,ipol,diag_ind[0],diag_ind[1],ispin]*pksp[:,jpol,diag_ind[0],diag_ind[1],ispin]))
        sq2_dk1 = 1.0/(np.sqrt(np.pi)*deltak[:,:,ispin])



        for e in xrange(ene.size):
            if smearing=='gauss':
                np.exp(-((ene[e])/dk_cont)**2,out=dfunc)
                dfunc *= sq2_dk1
            elif smearing=='m-p':
                dfunc = metpax(0.0,ene[e],deltak[:,:,ispin])
            epsi[ipol,jpol,e] += np.sum(1.0/(ene[e])*dfunc*fnF)

        sq2_dk1 = None
        dfunc   = None
        fnF     = None
        d2_cont = None

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)*kq_wght[0]
    jdos *= kq_wght[0]

    return(epsi,jdos)



def epsr_kramkron(ini_ie,end_ie,ene,epsi,shift,i,j):

    epsr = np.zeros((3,3,ene.size),dtype=float)
    de = ene[1]-ene[0]

    if end_ie == ini_ie: return
    if ini_ie < 3: ini_ie = 3
    if end_ie == ene.size: end_ie = ene.size-1
    f_ene = intmetpax(ene,shift,1.0)
    for ie in xrange(ini_ie,end_ie):
        #epsr[i,j,ie] = 2.0/np.pi * ( np.sum(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
        #               np.sum(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )
        epsr[i,j,ie] = 2.0/np.pi * ( tgr.simps(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]*f_ene[1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
                       tgr.simps(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]*f_ene[(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )

    return(epsr)

