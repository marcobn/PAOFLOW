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
from scipy import fftpack as FFT
import numpy as np
try:
    import psutil
except: pass

import cmath
import sys,os

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

#import matplotlib.pyplot as plt

from write_PAO_eigs import *
from kpnts_interpolation_mesh import *
#from new_kpoint_interpolation import *
from do_non_ortho import *
from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import scipy.linalg as LAN

def do_bands_calc(HRaux,SRaux,kq,R_wght,R,idx,read_S,inputpath):
    # Load balancing
    nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape    
    
    ini_ik, end_ik = load_balancing(size,rank,kq.shape[1]) 


    if read_S:
        Sks_aux = band_loop_S(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,SRaux,R_wght,kq,R,idx)
    else: Sks_aux = None

    Hks_aux = band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,HRaux,R_wght,kq,R,idx)

    E_kp_aux = np.zeros(((end_ik-ini_ik),nawf,nspin),dtype=float,order="C")
    v_kp_aux = np.zeros(((end_ik-ini_ik),nawf,nawf,nspin),dtype=complex,order="C")

    for ispin in xrange(nspin):
        for ik in xrange(end_ik-ini_ik):
            if read_S:
                E_kp_aux[ik,:,ispin],v_kp_aux[ik,:,:,ispin] = LAN.eigh(Hks_aux[:,:,ik,ispin], 
                                                                       b=Sks_aux[:,:,ik],lower=False, 
                                                                       overwrite_a=True,
                                                                       overwrite_b=True,
                                                                       turbo=True,check_finite=True)
            else:   
                E_kp_aux[ik,:,ispin],v_kp_aux[ik,:,:,ispin] = LAN.eigh(Hks_aux[:,:,ik,ispin], 
                                                                       b=None,lower=False, 
                                                                       overwrite_a=True,
                                                                       overwrite_b=True,
                                                                       turbo=True,check_finite=True)
                 
    Hks_aux = None
    Sks_aux = None

    if rank==0:
        E_kp = np.zeros((kq.shape[1],nawf,nspin),order="C")
        gather_array(E_kp,E_kp_aux)
    else:
        gather_array(None,E_kp_aux)

    vecs=False
    comm.Barrier()

    if vecs:
        if rank==0:
            v_kp = np.zeros((kq.shape[1],nawf,nawf,nspin),order="C")
            gather_array(v_kp,v_kp_aux)
        else:
            gather_array(None,v_kp_aux)

    else:
        v_kp = None
    v_kp_aux = None
    E_kp_aux = None
  
    if rank==0:
        for ispin in xrange(nspin):
            f=open(inputpath+'/bands_'+str(ispin)+'.dat','w')
            for ik in xrange(kq.shape[1]):
                s="%d\t"%ik
                for  j in E_kp[ik,:,ispin]:s += "% 3.5f\t"%j
                s+="\n"
                f.write(s)
            f.close()
            
        return E_kp,v_kp
    else: return None,None


def band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,HRaux,R_wght,kq,R,idx):

    HRaux = np.reshape(HRaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
    kdot = np.zeros(((end_ik-ini_ik),R.shape[0]),dtype=complex,order="C")
    kdot = np.tensordot(R,2.0j*np.pi*kq[:,ini_ik:end_ik],axes=([1],[0]))
    np.exp(kdot,kdot)

    auxh = np.zeros((nawf,nawf,(end_ik-ini_ik),nspin),dtype=complex,order="C")

    for ispin in xrange(nspin):
        auxh[:,:,:,ispin]=np.tensordot(HRaux[:,:,:,ispin],kdot,axes=([2],[0]))

    kdot  = None
    return auxh


# def band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,HRaux,R_wght,kq,R,idx):

#     nsize = end_ik - ini_ik
#     auxh = np.zeros((nawf,nawf,nsize,nspin),dtype=complex)
#     HRaux = np.reshape(HRaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')

#     for ik in xrange(ini_ik,end_ik):
#         for ispin in xrange(nspin):
#              auxh[:,:,ik-ini_ik,ispin] = np.sum(HRaux[:,:,:,ispin]*np.exp(2.0*np.pi*kq[:,ik].dot(R[:,:].T)*1j),axis=2)

#     return(auxh)

def band_loop_S(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,SRaux,R_wght,kq,R,idx):

    nsize = end_ik - ini_ik
    auxs = np.zeros((nawf,nawf,nsize),dtype=complex)

    for ik in xrange(ini_ik,end_ik):
        for i in xrange(nk1):
            for j in xrange(nk2):
                for k in xrange(nk3):
                    phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                    auxs[:,:,ik-ini_ik] += SRaux[:,:,i,j,k]*phase

    return(auxs)
