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
from write_PAO_eigs import *
from kpnts_interpolation_mesh import *
from do_non_ortho import *
from load_balancing import *
from communication import *
import scipy.linalg as LAN
# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()




def do_bands_calc(HRaux,SRaux,kq,R_wght,R,idx,read_S,inputpath,npool):
    # Load balancing
    nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape            
    kq_aux = scatter_full(kq.T,npool)
    kq_aux = kq_aux.T

 
    if read_S:
        Sks_aux = band_loop_S(nspin,nk1,nk2,nk3,nawf,SRaux,R_wght,kq_aux,R,idx)
    else: Sks_aux = None

    Hks_aux = band_loop_H(nspin,nk1,nk2,nk3,nawf,HRaux,R_wght,kq_aux,R,idx)

    E_kp_aux = np.zeros((kq_aux.shape[1],nawf,nspin),dtype=float,order="C")
    v_kp_aux = np.zeros((kq_aux.shape[1],nawf,nawf,nspin),dtype=complex,order="C")
    
    for ispin in range(nspin):
        for ik in xrange(kq_aux.shape[1]):
            if read_S:
                E_kp_aux[ik,:,ispin],v_kp_aux[ik,:,:,ispin] = LAN.eigh(Hks_aux[:,:,ik,ispin],
                                                                       b=Sks_aux[:,:,ik],lower=False,
                                                                       overwrite_a=False,
                                                                       overwrite_b=False,
                                                                       turbo=True,check_finite=True)
            else:   
                E_kp_aux[ik,:,ispin],v_kp_aux[ik,:,:,ispin] = LAN.eigh(Hks_aux[:,:,ik,ispin],
                                                                       lower=False,
                                                                       overwrite_a=True,
                                                                       overwrite_b=True,
                                                                       turbo=True,check_finite=True)


    Hks_aux = None
    Sks_aux = None

    if rank==0:
        E_kp = gather_full(E_kp_aux,npool)
    else:
        gather_full(E_kp_aux,npool)
        E_kp = None


    comm.Barrier()
  
    if rank==0:
        for ispin in xrange(nspin):
            f=open(os.path.join(inputpath,'bands_'+str(ispin)+'.dat'),'w')
            for ik in xrange(kq.shape[1]):
                s="%d\t"%ik
                for  j in E_kp[ik,:,ispin]:s += "% 3.5f\t"%j
                s+="\n"
                f.write(s)
            f.close()
            
    return E_kp_aux,v_kp_aux



def band_loop_H(nspin,nk1,nk2,nk3,nawf,HRaux,R_wght,kq,R,idx):

    HRaux = np.reshape(HRaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
    kdot = np.zeros((kq.shape[1],R.shape[0]),dtype=complex,order="C")
    kdot = np.tensordot(R,2.0j*np.pi*kq,axes=([1],[0]))
    np.exp(kdot,kdot)

    auxh = np.zeros((nawf,nawf,kq.shape[1],nspin),dtype=complex,order="C")

    for ispin in xrange(nspin):
        auxh[:,:,:,ispin]=np.tensordot(HRaux[:,:,:,ispin],kdot,axes=([2],[0]))

    kdot  = None
    return auxh


def band_loop_S(nspin,nk1,nk2,nk3,nawf,SRaux,R_wght,kq,R,idx):

    nsize = kq.shape[1]
    auxs  = np.zeros((nawf,nawf,nsize),dtype=complex)

    for ik in xrange(kq.shape[1]):
        for i in xrange(nk1):
            for j in xrange(nk2):
                for k in xrange(nk3):
                    phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                    auxs[:,:,ik] += SRaux[:,:,i,j,k]*phase

    return(auxs)
