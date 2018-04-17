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
import scipy.special as SPECIAL
import numpy as np
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import multiprocessing

from get_R_grid_fft import *
from kpnts_interpolation_mesh import *
from do_non_ortho import *
from load_balancing import *
from communication import *
from constants import *
from smearing import *
import time
# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_spin_Berry_curvature(E_k,jksp,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminSH,emaxSH,fermi_dw,fermi_up,deltak,smearing,writedata):
    #----------------------
    # Compute spin Berry curvature
    #----------------------



    nktot,_,nawf,nawf,nspin = pksp.shape

    # Compute only Omega_z(k)
    Om_znkaux = np.zeros((pksp.shape[0],nawf),dtype=float)

    start=time.time()
    deltap = 0.05
    for n in xrange(nawf):
        for m in xrange(nawf):
            if m!= n:
                Om_znkaux[:,n] += -2.0*np.imag(jksp[:,n,m,0]*pksp[:,jpol,m,n,0]) / \
                ((E_k[:,m,0] - E_k[:,n,0])**2 + deltap**2)


    de = (emaxSH-eminSH)/500
    ene = np.arange(eminSH,emaxSH,de,dtype=float)

    if rank == 0:
        Om_zk = np.zeros((nk1*nk2*nk3,ene.size),dtype=float)
    else:
        Om_zk = None

    Om_zkaux = np.zeros((pksp.shape[0],ene.size),dtype=float)

    for i in xrange(ene.size):
        if smearing == 'gauss':
            Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intgaussian(E_k[:,:,0],ene[i],deltak[:,:,0]),axis=1)
        elif smearing == 'm-p':
            Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intmetpax(E_k[:,:,0],ene[i],deltak[:,:,0]),axis=1)
        else:
            Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*(0.5 * (-np.sign(E_k[:,:,0]-ene[i]) + 1)),axis=1)


    if writedata:
        Om_zk = gather_full(Om_zkaux,npool)
        comm.Barrier()

        Om_zk_aux = None

        shc = None
        if rank == 0: shc = np.sum(Om_zk,axis=0)/float(nk1*nk2*nk3)


        n0 = 0
        if rank == 0:
            Om_k = np.zeros((nk1,nk2,nk3,ene.size),dtype=float)
            for i in xrange(ene.size-1):
                if ene[i] <= fermi_dw and ene[i+1] >= fermi_dw:
                    n0 = i
                if ene[i] <= fermi_up and ene[i+1] >= fermi_up:
                    n = i
            Om_k = np.reshape(Om_zk,(nk1,nk2,nk3,ene.size),order='C')
            Om_k = Om_k[:,:,:,n]-Om_k[:,:,:,n0]

        else: Om_k = None

        return(ene,shc,Om_k)

    else:
        
        shc_aux = np.sum(Om_zkaux,axis=0)/float(nk1*nk2*nk3)
        if rank==0:
            shc = np.zeros_like(ene)
        else:
            shc=None

        comm.Reduce(shc_aux,shc)

        return(ene,shc,None)
