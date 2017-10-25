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
from __future__ import print_function
from scipy import fftpack as FFT
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

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_Berry_curvature(E_k,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminSH,emaxSH,fermi_dw,fermi_up,deltak,smearing,writedata):
    #----------------------
    # Compute spin Berry curvature
    #----------------------

    _,_,nawf,nawf,nspin = pksp.shape

    Om_znk = np.zeros((pksp.shape[0],nawf),dtype=float)

    deltap = 0.05

    for n in xrange(nawf):
        for m in xrange(nawf):
            if m!= n:
                Om_znk[:,n] += -2.0*np.imag(pksp[:,ipol,n,m,0]*pksp[:,jpol,m,n,0]) / \
                ((E_k[:,m,0] - E_k[:,n,0])**2 + deltap**2)

    de = (emaxSH-eminSH)/500
    ene = np.arange(eminSH,emaxSH,de,dtype=float)

    Om_zkaux = np.zeros((pksp.shape[0],ene.size),dtype=float)

    for i in xrange(ene.size):
        if smearing == 'gauss':
            Om_zkaux[:,i] = np.sum(Om_znk[:,:]*intgaussian(E_k[:,:,0],ene[i],deltak[:,:,0]),axis=1)
        elif smearing == 'm-p':
            Om_zkaux[:,i] = np.sum(Om_znk[:,:]*intmetpax(E_k[:,:,0],ene[i],deltak[:,:,0]),axis=1)
        else:
            Om_zkaux[:,i] = np.sum(Om_znk[:,:]*(0.5 * (-np.sign(E_k[:,:,0]-ene[i]) + 1)),axis=1)


    Om_zk_sum = np.sum(Om_zkaux,axis=0)

    if rank==0:
        ahc = np.zeros((ene.size),dtype=float)
    else: ahc = None

    comm.Reduce(Om_zk_sum,ahc)
    comm.Barrier()
    
    Om_zk_sum=None


    if rank == 0:
        ahc/= float(nk1*nk2*nk3)
    else: Om_k = None

    if writedata:
        Om_zk = gather_full(Om_zkaux,npool)

        n0 = 0
        if rank == 0:
            for i in xrange(ene.size-1):
                if ene[i] <= fermi_dw and ene[i+1] >= fermi_dw:
                    n0 = i
                if ene[i] <= fermi_up and ene[i+1] >= fermi_up:
                    n = i

            Om_k = np.reshape(Om_zk,(nk1,nk2,nk3,ene.size),order='C')
            Om_k = Om_k[:,:,:,n]-Om_k[:,:,:,n0]
        comm.Barrier()

        return(ene,ahc,Om_k)

    else:
        return(ene,ahc,None)
