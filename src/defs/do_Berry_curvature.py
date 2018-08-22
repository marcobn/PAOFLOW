#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

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

def do_Berry_curvature(E_k,pksp_i,pksp_j,nk1,nk2,nk3,npool,eminSH,emaxSH,fermi_dw,fermi_up,deltak,smearing):
    #----------------------
    # Compute spin Berry curvature
    #----------------------

    _,nawf,nawf,nspin = pksp_i.shape

    Om_znk = np.zeros((pksp_i.shape[0],nawf),dtype=float)

    deltap = 0.05
    deltap = 0.00

    for ik in range(E_k.shape[0]):
        E_nm = (E_k[ik,:,0] - E_k[ik,:,0][:,None])**2
        E_nm[np.where(E_nm<1.e-4)] = np.inf
        Om_znk[ik] = -2.0*np.sum(np.imag(pksp_i[ik,:,:,0]*pksp_j[ik,:,:,0].T) / \
                                            E_nm,axis=1)
    E_nm = None

    de = (emaxSH-eminSH)/500
    ene = np.arange(eminSH,emaxSH,de,dtype=float)

    Om_zkaux = np.zeros((pksp_i.shape[0],ene.size),dtype=float)

    for i in range(ene.size):
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

    Om_zk = gather_full(Om_zkaux,npool)

    n0 = 0
    if rank == 0:
        for i in range(ene.size-1):
            if ene[i] <= fermi_dw and ene[i+1] >= fermi_dw:
                n0 = i
            if ene[i] <= fermi_up and ene[i+1] >= fermi_up:
                n = i

        Om_k = np.reshape(Om_zk,(nk1,nk2,nk3,ene.size),order='C')
        Om_k = Om_k[:,:,:,n]-Om_k[:,:,:,n0]
    comm.Barrier()

    return(ene,ahc,Om_k)
