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

def do_Berry_curvature(E_k,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminSH,emaxSH,fermi_dw,fermi_up,deltak,smearing):
  try:
    #----------------------
    # Compute spin Berry curvature
    #----------------------

    index = None

    if rank == 0:
        nktot,_,nawf,nawf,nspin = pksp.shape
        index = {'nktot':nktot,'nawf':nawf,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    # Compute only Omega_z(k)

    if rank == 0:
        Om_znk = np.zeros((nk1*nk2*nk3,nawf),dtype=float)
    else:
        Om_znk = None

    for pool in xrange(npool):
        if nk1*nk2*nk3%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nk1*nk2*nk3/npool

        if rank == 0:
            pksp_long = np.array_split(pksp,npool,axis=0)[pool]
            E_k_long= np.array_split(E_k,npool,axis=0)[pool]
            Om_znk_split = np.array_split(Om_znk,npool,axis=0)[pool]
        else:
            Om_znk_split = None
            pksp_long = None
            E_k_long = None

        comm.Barrier()
        pksaux = scatter_array(pksp_long)
        E_kaux = scatter_array(E_k_long)

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        Om_znkaux = np.zeros((nsize,nawf),dtype=float)

        deltap = 0.05
        for n in xrange(nawf):
            for m in xrange(nawf):
                if m!= n:
                    Om_znkaux[:,n] += -2.0*np.imag(pksaux[:,ipol,n,m,0]*pksaux[:,jpol,m,n,0]) / \
                    ((E_kaux[:,m,0] - E_kaux[:,n,0])**2 + deltap**2)
        comm.Barrier()
        gather_array(Om_znk_split, Om_znkaux)

        if rank == 0:
            Om_znk[pool*nkpool:(pool+1)*nkpool,:] = Om_znk_split[:,:]

    de = (emaxSH-eminSH)/500
    ene = np.arange(eminSH,emaxSH,de,dtype=float)

    if rank == 0:
        Om_zk = np.zeros((nk1*nk2*nk3,ene.size),dtype=float)
    else:
        Om_zk = None

    pksp_long = None
    E_k_long = None

    for pool in xrange(npool):
        if nk1*nk2*nk3%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nk1*nk2*nk3/npool

        if rank == 0:
            E_k_long= np.array_split(E_k,npool,axis=0)[pool]
            Om_znk_long = np.array_split(Om_znk,npool,axis=0)[pool]
            Om_zk_split = np.array_split(Om_zk,npool,axis=0)[pool]
            deltak_long= np.array_split(deltak,npool,axis=0)[pool]
        else:
            Om_znk_long = None
            Om_zk_split = None
            E_k_long = None
            deltak_long = None

        comm.Barrier()
        Om_znkaux = scatter_array(Om_znk_long)
        E_kaux = scatter_array(E_k_long)
        deltakaux = scatter_array(deltak_long)

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        Om_zkaux = np.zeros((nsize,ene.size),dtype=float)

        for i in xrange(ene.size):
            if smearing == 'gauss':
                Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intgaussian(E_kaux[:,:,0],ene[i],deltakaux[:,:,0]),axis=1)
            elif smearing == 'm-p':
                Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intmetpax(E_kaux[:,:,0],ene[i],deltakaux[:,:,0]),axis=1)
            else:
                Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*(0.5 * (-np.sign(E_kaux[:,:,0]-ene[i]) + 1)),axis=1)

        comm.Barrier()
        gather_array(Om_zk_split, Om_zkaux)

        if rank == 0:
            Om_zk[pool*nkpool:(pool+1)*nkpool,:] = Om_zk_split[:,:]

    ahc = None
    if rank == 0: ahc = np.sum(Om_zk,axis=0)/float(nk1*nk2*nk3)

    Om_k = np.zeros((nk1,nk2,nk3,ene.size),dtype=float)
    n0 = 0
    if rank == 0:
        for i in xrange(ene.size-1):
            if ene[i] <= fermi_dw and ene[i+1] >= fermi_dw:
                n0 = i
            if ene[i] <= fermi_up and ene[i+1] >= fermi_up:
                n = i

        Om_k = np.reshape(Om_zk,(nk1,nk2,nk3,ene.size),order='C')

    return(ene,ahc,Om_k[:,:,:,n]-Om_k[:,:,:,n0])
  except Exception as e:
    raise e
