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

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_spin_Berry_curvature(E_k,jksp,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminSH,emaxSH,fermi_dw,fermi_up,deltak,smearing):
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
        ini_ip, end_ip = load_balancing(npool, pool, nktot)
        nkpool = end_ip - ini_ip

        if rank == 0:
            pksp_long = pksp[ini_ip:end_ip]
            jksp_long = jksp[ini_ip:end_ip]
            E_k_long= E_k[ini_ip:end_ip]
            Om_znk_split = Om_znk[ini_ip:end_ip]
        else:
            Om_znk_split = None
            pksp_long = None
            jksp_long = None
            E_k_long = None

        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik - ini_ik

        Om_znkaux = np.zeros((nsize,nawf),dtype=float)

        pksaux = scatter_array(pksp_long)
        jksaux = scatter_array(jksp_long)
        E_kaux = scatter_array(E_k_long)

        deltap = 0.05
        for n in xrange(nawf):
            for m in xrange(nawf):
                if m!= n:
                    Om_znkaux[:,n] += -2.0*np.imag(jksaux[:,ipol,n,m,0]*pksaux[:,jpol,m,n,0]) / \
                    ((E_kaux[:,m,0] - E_kaux[:,n,0])**2 + deltap**2)

        gather_array(Om_znk_split, Om_znkaux)

        if rank == 0:
            Om_znk[ini_ip:end_ip,:] = Om_znk_split[:,:]

    de = (emaxSH-eminSH)/500
    ene = np.arange(eminSH,emaxSH,de,dtype=float)

    if rank == 0:
        Om_zk = np.zeros((nk1*nk2*nk3,ene.size),dtype=float)
    else:
        Om_zk = None

    pksp_long = None
    E_k_long = None

    for pool in xrange(npool):
        ini_ip, end_ip = load_balancing(npool, pool, nktot)
        nkpool = end_ip - ini_ip

        if rank == 0:
            E_k_long = E_k[ini_ip:end_ip]
            Om_znk_long = Om_znk[ini_ip:end_ip]
            Om_zk_split = Om_zk[ini_ip:end_ip]
            deltak_long= deltak[ini_ip:end_ip]
        else:
            Om_znk_long = None
            Om_zk_split = None
            E_k_long = None
            deltak_long = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        Om_zkaux = np.zeros((nsize,ene.size),dtype=float)

        Om_znkaux = scatter_array(Om_znk_long)
        E_kaux = scatter_array(E_k_long)
        deltakaux = scatter_array(deltak_long)

        for i in xrange(ene.size):
            if smearing == 'gauss':
                Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intgaussian(E_kaux[:,:,0],ene[i],deltakaux[:,:,0]),axis=1)
            elif smearing == 'm-p':
                Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intmetpax(E_kaux[:,:,0],ene[i],deltakaux[:,:,0]),axis=1)
            else:
                Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*(0.5 * (-np.sign(E_kaux[:,:,0]-ene[i]) + 1)),axis=1)

        gather_array(Om_zk_split, Om_zkaux)

        if rank == 0:
            Om_zk[ini_ip:end_ip,:] = Om_zk_split[:,:]

    shc = None
    if rank == 0: shc = np.sum(Om_zk,axis=0)/float(nk1*nk2*nk3)

    Om_k = np.zeros((nk1,nk2,nk3,ene.size),dtype=float)
    n0 = 0
    if rank == 0:
        for i in xrange(ene.size-1):
            if ene[i] <= fermi_dw and ene[i+1] >= fermi_dw:
                n0 = i
            if ene[i] <= fermi_up and ene[i+1] >= fermi_up:
                n = i
        Om_k = np.reshape(Om_zk,(nk1,nk2,nk3,ene.size),order='C')

    return(ene,shc,Om_k[:,:,:,n]-Om_k[:,:,:,n0])
  except Exception as e:
    raise e
