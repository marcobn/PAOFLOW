#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
#
from scipy import fftpack as FFT
import scipy.special as SPECIAL
import numpy as np
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import multiprocessing

from write_TB_eigs import write_TB_eigs
from get_R_grid_fft import *
from kpnts_interpolation_mesh import *
from do_non_ortho import *
from load_balancing import *
from constants import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_spin_Berry_curvature(E_k,jksp,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminSH,emaxSH,fermi_dw,fermi_up):
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
            jksp_long = np.array_split(jksp,npool,axis=0)[pool]
            E_k_long= np.array_split(E_k,npool,axis=0)[pool]
            Om_znk_split = np.array_split(Om_znk,npool,axis=0)[pool]
        else:
            Om_znk_split = None
            pksp_long = None
            jksp_long = None
            E_k_long = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        pksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        jksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        E_kaux = np.zeros((nsize,nawf,nspin),dtype = float)
        Om_znkaux = np.zeros((nsize,nawf),dtype=float)

        comm.Barrier()
        comm.Scatter(pksp_long,pksaux,root=0)
        comm.Scatter(jksp_long,jksaux,root=0)
        comm.Scatter(E_k_long,E_kaux,root=0)

        deltap = 0.05
        for n in xrange(nawf):
            for m in xrange(nawf):
                if m!= n:
                    Om_znkaux[:,n] += -2.0*np.imag(jksaux[:,ipol,n,m,0]*pksaux[:,jpol,m,n,0]) / \
                    ((E_kaux[:,m,0] - E_kaux[:,n,0])**2 + deltap**2)
        comm.Barrier()
        comm.Gather(Om_znkaux,Om_znk_split,root=0)

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

    delta = 0.05

    for pool in xrange(npool):
        if nk1*nk2*nk3%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nk1*nk2*nk3/npool

        if rank == 0:
            E_k_long= np.array_split(E_k,npool,axis=0)[pool]
            Om_znk_long = np.array_split(Om_znk,npool,axis=0)[pool]
            Om_zk_split = np.array_split(Om_zk,npool,axis=0)[pool]
        else:
            Om_znk_long = None
            Om_zk_split = None
            E_k_long = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        Om_znkaux = np.zeros((nsize,nawf),dtype=float)
        Om_zkaux = np.zeros((nsize,ene.size),dtype=float)
        E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)

        comm.Barrier()
        comm.Scatter(Om_znk_long,Om_znkaux,root= 0)
        comm.Scatter(E_k_long,E_kaux,root= 0)

        for i in xrange(ene.size):
            #Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*(0.5 * (-np.sign(E_kaux[:,:,0]-ene[i]) + 1)),axis=1)  # T=0.0K
            Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*0.5*(1-SPECIAL.erf((E_kaux[:,:,0]-ene[i])/delta)),axis=1)

        comm.Barrier()
        comm.Gather(Om_zkaux,Om_zk_split,root=0)

        if rank == 0:
            Om_zk[pool*nkpool:(pool+1)*nkpool,:] = Om_zk_split[:,:]

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
