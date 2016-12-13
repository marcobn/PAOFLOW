#
# AFLOWpi_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
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
import numpy as np
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import pyfftw
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

def do_Berry_curvature(E_k,pksp,nk1,nk2,nk3,npool):
    #----------------------
    # Compute Berry curvature
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

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        pksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        E_kaux = np.zeros((nsize,nawf,nspin),dtype = float)
        Om_znkaux = np.zeros((nsize,nawf),dtype=float)

        comm.Barrier()
        comm.Scatter(pksp_long,pksaux,root=0)
        comm.Scatter(E_k_long,E_kaux,root=0)

        ########NOTE The indeces of the polarizations (x,y,z) should be changed according to the direction of the magnetization
        ########     Here we enforce explicitly the antisymmetry of the conductivity tensor to minimize convergence errors.
        deltap = 0.05
        for nk in xrange(nsize):
            for n in xrange(nawf):
                for m in xrange(nawf):
                    if m!= n:
                        Om_znkaux[nk,n] += -1.0*np.imag(pksaux[nk,2,n,m,0]*pksaux[nk,1,m,n,0]- pksaux[nk,1,n,m,0]*pksaux[nk,2,m,n,0]) / \
                        ((E_kaux[nk,m,0] - E_kaux[nk,n,0])**2 + deltap**2)
        comm.Barrier()
        comm.Gather(Om_znkaux,Om_znk_split,root=0)

        if rank == 0:
            Om_znk[pool*nkpool:(pool+1)*nkpool,:] = Om_znk_split[:,:]

    if rank == 0:
        Om_zk = np.zeros((nk1*nk2*nk3),dtype=float)
    else:
        Om_znk = None
        Om_zk = None

    pksp_long = None
    E_k_long = None

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nk1*nk2*nk3)
    nsize = end_ik-ini_ik

    Om_znkaux = np.zeros((nsize,nawf),dtype=float)
    Om_zkaux = np.zeros((nsize),dtype=float)
    E_kaux = np.zeros((nsize,nawf,nspin),dtype = float)

    comm.Barrier()
    comm.Scatter(Om_znk,Om_znkaux,root= 0)
    comm.Scatter(Om_zk,Om_zkaux,root= 0)
    comm.Scatter(E_k,E_kaux,root= 0)

    for nk in xrange(nsize):
        Om_zkaux[nk] = np.sum(Om_znkaux[nk,:]*(0.5 * (-np.sign(E_kaux[nk,:,0]) + 1)))  # T=0.0K

    comm.Barrier()
    comm.Gather(Om_zkaux,Om_zk,root=0)

    ahc = None
    if rank == 0: ahc = -E2*np.sum(Om_zk)/float(nk1*nk2*nk3)

    return(ahc)
