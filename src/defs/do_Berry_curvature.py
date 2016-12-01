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

def do_Berry_curvature(E_k,pksp,nk1,nk2,nk3,delta,temp,ibrav,alat,a_vectors,b_vectors,dkres,iswitch,nthread,npool):
    #----------------------
    # Compute Berry curvature on a selected path in the BZ
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

    for pool in range(npool):
        if nk1*nk2*nk3%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nk1*nk2*nk3/npool

        if rank == 0:
            pksp = np.array_split(pksp,npool,axis=0)[pool]
            E_k= np.array_split(E_k,npool,axis=0)[pool]
            Om_znk_split = np.array_split(Om_znk,npool,axis=0)[pool]
        else:
            Om_znk_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        pksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        E_kaux = np.zeros((nsize,nawf,nspin),dtype = float)
        Om_znkaux = np.zeros((nsize,nawf),dtype=float)

        comm.Barrier()
        comm.Scatter(pksp,pksaux,root=0)
        comm.Scatter(E_k,E_kaux,root=0)

        for nk in range(nsize):
            for n in range(nawf):
                for m in range(nawf):
                    if n!= m:
                        Om_znkaux[nk,n] += -2.0*np.imag(pksaux[nk,0,n,m,0]*pksaux[nk,1,m,n,0]) / \
                        (E_kaux[nk,n,0]**2 - E_kaux[nk,m,0]**2 + delta**2)
        comm.Barrier()
        comm.Gather(Om_znkaux,Om_znk_split,root=0)

        if rank == 0:
            Om_znk[pool*nkpool:(pool+1)*nkpool,:] = Om_znk_split[:,:]

    if rank == 0:
        pksp = np.concatenate(pksp)
        E_k = np.concatenate(E_k)
    if rank == 0:
        Om_zk = np.zeros((nk1*nk2*nk3),dtype=float)
    else:
        Om_znk = None
        Om_zk = None

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

    for nk in range(nsize):
        for n in range(nawf):
            if E_kaux[nk,n,0] <= 0.0:
                Om_zkaux[nk] = Om_znkaux[nk,n] #* 1.0/2.0 * 1.0/(1.0+np.cosh((E_k[n,nk,0]/temp)))/temp

    comm.Barrier()
    comm.Gather(Om_zkaux,Om_zk,root=0)

    ahc = None
    if rank == 0: ahc = -E2*np.sum(Om_zk)/float(nk1*nk2*nk3)

    if iswitch == 0:

        # Define k-point mesh for bands interpolation

        kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
        nkpi=kq.shape[1]
        for n in range(nkpi):
            kq [:,n]=kq[:,n].dot(b_vectors)


        if rank == 0:
            # Compute Om_zR
            Om_zR = np.zeros((nk1*nk2*nk3),dtype=float)
            Om_zRc = np.zeros((nk1,nk2,nk3),dtype=complex)
            Om_zk = np.reshape(Om_zk,(nk1,nk2,nk3),order='C')+1.j
            fft = pyfftw.FFTW(Om_zk,Om_zRc,axes=(0,1,2), direction='FFTW_BACKWARD',\
                        flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
            Om_zRc = fft()
            Om_zR = np.real(np.reshape(Om_zRc,nk1*nk2*nk3,order='C'))
            R,_,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)
            Om_zk_disp = np.zeros((nkpi),dtype=float)
        else:
            Om_zR = None
            R = None
            R_wght = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nk1*nk2*nk3)
        nsize = end_ik-ini_ik

        for ik in range(nkpi):

            Om_zRaux = np.zeros(nsize,dtype=float)
            R_wghtaux = np.zeros(nsize,dtype=float)
            R_aux = np.zeros((nsize,3),dtype=float)
            Om_zk_sum = np.zeros(1,dtype=float)
            auxsum = np.zeros(1,dtype=float)

            comm.Barrier()
            comm.Scatter(R,R_aux,root=0)
            comm.Scatter(R_wght,R_wghtaux,root=0)
            comm.Scatter(Om_zR,Om_zRaux,root=0)

            for nk in range(nsize):
                phase=R_wghtaux[nk]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R_aux[nk,:])*1j)
                auxsum += np.real(Om_zRaux[nk]*phase)

            comm.Barrier()
            comm.Reduce(auxsum,Om_zk_sum,op=MPI.SUM)
            if rank == 0: Om_zk_disp[ik] = Om_zk_sum

        if rank == 0:
            f=open('Omega_z'+'.dat','w')
            for ik in range(nkpi):
                f.write('%3d  %.5f \n' %(ik,-Om_zk_disp[ik]))
            f.close()

    return(Om_zk,ahc)
