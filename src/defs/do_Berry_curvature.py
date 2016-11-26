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

def do_Berry_curvature(E_k,pksp,delta,temp,ibrav,alat,a_vectors,b_vectors,dkres,iswitch):
    #----------------------
    # Compute Berry curvature on a selected path in the BZ
    #----------------------

    _,nawf,nawf,nk1,nk2,nk3,_ = pksp.shape

    # Compute only Omega_z(k)

    Om_znk = np.zeros((nawf,nk1*nk2*nk3),dtype=float)

    for n in range(nawf):
        for m in range(nawf):
            if n!= m:
                Om_znk[n,:] += -2.0*np.imag(np.reshape(pksp[0,n,m,:,:,:,0],nk1*nk2*nk3,order='C')* \
                np.reshape(pksp[1,m,n,:,:,:,0],nk1*nk2*nk3,order='C')) / \
                (E_k[:,n,0]**2 - E_k[:,m,0]**2 + delta**2)


    Om_zk = np.zeros((nk1*nk2*nk3),dtype=float)
    for nk in range(nk1*nk2*nk3):
        for n in range(nawf):
            if E_k[nk,n,0] <= 0.0:
                Om_zk[nk] += Om_znk[n,nk] #* 1.0/2.0 * 1.0/(1.0+np.cosh((E_k[n,nk,0]/temp)))/temp

    ahc = E2*np.sum(Om_zk)/float(nk1*nk2*nk3)

    if iswitch == 0:

        # Define k-point mesh for bands interpolation

        kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
        nkpi=kq.shape[1]
        for n in range(nkpi):
            kq [:,n]=kq[:,n].dot(b_vectors)

        # Compute Om_zR
        Om_zR = np.zeros((nk1,nk2,nk3),dtype=float)
        Om_zR = FFT.ifftn(np.reshape(Om_zk,(nk1,nk2,nk3),order='C'))

        R,_,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

        Om_zk_disp = np.zeros((nkpi),dtype=float)

        for ik in range(nkpi):
            for i in range(nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                        Om_zk_disp[ik] += np.real(Om_zR[i,j,k]*phase)

        f=open('Omega_z'+'.dat','w')
        for ik in range(nkpi):
            f.write('%3d  %.5f \n' %(ik,-Om_zk_disp[ik]))
        f.close()

    return(Om_zk,ahc)
