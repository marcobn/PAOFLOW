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
import sys, time
from mpi4py import MPI

sys.path.append('./')

from zero_pad import zero_pad

comm=MPI.COMM_WORLD
rank = comm.Get_rank()

def do_double_grid(nfft1,nfft2,nfft3,HRaux):
    # Fourier interpolation on extended grid (zero padding)
    if HRaux.shape[0] != 3 and HRaux.shape[1] == HRaux.shape[0]:
        nawf = HRaux.shape[0]
        nk1 = HRaux.shape[2]
        nk2 = HRaux.shape[3]
        nk3 = HRaux.shape[4]
        nspin = HRaux.shape[5]
        nk1p = nfft1+nk1
        nk2p = nfft2+nk2
        nk3p = nfft3+nk3
        nktotp= nk1p*nk2p*nk3p
        #if rank == 0: print('Number of k vectors for zero padding Fourier interpolation ',nktotp)

        # Extended R to k (with zero padding)
        HRauxp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
        Hksp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
        aux = np.zeros((nk1,nk2,nk3),dtype=complex)

        for ispin in range(nspin):
            for i in range(nawf):
                aux = HRaux[i,i,:,:,:,ispin]
                Hksp[i,i,:,:,:,ispin] = FFT.fftn(zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3))

        for ispin in range(nspin):
            for i in range(0,nawf-1):
                for j in range(i,nawf):
                    aux = HRaux[i,j,:,:,:,ispin]
                    Hksp[i,j,:,:,:,ispin] = FFT.fftn(zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3))
    elif HRaux.shape[0] == 3:
        # This works for the momentum tensor
        nawf = HRaux.shape[1]
        nk1 = HRaux.shape[3]
        nk2 = HRaux.shape[4]
        nk3 = HRaux.shape[5]
        nspin = HRaux.shape[6]
        nk1p = nfft1+nk1
        nk2p = nfft2+nk2
        nk3p = nfft3+nk3
        nktotp= nk1p*nk2p*nk3p
        #if rank == 0: print('Number of k vectors for zero padding Fourier interpolation ',nktotp)

        # Extended R to k (with zero padding)
        HRauxp  = np.zeros((3,nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
        Hksp  = np.zeros((3,nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
        aux = np.zeros((nk1,nk2,nk3),dtype=complex)

        for l in range(3):
            for ispin in range(nspin):
                for i in range(nawf):
                    aux = HRaux[l,i,i,:,:,:,ispin]
                    Hksp[l,i,i,:,:,:,ispin] = FFT.fftn(zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3))

            for ispin in range(nspin):
                for i in range(nawf):
                    for j in range(nawf):
                        aux = HRaux[l,i,j,:,:,:,ispin]
                        Hksp[l,i,j,:,:,:,ispin] = FFT.fftn(zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3))
    
    else: 
        sys.exit('wrong dimensions in input array')

    nk1 = nk1p
    nk2 = nk2p
    nk3 = nk3p
    aux = None
    return(Hksp,nk1,nk2,nk3)

