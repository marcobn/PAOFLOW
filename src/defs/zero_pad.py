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
import numpy as np

def zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3):
    # zero padding for FFT interpolation in 3D
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    # first dimension
    auxp1 = np.zeros((nk1,nk2,nk3p),dtype=complex)
    auxp1[:,:,:(nk3/2)]=aux[:,:,:(nk3/2)]
    auxp1[:,:,(nfft3+nk3/2):]=aux[:,:,(nk3/2):]
    # second dimension
    auxp2 = np.zeros((nk1,nk2p,nk3p),dtype=complex)
    auxp2[:,:(nk2/2),:]=auxp1[:,:(nk2/2),:]
    auxp2[:,(nfft2+nk2/2):,:]=auxp1[:,(nk2/2):,:]
    # third dimension
    auxp3 = np.zeros((nk1p,nk2p,nk3p),dtype=complex)
    auxp3[:(nk1/2),:,:]=auxp2[:(nk1/2),:,:]
    auxp3[(nfft1+nk1/2):,:,:]=auxp2[(nk1/2):,:,:]

    return(auxp3)

def zero_pad_float(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3):
    # zero padding for FFT interpolation in 3D
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    # first dimension
    auxp1 = np.zeros((nk1,nk2,nk3p),dtype=float)
    auxp1[:,:,:(nk3/2)]=aux[:,:,:(nk3/2)]
    auxp1[:,:,(nfft3+nk3/2):]=aux[:,:,(nk3/2):]
    # second dimension
    auxp2 = np.zeros((nk1,nk2p,nk3p),dtype=float)
    auxp2[:,:(nk2/2),:]=auxp1[:,:(nk2/2),:]
    auxp2[:,(nfft2+nk2/2):,:]=auxp1[:,(nk2/2):,:]
    # third dimension
    auxp3 = np.zeros((nk1p,nk2p,nk3p),dtype=float)
    auxp3[:(nk1/2),:,:]=auxp2[:(nk1/2),:,:]
    auxp3[(nfft1+nk1/2):,:,:]=auxp2[(nk1/2):,:,:]

    return(auxp3)
