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
import numpy as np

def zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3):
    # zero padding for FFT interpolation in 3D
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    # first dimension
    auxp1 = np.zeros((nk1,nk2,nk3p),dtype=complex)
    auxp1[:,:,:int(nk3/2)]=aux[:,:,:int(nk3/2)]
    auxp1[:,:,int(nfft3+nk3/2):]=aux[:,:,int(nk3/2):]
    # second dimension
    auxp2 = np.zeros((nk1,nk2p,nk3p),dtype=complex)
    auxp2[:,:int(nk2/2),:]=auxp1[:,:int(nk2/2),:]
    auxp2[:,int(nfft2+nk2/2):,:]=auxp1[:,int(nk2/2):,:]
    # third dimension
    auxp3 = np.zeros((nk1p,nk2p,nk3p),dtype=complex)
    auxp3[:int(nk1/2),:,:]=auxp2[:int(nk1/2),:,:]
    auxp3[int(nfft1+nk1/2):,:,:]=auxp2[int(nk1/2):,:,:]

    return(auxp3)

def zero_pad_float(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3):
    # zero padding for FFT interpolation in 3D
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    # first dimension
    auxp1 = np.zeros((nk1,nk2,nk3p),dtype=float)
    auxp1[:,:,:int(nk3/2)]=aux[:,:,:int(nk3/2)]
    auxp1[:,:,int(nfft3+nk3/2):]=aux[:,:,int(nk3/2):]
    # second dimension
    auxp2 = np.zeros((nk1,nk2p,nk3p),dtype=float)
    auxp2[:,:int(nk2/2),:]=auxp1[:,:int(nk2/2),:]
    auxp2[:,int(nfft2+nk2/2):,:]=auxp1[:,int(nk2/2):,:]
    # third dimension
    auxp3 = np.zeros((nk1p,nk2p,nk3p),dtype=float)
    auxp3[:int(nk1/2),:,:]=auxp2[:int(nk1/2),:,:]
    auxp3[int(nfft1+nk1/2):,:,:]=auxp2[int(nk1/2):,:,:]

    return(auxp3)
