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
    '''
    Pad frequency domain with zeroes, such that any relationship between
        aux[k] and aux[N-k] is preserved.

    Arguments:
        aux (ndarray): unpadded frequency domain data
        nk1 (int): current size of aux along axis 0
        nk2 (int): current size of aux along axis 1
        nk3 (int): current size of aux along axis 2
        nfft1 (int): number of zeroes to pad axis 0 by
        nfft1 (int): number of zeroes to pad axis 1 by
        nfft1 (int): number of zeroes to pad axis 2 by

    Returns:
        auxp3 (ndarray): padded frequency domain data
    '''
    # post-padding dimensions
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    # halfway points
    sk1 = int((nk1+1)/2)
    sk2 = int((nk2+1)/2)
    sk3 = int((nk3+1)/2)
    # parities (even <-> p==1)
    p1 = (nk1 & 1)^1
    p2 = (nk2 & 1)^1
    p3 = (nk3 & 1)^1

    # accomodate nfft==0
    if nfft1 == 0:  p1 = 0
    if nfft2 == 0:  p2 = 0
    if nfft3 == 0:  p3 = 0

    # first dimension
    auxp1 = np.zeros((nk1,nk2,nk3p),dtype=complex)
    auxp1[:,:,:sk3+p3]=aux[:,:,:sk3+p3]
    auxp1[:,:,nfft3+sk3:]=aux[:,:,sk3:]
    # second dimension
    auxp2 = np.zeros((nk1,nk2p,nk3p),dtype=complex)
    auxp2[:,:sk2+p2,:]=auxp1[:,:sk2+p2,:]
    auxp2[:,nfft2+sk2:,:]=auxp1[:,sk2:,:]
    # third dimension
    auxp3 = np.zeros((nk1p,nk2p,nk3p),dtype=complex)
    auxp3[:sk1+p1,:,:]=auxp2[:sk1+p1,:,:]
    auxp3[nfft1+sk1:,:,:]=auxp2[sk1:,:,:]

    # halve Nyquist axes
    if p1:
        auxp3[ sk1,:,:] /= 2
        auxp3[-sk1,:,:] /= 2
    if p2:
        auxp3[:, sk2,:] /= 2
        auxp3[:,-sk2,:] /= 2
    if p3:
        auxp3[:,:, sk3] /= 2
        auxp3[:,:,-sk3] /= 2

    return(auxp3)


def zero_pad_float(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3):
    """ Deprecated. Use zero_pad instead.

    Note that this function uses the old padding algorithm, which
        1) does not (quite) preserve symmetry of DFT for even nk
        2) puts the zeros in the wrong spot altogether for odd nk...
    Besides that, it only works with real numbers...
    """
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
