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
from scipy import fftpack as FFT

def get_R_grid_fft(nk1,nk2,nk3,a_vectors):
    nrtot = nk1*nk2*nk3
    R = np.zeros((nrtot,3),dtype=float)
    Rfft = np.zeros((nk1,nk2,nk3,3),dtype=float)
    R_wght = np.ones((nrtot),dtype=float)
    idx = np.zeros((nk1,nk2,nk3),dtype=int)

    for i in xrange(nk1):
        for j in xrange(nk2):
            for k in xrange(nk3):
                n = k + j*nk3 + i*nk2*nk3
                Rx = float(i)/float(nk1)
                Ry = float(j)/float(nk2)
                Rz = float(k)/float(nk3)
                if Rx >= 0.5: Rx=Rx-1.0
                if Ry >= 0.5: Ry=Ry-1.0
                if Rz >= 0.5: Rz=Rz-1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)
                R[n,:] = Rx*nk1*a_vectors[0,:]+Ry*nk2*a_vectors[1,:]+Rz*nk3*a_vectors[2,:]
                Rfft[i,j,k,:] = R[n,:]
                idx[i,j,k]=n

    Rfft = FFT.fftshift(Rfft,axes=(0,1,2))

    return(R,Rfft,R_wght,nrtot,idx)
