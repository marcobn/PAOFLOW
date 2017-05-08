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

def get_K_grid_fft(nk1,nk2,nk3,b_vectors):
    nktot = nk1*nk2*nk3
    Kint = np.zeros((3,nktot),dtype=float)
    K_wght = np.ones((nktot),dtype=float)
    K_wght /= nktot
    idk = np.zeros((nk1,nk2,nk3),dtype=int)

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
                idk[i,j,k]=n
                Kint[:,n] = Rx*b_vectors[0,:]+Ry*b_vectors[1,:]+Rz*b_vectors[2,:]

    return(Kint,K_wght,nktot,idk)
