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
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys, time
from get_R_grid_fft import *

sys.path.append('./')

def do_z2pack_hamiltonian(nawf,nk1,nk2,nk3,a_vectors,HRs):
    f=open('z2pack_hamiltonian.dat','w')
    f.write ("PAOFLOW Generated \n")
    f.write (('%5d \n')%(nawf))

    nkpts=nk1*nk2*nk3
    f.write (('%5d \n')%(nkpts))

    nl=15 # z2pack read the weights in lines of 15 items

    nlines=int(nkpts/nl) # number of lines
    nlast=nkpts%nl   # number of items of laste line if needed

    # the weight is always one
    kq_wght = np.ones((nkpts),dtype=int)

    # print each cell weight
    j=0
    for i in xrange(nlines):
        j=i*nl
        f.write ('   '.join('{:d} '.format(j) for j in kq_wght[j:j+nl]))
        f.write ('\n')

    # Last line if needed
    if (nlast != 0):
        f.write ('   '.join('{:d} '.format(j) for j in kq_wght[nlines*nl:nkpts]))
        f.write ('\n')

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
                # the minus sign in Rx*nk1 is due to the Fourier transformation (Ri-Rj)
                ix=-round(Rx*nk1,0)
                iy=-round(Ry*nk2,0)
                iz=-round(Rz*nk3,0)
                for m in xrange(nawf):
                    for l in xrange(nawf):
                        # l+1,m+1 just to start from 1 not zero
                        f.write (('%3d %3d %3d %5d %5d %14f %14f \n') %(ix,iy,iz,l+1,m+1,HRs[l,m,i,j,k,0].real,HRs[l,m,i,j,k,0].imag))
