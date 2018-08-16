#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys

sys.path.append('./')

from write_PAO_eigs import *


# THIS MODULE **CANNOT** BE FUNCTOINAL
def do_bands_calc_1D(Hkaux,inputpath):
    # FFT interpolation along a single directions in the BZ

    nawf,_,nk1,nk2,nk3,nspin = Hksp.shape
    nktot = nk1*nk2*nk3

    Hkaux  = np.zeros((nawf,nawf,nktot,nspin), dtype=complex)
    for ispin in range(nspin):
        for i in range(nawf):
            for j in range(nawf):
                nL=0
                for ik1 in range(nk1):
                    for ik2 in range(nk2):
                        for ik3 in range(nk3):
                            Hkaux[i,j,nL,ispin] = Hksp[i,j,ik1,ik2,ik3,ispin]
                            nL += 1

    # zero padding interpolation
    # k to R
    npad = 500
    HRaux  = np.zeros((nawf,nawf,nktot,nspin), dtype=complex)
    for ispin in range(nspin):
        for i in range(nawf):
            for j in range(nawf):
                HRaux[i,j,:,ispin] = FFT.ifft(Hkaux[i,j,:,ispin])

    Hkaux = None
    Hkaux  = np.zeros((nawf,nawf,npad+nktot,nspin),dtype=complex)
    HRauxp  = np.zeros((nawf,nawf,npad+nktot,nspin),dtype=complex)

    for ispin in range(nspin):
        for i in range(nawf):
            for j in range(nawf):
                HRauxp[i,j,:(nktot/2),ispin] = HRaux[i,j,:(nktot/2),ispin]
                HRauxp[i,j,(npad+nktot/2):,ispin] = HRaux[i,j,(nktot/2):,ispin]
                Hkaux[i,j,:,ispin] = FFT.fft(HRauxp[i,j,:,ispin])

    # Print PAO eigenvalues on interpolated mesh
    for ispin in range(nspin):
        write_PAO_eigs(Hkaux,ispin,inputpath)
