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
import numpy as np
import cmath
import sys
from numpy import linalg as LAN
from scipy import linalg as LA

def do_ortho(Hks,Sks):
    # If orthogonality is required, we have to apply a basis change to Hks as
    # Hks -> Sks^(-1/2)*Hks*Sks^(-1/2)

    nawf,_,nkpnts,nspin = Hks.shape
    S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
    for ik in range(nkpnts):
        S2k[:,:,ik] = LAN.inv(LA.sqrtm(Sks[:,:,ik]))

    Hks_o = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    for ispin in range(nspin):
        for ik in range(nkpnts):
            Hks_o[:,:,ik,ispin] = np.dot(S2k[:,:,ik],Hks[:,:,ik,ispin]).dot(S2k[:,:,ik])

    return(Hks_o)
