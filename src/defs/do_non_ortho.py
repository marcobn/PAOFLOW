#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital basis (PAO)
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
import cmath
import sys
from numpy import linalg as LAN
from scipy import linalg as LA


sys.path.append('./')

def do_non_ortho(Hks,Sks):
    # Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
    # Hks -> Sks^(1/2)*Hks*Sks^(1/2)+

    if len(Hks.shape) > 4:

        nawf = Hks.shape[0]
        nk1 = Hks.shape[2]
        nk2 = Hks.shape[3]
        nk3 = Hks.shape[4]
        nspin = Hks.shape[5]
        aux = np.zeros((nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
        saux = np.zeros((nawf,nawf,nk1*nk2*nk3),dtype=complex)
        idk = np.zeros((nk1,nk2,nk3),dtype=int)
        nkpnts = 0
        for i in xrange(nk1):
            for j in xrange(nk2):
                for k in xrange(nk3):
                    aux[:,:,nkpnts,:] = Hks[:,:,i,j,k,:]
                    saux[:,:,nkpnts] = Sks[:,:,i,j,k]
                    idk[i,j,k] = nkpnts
                    nkpnts += 1

        S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        for ik in xrange(nkpnts):
            w, v = LAN.eigh(saux[:,:,ik],UPLO='U')
            #w, v = LA.eigh(saux[:,:,ik])
            w = np.sqrt(w)
            for j in xrange(nawf):
                for i in xrange(nawf):
                    S2k[i,j,ik] = v[i,j]*w[j]
            S2k[:,:,ik] = S2k[:,:,ik].dot(np.conj(v).T)

        Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
        for ispin in xrange(nspin):
            for ik in xrange(nkpnts):
                Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(aux[:,:,ik,ispin]).dot(S2k[:,:,ik])

        aux = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
        for i in xrange(nk1):
            for j in xrange(nk2):
                for k in xrange(nk3):
                    aux[:,:,i,j,k,:]=Hks_no[:,:,idk[i,j,k],:]
        return(aux)

    else:

        nawf = Hks.shape[0]
        nkpnts = Hks.shape[2]
        nspin = Hks.shape[3]
        S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        for ik in xrange(nkpnts):
            w, v = LAN.eigh(Sks[:,:,ik],UPLO='U')
            #w, v = LA.eigh(Sks[:,:,ik])
            w = np.sqrt(w)
            for j in xrange(nawf):
                for i in xrange(nawf):
                    S2k[i,j,ik] = v[i,j]*w[j]
            S2k[:,:,ik] = S2k[:,:,ik].dot(np.conj(v).T)

        Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
        for ispin in xrange(nspin):
            for ik in xrange(nkpnts):
                Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(Hks[:,:,ik,ispin]).dot(S2k[:,:,ik])
        return(Hks_no)
