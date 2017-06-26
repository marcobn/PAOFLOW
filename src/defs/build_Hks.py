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
from scipy import linalg as LA
import numpy as np
from numpy import linalg as LAN
import sys

def build_Hks(nawf,bnd,nkpnts,nspin,eta,my_eigsmat,shift_type,U,Sks):
    Hks = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    for ik in xrange(nkpnts):
        for ispin in xrange(nspin):
            my_eigs=my_eigsmat[:,ik,ispin]
            #Building the Hamiltonian matrix
            E = np.diag(my_eigs)
            UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
            norms = 1/np.sqrt(np.real(np.sum(np.conj(UU)*UU,axis=0)))
            UU[:,:nawf] = UU[:,:nawf]*norms[:nawf]
            # Choose only the eigenvalues that are below the energy shift
            bnd_ik=0
            for n in xrange(bnd):
                if my_eigs[n] <= eta:
                    bnd_ik += 1
            if bnd_ik == 0: sys.exit('no eigenvalues in selected energy range')
            ac = UU[:,:bnd_ik]  # filtering: bnd is defined by the projectabilities
            ee1 = E[:bnd_ik,:bnd_ik]
            if shift_type ==0:
                #option 1 (PRB 2013)
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(np.conj(ac).T))
            elif shift_type==1:
                #option 2 (PRB 2016)
                aux_p=LA.inv(np.dot(np.conj(ac).T,ac))
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(aux_p).dot(np.conj(ac).T))
            elif shift_type==2:
                # no shift
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T)
            else:
                sys.exit('shift_type not recognized')
            # Enforce Hermiticity (just in case...)
            Hks[:,:,ik,ispin] = 0.5*(Hks[:,:,ik,ispin] + np.conj(Hks[:,:,ik,ispin].T))

        # This is needed for consistency of the ordering of the matrix elements (see "transposition" above)
        # Important in ACBN0 file writing
        Sks[:,:,ik] = Sks[:,:,ik].T

    return(Hks,Sks)
