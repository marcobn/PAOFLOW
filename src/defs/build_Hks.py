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
from scipy import linalg as LA
import numpy as np
import sys

def build_Hks(nawf,bnd,nbnds,nbnds_norm,nkpnts,nspin,shift,my_eigsmat,shift_type,U):
    Hks = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    for ispin in xrange(nspin):
        for ik in xrange(nkpnts):
            my_eigs=my_eigsmat[:,ik,ispin]
            #Building the Hamiltonian matrix
            E = np.diag(my_eigs)
            UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
            norms = 1/np.sqrt(np.real(np.sum(np.conj(UU)*UU,axis=0)))
            UU[:,:nbnds_norm] = UU[:,:nbnds_norm]*norms[:nbnds_norm]
            eta=shift
            # Choose only the eigenvalues that are below the energy shift
            bnd_ik=0
            for n in xrange(bnd):
                if my_eigs[n] <= eta:
                    bnd_ik += 1
            if bnd_ik == 0: sys.exit('no eigenvalues in selected energy range')
            ac = UU[:,:bnd_ik]  # filtering: bnd is defined by the projectabilities
            ee1 = E[:bnd_ik,:bnd_ik]
            #if bnd == nbnds:
            #    bd = np.zeros((nawf,1))
            #    ee2 = 0
            #else:
            #    bd = UU[:,bnd:nbnds]
            #    ee2= E[bnd:nbnds,bnd:nbnds]
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

            reordering = False
            if reordering:
                # Reordering of the Hamiltonian in 2D spinors for spin-orbit calculations - SYSTEM DEPENDENT!!!
                aux = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
                perm = np.array([0,1,2,3,5,6,4,7,9,10,8,11,14,15,13,16,12,17])
                i = np.argsort(perm)
                aux[:,:,ik,ispin] = Hks[:,i,ik,ispin]
                Hks[:,:,ik,ispin] = aux[i,:,ik,ispin]
    return(Hks)
