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
from scipy import linalg as LA
import numpy as np
from numpy import linalg as LAN
import numpy.random as rd
import sys

def build_Hks(nawf,bnd,nkpnts,nspin,eta,my_eigsmat,shift_type,U):
    minimal = False
    Hksaux = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    if minimal:
        Hks = np.zeros((bnd,bnd,nkpnts,nspin),dtype=complex)
    else:
        Hks = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    for ik in range(nkpnts):
        for ispin in range(nspin):
            my_eigs=my_eigsmat[:,ik,ispin]
            #Building the Hamiltonian matrix
            E = np.diag(my_eigs)
            UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
            norms = 1/np.sqrt(np.real(np.sum(np.conj(UU)*UU,axis=0)))
            UU[:,:nawf] = UU[:,:nawf]*norms[:nawf]
            # Choose only the eigenvalues that are below the energy shift
            bnd_ik=0
            for n in range(bnd):
                if my_eigs[n] <= eta:
                    bnd_ik += 1
            if bnd_ik == 0: sys.exit('no eigenvalues in selected energy range')
            ac = UU[:,:bnd_ik]  # filtering: bnd is defined by the projectabilities
            ee1 = E[:bnd_ik,:bnd_ik]
            if shift_type ==0:
                #option 1 (PRB 2013)
                Hksaux[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(np.conj(ac).T))
            elif shift_type==1:
                #option 2 (PRB 2016)
                aux_p=LA.inv(np.dot(np.conj(ac).T,ac))
                Hksaux[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(aux_p).dot(np.conj(ac).T))
            elif shift_type==2:
                # no shift
                Hksaux[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T)
            else:
                sys.exit('shift_type not recognized')
            # Enforce Hermiticity (just in case...)
            Hksaux[:,:,ik,ispin] = 0.5*(Hksaux[:,:,ik,ispin] + np.conj(Hksaux[:,:,ik,ispin].T))

            if minimal:
                Sbd = np.zeros((nawf,nawf),dtype=complex)
                Sbdi = np.zeros((nawf,nawf),dtype=complex)
                S = sv = np.zeros((nawf,nawf),dtype=complex)
                e = se = np.zeros(nawf,dtype=float)

                e,S = LAN.eigh(Hksaux[:,:,ik,ispin])
                S11 = S[:bnd,:bnd] + 1.0*rd.random(bnd)/10000.
                S21 = S[:bnd,bnd:] + 1.0*rd.random(nawf-bnd)/10000.
                S12 = S21.T
                S22 = S[bnd:,bnd:] + 1.0*rd.random(nawf-bnd)/10000.
                S22 = S22 + S21.T.dot(np.dot(LA.inv(S11),S12.T))
                Sbd[:bnd,:bnd] = 0.5*(S11+np.conj(S11.T))
                Sbd[bnd:,bnd:] = 0.5*(S22+np.conj(S22.T))
                Sbdi = LA.inv(np.dot(Sbd,np.conj(Sbd.T)))
                se,sv = LAN.eigh(Sbdi)
                se = np.sqrt(se+0.0j)*np.identity(nawf,dtype=complex)
                Sbdi = sv.dot(se).dot(np.conj(sv).T)
                T = S.dot(np.conj(Sbd.T)).dot(Sbdi)
                Hbd = np.conj(T.T).dot(np.dot(Hksaux[:,:,ik,ispin],T))
                Hks[:,:,ik,ispin] = 0.5*(Hbd[:bnd,:bnd]+np.conj(Hbd[:bnd,:bnd].T))
            else:
                Hks = Hksaux

    return(Hks)
