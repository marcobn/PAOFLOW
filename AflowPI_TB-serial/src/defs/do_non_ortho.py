#
# AflowPI_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
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

sys.path.append('./')
 
def do_non_ortho(Hks,Sks):
	# Take care of non-orthogonality, if needed
	# Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
	# Hks -> Sks^(1/2)*Hks*Sks^(1/2)+

	nawf = Hks.shape[0]
	nkpnts = Hks.shape[2]
	nspin = Hks,shape[3]
	S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
	for ik in range(nkpnts):
		w, v = LAN.eigh(Sks[:,:,ik],UPLO='U')
		w = np.sqrt(w)
		S2k[:,:,ik] = v*w

	Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
	for ispin in range(nspin):
		for ik in range(nkpnts):
			Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(Hks[:,:,ik,ispin]).dot(np.conj(S2k[:,:,ik]).T)
	return(Hks_no)

