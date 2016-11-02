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
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys

sys.path.append('./')

from write_TB_eigs import write_TB_eigs
from kpnts_interpolation_mesh import kpnts_interpolation_mesh
 
def do_bands_calc(HRaux,SRaux,R_wght,R,idx,read_S):
	# Compute bands on a selected path in the BZ
	print('... computing bands')

	# Define k-point mesh for bands interpolation
	kq,kmod,nkpi = kpnts_interpolation_mesh()

	nawf = HRaux.shape[0]
	nk1 = HRaux.shape[2]
	nk2 = HRaux.shape[3]
	nk3 = HRaux.shape[4]
	nspin = HRaux.shape[5]
	Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)
	Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
	for ispin in range(nspin):
        	for ik in range(nkpi):
			for i in range(nk1):
				for j in range(nk2):
					for k in range(nk3):
                      				phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                       				Hks_int[:,:,ik,ispin] += HRaux[:,:,i,j,k,ispin]*phase
                       				if read_S and ispin == 0:
                               				Sks_int[:,:,ik] += SRaux[:,:,nr]*phase
	
	write_TB_eigs(Hks_int,Sks_int,read_S)
	return()

