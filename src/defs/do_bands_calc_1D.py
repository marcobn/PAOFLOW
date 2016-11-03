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

def do_bands_calc_1D(Hkaux,Skaux,read_S):
	# FFT interpolation along a single directions in the BZ

	nawf = Hksp.shape[0]
	nk1 = Hksp.shape[2]
	nk2 = Hksp.shape[3]
	nk3 = Hksp.shape[4]
	nspin = Hksp.shape[5]
	
	# Count points along symmetry direction
	nL = 0
	for ik1 in range(nk1):
		for ik2 in range(nk2):
			for ik3 in range(nk3):
				nL += 1
	
	Hkaux  = np.zeros((nawf,nawf,nL,nspin),dtype=complex)
	Skaux  = np.zeros((nawf,nawf,nL),dtype=complex)
	for ispin in range(nspin):
        	for i in range(nawf):
                	for j in range(nawf):
				nL=0
				for ik1 in range(nk1):
					for ik2 in range(nk2):
						for ik3 in range(nk3):
							Hkaux[i,j,nL,ispin]=Hksp[i,j,ik1,ik2,ik3,ispin]	
							if (read_S and ispin == 0):
								Skaux[i,j,nL] = Sksp[i,j,ik1,ik2,ik3]
							nL += 1

	# zero padding interpolation
	# k to R
	npad = 500
	HRaux  = np.zeros((nawf,nawf,nL,nspin),dtype=complex)
	SRaux  = np.zeros((nawf,nawf,nL),dtype=complex)
	for ispin in range(nspin):
		for i in range(nawf):
			for j in range(nawf):
				HRaux[i,j,:,ispin] = FFT.ifft(Hkaux[i,j,:,ispin])
				if read_S and ispin == 0:
					SRaux[i,j,:] = FFT.ifft(Skaux[i,j,:])

	Hkaux = None
	Skaux = None
	Hkaux  = np.zeros((nawf,nawf,npad+nL,nspin),dtype=complex)
	Skaux  = np.zeros((nawf,nawf,npad+nL),dtype=complex)
	HRauxp  = np.zeros((nawf,nawf,npad+nL,nspin),dtype=complex)
	SRauxp  = np.zeros((nawf,nawf,npad+nL),dtype=complex)

	for ispin in range(nspin):
        	for i in range(nawf):
                	for j in range(nawf):
				HRauxp[i,j,:(nL/2),ispin]=HRaux[i,j,:(nL/2),ispin]
				HRauxp[i,j,(npad+nL/2):,ispin]=HRaux[i,j,(nL/2):,ispin]
                        	Hkaux[i,j,:,ispin] = FFT.fft(HRauxp[i,j,:,ispin])
                        	if read_S and ispin == 0:
					SRauxp[i,j,:(nL/2)]=SRaux[i,j,:(nL/2)]
					SRauxp[i,j,(npad+nL/2):]=SRaux[i,j,(nL/2):]
                                	Skaux[i,j,:] = FFT.fft(SRauxp[i,j,:])


	# Print TB eigenvalues on interpolated mesh
	write_TB_eigs(Hkaux,Skaux,read_S)

	return()

