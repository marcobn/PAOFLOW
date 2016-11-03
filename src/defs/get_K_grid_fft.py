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

def get_K_grid_fft(nk1,nk2,nk3,b_vectors, print_kgrid):
	nktot = nk1*nk2*nk3
	Kint = np.zeros((nktot,3),dtype=float)
	K_wght = np.ones((nktot),dtype=float)
	K_wght /= nktot
	idk = np.zeros((nk1,nk2,nk3),dtype=int)

	for i in range(nk1):
		for j in range(nk2):
        	        for k in range(nk3):
                	        n = k + j*nk3 + i*nk2*nk3
                        	Rx = float(i)/float(nk1)
                        	Ry = float(j)/float(nk1)
                        	Rz = float(k)/float(nk1)
                        	if Rx >= 0.5: Rx=Rx-1.0
                        	if Ry >= 0.5: Ry=Ry-1.0
                        	if Rz >= 0.5: Rz=Rz-1.0
                        	Rx -= int(Rx)
                        	Ry -= int(Ry)
                        	Rz -= int(Rz)
				idk[i,j,k]=n
	                       	Kint[n,:] = Rx*b_vectors[0,:]+Ry*b_vectors[1,:]+Rz*b_vectors[2,:]
                      
	return(Kint,K_wght,nktot,idk)

