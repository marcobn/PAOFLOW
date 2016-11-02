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

def kpnts_interpolation_mesh():
	# To be made general reading boundary points from input. For now:
	# define k vectors (L-Gamma-X-K-Gamma) by hand

	# L - Gamma
	nk=60
	kx=np.linspace(-0.5, 0.0, nk)
	ky=np.linspace(0.5, 0.0, nk)
	kz=np.linspace(0.5, 0.0, nk)
	k1=np.array([kx,ky,kz])
	# Gamma - X
	kx=np.linspace(0.0, -0.75, nk)
	ky=np.linspace(0.0, 0.75, nk)
	kz=np.zeros(nk)
	k2=np.array([kx,ky,kz])
	# X - K 
	kx=np.linspace(-0.75, -1.0, nk)
	ky=np.linspace(0.75,  0.0, nk)
	kz=np.zeros(nk)
	k3=np.array([kx,ky,kz])
	# K - Gamma
	kx=np.linspace(-1.0, 0.0, nk)
	ky=np.zeros(nk)
	kz=np.zeros(nk)
	k4=np.array([kx,ky,kz])

	k=np.concatenate((k1,k2,k3,k4),1)

	# Define path for plotting
	nkpi = 0
	kmod = np.zeros(k.shape[1],dtype=float)
	for ik in range(k.shape[1]):
        	if ik < nk:
                	kmod[nkpi]=-np.sqrt(np.absolute(np.dot(k[:,ik],k[:,ik])))
        	elif ik >= nk and ik < 2*nk:
                	kmod[nkpi]=np.sqrt(np.absolute(np.dot(k[:,ik],k[:,ik])))
        	elif ik >= 2*nk:
                	kmod[nkpi]=1+np.sqrt(2)-np.sqrt(np.absolute(np.dot(k[:,ik],k[:,ik])))
		nkpi += 1
	return (k,kmod,nkpi)

