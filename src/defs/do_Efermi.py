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
from .smearing import intmetpax

def E_Fermi(data_controller,ham='Hks'):
	arry, attr = data_controller.data_dicts()
	
	# Calculate the Fermi energy using a braketing algorithm
	
	nawf,_,nk1,nk2,nk3,nspin = arry[ham].shape
	nktot = nk1*nk2*nk3
	Hksp = arry[ham].reshape((nawf,nawf,nktot,nspin),order='C')
	eig = np.zeros((Hksp.shape[1],Hksp.shape[2],Hksp.shape[3]))
	
	dinds = np.diag_indices(Hksp.shape[1])
	
	for ispin in range(nspin):
		for kp in range(nktot):
			eig[:,kp,ispin] = np.linalg.eigvalsh(Hksp[:,:,kp,ispin])
			
	Elw = 1.0e+8
	Eup =-1.0e+8
	degauss = 0.01
	nbnd = attr['bnd']
	nelec = attr['nelec']
	eps = 1.0e-10
	
	for ispin in range(nspin):
		for kp in range(nktot):
			Elw = min(Elw,eig[0,kp,ispin])
			Eup = max(Elw,eig[nbnd,kp,ispin])
	Eup = Eup + 2 * degauss
	Elw = Elw - 2 * degauss
	
	# bisection method
	if attr['dftSO']:
		fac = 1
	else:
		fac = 2
	sumkup = fac*np.sum(intmetpax(eig[:nbnd,:,:],Eup,degauss))/nktot
	sumklw = fac*np.sum(intmetpax(eig[:nbnd,:,:],Elw,degauss))/nktot
	if (sumkup - nelec) < -eps or (sumklw - nelec) > eps:
		print('Error: cannot bracket Ef')
		
	maxiter = 100
	for i in range(maxiter):
		Ef = (Eup + Elw)/2
		sumkmid = fac*np.sum(intmetpax(eig[:,:,:],Ef,degauss))/nktot
		if np.abs( sumkmid-nelec ) < eps:
#			print(' Fermi Energy = ',Ef)
			break
		elif sumkmid-nelec < -eps:
			Elw = Ef
		else:
			Eup = Ef
#	attr['Efermi'] = Ef
	
	# rescale Hksp
	Hksp[dinds[0],dinds[1]] -= Ef
	arry[ham] = Hksp.reshape((nawf,nawf,nk1,nk2,nk3,nspin),order='C')
	return(Ef)
	