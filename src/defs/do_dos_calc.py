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
import sys, time

sys.path.append('./')

from calc_TB_eigs import calc_TB_eigs
 
def do_dos_calc(Hksp,Sksp,read_S,shift,delta):
	# DOS calculation with gaussian smearing

	eig,ndos = calc_TB_eigs(Hksp,Sksp,read_S)
	emin = np.min(eig)-1.0
	emax = np.max(eig)-shift/2.0
	de = (emax-emin)/1000
	ene = np.arange(emin,emax,de,dtype=float)
	dos = np.zeros((ene.size),dtype=float)
	dosvec = np.zeros((eig.size),dtype=float)

	for ne in range(ene.size):
		dosvec = 1.0/np.sqrt(np.pi)*np.exp(-((ene[ne]-eig)/delta)**2)/delta
		dos[ne] = np.sum(dosvec)

	f=open('dos.dat','w')
        for ne in range(ene.size):
                f.write('%.5f  %.5f \n' %(ene[ne],dos[ne]))
	f.close()

	return()
