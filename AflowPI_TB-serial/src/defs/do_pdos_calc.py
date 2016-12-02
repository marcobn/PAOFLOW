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

from calc_TB_eigvecs import calc_TB_eigvecs
 
def do_pdos_calc(Hksp,Sksp,read_S,shift,delta):
    # DOS calculation with gaussian smearing
    print('... computing PDOS')
    nawf,nawf,nk1,nk2,nk3,nspin = Hksp.shape
    eig,vec,ndos,nawf = calc_TB_eigvecs(Hksp,Sksp,read_S)
    emin = np.min(eig)-1.0
    emax = np.max(eig)-shift/2.0
    de = (emax-emin)/50
    ene = np.arange(emin,emax,de,dtype=float)
    pdos = np.zeros((nawf,ene.size),dtype=float)
#!    pdosvec = np.zeros((eig.size),dtype=float)


    for ne in range(ene.size):
        count = 0
        print ne
        for m in range(nawf):
            for k in range(nk1*nk2*nk3):
                pdos[m,ne] += 1.0/np.sqrt(np.pi)*np.exp(-((ene[ne]-eig[count])/delta)**2)/delta*np.real(np.matrix(vec[count])*np.matrix.getH(np.matrix(vec[count]))) 
                count += 1
            
    for m in range(nawf):            
        f=open('pdos.{0}.dat'.format(m),'w')
        for ne in range(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],pdos[m,ne]))
        f.close()     

           
    return()
