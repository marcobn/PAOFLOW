#
# AFLOWpi_TB
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
from scipy import linalg as LA
from numpy import linalg as LAN
import numpy as np
import os

def write_TB_eigs(Hks,Sks,read_S,ispin):

    nawf,nawf,nkpnts,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nbnds_tb,nkpnts,nspin))

    for ik in range(nkpnts):
        if read_S:
            eigval,_ = LA.eigh(Hks[:,:,ik,ispin],Sks[:,:,ik])
        else:
            eigval,_ = LAN.eigh(Hks[:,:,ik,ispin],UPLO='U')
        E_k[:,ik,ispin] = np.sort(np.real(eigval))

    f=open('bands_'+str(ispin)+'.dat','w')
    for ik in range(nkpnts):
        for nb in range(nawf):
            f.write('%3d  %.5f \n' %(ik,E_k[nb,ik,ispin]))
    return()
