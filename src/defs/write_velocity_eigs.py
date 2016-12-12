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

def write_velocity_eigs(Hks,ipol,ispin,index):

    nawf,nawf,nkpnts,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nkpnts,nbnds_tb,nspin))
    v_k = np.zeros((nkpnts,nbnds_tb,nbnds_tb,nspin),dtype=complex)
    E_kaux = np.zeros((nbnds_tb,nkpnts,nspin))

    for ik in xrange(nkpnts):
        eigval,eigvec = LAN.eigh(Hks[:,:,ik,ispin],UPLO='U')
        E_k[ik,:,ispin] = np.real(eigval)
        v_k[ik,:,:,ispin] = eigvec
        for n in xrange(nawf):
            E_kaux[n,ik,ispin] = E_k[ik,index[n,ik,ispin],ispin]

    ipad = False
    if ipad:
        f=open('velocity_'+str(ipol)+'_'+str(ispin)+'.dat','w')
        for ik in xrange(nkpnts):
            for nb in xrange(nawf):
                f.write('%3d  %.5f \n' %(ik,E_kaux[nb,ik,ispin]))
        f.close()
    else:
        f=open('velocity_'+str(ipol)+'_'+str(ispin)+'.dat','w')
        for ik in xrange(nkpnts):
            s="%d\t"%ik
            for  j in E_kaux[:,ik,ispin]:s += "%3.5f\t"%j
            s+="\n"
            f.write(s)
        f.close()

    return(E_k,v_k)
