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
import PySide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#units
Ry2eV   = 13.60569193

def plot_compare_TB_DFT_eigs(Hks,Sks,my_eigsmat,read_S):

    nawf,nawf,nkpnts,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nbnds_tb,nkpnts,nspin))

    ispin = 0 #plots only 1 spin channel
    #for ispin in range(nspin):
    for ik in range(nkpnts):
        if read_S:
            eigval,_ = LA.eigh(Hks[:,:,ik,ispin],Sks[:,:,ik])
        else:
            eigval,_ = LAN.eigh(Hks[:,:,ik,ispin],UPLO='U')
        E_k[:,ik,ispin] = np.sort(np.real(eigval))

    fig=plt.figure
    nbnds_dft,_,_=my_eigsmat.shape
    for i in range(nbnds_dft):
        #print("{0:d}".format(i))
        yy = my_eigsmat[i,:,ispin]
        if i==0:
            plt.plot(yy,'ok',markersize=3,markeredgecolor='lime',markerfacecolor='lime',label='DFT')
        else:
            plt.plot(yy,'ok',markersize=3,markeredgecolor='lime',markerfacecolor='lime')

    for i in range(nbnds_tb):
        yy = E_k[i,:,ispin]
        if i==0:
            plt.plot(yy,'ok',markersize=2,markeredgecolor='None',label='TB')
        else:
            plt.plot(yy,'ok',markersize=2,markeredgecolor='None')

    plt.xlabel('k-points')
    plt.ylabel('Energy - E$_F$ (eV)')
    plt.legend()
    plt.title('Comparison of TB vs. DFT eigenvalues')
    plt.savefig('comparison.pdf',format='pdf')
    return()
