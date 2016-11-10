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
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 

import numpy as np

def get_R_grid_regular(nk1,nk2,nk3,a_vectors):

    # Generate a regular grid in real space (crystal coordinates)
    # This is the algorithm in WanT
    nr = 0
    nrtot = 100000
    R = np.zeros((nrtot,3),dtype=float)
    R_wght = np.zeros((nrtot),dtype=float)
    for i in range(1,nk1+1):
        for j in range(1,nk2+1):
            for k in range(1,nk3+1):
                ri=i-((nk1+1)/2)
                rj=j-((nk2+1)/2)
                rk=k-((nk3+1)/2)
                R[nr,:] = ri*a_vectors[0,:]+rj*a_vectors[1,:]+rk*a_vectors[2,:]
                R_wght[nr] = 1.0
                nr += 1
    nrtot = nr-1
    counter = nr-1
    # Check that -R is always present
    for ir in range(0,nrtot+1):
        found = False
        for ir2 in range(0,nrtot+1):
            test_opp = all(R[ir2,:] == -R[ir,:])
            if test_opp == True:
                found = True
                break
        if found == False:
            counter += 1
            R[counter,:] = -R[ir,:]
            R_wght[counter] = R_wght[ir]/2.0
            R_wght[ir] = R_wght[ir]/2.0
    nrtot = counter+1
    Rreg = np.zeros((nrtot,3),dtype=float)
    Rreg_wght = np.zeros((nrtot),dtype=float)
    Rreg = R[:nrtot,:]
    Rreg_wght = R_wght[:nrtot]

    return(Rreg,Rreg_wght,nrtot)
