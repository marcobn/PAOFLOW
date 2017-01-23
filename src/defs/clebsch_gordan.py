#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
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

import cmath
import sys

import numpy as np
from numpy import linalg as LAN
from scipy import linalg as LA
import matplotlib.pyplot as plt


def spinor(l,j,m,spin):
# This function calculates the numerical Clebsch-Gordan coefficients of a spinor
# with orbital angular momentum l, total angular momentum j, 
# projection along z of the total angular momentum m+-1/2. Spin selects
# the up (spin=0) or down (spin=1) coefficient.

    if spin != 0 and spin != 1: sys.exit('spinor - spin direction unknown')
    if m < -l-1 or m > l: sys.exit('spinor - m not allowed')

    denom=1./(2.*l+1.)
    if (abs(j-l-0.5) < 1.e-8):
        if spin == 0: spinor = np.sqrt((l+m+1.)*denom)
        if spin == 1: spinor = np.sqrt((l-m)*denom)
    elif abs(j-l+0.5) < 1.e-8:
        if m < -l+1:
            spinor=0.
        else:
            if spin == 0: spinor = np.sqrt((l-m+1.)*denom)
            if spin == 1: spinor = -np.sqrt((l+m)*denom)

    else:
        sys.exit('spinor - j and l not compatible')

    return(spinor)

def clebsch_gordan(nawf,nl,spol):
    #
    # Transformation matrices from the | l m s s_z > basis to the
    # | j mj l s > basis in the l-subspace
    #
    l = 0
    Ul0 = np.zeros((2*(2*l+1),2*(2*l+1)),dtype=float)
    Ul0[0,1] = 1
    Ul0[1,0] = 1

    l = 1
    Ul1 = np.zeros((2*(2*l+1),2*(2*l+1)),dtype=float)
    j = l - 0.5
    for m1 in range(1,2*l+1):
        m = m1 - l
        Ul1[m1-1,2*(m1-1)+1-1] = spinor(l,j,m,0)
        Ul1[m1-1,2*(m1-1)+4-1] = spinor(l,j,m,1)
    j = l + 0.5
    for m1 in xrange (1,2*l+2+1):
        m = m1 - l - 2
        if (m1 == 1):
           Ul1[m1+2*l-1,2*(m1-1)+2-1] = spinor(l,j,m,1)
        elif (m1==2*l+2):
           Ul1[m1+2*l-1,2*(m1-1)-1-1] = spinor(l,j,m,0)
        else:
           Ul1[m1+2*l-1,2*(m1-1)-1-1] = spinor(l,j,m,0)
           Ul1[m1+2*l-1,2*(m1-1)+2-1] = spinor(l,j,m,1)

    l = 2
    Ul2 = np.zeros((2*(2*l+1),2*(2*l+1)),dtype=float)
    j = l - 0.5
    for m1 in range(1,2*l+1):
        m = m1 - l
        Ul2[m1-1,2*(m1-1)+1-1] = spinor(l,j,m,0)
        Ul2[m1-1,2*(m1-1)+4-1] = spinor(l,j,m,1)
    j = l + 0.5
    for m1 in xrange (1,2*l+2+1):
        m = m1 - l - 2
        if (m1 == 1):
           Ul2[m1+2*l-1,2*(m1-1)+2-1] = spinor(l,j,m,1)
        elif (m1==2*l+2):
           Ul2[m1+2*l-1,2*(m1-1)-1-1] = spinor(l,j,m,0)
        else:
           Ul2[m1+2*l-1,2*(m1-1)-1-1] = spinor(l,j,m,0)
           Ul2[m1+2*l-1,2*(m1-1)+2-1] = spinor(l,j,m,1)

    l = 3
    Ul3 = np.zeros((2*(2*l+1),2*(2*l+1)),dtype=float)
    j = l - 0.5
    for m1 in range(1,2*l+1):
        m = m1 - l
        Ul3[m1-1,2*(m1-1)+1-1] = spinor(l,j,m,0)
        Ul3[m1-1,2*(m1-1)+4-1] = spinor(l,j,m,1)
    j = l + 0.5
    for m1 in xrange (1,2*l+2+1):
        m = m1 - l - 2
        if (m1 == 1):
           Ul3[m1+2*l-1,2*(m1-1)+2-1] = spinor(l,j,m,1)
        elif (m1==2*l+2):
           Ul3[m1+2*l-1,2*(m1-1)-1-1] = spinor(l,j,m,0)
        else:
           Ul3[m1+2*l-1,2*(m1-1)-1-1] = spinor(l,j,m,0)
           Ul3[m1+2*l-1,2*(m1-1)+2-1] = spinor(l,j,m,1)

    Ul = [Ul0,Ul1,Ul2,Ul3]

    # Build the full transformation matrix

    occ = [2,6,10,14]

    ntot = np.dot(nl,occ)
    if ntot != nawf: sys.exit('wrong number of shells in reading')
    Tn = np.zeros((ntot,ntot),dtype=float)

    n = 0
    for l in range(4):
        for i in range(nl[l]):
            Tn[n:n+occ[l],n:n+occ[l]] = Ul[l]
            n += occ[l]

    # Pauli matrices (x,y,z) 
    sP=0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])

    # Spin operator matrix  in the basis of |l,m,s,s_z>
    Sl = np.zeros((nawf,nawf),dtype=complex)
    for i in xrange(0,nawf,2):
        Sl[i,i] = sP[spol][0,0]
        Sl[i,i+1] = sP[spol][0,1]
        Sl[i+1,i] = sP[spol][1,0]
        Sl[i+1,i+1] = sP[spol][1,1]

    # Spin operator matrix  in the basis of |j,m_j,l,s>
    Sj = np.zeros((nawf,nawf),dtype=complex)
    Sj = np.dot(Tn,Sl).dot(Tn.T)

    return(Sj)
