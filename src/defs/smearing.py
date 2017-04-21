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

import numpy as np
import numpy.polynomial.hermite as HERMITE
import scipy.special as SPECIAL
import math, cmath
import sys, time


def gaussian(eig,ene,delta):

    # gaussian smearing
    return (1.0/np.sqrt(np.pi)*np.exp(-((ene-eig)/delta)**2)/delta)

def metpax(eig,ene,delta):

    # Methfessel and Paxton smearing
    nh = 5
    coeff = np.zeros(2*nh)
    coeff[0] = 1.
    for n in xrange(2,2*nh,2):
        m = n/2
        coeff[n] = (-1.)**m/(math.factorial(m)*4.0**m*np.sqrt(np.pi))

    return (HERMITE.hermval((ene-eig)/delta,coeff)*np.exp(-((ene-eig)/delta)**2)/delta/np.sqrt(np.pi))

def intgaussian(eig,ene,delta):

    # integral of the gaussian function as approximation of the Fermi-Dirac distribution 
    return(0.5*(1-SPECIAL.erf((eig-ene)/delta)))

def intmetpax(eig,ene,delta):

    # Methfessel and Paxton correction to the Fermi-Dirac distribution
    nh = 5
    coeff = np.zeros(2*nh)
    coeff[0] = 0.
    for n in xrange(2,2*nh,2):
        m = n/2
        coeff[n-1] = (-1.)**m/(math.factorial(m)*4.0**m*np.sqrt(np.pi))

    return(0.5*(1-SPECIAL.erf((eig-ene)/delta)) + HERMITE.hermval((eig-ene)/delta,coeff)*np.exp(-((eig-ene)/delta)**2)/np.sqrt(np.pi))
