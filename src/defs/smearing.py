# 
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
import numpy.polynomial.hermite as HERMITE
import scipy.special as SPECIAL
import math, cmath
import sys, time


one_over_sqrt_pi = 1.0/np.sqrt(np.pi)



def gaussian(eig,ene,delta):

    # gaussian smearing
    return one_over_sqrt_pi*(np.exp(-((ene-eig)/delta)**2)/delta)

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
