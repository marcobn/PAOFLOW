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


def gaussian ( eig, ene, delta ):
    import numpy as np

    # Gaussian Smearing
    return (np.exp(-((ene-eig)/delta)**2)/delta)/np.sqrt(np.pi)



def metpax ( eig, ene, delta ):
    import numpy as np
    from math import factorial
    from numpy.polynomial.hermite import hermval

    # Methfessel and Paxton smearing
    nh = 5
    coeff = np.zeros(2*nh)
    coeff[0] = 1.
    for n in range(2,2*nh,2):
        m = n/2
        coeff[n] = (-1.)**m/(factorial(m)*(4.0**m)*np.sqrt(np.pi))

    x = (ene-eig)/delta
    return hermval(x, coeff)*np.exp(-(x)**2)/(delta*np.sqrt(np.pi))

def intgaussian ( eig, ene, delta ):
    from scipy.special import erf

    # integral of the gaussian function as approximation of the Fermi-Dirac distribution 
    return (1.-erf((eig-ene)/delta))/2.

def intmetpax ( eig, ene, delta ):
    import numpy as np
    from math import factorial
    from scipy.special import erf
    from numpy.polynomial.hermite import hermval

    # Methfessel and Paxton correction to the Fermi-Dirac distribution
    nh = 5
    coeff = np.zeros(2*nh)
    coeff[0] = 0.
    for n in range(2,2*nh,2):
        m = n/2
        coeff[n-1] = (-1.)**m/(factorial(m)*(4.0**m)*np.sqrt(np.pi))

    x = (eig-ene)/delta
    return (1.-erf(x))/2. + hermval(x, coeff)*np.exp(-(x**2))/np.sqrt(np.pi)
