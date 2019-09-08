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




def FermiDirac( eig, ene, degauss):
    import numpy as np

    x = (ene-eig)/degauss
    wgauss = 1.0 / (1.0 + np.exp(x))
    return wgauss


def ColdSmearing( eig, ene, degauss):
    import numpy as np
    from scipy.special import erf

    x = (ene-eig)/degauss
    xp = x - 1.0/np.sqrt(2.0)
    maxarg = 200.0
    arg = np.minimum(xp**2, maxarg)
    wgauss = 0.5*erf(xp) + 1.0/np.sqrt(2.0*np.pi)*np.exp(-arg) + 0.5
    return wgauss

def Gaussian( eig, ene, degauss):
    import numpy as np
    from scipy.special import erf
    
    x = (ene-eig)/degauss
    wgauss = (1 + erf(x))/2.
    return wgauss
        
def MethPaxton( eig, ene, degauss):
    import numpy as np
    from scipy.special import erf

    x = (ene-eig)/degauss
    wgauss = (1 + erf(x))/2.
    hd = 0.0
    maxarg = 200.0
    arg = np.minnimum(x**2, maxarg)
    hp = np.exp(-arg)
    ni = 0
    a = 1.0/np.sqrt(np.pi)
    hd = 2.0 * x * hp
    a = -a / 4.0
    wgauss = wgauss - a*hd
    return wgauss

