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


# Physical constants
def K_BOLTZMAN_SI   (): return(1.38066e-23)       # J K^-1
def K_BOLTZMAN_AU   (): return(3.1667e-6)         # Hartree K^-1
def K_BOLTZMAN_M1_AU(): return(315795.26e0)       # Hartree^-1 K
def FACTEM          (): return(315795.26e0)       # Hartree^-1 K
def H_OVER_TPI      (): return(1.054571e-34)      # J sec

def BOHR_RADIUS_SI  (): return(0.529177e-10)      # m
def BOHR_RADIUS_CM  (): return(0.529177e-8)       # cm
def BOHR_RADIUS_ANGS(): return(0.529177e0)        # angstrom
def ELECTRONMASS_SI (): return(9.10953e-31)       # Kg
def ELECTRONMASS_UMA(): return(5.4858e-4)         # uma

def ELECTRONVOLT_SI (): return(1.6021892e-19)     # J
def UMA_SI          (): return(1.66057e-27)       # Kg
def DEBYE_SI        (): return(3.33564e-30)       # Coulomb meter
def DEBYE_AU        (): return(0.393427228)       # e * Bohr
def ANGSTROM_AU     (): return(1.889727e0)        # au
def AU_TO_OHMCMM1   (): return(46000.0e0)         # (ohm cm)^-1
def AU_KB           (): return(294210.0e0)        # Kbar
def KB_AU           (): return(1.0e0/294210.0e0)  # au
def AU              (): return(27.211652e0)       # eV
def RYD             (): return(13.605826e0)       # eV
def SCMASS          (): return(1822.89e0)         # uma to au
def UMA_AU          (): return(1822.89e0)         # au
def AU_TERAHERTZ    (): return(2.418e-5)          # THz
def TERAHERTZ       (): return(2.418e-5)          # from au to THz
def AU_SEC          (): return(2.4189e-17)        # sec

def EPS0            (): return(1.0/(4.0 * 3.14159265358979323846)) # vacuum dielectric constant in Ry
def RYTOEV          (): return(13.6058e0)    # conversion from Ry to eV
def EVTORY          (): return(1.0/13.6058e0)  # conversion from eV to Ry

def AMCONV          (): return(1.66042e-24/9.1095e-28*0.5e0) # mass conversion: a.m.u to a.u. (Ry)
def UAKBAR          (): return(147105.e0) # pressure conversion from Ry/(a.u)^3 to K

