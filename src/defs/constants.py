#
# PAOpy
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO 
#
# Copyright (C  2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013 .
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016 .
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016 .
#
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016 .
# 


# Physical constants

K_BOLTZMAN_SI    = 1.38066e-23        # J K^-1
K_BOLTZMAN_AU    = 3.1667e-6          # Hartree K^-1
K_BOLTZMAN_M1_AU = 315795.26e0        # Hartree^-1 K
FACTEM           = 315795.26e0        # Hartree^-1 K
H_OVER_TPI       = 1.054571e-34       # J sec

BOHR_RADIUS_SI   = 0.529177e-10       # m
BOHR_RADIUS_CM   = 0.529177e-8        # cm
BOHR_RADIUS_ANGS = 0.529177e0         # angstrom
ELECTRONMASS_SI  = 9.10953e-31        # Kg
ELECTRONMASS_UMA = 5.4858e-4          # uma

ELECTRONVOLT_SI  = 1.6021892e-19      # J
UMA_SI           = 1.66057e-27        # Kg
DEBYE_SI         = 3.33564e-30        # Coulomb meter
DEBYE_AU         = 0.393427228        # e * Bohr
ANGSTROM_AU      = 1.889727e0         # au
AU_TO_OHMCMM1    = 46000.0e0          # (ohm cm)^-1
AU_KB            = 294210.0e0         # Kbar
KB_AU            = 1.0e0/294210.0e0   # au
AU               = 27.211652e0        # eV
RYD              = 13.605826e0        # eV
SCMASS           = 1822.89e0          # uma to au
UMA_AU           = 1822.89e0          # au
AU_TERAHERTZ     = 2.418e-5           # THz
TERAHERTZ        = 2.418e-5           # from au to THz
AU_SEC           = 2.4189e-17         # sec

EPS0             = 1.0/(4.0 * 3.14159265358979323846 ) # vacuum dielectric constant in Ry
RYTOEV           = 13.6058e0     # conversion from Ry to eV
EVTORY           = 1.0/13.6058e0   # conversion from eV to Ry
AMCONV           = 1.66042e-24/9.1095e-28*0.5e0  # mass conversion: a.m.u to a.u. (Ry)
UAKBAR           = 147105.e0  # pressure conversion from Ry/(a.u)^3 to K
DEGTORAD         = (3.14159265358979323846)/(180)  # Degrees to radians
CM1TOEV          = 1.23981e-4  # cm^-1 to eV
EVTOCM1          = 1.0/1.23981e-4  # eV to cm^-1

E2               = 2.0 # e^2

# Logos
AFLOW = 'AFLOW'
PAO = 'PAO'
p = u"\u03C0"
pp = p.encode('utf8')
TB = '(TB)'
AFLOWPITB = str(AFLOW)+str(pp)+str(TB)
PAOPY = str(PAO)+str(pp)

C = u"\u00A9"
CC = C.encode('utf8')
