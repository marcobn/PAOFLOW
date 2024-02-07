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
AU_TO_VMM1       = 36.3609*10e10      # V/m
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
PAO = 'PAO'
p = "\u03C0"
pp = p.encode('utf8')
PAOPY = str(PAO)+str(pp)

C = "\u00A9"
CC = C.encode('utf8')

LL = ['x','y','z']
