"""Physical constants and unit conversions of PAOFLOW: CODATA 2018.

The base values are those of Quantum ESPRESSO 7.x (``Modules/constants.f90``), so that energies,
lengths and masses read from a QE run convert with the constants QE used to write them. Every
derived conversion is computed from the base values below, not typed in. Until Nov 2026 the module
held older values (e.g. ``AU`` = 27.211652 eV, ``ELECTRONVOLT_SI`` = 1.6021892e-19 J), a few parts
in 1e5 to 1e4 away; results that use them move by that much.
"""

import math

# --- CODATA 2018 base values -------------------------------------------------------------------
H_PLANCK_SI = 6.62607015e-34  # J s (exact)
H_OVER_TPI = 1.054571817e-34  # J s, hbar
ELECTRONVOLT_SI = 1.602176634e-19  # J (exact); also the elementary charge in C
K_BOLTZMAN_SI = 1.380649e-23  # J K^-1 (exact)
ELECTRONMASS_SI = 9.1093837015e-31  # kg
HARTREE_SI = 4.3597447222071e-18  # J
BOHR_RADIUS_SI = 0.529177210903e-10  # m
UMA_SI = 1.66053906660e-27  # kg, atomic mass unit
SPEED_OF_LIGHT = 299792458.0  # m / s (exact)

# --- Derived ---------------------------------------------------------------------------------
AU = HARTREE_SI / ELECTRONVOLT_SI  # eV per Hartree, 27.211386...
RYD = AU / 2.0  # eV per Rydberg
RYTOEV = RYD  # conversion from Ry to eV
EVTORY = 1.0 / RYTOEV  # conversion from eV to Ry
HBAR = H_OVER_TPI / ELECTRONVOLT_SI  # eV s

K_BOLTZMAN_AU = K_BOLTZMAN_SI / HARTREE_SI  # Hartree K^-1
K_BOLTZMAN_M1_AU = 1.0 / K_BOLTZMAN_AU  # Hartree^-1 K
FACTEM = K_BOLTZMAN_M1_AU  # Hartree^-1 K

BOHR_RADIUS_CM = BOHR_RADIUS_SI * 1.0e2  # cm
BOHR_RADIUS_ANGS = BOHR_RADIUS_SI * 1.0e10  # angstrom
ANGSTROM_AU = 1.0 / BOHR_RADIUS_ANGS  # bohr per angstrom

ELECTRONMASS_UMA = ELECTRONMASS_SI / UMA_SI  # electron mass in amu
UMA_AU = UMA_SI / ELECTRONMASS_SI  # amu in electron masses (Hartree atomic units)
SCMASS = UMA_AU  # amu to au
AMCONV = UMA_AU / 2.0  # amu in Rydberg atomic units of mass (2 m_e)

AU_SEC = H_OVER_TPI / HARTREE_SI  # s, Hartree atomic unit of time
AU_TERAHERTZ = AU_SEC * 1.0e12  # the atomic unit of time in ps (QE's AU_PS), as before
TERAHERTZ = AU_TERAHERTZ

DEBYE_SI = 1.0e-21 / SPEED_OF_LIGHT  # C m
DEBYE_AU = DEBYE_SI / (ELECTRONVOLT_SI * BOHR_RADIUS_SI)  # e bohr
# V/m: the Rydberg atomic unit of electric field (e = sqrt(2)), as before
AU_TO_VMM1 = HARTREE_SI / (ELECTRONVOLT_SI * BOHR_RADIUS_SI) / math.sqrt(2.0)
AU_TO_OHMCMM1 = ELECTRONVOLT_SI**2 / (H_OVER_TPI * BOHR_RADIUS_SI) / 1.0e2  # (ohm cm)^-1
AU_KB = HARTREE_SI / BOHR_RADIUS_SI**3 / 1.0e8  # kbar per Hartree/bohr^3
KB_AU = 1.0 / AU_KB  # au
UAKBAR = AU_KB / 2.0  # kbar per Ry/bohr^3

EVTOCM1 = ELECTRONVOLT_SI / (H_PLANCK_SI * SPEED_OF_LIGHT * 1.0e2)  # cm^-1 per eV
CM1TOEV = 1.0 / EVTOCM1  # eV per cm^-1
RYDCM1 = RYTOEV * EVTOCM1  # Rydberg to cm^-1

EPS0 = 1.0 / (4.0 * math.pi)  # vacuum dielectric constant in Ry
DEGTORAD = math.pi / 180.0  # Degrees to radians

E2 = 2.0  # e^2

# Logos
PAO = 'PAO'
p = '\u03c0'
pp = p.encode('utf8')
PAOPY = str(PAO) + str(pp)

C = '\u00a9'
CC = C.encode('utf8')

LL = ['x', 'y', 'z']

# SI constants of the non-linear optical responses (response/nonlinear_optics.py)
EPSILON0_SI = 8.8541878128e-12  # F/m, vacuum permittivity
ELECTRON_CHARGE_SI = ELECTRONVOLT_SI  # C
HBAR_SI = H_OVER_TPI  # J s
