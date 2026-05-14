"""Physical and SKEAF-specific constants.

Values copied verbatim from the Fortran source (skeaf_v1p3p0_r149.F90, lines ~50-58)
to preserve numerical equivalence with the reference implementation.
"""

import math

# Build identifier — kept identical to Fortran for output-file headers.
BUILD_NUMBER = "v1.3.0 r149 (Python port)"

PI = math.pi  # Fortran uses 3.1415926535897932D0 — the same as math.pi to ~16 digits.

# Bohr radius in angstroms (a.u. -> Å length conversion).
# BXSF reciprocal lattice vectors are stored in (a.u.)^-1.
CONV_AU_TO_ANG = 0.529177209

# Multiplicative constant hbar / (2 * pi * e) in units that take a Fermi-surface
# cross-sectional area in Å^-2 and return the dHvA frequency F in kT.
CONV_FSAREA_TO_KT = 10.47576797

# Multiplicative constant hbar^2 / (2 * pi * m_e) in units that take dA/dE in
# (Å^-2 * Ryd^-1) and return the cyclotron effective mass m* / m_e (dimensionless).
CONV_FSDADE_TO_MSTAR = 0.089135845
