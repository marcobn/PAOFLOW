############################
#
#  inputfile.py for PAOpy.py
#
############################

import numpy as np

# Control 

fpath = './pt.save'
verbose = False
non_ortho  = False
write2file = False
shift_type = 1
shift      = 'auto'
pthr       = 0.95
npool = 1

# Calculations 

# Compare PAO bands with original DFT bands on the original MP mesh
do_comparison = False

# Bands interpolation on a path from the original MP mesh 
do_bands = False
ibrav = 2
dkres = 0.01

# Hamiltonian interpolation on finer MP mesh
double_grid = True
nfft1 = 24
nfft2 = 24
nfft3 = 24

# Adaptive smearing
smearing = 'gauss'

# DOS(PDOS) calculation
do_dos = True
do_pdos = True
emin = -8.
emax = 4
delta = 0.2

# Tensor components
# Dielectric function
d_tensor = np.array([[0,0]])
# Boltzmann transport
t_tensor = np.array([[0,0]])

# Boltzmann transport calculation
Boltzmann = True

# Dielectric function calculation
epsilon = True
metal = True
epsmin=0.5
epsmax=10.0
ne = 500
