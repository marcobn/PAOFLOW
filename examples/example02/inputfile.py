############################INPUT variables########################

import numpy as np

# Control 

fpath = './al.save'
verbose = False
non_ortho  = False
write2file = False
use_cuda = True
shift_type = 1
shift      = 'auto'
pthr       = 0.97
npool = 8

# Calculations 

# Compare PAO bands with original DFT bands on the original MP mesh
do_comparison = False

# Bands interpolation along a path from a 1D string of k points
onedim = False
# Bands interpolation on a path from the original MP mesh 
do_bands = False
ibrav = 2
dkres = 0.01

# Hamiltonian interpolation on finer MP mesh
double_grid = True
nfft1 = 24
nfft2 = 24
nfft3 = 24

# DOS(PDOS) calculation
do_dos = True
do_pdos = False
emin = -12.
emax = 3
delta = 0.1

smearing = 'gauss'

# Tensor components
# Dielectric function
d_tensor = np.array([[0,0]])
# Boltzmann transport
t_tensor = np.array([[0,0]])

# Set temperature in eV
temp = 0.025852  # room temperature

# Boltzmann transport calculation
Boltzmann = True

# Dielectric function calculation
epsilon = True
metal = True
epsmin = 0.05
epsmax = 6.0
ne = 500
