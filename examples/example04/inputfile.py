############################
#
#  inputfile.py for PAOpy.py
#
############################

import numpy as np

# Control 

fpath = './fe.save/'
restart = False
verbose = False
non_ortho  = False
write2file = False
shift_type = 1
shift      = 'auto'
pthr       = 0.95
npool = 8

# Calculations 

# Compare PAO bands with original DFT bands on the original MP mesh
do_comparison = False

# Shell order and degeneracy for SO (order must be the same as in the output of projwfc.x)
sh = [0,1,2]    # order of shells with l angular momentum
nl = [1,1,1]    # multiplicity of each l shell

# Bands interpolation on a path from the original MP mesh 
do_bands = True
ibrav = 3
nk = 2000
# Band topology
band_topology = True
ipol = 1
jpol = 2
spol = 2

# Hamiltonian interpolation on finer MP mesh
double_grid = True
nfft1 = 24
nfft2 = 24
nfft3 = 24

# DOS(PDOS) calculation
do_dos = True
do_pdos = False
emin = -10.
emax = 2
delta = 0.2

# Smearing
smearing = 'gauss'

# Tensor components
# Berry curvature tensor
a_tensor = np.array([[0,1]])

# Berry curvature and AHC
Berry = True
ac_cond_Berry = True
eminAH = -8.0
emaxAH = 4.0
