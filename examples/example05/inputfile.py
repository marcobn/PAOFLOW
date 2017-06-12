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
use_cuda = True
shift_type = 1
shift      = 'auto'
pthr       = 0.95
npool = 1

# Calculations 

# Compare PAO bands with original DFT bands on the original MP mesh
do_comparison = False

# Shell order and degeneracy for SO (order must be the same as in the output of projwfc.x)
sh = [0,1,2]    # order of shells with l angular momentum
nl = [1,1,1]    # multiplicity of each l shell

# Bands interpolation on a path from the original MP mesh 
do_bands = False
ibrav = 2
dkres = 0.01
# Band topology analysis 
band_topology = True
ipol = 0
jpol = 1
spol = 2

# Hamiltonian interpolation on finer MP mesh
double_grid = True
nfft1 = 24
nfft2 = 24
nfft3 = 24

smearing = 'gauss' #'gauss'

# DOS(PDOS) calculation
do_dos = True
do_pdos = False
emin = -8.
emax = 4
delta = 0.2

# Tensor components
# Spin Hall tensor
s_tensor = np.array([[0,1,2]])

# Spin Berry curvature and SHC
spin_Hall = True
ac_cond_spin = True
eminSH = -8.0
emaxSH = 4.0
writedata = False
