############################
#
#  inputfile.py for PAOpy.py
#
############################
import numpy as np

# Control 

fpath = './mgo.save/'
verbose = False
restart = False
non_ortho  = True
write2file = True
shift_type = 1
shift      = 'auto'
pthr       = 0.95
npool = 1

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
double_grid = False
nfft1 = 16
nfft2 = 16
nfft3 = 16

# DOS(PDOS) calculation
do_dos = False
do_pdos = False
emin = -12.
emax = 6.0
delta = 0.1

# Adaptive smearing
smearing = 'gauss' # allowed values: None, 'gauss', 'm-p'

# Tensor components
# Dielectric function
d_tensor = np.array([[0,0]])
# Boltzmann transport
t_tensor = np.array([[0,0]])

# Fermi surface
do_fermisurf = False
fermi_dw = -1.0
fermi_up = 1.0

# Boltzmann transport calculation
Boltzmann = False

# Set temperature in eV
temp = 0.025852  # room temperature

# Dielectric function calculation
epsilon =False
metal = False
epsmin=0.0
epsmax=6.0
ne = 500

# Critical points
critical_points = False
