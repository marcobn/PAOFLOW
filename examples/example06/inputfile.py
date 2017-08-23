############################
#
#  inputfile.py for PAOpy.py
#
############################
import numpy as np

# Control 

fpath = './alp.save/'
verbose = False
restart = False
non_ortho  = False
write2file = False
shift_type = 1
shift      = 'auto'
pthr       = 0.95
npool = 1

# Calculations 

# Compare PAO bands with original DFT bands on the original MP mesh
do_comparison = False

# Dimensions of the atomic basis for each atom (order must be the same as in the output of projwfc.x)
naw=np.array([16,16]) # naw.shape[0] = natom

# ACBN0 U values for each orbtal
HubbardU = np.zeros(32,dtype=float)
HubbardU[1:4] = 0.01
HubbardU[17:21] = 2.31

# Bands interpolation along a path from a 1D string of k points
onedim = False
# Bands interpolation on a path from the original MP mesh 
do_bands = True
ibrav = 2
nk = 2000

# Hamiltonian interpolation on finer MP mesh
double_grid = True
nfft1 = 24
nfft2 = 24
nfft3 = 24

# DOS(PDOS) calculation
do_dos = True
do_pdos = False
emin = -8.
emax = 5.
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
do_spintexture = False

# Boltzmann transport calculation
Boltzmann = True

# Set temperature in eV
temp = 0.025852  # room temperature

# Dielectric function calculation
epsilon = True
metal = False
epsmin=0.0
epsmax=6.0
ne = 500

# Critical points
critical_points = False
