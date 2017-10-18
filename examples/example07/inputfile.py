############################
#
#  inputfile.py for PAOpy.py
#
############################
import numpy as np

# Control

fpath = './bi.save'
verbose = False
restart = False
non_ortho  = False
write2file = False
shift_type = 1
shift      = 'auto'
pthr       = 0.9
npool = 1

# Calculations

# Compare PAO bands with original DFT bands on the original MP mesh
do_comparison = False

# Construct TB spin-orbit Hamiltonian
do_spin_orbit = True
theta = 0.0
phi = 0.0

naw=np.array([ 9 ]) 
orb_pseudo = [ 'spd' ]


lambda_p = [ 1.65 ]
lambda_d = [  0.0  ] 

# Bands interpolation along a path from a 1D string of k points
onedim = False
# Bands interpolation on a path from the original MP mesh
do_bands = True
ibrav = 3
dkres = 0.001

# Hamiltonian interpolation on finer MP mesh
double_grid = False
nfft1 = 24
nfft2 = 24
nfft3 = 24

# DOS(PDOS) calculation
do_dos = False
do_pdos = False
emin = -4.
emax = 2.2
delta = 0.05

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
temp = 0.05852  # room temperature

# Dielectric function calculation
epsilon =False
metal = False
epsmin=0.0
epsmax=6.0
ne = 500

# Critical points
critical_points = False
