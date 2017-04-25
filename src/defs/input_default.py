#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
#

############################INPUT variables########################

# Control 

fpath = 'dir.save'
restart = False
verbose = False
non_ortho  = False
write2file = False
write_binary = False
writedata = False
shift_type = 1
shift      = 'auto' # if 'auto' shift is selected automatically; else, give numerical value 
pthr       = 0.95
npool = 1

# Calculations 

# Compare PAO bands with original DFT bands on the original MP mesh
do_comparison = False

# Construct TB spin-orbit Hamiltonian
do_spin_orbit = False
theta = 0.0
phi = 0.0
lambda_p = 0.0
lambda_d = 0.0

# Bands interpolation along a path from a 1D string of k points
onedim = False
# Bands interpolation on a path from the original MP mesh 
do_bands = False
ibrav = 0
dkres = 0.1
# Band topology analysis
band_topology = False

# Hamiltonian interpolation on finer MP mesh
double_grid = False
nfft1 = 0
nfft2 = 0
nfft3 = 0

# DOS(PDOS) calculation
do_dos = False
do_pdos = False
emin = -10.
emax = 2
delta = 0.01

# Adaptive smearing
smearing = 'gauss' # other available values are None or 'm-p'

# Plot Fermi Surface (spin texture)
do_fermisurf = False
fermi_up = 0.1
fermi_dw = -0.1
do_spintexture = False

# Tensor components
spol = 0  # spin
ipol = 0
jpol = 0

# Shell order and degeneracy for SO
sh = [0,1,2,0,1,2]    # order of shells with l angular momentum
nl = [2,1,1,1,1,1]    # multiplicity of each l shell

# Set temperature in eV
temp = 0.025852  # room temperature

# Boltzmann transport calculation
Boltzmann = False

# Dielectric function calculation
epsilon = False
metal = False
epsmin=0.0
epsmax=10.0
ne = 500

# Critical points
critical_points = False

# Berry curvature and AHC
Berry = False
ac_cond_Berry = False

# Spin Berry curvature and SHC
spin_Hall = False
eminSH = -1.0
emaxSH = 1.0
ac_cond_spin = False
writedata = False
