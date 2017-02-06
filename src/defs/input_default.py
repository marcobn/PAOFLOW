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
verbose = False
non_ortho  = False
write2file = False
shift_type = 1
shift      = 5.0
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

# Plot Fermi Surface
do_fermisurf = False

# Boltzmann transport calculation
Boltzmann = False

# Dielectric function calculation
epsilon = False

# Band topology analysis (also in do_bands_calc)
band_topology = False
# Berry curvature and AHC
Berry = False
ipol = 0
jpol = 1
ac_cond_Berry = False
# Spin Berry curvature and SHC
spin_Hall = False
spol = 2
eminSH = -1.0
emaxSH = 1.0
sh = [0,1,2]    # order of shells with l angular momentum
nl = [1,1,1]    # multiplicity of each l shell
ac_cond_spin = False
