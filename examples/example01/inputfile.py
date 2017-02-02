############################INPUT variables########################

# Control 

fpath = './silicon.save'
verbose = False
non_ortho  = False
write2file = False
shift_type = 1
shift      = 5.0
pthr       = 0.97
Efermi_corr = 0.0
npool = 8

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
do_bands = True
ibrav = 2
dkres = 0.01

# Hamiltonian interpolation on finer MP mesh
double_grid = True
nfft1 = 16
nfft2 = 16
nfft3 = 16

# DOS(PDOS) calculation
do_dos = True
do_pdos = True
emin = -12.
emax = 3
delta = 0.1

# Boltzmann transport calculation
Boltzmann = True

# Dielectric function calculation
epsilon = True

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
