import sys,time
import re
import numpy as np
from scipy import constants

sys.path.append('/home/marco/Programs/PAOpy/src/defs')
from constants import *

#print "\nStarted at %s\n" %time.ctime()
reset = time.time()
H0path = 'H0/'
ofile = open('lambda_q','a')

# Get degauss variable from scf input file
try:
    inputscf = open(H0path+'al.scf.in','r')
except:
    sys.exit("\nCannot find scf input file from directory './H0'\n")

lines = inputscf.readlines()
inputscf.close()

check = False
for line in lines:
    if re.search('degauss',line):
        x = line.split()
        check = True
if not check:
    sys.exit('\nCannot find "degauss" inside scf input file\n')

x = np.array(x)
if x.shape[0]!=3: sys.exit('\nCannot read "degauss" variable from scf input file.\nThe number should be separated from other symbols\n')
degauss = float(x[2])                  # degauss is defined in Ry          


# Define useful quantities
freq = np.array((4.220014,4.220014,9.435680))*10**12   # in Hz
delta = degauss*13.6058                          # delta in eV
delta = sys.argv[1]                          # delta in eV
delta = float(delta)
print 'delta = ',delta
alat = 4.039e-10                             # lattice parameter in m
N_Ef = 0.202                                  # in eV^-1 (2.75 Ry^-1 per atom per spin)
mass = 2.513314e+10                           # in eV/c^2
mass =  26.9815
c = constants.c                               # speed of light in m/s


# Read output files from PAOpy
ifile0 = np.load(H0path+'aluminum.save/PAOdump.npz')
kq = ifile0['kq']
Hksp0 = ifile0['Hksp']
v_k = ifile0['v_k']
E_k = ifile0['E_k']
#print E_k.shape
nk1 = ifile0['nk1']
nk2 = ifile0['nk2']
nk3 = ifile0['nk3']

Hksp0 = np.reshape(Hksp0,(kq.shape[1],Hksp0.shape[3],Hksp0.shape[4],Hksp0.shape[5]),order='C')
nawf = Hksp0.shape[1]                  # number of atomic orbitals
nspin = Hksp0.shape[3]                 # number of spin components
nktot = kq.shape[1]                    # number of k-points
print 'nktot = ', nktot
lambda_q = np.zeros(Hksp0.shape[3])

ofile.write('\n**************************************************************************************************')
ofile.write('\nNat = 2\t\t k-mesh = %dx%dx%d\t\t delta = %f eV\t\t ' %(nk1,nk2,nk3,delta))

#print "Reading input files in:                         %.3f sec" %(time.time()-reset)
start = time.time()


# Loop on phonon modes
for v in range(3):
    Hphpath = 'H'+str(v+1)+'/'                              # load the perturbed Hamiltonian directory named H1,H2,H3,...
    ifile1 = np.load(Hphpath+'aluminum.save/PAOdump.npz')

    # Check on the k-point mesh
    kq1 = ifile1['kq']
    for i in range(3):
        for kn in range(nktot):
            try:
                kq[i,kn] == kq1[i,kn]
            except:
                sys.exit('\nThe k-point meshes for the two hamiltonians do not match\n')
    #print "Matching the two hamiltonians meshes in:        %.3f sec" %(time.time()-start)
    reset = time.time()

    Hksp1 = ifile1['Hksp']
    Hksp1 = np.reshape(Hksp1,(kq.shape[1],Hksp1.shape[3],Hksp1.shape[4],Hksp1.shape[5]),order='C')

    # Check on the atomic orbital basis dimension for the two Hamiltonians
    if Hksp1.shape[1] != nawf or Hksp1.shape[2] != nawf:
        sys.exit("\nThe atomic orbitals basis of the two Hamiltonians do not match\n")

    # Calculation of lambda_q(v)

    gkk = np.zeros((nktot,Hksp1.shape[1],Hksp1.shape[2],Hksp1.shape[3]),dtype=complex)
    for spin in range(nspin):
        for kn in range(nktot):
            gkk[kn,:,:,spin] = v_k[kn,:,:,spin].T.conj().dot(np.dot(((Hksp0[kn,:,:,spin]-Hksp1[kn,:,:,spin])/(.01*alat)),v_k[kn,:,:,spin]))

        num = np.sum(np.exp(-(E_k[:,:,spin]/delta)**2) * \
                     np.swapaxes(np.exp(-(E_k[:,:,spin]/delta)**2)*np.absolute(gkk.swapaxes(0,1)[:,:,:,spin])**2,0,2))*nktot

        den = np.sum(np.exp(-(E_k[:,:,spin]/delta)**2))

    gamma_q = num/den**2*N_Ef**2*np.pi/mass * H_OVER_TPI /UMA_SI/2.0  # the factor 1./2. comes from the number of atoms (unit cells in the supercell)

    lambda_q = gamma_q/2.0/np.pi**2/N_Ef*ELECTRONVOLT_SI/H_OVER_TPI/freq[v]**2 # the factor 1./2./np.pi comes from conversion from rad to Hz.

    print 'lambda_q = ',lambda_q, ', freq = ',freq[v]/1.e+12,' THz,  gamma_q = ',gamma_q/1.0e+9,' GHz'
    ofile.write('\n%f' %lambda_q)
    #start = time.time()

ofile.close()

