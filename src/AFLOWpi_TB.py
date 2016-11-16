#
# AFLOWpi_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
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
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli,
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
#

# import general modules
from __future__ import print_function
from scipy import fftpack as FFT
from scipy import linalg as LA
from numpy import linalg as LAN
import xml.etree.ElementTree as ET
import numpy as np
import sys, time
from mpi4py import MPI

# Define paths
sys.path.append('./')
sys.path.append('/home/marco/Programs/AFLOWpi_TB/src/defs')
# Import TB specific functions
from read_input import *
from build_Pn import *
from build_Hks import *
from do_non_ortho import *
from do_ortho import *
from get_R_grid_fft import *
from get_K_grid_fft import *
from do_bands_calc import *
from do_bands_calc_1D import *
from do_double_grid import *
from do_dos_calc import *
from do_spin_orbit import *
from constants import *

#----------------------
# initialize parallel execution
#----------------------
comm=MPI.COMM_WORLD
size=comm.Get_size()
if size > 1:
    rank = comm.Get_rank()
    if rank == 0: print('parallel execution on ',size,' processors')
    from read_QE_output_xml import *
else:
    rank=0
    #from read_QE_output_xml_parse import *
    from read_QE_output_xml import *

#----------------------
# Read input and DFT data
#----------------------
input_file = sys.argv[1]

non_ortho, shift_type, fpath, shift, pthr, do_comparison, double_grid,\
        do_bands, onedim, do_dos, delta, do_spin_orbit,nfft1, nfft2, \
        nfft3, ibrav, dkres, Boltzmann, epsilon, theta, phi,        \
        lambda_p, lambda_d = read_input(input_file)

if (not non_ortho):
    U, my_eigsmat, alat, a_vectors, b_vectors, \
    nkpnts, nspin, kpnts, kpnts_wght, \
    nbnds, Efermi, nawf, nk1, nk2, nk3,natoms  =  read_QE_output_xml(fpath,non_ortho)
    Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
    sumk = np.sum(kpnts_wght)
    kpnts_wght /= sumk
    for ik in range(nkpnts):
        Sks[:,:,ik]=np.identity(nawf)
    if rank == 0: print('...using orthogonal algorithm')
else:
    U, Sks, my_eigsmat, alat, a_vectors, b_vectors, \
    nkpnts, nspin, kpnts, kpnts_wght, \
    nbnds, Efermi, nawf, nk1, nk2, nk3,natoms  =  read_QE_output_xml(fpath,non_ortho)
    if rank == 0: print('...using non-orthogonal algorithm')

if rank == 0: print('reading in ',time.clock(),' sec')
reset=time.clock()

#----------------------
# Building the Projectability
#----------------------
Pn = build_Pn(nawf,nbnds,nkpnts,nspin,U)

if rank == 0: print('Projectability vector ',Pn)

# Check projectability and decide bnd

bnd = 0
for n in range(nbnds):
    if Pn[n] > pthr:
        bnd += 1
if rank == 0: print('# of bands with good projectability (>',pthr,') = ',bnd)

#----------------------
# Building the TB Hamiltonian
#----------------------
nbnds_norm = nawf
Hks = build_Hks(nawf,bnd,nbnds,nbnds_norm,nkpnts,nspin,shift,my_eigsmat,shift_type,U)

if rank == 0: print('building Hks in ',time.clock()-reset,' sec')
reset=time.clock()

# NOTE: Take care of non-orthogonality, if needed
# Hks from projwfc is orthogonal. If non-orthogonality is required, we have to 
# apply a basis change to Hks as Hks -> Sks^(1/2)+*Hks*Sks^(1/2)
# non_ortho flag == 0 - makes H non orthogonal (original basis of the atomic pseudo-orbitals)
# non_ortho flag == 1 - makes H orthogonal (rotated basis) 
#    Hks = do_non_ortho(Hks,Sks)
#    Hks = do_ortho(Hks,Sks)

if non_ortho:
    Hks = do_non_ortho(Hks,Sks)

#----------------------
# Plot the TB and DFT eigevalues. Writes to comparison.pdf
#----------------------
if do_comparison:
    from plot_compare_TB_DFT_eigs import *
    plot_compare_TB_DFT_eigs(Hks,Sks,my_eigsmat,non_ortho)
    quit()

#----------------------
# Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)
#----------------------

# Define real space lattice vectors for FFT ordering of Hks
R,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

# Original k grid to R grid
reset=time.clock()
Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
for i in range(nk1):
    for j in range(nk2):
        for k in range(nk3):
            Hkaux[:,:,i,j,k,:] = Hks[:,:,idx[i,j,k],:]
            if (non_ortho):
                Skaux[:,:,i,j,k] = Sks[:,:,idx[i,j,k]]

HRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
SRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)

HRaux[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
if non_ortho:
    SRaux[:,:,:,:,:] = FFT.ifftn(Skaux[:,:,:,:,:],axes=[2,3,4])

# NOTE: Naming convention (from here):
# Hks = k-space Hamiltonian on original MP grid
# HRs = R-space Hamiltonian on original MP grid
if non_ortho:
    if Boltzmann or epsilon:
        Sks_long = Sks
Hks = None
Sks = None
Hks = Hkaux
Sks = Skaux
HRs = HRaux
SRs = SRaux

if rank == 0: print('k -> R in ',time.clock()-reset,' sec')
reset=time.clock()

if Boltzmann or epsilon:
    #----------------------
    # Compute the gradient of the k-space Hamiltonian
    #----------------------
    from do_gradient import *
    from get_R_grid_regular import *
    from do_ortho import *

    if non_ortho:
        sys.exit('H must be orthogonal')

    Rreg,Rreg_wght,nrreg = get_R_grid_regular(nk1,nk2,nk3,a_vectors)

    dHks = do_gradient(Hks_long,Rreg_wght,Rreg,b_vectors,nk1,nk2,nk3,alat)

    if rank == 0: print('gradient in ',time.clock()-reset,' sec')
    reset=time.clock()

    #----------------------
    # Compute the momentum operator p_n,m(k) and interpolate on extended grid
    #----------------------
    from do_momentum import *
    pks,E_k = do_momentum(Hks,dHks)
    if double_grid:
        pRs = np.zeros((3,nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
        pRs[:,:,:,:,:,:,:] = FFT.ifftn(pks[:,:,:,:,:,:,:],axes=[3,4,5])
        pksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,pRs)
    else:
        pksp = pks

    if rank == 0: print('momenta in ',time.clock()-reset,' sec')
    reset=time.clock()

    if Boltzmann:
        #----------------------
        # Compute velocities for Boltzmann transport
        #----------------------
        velkp = np.zeros((3,nawf,nk1*nk2*nk3,nspin),dtype=float)
        for n in range(nawf):
            nkb = 0
            for i in range (nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        velkp[:,n,nkb,:] = np.real(pksp[:,n,n,i,j,k,:])
                        nkb += 1

if do_spin_orbit:
    #----------------------
    # Compute bands with spin-orbit coupling
    #----------------------

    # NOTE: HRs is now in the original non-orthogonal basis of the PAOs

    socStrengh = np.zeros((natoms,2),dtype=float)
    socStrengh [:,0] =  lambda_p[:]
    socStrengh [:,1] =  lambda_d[:]

    HRs = do_spin_orbit_calc(HRs,natoms,theta,phi,socStrengh)

    nawf=2*nawf
    SRaux = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype= complex)
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                SRaux[:,:,i,j,k] = LA.block_diag(SRs[:,:,i,j,k],SRs[:,:,i,j,k])

    SRs = None
    SRs = SRaux

if do_bands and not(onedim):
    #----------------------
    # Compute bands on a selected path in the BZ
    #----------------------
    alat *= 0.529177

    do_bands_calc(HRs,SRs,R_wght,R,idx,non_ortho,ibrav,alat,a_vectors,b_vectors,dkres)

    if rank == 0: print('bands in ',time.clock()-reset,' sec')
    reset=time.clock()

elif do_bands and onedim:
    #----------------------
    # FFT interpolation along a single directions in the BZ
    #----------------------
    if rank == 0: print('... computing bands along a line')
    do_bands_calc_1D(Hks)

    if rank ==0: print('bands in ',time.clock()-reset,' sec')
    reset=time.clock()

if double_grid:

    if non_ortho:
        sys.exit('H must be orthogonal')

    #----------------------
    # Fourier interpolation on extended grid (zero padding)
    #----------------------
    # Returns only the U(pper) triangle of the Hermitian matrices. If the whole matrix is needed add L
    # def symmetrize(Hksp):
    #     return Hksp + Hksp.getH() - np.diag(Hksp.diagonal())
    Hksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,HRs)
    # Naming convention (from here): 
    # Hksp = k-space Hamiltonian on interpolated grid
    if rank == 0: print('Number of k vectors for zero padding Fourier interpolation ',nk1*nk2*nk3),

    kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)

    if rank ==0: print('R -> k zero padding in ',time.clock()-reset,' sec')
    reset=time.clock()
else:
    kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)
    Hksp = Hks

if do_dos or Boltzmann or epsilon:
    #----------------------
    # Compute eigenvalues of the interpolated Hamiltonian
    #----------------------
    eig = np.zeros((nawf*nk1*nk2*nk3,nspin))
    for ispin in range(nspin):
        eig, E_k = calc_TB_eigs(Hksp,ispin)

if do_dos:

    if non_ortho:
        sys.exit('H must be orthogonal')

    #----------------------
    # DOS calculation with gaussian smearing on double_grid Hksp
    #----------------------
    for ispin in range(nspin):
        do_dos_calc(eig[:,ispin],shift,delta,ispin,kq_wght)

    if rank ==0: print('dos in ',time.clock()-reset,' sec')
    reset=time.clock()

if Boltzmann:
    #----------------------
    # Compute transport quantities (conductivity, Seebeck and thermal electrical conductivity)
    #----------------------
    from do_Boltz_tensors import *
    temp = 0.025852  # set room temperature in eV

    if non_ortho:
        sys.exit('H must be orthogonal')

    for ispin in range(nspin):
        ene,L0,L1,L2 = do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin)

        #----------------------
        # Conductivity (in units of 1.e21/Ohm/m/s)
        #----------------------

        L0 *= ELECTRONVOLT_SI**2/(4.0*np.pi**3)* \
              (ELECTRONVOLT_SI/(H_OVER_TPI**2*BOHR_RADIUS_SI))*1.0e-21
        if rank == 0:
            f=open('sigma_'+str(ispin)+'.dat','w')
            for n in range(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                    %(ene[n],L0[0,0,n],L0[1,1,n],L0[2,2,n],L0[0,1,n],L0[0,2,n],L0[1,2,n]))
            f.close()

        #----------------------
        # Seebeck (in units of 1.e-4 V/K)
        #----------------------

        S = np.zeros((3,3,ene.size),dtype=float)

        L0 *= 1.0e21
        L1 *= (ELECTRONVOLT_SI**2/(4.0*np.pi**3))*(ELECTRONVOLT_SI**2/(H_OVER_TPI**2*BOHR_RADIUS_SI))

        for n in range(ene.size):
            S[:,:,n] = LAN.inv(L0[:,:,n])*L1[:,:,n]*(-K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**2))*1.e4

        if rank == 0:
            f=open('Seebeck_'+str(ispin)+'.dat','w')
            for n in range(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                        %(ene[n],S[0,0,n],S[1,1,n],S[2,2,n],S[0,1,n],S[0,2,n],S[1,2,n]))
            f.close()

        #----------------------
        # Electron thermal conductivity ((in units of 1.e15 W/m/K/s)
        #----------------------

        kappa = np.zeros((3,3,ene.size),dtype=float)

        L2 *= (ELECTRONVOLT_SI**2/(4.0*np.pi**3))*(ELECTRONVOLT_SI**3/(H_OVER_TPI**2*BOHR_RADIUS_SI))

        for n in range(ene.size):
            kappa[:,:,n] = (L2[:,:,n] - L1[:,:,n]*LAN.inv(L0[:,:,n])*L1[:,:,n])*(K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**3))*1.e-15

        if rank == 0:
            f=open('kappa_'+str(ispin)+'.dat','w')
            for n in range(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                        %(ene[n],kappa[0,0,n],kappa[1,1,n],kappa[2,2,n],kappa[0,1,n],kappa[0,2,n],kappa[1,2,n]))
            f.close()

    if rank ==0: print('transport in ',time.clock()-reset,' sec')
    reset=time.clock()

if epsilon:
    from do_epsilon import *

    if non_ortho:
            sys.exit('H must be orthogonal')

    temp = 0.025852  # set room temperature in eV
    # Symmetrize pksp and build long vector on k-points
    pksp_long = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
    omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

    for ispin in range(nspin):
        for l in range(3):
            for i in range(nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        n = k + j*nk3 + i*nk2*nk3
                        pksp_long[l,:,:,n,ispin] = pksp[l,:,:,i,j,k,ispin]

        ene, epsi, epsr = do_epsilon(E_k,pksp_long,kq_wght,omega,delta,temp,ispin)

        if rank == 0:
            f=open('epsi_'+str(ispin)+'.dat','w')
            for n in range(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                        %(ene[n],epsi[0,0,n],epsi[1,1,n],epsi[2,2,n],epsi[0,1,n],epsi[0,2,n],epsi[1,2,n]))
            f.close()
            f=open('epsr_'+str(ispin)+'.dat','w')
            for n in range(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                        %(ene[n],epsr[0,0,n],epsr[1,1,n],epsr[2,2,n],epsr[0,1,n],epsr[0,2,n],epsr[1,2,n]))
            f.close()


    if rank ==0: print('epsilon in ',time.clock()-reset,' sec')
    reset=time.clock()

# Timing
if rank ==0: print('Total CPU time =', time.clock(),' sec')
