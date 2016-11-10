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
from get_R_grid_fft import *
from do_bands_calc import *
from do_bands_calc_1D import *
from do_double_grid import *
from do_dos_calc import *
from do_spin_orbit import *
from constants import *

# initialize parallel execution
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

input_file = sys.argv[1]

non_ortho, shift_type, fpath, shift, pthr, do_comparison, double_grid,\
        do_bands, onedim, do_dos, delta, do_spin_orbit,nfft1, nfft2, \
        nfft3, ibrav, dkres, Boltzmann, epsilon = read_input(input_file)

if (not non_ortho):
    U, my_eigsmat, alat, a_vectors, b_vectors, \
    nkpnts, nspin, kpnts, kpnts_wght, \
    nbnds, Efermi, nawf, nk1, nk2, nk3 =  read_QE_output_xml(fpath)
    Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
    sumk = np.sum(kpnts_wght)
    kpnts_wght /= sumk
    for ik in range(nkpnts):
        Sks[:,:,ik]=np.identity(nawf)
    if rank == 0: print('...using orthogonal algorithm')
else:
    if rank == 0: print('...using non-orthogonal algorithm')
    sys.exit('must read overlap matrix from file')

if rank == 0: print('reading in ',time.clock(),' sec')
reset=time.clock()

# Building the Projectability
Pn = build_Pn(nawf,nbnds,nkpnts,nspin,U)

if rank == 0: print('Projectability vector ',Pn)

# Check projectability and decide bnd

bnd = 0
for n in range(nbnds):
    if Pn[n] > pthr:
        bnd += 1
if rank == 0: print('# of bands with good projectability (>',pthr,') = ',bnd)

# Building the TB Hamiltonian
nbnds_norm = nawf
Hks = build_Hks(nawf,bnd,nbnds,nbnds_norm,nkpnts,nspin,shift,my_eigsmat,shift_type,U)

if rank == 0: print('building Hks in ',time.clock()-reset,' sec')
reset=time.clock()

# Take care of non-orthogonality, if needed
# Hks from projwfc is orthogonal. If non-orthogonality is required, we have to read a non orthogonal Hks and
# apply a basis change to Hks as Hks -> Sks^(1/2)+*Hks*Sks^(1/2)
if non_ortho:
    Hks = do_non_ortho(Hks,Sks)

# Plot the TB and DFT eigevalues. Writes to comparison.pdf
if do_comparison:
    from plot_compare_TB_DFT_eigs import *
    plot_compare_TB_DFT_eigs(Hks,Sks,my_eigsmat)
    quit()

# Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)

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

HRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
HRaux[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])

# Naming convention (from here): 
# Hks = k-space Hamiltonian on original MP grid
# HRs = R-space Hamiltonian on original MP grid
if Boltzmann or epsilon:
    Hks_long = Hks
Hks = None
Hks = Hkaux
HRs = HRaux

if rank == 0: print('k -> R in ',time.clock()-reset,' sec')
reset=time.clock()

if Boltzmann or epsilon:
    # Compute the gradient of the k-space Hamiltonian
    from do_gradient import *
    from get_R_grid_regular import *

    Rreg,Rreg_wght,nrreg = get_R_grid_regular(nk1,nk2,nk3,a_vectors)

    dHks = do_gradient(Hks_long,Rreg_wght,Rreg,b_vectors,nk1,nk2,nk3,alat)

    #kq,kq_wght,_,_ = get_K_grid_fft(nk1,nk2,nk3,b_vectors)
    #nq=0
    #for i in range (nk1):
    #    for j in range(nk2):
    #        for k in range(nk3):
    #            np.set_printoptions(precision=5, suppress=True, formatter={'complex': '{: 0.5f : 0.5f}'.format})
    #            if rank == 0: print(nq,kq[:,nq],kq_wght[nq])
    #            for l in range(3):
    #                if rank == 0: print(dHks[l,:,:,i,j,k,0])
    #                if rank == 0: print('   ')
    #            #if rank == 0: print(Hks[:,:,i,j,k,0])
    #            if rank == 0: print('======')
    #            nq += 1

    # Compute the momentum operator p_n,m(k) and interpolate on extended grid
    from do_momentum import *
    pks,E_k = do_momentum(Hks,dHks)
    if double_grid:
        pRs = np.zeros((3,nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
        pRs[:,:,:,:,:,:,:] = FFT.ifftn(pks[:,:,:,:,:,:,:],axes=[3,4,5])
        pksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,pRs)  # REM: this is only 'U'
    else:
        pksp = pks

    # Compute velocities for Boltzmann transport
    velkp = np.zeros((3,nawf,nk1*nk2*nk3,nspin),dtype=float)
    for n in range(nawf):
        nkb = 0 
        for i in range (nk1):
            for j in range(nk2):
                for k in range(nk3):
                    velkp[:,n,nkb,:] = np.real(pksp[:,n,n,i,j,k,:])
                    nkb += 1

    #for nq in range (nkb):
    #    if rank == 0: print(nq,kq[:,nq],kq_wght[nq])
    #    for l in range(3):
    #        for n in range(nawf):
    #            if rank == 0: print(velkp[l,n,nq,:])
    #        if rank == 0: print('   ')
    #if rank == 0: print('======') 

if rank == 0: print('Boltzmann in ',time.clock()-reset,' sec')
reset=time.clock()

if do_spin_orbit:
    do_spin_orbit()

if do_bands and not(onedim):
    # Compute bands on a selected path in the BZ
    alat *= 0.529177
    do_bands_calc(HRs,R_wght,R,idx,ibrav,alat,a_vectors,b_vectors,dkres)

    if rank == 0: print('bands in ',time.clock()-reset,' sec')
    reset=time.clock()

elif do_bands and onedim:
    # FFT interpolation along a single directions in the BZ
    if rank == 0: print('... computing bands along a line')
    do_bands_calc_1D(Hks)

    if rank ==0: print('bands in ',time.clock()-reset,' sec')
    reset=time.clock()

if double_grid:
    # Fourier interpolation on extended grid (zero padding)
    # Returns only the U(pper) triangle of the Hermitian matrices. If the whole matrix is needed add L
    # def symmetrize(Hksp):
    #     return Hksp + Hksp.getH() - np.diag(Hksp.diagonal())
    Hksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,HRs)
    # Naming convention (from here): 
    # Hksp = k-space Hamiltonian on interpolated grid
    if rank == 0: print('Number of k vectors for zero padding Fourier interpolation ',nk1*nk2*nk3),

    kq,kq_wght,_,_ = get_K_grid_fft(nk1,nk2,nk3,b_vectors)

    if rank ==0: print('R -> k zero padding in ',time.clock()-reset,' sec')
    reset=time.clock()
else:
    Hksp = Hks

if do_dos:
    # DOS calculation with gaussian smearing on double_grid Hksp
    for ispin in range(nspin):
        E_k = do_dos_calc(Hksp,shift,delta,ispin,kq_wght)

    if rank ==0: print('dos in ',time.clock()-reset,' sec')
    reset=time.clock()

if Boltzmann and do_dos:
    # Compute transport quantities (conductivity, Seebeck and thermal electrical conductivity)
    from do_Boltz_tensors import *
    temp = 0.025852  # set room temperature in eV

    for ispin in range(nspin):
        ene,L0,L1,L2 = do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin)

        #----------------------
        # Conductivity
        #----------------------

        L0 *= ELECTRONVOLT_SI**2/(4.0*np.pi**3)* \
              (ELECTRONVOLT_SI/(H_OVER_TPI**2*BOHR_RADIUS_SI))
        if rank == 0:
            f=open('sigma_'+str(ispin)+'.dat','w')
            for n in range(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                    %(ene[n],L0[0,0,n],L0[1,1,n],L0[2,2,n],L0[0,1,n],L0[0,2,n],L0[1,2,n]))
            f.close()

        #----------------------
        # Seebeck
        #----------------------

        S = np.zeros((3,3,ene.size),dtype=float)

        L1 *= (ELECTRONVOLT_SI**2/(4.0*np.pi**3))*(ELECTRONVOLT_SI**2/(H_OVER_TPI**2*BOHR_RADIUS_SI))

        for n in range(ene.size):
            S[:,:,n] = LAN.inv(L0[:,:,n])*L1[:,:,n]*(-K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**2))

        if rank == 0:
            f=open('Seebeck_'+str(ispin)+'.dat','w')
            for n in range(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                        %(ene[n],S[0,0,n],S[1,1,n],S[2,2,n],S[0,1,n],S[0,2,n],S[1,2,n]))
            f.close()

        #----------------------
        # Electron thermal conductivity
        #----------------------

        kappa = np.zeros((3,3,ene.size),dtype=float)

        L2 *= (ELECTRONVOLT_SI**2/(4.0*np.pi**3))*(ELECTRONVOLT_SI**3/(H_OVER_TPI**2*BOHR_RADIUS_SI))

        for n in range(ene.size):
            kappa[:,:,n] = (L2[:,:,n] - L1[:,:,n]*LAN.inv(L0[:,:,n])*L1[:,:,n])*(K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**3))

        if rank == 0:
            f=open('kappa_'+str(ispin)+'.dat','w')
            for n in range(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                        %(ene[n],kappa[0,0,n],kappa[1,1,n],kappa[2,2,n],kappa[0,1,n],kappa[0,2,n],kappa[1,2,n]))
            f.close()

    if rank ==0: print('transport in ',time.clock()-reset,' sec')
    reset=time.clock()
else:
    sys.exit('missing eigenvalues - compute dos!')

# Timing
if rank ==0: print('Total CPU time =', time.clock(),' sec')
