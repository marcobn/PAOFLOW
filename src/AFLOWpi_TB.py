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

#units
Ry2eV      = 13.60569193

# initialize parallel execution
comm=MPI.COMM_WORLD
size=comm.Get_size()
if size > 1:
    rank = comm.Get_rank()
    if rank == 0: print('parallel execution on ',size,' processors')
    from read_QE_output_xml import *
else:
    rank=0
#    from read_QE_output_xml_parse import read_QE_output_xml
    from read_QE_output_xml import read_QE_output_xml

input_file = sys.argv[1]

read_S, shift_type, fpath, shift, pthr, do_comparison, double_grid,\
        do_bands, onedim, do_dos, delta, do_spin_orbit,nfft1, nfft2, nfft3, ibrav, dkres = read_input(input_file)

if (not read_S):
    U, my_eigsmat, alat, a_vectors, b_vectors, \
    nkpnts, nspin, kpnts, kpnts_wght, \
    nbnds, Efermi, nawf, nk1, nk2, nk3 =  read_QE_output_xml(fpath,read_S)
    Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
    sumk = np.sum(kpnts_wght)
    kpnts_wght /= sumk
    for ik in range(nkpnts):
        Sks[:,:,ik]=np.identity(nawf)
    if rank == 0: print('...using orthogonal algorithm')
else:
    U,Sks, my_eigsmat, alat, a_vectors, b_vectors, \
    nkpnts, nspin, kpnts, kpnts_wght, \
    nbnds, Efermi, nawf, nk1, nk2, nk3 =  read_QE_output_xml(fpath,read_S)
    sumk = np.sum(kpnts_wght)
    kpnts_wght /= sumk
    if rank == 0: print('...using non-orthogonal algorithm')

if rank == 0: print('reading in ',time.clock(),' sec')
reset=time.clock()

# Get grid of k-vectors in the fft order for the nscf calculation
#if print_kgrid:
#       get_K_grid_fft(nk1,nk2,nk3,b_vectors, print_kgrid)

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
# Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
# Hks -> Sks^(1/2)*Hks*Sks^(1/2)+
if read_S and do_comparison:
    Hks = do_non_ortho(Hks,Sks)

# Plot the TB and DFT eigevalues. Writes to comparison.pdf
if do_comparison:
    from plot_compare_TB_DFT_eigs import *
    plot_compare_TB_DFT_eigs(Hks,Sks,my_eigsmat,read_S)
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
            if (read_S):
                Skaux[:,:,i,j,k] = Sks[:,:,idx[i,j,k]]

HRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
SRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)

HRaux[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
if read_S and ispin == 0:
    SRaux[:,:,:,:,:] = FFT.ifftn(Skaux[:,:,:,:,:],axes=[2,3,4])

if rank == 0: print('k -> R in ',time.clock()-reset,' sec')
reset=time.clock()

if do_spin_orbit and not(read_S):
    do_spin_orbit()
elif do_spin_orbit and read_S:
    sys.exit('spin orbit interaction only with orthogonal basis')

if do_bands and not(onedim):
    # Compute bands on a selected path in the BZ
    alat *= 0.529177
    do_bands_calc(HRaux,SRaux,R_wght,R,idx,read_S,ibrav,alat,a_vectors,b_vectors,dkres)

    if rank == 0: print('bands in ',time.clock()-reset,' sec')
    reset=time.clock()

if double_grid:
    # Fourier interpolation on extended grid (zero padding)
    parallel = False #flag for paralellization of the fft on the awf elements
    if parallel: from do_double_grid_para import *
    Hksp,Sksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,HRaux,SRaux,read_S)

    if rank ==0: print('R -> k zero padding in ',time.clock()-reset,' sec')
    reset=time.clock()
else:
    Hksp = Hkaux
    Sksp = Skaux

if do_bands and onedim:
    # FFT interpolation along a single directions in the BZ
    if rank == 0: print('... computing bands along a line')
    do_bands_calc_1D(Hkaux,Skaux,read_S)

    if rank ==0: print('bands in ',time.clock()-reset,' sec')
    reset=time.clock()

if do_dos:
    # DOS calculation with gaussian smearing on double_grid Hksp
    for ispin in range(nspin):
        do_dos_calc(Hksp,Sksp,read_S,shift,delta,ispin)

    if rank ==0: print('dos in ',time.clock()-reset,' sec')
    reset=time.clock()

# Timing
if rank ==0: print('Total CPU time =', time.clock(),' sec')
