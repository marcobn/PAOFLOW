#
# AflowPI_TB
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

# import general modules
from __future__ import print_function
from scipy import fftpack as FFT
import xml.etree.ElementTree as ET
import numpy as np
import sys, time
from mpi4py import MPI
# Define paths
sys.path.append('./')
sys.path.append('/home/marco/Programs/AflowPI_TB/src/defs')
# Import TB specific functions
from read_input import read_input
from build_Pn import build_Pn
from build_Hks import build_Hks
from do_non_ortho import do_non_ortho
from plot_compare_TB_DFT_eigs import plot_compare_TB_DFT_eigs
from get_R_grid_fft import get_R_grid_fft
from do_bands_calc import do_bands_calc
from do_bands_calc_1D import do_bands_calc_1D
from do_double_grid import do_double_grid
from do_dos_calc import do_dos_calc
 
#units
Ry2eV      = 13.60569193

# initialize parallel execution
comm=MPI.COMM_WORLD
size=comm.Get_size()
if size > 1:
	rank = comm.Get_rank()
	if rank == 0: print('parallel execution on ',size,' processors')
	from read_QE_output_xml import read_QE_output_xml
else:
	rank=0
	from read_QE_output_xml_parse import read_QE_output_xml


input_file = sys.argv[1]

read_S, shift_type, fpath, shift, pthr, do_comparison, double_grid,\
	do_bands, onedim, do_dos, delta, nfft1, nfft2, nfft3 = read_input(input_file)

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
#	get_K_grid_fft(nk1,nk2,nk3,b_vectors, print_kgrid)

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
if read_S:
	Hks = do_non_ortho(Hks,Sks)

# Plot the TB and DFT eigevalues. Writes to comparison.pdf
if do_comparison:
	plot_compare_TB_DFT_eigs(Hks,Sks,my_eigsmat,read_S)
	quit()

# Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)

# Define real space lattice vectors for FFT ordering of Hks
R,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

# Original k grid to R grid
Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
for i in range(nk1):
	for j in range(nk2):
		for k in range(nk3):
			Hkaux[:,:,i,j,k,:] = Hks[:,:,idx[i,j,k],:]	
			Skaux[:,:,i,j,k] = Sks[:,:,idx[i,j,k]]	

	HRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
	SRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
	for ispin in range(nspin):
		for i in range(nawf):
			for j in range(nawf):
				HRaux[i,j,:,:,:,ispin] = FFT.ifftn(Hkaux[i,j,:,:,:,ispin])
				if read_S and ispin == 0:
					SRaux[i,j,:,:,:] = FFT.ifftn(Skaux[i,j,:,:,:])

if rank == 0: print('k -> R in ',time.clock()-reset,' sec')
reset=time.clock()

if do_bands and not(onedim):
	# Compute bands on a selected path in the BZ
	do_bands_calc(HRaux,SRaux,R_wght,R,idx,read_S)

	if rank == 0: print('bands in ',time.clock()-reset,' sec')
        reset=time.clock()

if double_grid:
	# Fourier interpolation on extended grid (zero padding)
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
	do_dos_calc(Hksp,Sksp,read_S,shift,delta)

        if rank ==0: print('dos in ',time.clock()-reset,' sec')
        reset=time.clock()

# Timing
if rank ==0: print('Total CPU time =', time.clock(),' sec')

