#
# AflowPI_TB.py
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
#
# Copyright (C) 2015 Luis A. Agapito, 2016 Marco Buongiorno Nardelli
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).

# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).

# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).

from __future__ import print_function
from scipy import linalg as LA
from scipy import fftpack as FFT
from numpy import linalg as LAN
import xml.etree.ElementTree as ET
import multiprocessing
import numpy as np
import cmath
import sys, time
import re
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
sys.path.append('/home/marco/Programs/AflowPI_TB/')
import AflowPI_TB_lib as API
sys.path.append('./')
 
#units
Ry2eV      = 13.60569193

input_file = sys.argv[1]

read_S, shift_type, fpath, shift, pthr, do_comparison, double_grid,\
	do_bands, onedim, do_dos, delta, nfft1, nfft2, nfft3 = API.read_input(input_file)

if (not read_S):
	U, my_eigsmat, alat, a_vectors, b_vectors, \
	nkpnts, nspin, kpnts, kpnts_wght, \
	nbnds, Efermi, nawf, nk1, nk2, nk3 =  API.read_QE_output_xml(fpath,read_S)
   	Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
	sumk = np.sum(kpnts_wght)
	kpnts_wght /= sumk
	for ik in range(nkpnts):
        	Sks[:,:,ik]=np.identity(nawf)
	print('...using orthogonal algorithm')
else:
	U,Sks, my_eigsmat, alat, a_vectors, b_vectors, \
	nkpnts, nspin, kpnts, kpnts_wght, \
	nbnds, Efermi, nawf, nk1, nk2, nk3 =  API.read_QE_output_xml(fpath,read_S)
	sumk = np.sum(kpnts_wght)
	kpnts_wght /= sumk
	print('...using non-orthogonal algorithm')

print('reading in ',time.clock(),' sec')
reset=time.clock()

# Get grid of k-vectors in the fft order for the nscf calculation
#if print_kgrid:
#	API.get_K_grid_fft(nk1,nk2,nk3,b_vectors, print_kgrid)

# Building the Projectability
Pn = API.build_Pn(nawf,nbnds,nkpnts,nspin,U)

print('Projectability vector ',Pn)

# Check projectability and decide bnd

bnd = 0
for n in range(nbnds):
   if Pn[n] > pthr:
      bnd += 1
print('# of bands with good projectability (>',pthr,') = ',bnd)
 
# Building the TB Hamiltonian 
nbnds_norm = nawf
Hks = API.build_Hks(nawf,bnd,nbnds,nbnds_norm,nkpnts,nspin,shift,my_eigsmat,shift_type,U)

print('building Hks in ',time.clock()-reset,' sec')
reset=time.clock()

# Take care of non-orthogonality, if needed
# Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
# Hks -> Sks^(1/2)*Hks*Sks^(1/2)+

if read_S:
	S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
	for ik in range(nkpnts):
		w, v = LAN.eigh(Sks[:,:,ik],UPLO='U')
		w = np.sqrt(w)
		S2k[:,:,ik] = v*w

	Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
	for ispin in range(nspin):
		for ik in range(nkpnts):
			Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(Hks[:,:,ik,ispin]).dot(np.conj(S2k[:,:,ik]).T)
	Hks = Hks_no

# Plot the TB and DFT eigevalues. Writes to comparison.pdf
if do_comparison:
	API.plot_compare_TB_DFT_eigs(Hks,Sks,my_eigsmat,read_S)
	quit()

# Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)

# Define real space lattice vectors for Fourier transform of Hks
R,R_wght,nrtot,idx = API.get_R_grid_fft(nk1,nk2,nk3,a_vectors)
if abs(np.sum(R_wght)-float(nk1*nk2*nk3)) > 1.0e-8:
	print(np.sum(R_wght), float(nr1*nr2*nr3))
	sys.exit('wrong sum rule on R weights')

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

print('k -> R in ',time.clock()-reset,' sec')
reset=time.clock()

if do_bands and not(onedim):
	# Compute bands on a selected path in the BZ
	print('... computing bands')

	# Define k-point mesh for bands interpolation
	kq,kmod,nkpi = API.kpnts_interpolation_mesh()

	Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)
	Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
	for ispin in range(nspin):
        	for ik in range(nkpi):
			for i in range(nk1):
				for j in range(nk2):
					for k in range(nk3):
                       				phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                       				Hks_int[:,:,ik,ispin] += HRaux[:,:,i,j,k,ispin]*phase
                       				if read_S and ispin == 0:
                               				Sks_int[:,:,ik] += SRaux[:,:,nr]*phase
	
	API.write_TB_eigs(Hks_int,Sks_int,read_S)
        print('bands in ',time.clock()-reset,' sec')
        reset=time.clock()

if double_grid:
	# Fourier interpolation on extended grid (zero padding)
	nk1p = nfft1+nk1
	nk2p = nfft2+nk2
	nk3p = nfft3+nk3
	nktotp= nk1p*nk2p*nk3p
	print('Number of k vectors for zero padding Fourier interpolation ',nktotp)

	# Extended R to k (with zero padding)
	HRauxp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
	SRauxp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p),dtype=complex)
	Hksp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
	Sksp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p),dtype=complex)
	aux = np.zeros((nk1,nk2,nk3),dtype=complex)
			
	for ispin in range(nspin):
        	for i in range(nawf):
                	for j in range(nawf):
				aux = HRaux[i,j,:,:,:,ispin]
				HRauxp[i,j,:,:,:,ispin] = API.zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3)
                        	Hksp[i,j,:,:,:,ispin] = FFT.fftn(HRauxp[i,j,:,:,:,ispin])
                        	if read_S and ispin == 0:
					aux = HRaux[i,j,:,:,:,ispin]
					SRauxp = API.zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3)
                                	Sksp[i,j,:,:,:] = FFT.fftn(SRauxp[i,j,:,:,:])

	nk1 = nk1p
	nk2 = nk2p
	nk3 = nk3p
	aux = None
	print('R -> k zero padding in ',time.clock()-reset,' sec')
	reset=time.clock()
else:
	Hksp = Hkaux
	Sksp = Skaux
	HRauxp = HRaux
	SRauxp = SRaux

if do_bands and onedim:
	# FFT interpolation along a single directions in the BZ
	print('... computing bands along a line')

	# Count points along symmetry direction
	nL = 0
	for ik1 in range(nk1):
		for ik2 in range(nk2):
			for ik3 in range(nk3):
				nL += 1
	
	Hkaux  = np.zeros((nawf,nawf,nL,nspin),dtype=complex)
	Skaux  = np.zeros((nawf,nawf,nL),dtype=complex)
	for ispin in range(nspin):
        	for i in range(nawf):
                	for j in range(nawf):
				nL=0
				for ik1 in range(nk1):
					for ik2 in range(nk2):
						for ik3 in range(nk3):
							Hkaux[i,j,nL,ispin]=Hksp[i,j,ik1,ik2,ik3,ispin]	
							if (read_S and ispin == 0):
								Skaux[i,j,nL] = Sksp[i,j,ik1,ik2,ik3]
							nL += 1

	# zero padding interpolation
	# k to R
	npad = 500
	HRaux  = np.zeros((nawf,nawf,nL,nspin),dtype=complex)
	SRaux  = np.zeros((nawf,nawf,nL),dtype=complex)
	for ispin in range(nspin):
		for i in range(nawf):
			for j in range(nawf):
				HRaux[i,j,:,ispin] = FFT.ifft(Hkaux[i,j,:,ispin])
				if read_S and ispin == 0:
					SRaux[i,j,:] = FFT.ifft(Skaux[i,j,:])

	Hkaux = None
	Skaux = None
	Hkaux  = np.zeros((nawf,nawf,npad+nL,nspin),dtype=complex)
	Skaux  = np.zeros((nawf,nawf,npad+nL),dtype=complex)
	HRauxp  = np.zeros((nawf,nawf,npad+nL,nspin),dtype=complex)
	SRauxp  = np.zeros((nawf,nawf,npad+nL),dtype=complex)

	for ispin in range(nspin):
        	for i in range(nawf):
                	for j in range(nawf):
				HRauxp[i,j,:(nL/2),ispin]=HRaux[i,j,:(nL/2),ispin]
				HRauxp[i,j,(npad+nL/2):,ispin]=HRaux[i,j,(nL/2):,ispin]
                        	Hkaux[i,j,:,ispin] = FFT.fft(HRauxp[i,j,:,ispin])
                        	if read_S and ispin == 0:
					SRauxp[i,j,:(nL/2)]=SRaux[i,j,:(nL/2)]
					SRauxp[i,j,(npad+nL/2):]=SRaux[i,j,(nL/2):]
                                	Skaux[i,j,:] = FFT.fft(SRauxp[i,j,:])


	# Print TB eigenvalues on interpolated mesh
	API.write_TB_eigs(Hkaux,Skaux,read_S)

        print('bands in ',time.clock()-reset,' sec')
        reset=time.clock()

if do_dos:
	# DOS calculation with gaussian smearing
	print('... computing DOS')
	eig,ndos = API.calc_TB_eigs(Hksp,Sksp,read_S)
	emin = np.min(eig)-1.0
	emax = np.max(eig)-shift/2.0
	de = (emax-emin)/1000
	ene = np.arange(emin,emax,de,dtype=float)
	dos = np.zeros((ene.size),dtype=float)
	dosvec = np.zeros((eig.size),dtype=float)

	for ne in range(ene.size):
		dosvec = 1.0/np.sqrt(np.pi)*np.exp(-((ene[ne]-eig)/delta)**2)/delta
		dos[ne] = np.sum(dosvec)

	f=open('dos.dat','w')
        for ne in range(ene.size):
                f.write('%.5f  %.5fi \n' %(ene[ne],dos[ne]))

        print('dos in ',time.clock()-reset,' sec')
        reset=time.clock()

# Timing
print('Total CPU time =', time.clock(),' sec')

