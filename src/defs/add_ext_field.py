#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

def add_ext_field ( data_controller ):
  import numpy as np
  #from constants import ANGSTROM_AU

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape
  arrays['HRs'] = np.reshape(arrays['HRs'],(nawf,nawf,nk1*nk2*nk3,nspin), order='C')

  tau_wf = np.zeros((nawf,3),dtype=float)
  l=0
  for n in range(attributes['natoms']):
    for i in range(arrays['naw'][n]):
      tau_wf[l,:] = arrays['tau'][n,:]
      l += 1

  tau_wf /= ANGSTROM_AU

  if arrays['Efield'].any() != 0.0:
    for n in range(nawf):
      arrays['HRs'][n,n,0,:] -= arrays['Efield'].dot(tau_wf[n,:])

  # --- Add Magnetic Field ---
  # Define real space lattice vectors
  #get_R_grid_fft(data_controller)
  #alat = attributes['alat']/ANGSTROM_AU
  if arrays['Bfield'].any() != 0.0:
    print('calculation in magnetic supercell not implemented')
  # Magnetic field in units of magnetic flux quantum (hc/e)
  #for i in xrange(nk1*nk2*nk3):
  #    for n in xrange(nawf):
  #        for m in xrange(nawf):
  #            arg = 0.5*np.dot((np.cross(Bfield,alat*R[i,:]+tau_wf[m,:])+np.cross(Bfield,tau_wf[n,:])),(alat*R[i,:]+tau_wf[m,:]-tau_wf[n,:]))
  #            HRs[n,m,i,:] *= np.exp(-np.pi*arg*1.j)

  if arrays['HubbardU'].any() != 0:
    for n in range(nawf):
      arrays['HRs'][n,n,0,:] -= arrays['HubbardU'][n]/2.0

  arrays['HRs'] = np.reshape(HRs,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
