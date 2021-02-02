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
  from .constants import ANGSTROM_AU

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape
  arrays['HRs'] = np.reshape(arrays['HRs'], (nawf,nawf,nk1*nk2*nk3,nspin), order='C')

  l=0
  natoms = attributes['natoms']
  nwf = nawf//natoms
  tau_wf = np.zeros((nawf,3), dtype=float)
  for n in range(attributes['natoms']):
    for i in range(nwf):
      tau_wf[l,:] = arrays['tau'][n,:]
      l += 1

  tau_wf /= ANGSTROM_AU

  if arrays['Efield'].any() != 0.0:
    for n in range(nawf):
      arrays['HRs'][n,n,0,:] -= arrays['Efield'].dot(tau_wf[n,:])

  if arrays['HubbardU'].any() != 0:
    for n in range(nawf):
      arrays['HRs'][n,n,0,:] -= arrays['HubbardU'][n]/2.0

  arrays['HRs'] = np.reshape(arrays['HRs'], (nawf,nawf,nk1,nk2,nk3,nspin), order='C')
