#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

def doubling_HRs ( data_controller, ndx, ndy, ndz ):
  import numpy as np

  arry,attr = data_controller.data_dicts()

  nspin = attr['nspin']
  nx = np.array([ndx,ndy,ndz])
  nk = np.array([attr['nk1'],attr['nk2'],attr['nk3']])

  nkpts = np.prod(nk)
  new_index = np.zeros((3,nkpts), dtype=int)
  cell_index = np.zeros((nk[0],nk[1],nk[2],3), dtype=int)
    
  for i in range(nk[0]):
    for j in range(nk[1]):
      for k in range(nk[2]):
        n = k + j*nk[2] + i*nk[1]*nk[2]
        ijk = np.array([i,j,k])

        R = np.array([vR if vR<.5 else vR-1 for vR in ijk/nk])
        R -= R.astype(int)
        # the minus sign in Rx*nk1 is due to the Fourier transformation (Ri-Rj)

        ix = np.round(R*nk).astype(int)
        new_index[:,n] = ix[:]
        cell_index[ix[0],ix[1],ix[2],:] = ijk[:]

  min_inds = np.amin(new_index, axis=1)
  max_inds = np.amax(new_index, axis=1)

  # Loop across dimnsions in the order y,z,x
  for dim in range(-2,1):
    # Construct the doubled Hamiltonian nx[dim] times for this dimension    
    for _ in range(nx[dim]):

      nawf = attr['nawf']
      HR_double = np.zeros((2*nawf,2*nawf,nk[0],nk[1],nk[2],nspin), dtype=complex)
      for ix in range(min_inds[0], max_inds[0]+1):
        for iy in range(min_inds[1], max_inds[1]+1):
          for iz in range(min_inds[2], max_inds[2]+1):

            i,j,k = cell_index[ix,iy,iz,:] # Doubled cell index
            iwa = np.array([ix,iy,iz])
            i2w = 2*iwa[dim]

            if i2w+1 >= min_inds[dim] and i2w-1 <= max_inds[dim]:
              if i2w >= min_inds[dim] and i2w <= max_inds[dim]:
                iwa[dim] = i2w
                m,n,l = cell_index[tuple(iwa)][:]
                # Upper Left and Lower Right HR_double block
                HR_double[:nawf,:nawf,i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
                HR_double[nawf:2*nawf,nawf:2*nawf,i,j,k,:] = arry['HRs'][:,:,m,n,l,:]    
              if i2w+1 >= min_inds[dim] and i2w+1 <= max_inds[dim]:
                # Upper Right HR_double block                
                iwa[dim] = i2w+1
                m,n,l = cell_index[tuple(iwa)][:]
                HR_double[:nawf,nawf:2*nawf,i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
              if i2w-1 >= min_inds[dim] and i2w-1 <= max_inds[dim]:
                iwa[dim] = i2w-1
                m,n,l = cell_index[tuple(iwa)][:]
                # Lower Left HR_double block
                HR_double[nawf:2*nawf,:nawf,i,j,k,:] = arry['HRs'][:,:,m,n,l,:]

      arry['HRs'] = HR_double
      doubling_attr_arry(data_controller, dim)

  omega,alat = attr['omega'],attr['alat']
  a_vec = lambda n : arry['a_vectors'][n,:]
  arry['b_vectors'][0,:] = (np.cross(a_vec(1),a_vec(2)))/omega*alat**3
  arry['b_vectors'][1,:] = (np.cross(a_vec(2),a_vec(0)))/omega*alat**3
  arry['b_vectors'][2,:] = (np.cross(a_vec(0),a_vec(1)))/omega*alat**3


def doubling_attr_arry ( data_controller, dimension ):
  import numpy as np
  from .constants import ANGSTROM_AU

  def double_array ( arrays, key ):
    if key in arrays:
      arrays[key] = np.append(arrays[key], arrays[key])

  arry,attr = data_controller.data_dicts()
  tau,a_vecs = arry['tau'],arry['a_vectors']

  # Increassing nawf/natoms
  attr['bnd'] *= 2
  attr['nawf'] *= 2
  attr['nelec'] *= 2
  attr['natoms'] *= 2

  # Add new atom species and positions
  double_array(arry, 'species')
  arry['tau'] = np.append(tau, tau[:,:]+a_vecs[dimension,:]*ANGSTROM_AU, axis=0)

  # Doubling lattice vectors and cell volume
  attr['omega'] *= 2
  arry['a_vectors'][dimension,:] *= 2

  # Doubling the atom number of orbitals / orbital character / multiplicity
  double_array(arry, 'sh')
  double_array(arry, 'nl')
  double_array(arry, 'naw')

  # If the SOC is included pertubative
  if 'do_spin_orbit' in attr and attr['do_spin_orbit']:
    double_array(arry, 'lambda_p')
    double_array(arry, 'lambda_d')
    double_array(arry, 'orb_pseudo')
