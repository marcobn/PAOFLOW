#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import sys
import numpy as np
from scipy import linalg as spl
from numpy import linalg as npl

def bands_calc ( data_controller ):
  from .communication import scatter_full, gather_full

  arrays,attributes = data_controller.data_dicts()

  npool = attributes['npool']
  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape

  kq_aux = scatter_full(arrays['kq'].T, npool).T
 
  Hks_aux = band_loop_H(data_controller, kq_aux)
  
  if np.iscomplex(arrays['kq']).any():
    E_kp_aux = np.zeros((kq_aux.shape[1],nawf,nspin), dtype=complex, order="C")
  else:
    E_kp_aux = np.zeros((kq_aux.shape[1],nawf,nspin), dtype=float, order="C")
  v_kp_aux = np.zeros((kq_aux.shape[1],nawf,nawf,nspin), dtype=complex, order="C")

  for ispin in range(nspin):
    for ik in range(kq_aux.shape[1]):
      if np.iscomplex(arrays['kq']).any():
        E_kp_aux[ik,:,ispin],v_kp_aux[ik,:,:,ispin] = spl.eig(Hks_aux[:,:,ik,ispin], b=(None), overwrite_a=True, overwrite_b=True, check_finite=True)
      else:
        E_kp_aux[ik,:,ispin],v_kp_aux[ik,:,:,ispin] = spl.eigh(Hks_aux[:,:,ik,ispin], b=(None), lower=False, overwrite_a=True, overwrite_b=True, turbo=True, check_finite=True)

  Hks_aux = Sks_aux = None
  return E_kp_aux, v_kp_aux


def band_loop_H ( data_controller, kq_aux ):

  arrays,attributes = data_controller.data_dicts()

  nksize = kq_aux.shape[1]
  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape

  HRs = np.reshape(arrays['HRs'], (nawf,nawf,nk1*nk2*nk3,nspin), order='C')
  kdot = np.tensordot(arrays['R'], 2.0j*np.pi*kq_aux, axes=([1],[0]))
  np.exp(kdot, kdot)
  Haux = np.zeros((nawf,nawf,nksize,nspin), dtype=complex, order="C")

  for ispin in range(nspin):
    Haux[...,ispin] = np.tensordot(HRs[...,ispin], kdot, axes=([2],[0]))

  return Haux


def do_bands ( data_controller ):
  from mpi4py import MPI
  from .constants import ANGSTROM_AU
  from .communication import gather_full
  from .get_R_grid_fft import get_R_grid_fft
  from .kpnts_interpolation_mesh import kpnts_interpolation_mesh

  rank = MPI.COMM_WORLD.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  # Bohr to Angstrom
  attributes['alat'] /= ANGSTROM_AU

#### What about ONEDIM?
  attributes['onedim'] = False
  if not attributes['onedim']:
    #--------------------------------------------
    # Compute bands on a selected path in the BZ
    #--------------------------------------------

    alat = attributes['alat']
    nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape
    nktot = nk1*nk2*nk3

    # Define real space lattice vectors
    get_R_grid_fft(data_controller, nk1, nk2, nk3)

    # Define k-point mesh for bands interpolation
    if 'ibrav' in attributes:
      kpnts_interpolation_mesh(data_controller)
    if 'kq' not in arrays:
      print('need external kq for bands')

    nkpi = arrays['kq'].shape[1]
    for n in range(nkpi):
      arrays['kq'][:,n] = np.dot(arrays['kq'][:,n], arrays['b_vectors'])

    # Compute the bands along the path in the IBZ
    arrays['E_k'],arrays['v_k'] = bands_calc(data_controller)
    if np.iscomplex(arrays['kq']).any():
      for ispin in range(nspin):
        sorted_E_k = []
        for k in range(arrays['kq'].shape[1]):
          E = np.array([item for sublist in arrays['E_k'][k] for item in sublist])
          idx = np.argsort(E)
          sorted_E_k.append(E[idx])
        arrays['E_k'][:,:,ispin] = np.array(sorted_E_k)
      arrays['v_k'] = arrays['v_k'][:,idx]

  else:
    if rank == 0:
      print('OneDim bands deprecated. Use band_path and high_sym_points')

  # Angstrom to Bohr
  attributes['alat'] *= ANGSTROM_AU
  
