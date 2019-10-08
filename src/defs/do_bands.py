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

  E_kp_aux = np.zeros((kq_aux.shape[1],nawf,nspin), dtype=float, order="C")
  v_kp_aux = np.zeros((kq_aux.shape[1],nawf,nawf,nspin), dtype=complex, order="C")

  for ispin in range(nspin):
    for ik in range(kq_aux.shape[1]):
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
    Haux[:,:,:,ispin] = np.tensordot(HRs[:,:,:,ispin], kdot, axes=([2],[0]))

  kdot  = None
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
    kpnts_interpolation_mesh(data_controller)

    nkpi = arrays['kq'].shape[1]
    for n in range(nkpi):
      arrays['kq'][:,n] = np.dot(arrays['kq'][:,n], arrays['b_vectors'])

    # Compute the bands along the path in the IBZ
    arrays['E_k'],arrays['v_k'] = bands_calc(data_controller)

#### 1D Bands not implemented
  else:
    if rank == 0:
      print('OneDim bands not implemented in PAOFLOW_Class')
      #----------------------
      # FFT interpolation along a single directions in the BZ
      #----------------------
      #if rank == 0 and verbose: print('... computing bands along a line')
      #if rank == 0: bands_calc_1D(Hks,inputpath)

  # Angstrom to Bohr
  attributes['alat'] *= ANGSTROM_AU
