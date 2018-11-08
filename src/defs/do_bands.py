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


def bands_calc ( data_controller ):
  import numpy as np
  from scipy import linalg as LA
  from communication import scatter_full, gather_full

  arrays,attributes = data_controller.data_dicts()

  npool = attributes['npool']
  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape

  kq_aux = scatter_full(arrays['kq'].T, npool).T
 
## Ortho before bands
##  Sks_aux = (None if not attributes['non_ortho'] else band_loop_S(data_controller))
  Hks_aux = band_loop_H(data_controller)

  E_kp_aux = np.zeros((kq_aux.shape[1],nawf,nspin), dtype=float,order="C")
  v_kp_aux = np.zeros((kq_aux.shape[1],nawf,nawf,nspin), dtype=complex,order="C")

  for ispin in range(nspin):
    for ik in range(kq_aux.shape[1]):
      E_kp_aux[ik,:,ispin],v_kp_aux[ik,:,:,ispin] = LA.eigh(Hks_aux[:,:,ik,ispin], b=(None), lower=False, overwrite_a=True, overwrite_b=True, turbo=True, check_finite=True)
## Orthogonalize before bands?
##      E_kp_aux[ik,:,ispin], v_kp_aux[ik,:,:,ispin] = LA.eigh(Hks_aux[:,:,ik,ispin], b=(Sks_aux[:,:,ik] if attributes['non_ortho'] else None), lower=False, overwrite_a=True, overwrite_b=True, turbo=True, check_finite=True)

  Hks_aux = None
  Sks_aux = None

  return E_kp_aux, v_kp_aux


def band_loop_H ( data_controller ):
  import numpy as np

  arrays,attributes = data_controller.data_dicts()

  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape

  arrays['HRs'] = np.reshape(arrays['HRs'], (nawf,nawf,attributes['nkpnts'],nspin), order='C')
  kdot = np.tensordot(arrays['R'], 2.0j*np.pi*arrays['kq'], axes=([1],[0]))
  np.exp(kdot, kdot)
  Haux = np.zeros((nawf,nawf,arrays['kq'].shape[1],nspin), dtype=complex, order="C")

  for ispin in range(nspin):
    Haux[:,:,:,ispin] = np.tensordot(arrays['HRs'][:,:,:,ispin], kdot, axes=([2],[0]))

  arrays['HRs'] = np.reshape(arrays['HRs'], (nawf,nawf,nk1,nk2,nk3,nspin), order='C')

  kdot  = None
  return Haux


def band_loop_S ( data_controller ):
  import cmath
  import numpy as np

  arrays,attributes = data_controller.data_dicts()

  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape

  nsize = arrays['kq'].shape[1]
  Saux  = np.zeros((nawf,nawf,nsize), dtype=complex)

  for ik in range(nsize):
    for i in range(nk1):
      for j in range(nk2):
        for k in range(nk3):
#### Why cmath here?
          phase = arrays['R_wght'][arrays['idx'][i,j,k]]*cmath.exp(2.0*np.pi*arrays['kq'][:,ik].dot(arrays['R'][arrays['idx'][i,j,k],:])*1j)
          Saux[:,:,ik] += phase*arrays['SRs'][:,:,i,j,k]

  return Saux


def do_ortho ( Hks, Sks ):
  import numpy as np
  from scipy import linalg as LA
  from numpy import linalg as LAN

  # If orthogonality is required, we have to apply a basis change to Hks as
  # Hks -> Sks^(-1/2)*Hks*Sks^(-1/2)

  nawf,_,nkpnts,nspin = Hks.shape
  S2k  = np.zeros((nawf,nawf,nkpnts), dtype=complex)
  for ik in range(nkpnts):
    S2k[:,:,ik] = LAN.inv(LA.sqrtm(Sks[:,:,ik]))

  Hks_o = np.zeros((nawf,nawf,nkpnts,nspin), dtype=complex)
  for ispin in range(nspin):
    for ik in range(nkpnts):
      Hks_o[:,:,ik,ispin] = np.dot(S2k[:,:,ik], Hks[:,:,ik,ispin]).dot(S2k[:,:,ik])

  return Hks_o


def do_bands ( data_controller ):
  import numpy as np
  from mpi4py import MPI
  from constants import ANGSTROM_AU
  from communication import gather_full
  from get_R_grid_fft import get_R_grid_fft
  from kpnts_interpolation_mesh import kpnts_interpolation_mesh

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
    nawf = attributes['nawf']
    nk1 = attributes['nk1']
    nk2 = attributes['nk2']
    nk3 = attributes['nk3']
    nspin = attributes['nspin']
    nktot = attributes['nkpnts']

    # Define real space lattice vectors
    get_R_grid_fft(data_controller)

    # Define k-point mesh for bands interpolation
    kpnts_interpolation_mesh(data_controller)

    nkpi = arrays['kq'].shape[1]
    for n in range(nkpi):
      arrays['kq'][:,n] = np.dot(arrays['kq'][:,n], arrays['b_vectors'])

    # Compute the bands along the path in the IBZ
    arrays['E_k'],arrays['v_k'] = bands_calc(data_controller)

    E_kp = gather_full(arrays['E_k'], attributes['npool'])
    data_controller.write_bands('bands', E_kp)
    E_kp = None

  # 1D Bands not implemented
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
