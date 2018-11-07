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


def do_pao_eigh ( data_controller ):
  from communication import gather_scatter
  from numpy.linalg import eigh
  from mpi4py import MPI
  import numpy as np

  rank = MPI.COMM_WORLD.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  snktot,nawf,_,nspin = arrays['Hksp'].shape

  arrays['E_k'] = np.zeros((snktot,nawf,nspin), dtype=float)
  arrays['v_k'] = np.zeros((snktot,nawf,nawf,nspin), dtype=complex)

  for ispin in range(nspin):
    for n in range(snktot):
      arrays['E_k'][n,:,ispin],arrays['v_k'][n,:,:,ispin] = eigh(arrays['Hksp'][n,:,:,ispin], UPLO='U')


def do_eigh_calc ( HRaux, SRaux, kq, R, read_S ):
  import numpy as np
  from numpy import linalg as LAN

  # Compute bands on a selected mesh in the BZ

  nkpi = kq.shape[0]
  nawf = HRaux.shape[0]
  nspin = HRaux.shape[-1]

  Hks_int = band_loop_H(HRaux, kq, R)

  if read_S:
    Sks_int = band_loop_S(SRaux, kq, R)

  E_kp = np.empty((nkpi,nawf,nspin), dtype=float)
  v_kp = np.empty((nkpi,nawf,nawf,nspin), dtype=complex)

  for ispin in range(nspin):
    for ik in range(nkpi):
      if read_S:
        E_kp[ik,:,ispin],v_kp[ik,:,:,ispin] = LA.eigh(Hks_int[:,:,ik,ispin],Sks_int[:,:,ik])
      else:
        E_kp[ik,:,ispin],v_kp[ik,:,:,ispin] = LAN.eigh(Hks_int[:,:,ik,ispin],UPLO='U')

  return (E_kp, v_kp)


### R_wght assumed to be 1
def band_loop_H ( HRaux, kq, R ):
  import numpy as np

  nkpi = kq.shape[0]
  nawf,_,nk1,nk2,nk3,nspin = HRaux.shape
  auxh = np.empty((nawf,nawf,nkpi,nspin), dtype=complex)
  HRaux = np.reshape(HRaux, (nawf,nawf,nk1*nk2*nk3,nspin), order='C')

  for ik in range(nkpi):
    for ispin in range(nspin):
       auxh[:,:,ik,ispin] = np.sum(HRaux[:,:,:,ispin]*np.exp(2.j*np.pi*kq[ik,:].dot(R[:,:].T)), axis=2)

  return auxh


def band_loop_S ( SRaux, kq, R ):
  import numpy as np

  nkpi = kq.shape[0]
  nawf,_,nk1,nk2,nk3 = SRaux.shape
  auxs = np.empty((nawf,nawf,nkpi), dtype=complex)
  SRaux = np.reshape(SRaux, (nawf,nawf,nk1*nk2*nk3), order='C')

  for ik in range(nkpi):
    auxs[:,:,ik] = np.sum(SRaux[:,:,:]*np.exp(2.j*np.pi*kq[ik,:].dot(R[:,:].T)), axis=2)

  return auxs
