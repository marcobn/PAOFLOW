# *************************************************************************************
# *                                                                                   *
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2018      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
#
#  This file is part of AFLOW software.
#
#  AFLOW is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *************************************************************************************

from PAOFLOW import PAOFLOW
import numpy as np

if __name__== '__main__':

  paoflow = PAOFLOW.PAOFLOW(savedir='pt.save')
  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()

  data_controller = paoflow.data_controller
  arry,attr = data_controller.data_dicts()

  # Grid size
  ck1,ck2 = 20,20
  vk1,vk2 = np.array([1,0,0]),np.array([0,1,0])

  # L point
  origin = -.5 * arry['b_vectors'][0]
  origin -= .5 * (vk1 + vk2)

  # k-points
  kq = []
  for x in np.linspace(0, 1, ck1):
    for y in np.linspace(0, 1, ck2):
      kq.append(arry['b_vectors'].dot(x*vk1 + y*vk2))
  kq = np.array(kq)
  arry['kq'] = kq.T
  #for n in range(ck1*ck2):
  #  kq[n,:] = kq[n,:].dot(arry['b_vectors'])

  npool = attr['npool']
  nawf,_,nk1,nk2,nk3,nspin = arry['HRs'].shape

  # Define real space lattice vectors
  from PAOFLOW.defs.get_R_grid_fft import get_R_grid_fft
  get_R_grid_fft(data_controller, nk1, nk2, nk3)

  from PAOFLOW.defs.communication import scatter_full
  kq_aux = scatter_full(kq, npool).T
  nks = kq_aux.shape[1]

  HRs = np.reshape(arry['HRs'], (nawf,nawf,nk1*nk2*nk3,nspin))
  kdot = np.tensordot(arry['R'], 2j*np.pi*kq_aux, axes=([1],[0]))
  np.exp(kdot, kdot)
  Hks_aux = np.zeros((nawf,nawf,nks,nspin), dtype=complex)

  for ispin in range(nspin):
    Hks_aux[...,ispin] = np.tensordot(HRs[...,ispin], kdot, axes=([2],[0]))

  kdot  = None

  E_k_aux = np.zeros((nks,nawf,nspin), dtype=float)
  v_k_aux = np.zeros((nks,nawf,nawf,nspin), dtype=complex)

  from scipy import linalg as spl
  for ispin in range(nspin):
    for ik in range(nks):
      E_k_aux[ik,:,ispin],v_k_aux[ik,:,:,ispin] = spl.eigh(Hks_aux[:,:,ik,ispin], b=(None), lower=False, overwrite_a=True, overwrite_b=True, turbo=True, check_finite=True)

  arry['E_k'] = E_k_aux
  arry['v_k'] = v_k_aux

  paoflow.spin_texture()
  paoflow.finish_execution()

