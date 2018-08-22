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

def do_momentum ( data_controller ):
  import numpy as np

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  nktot,_,nawf,nawf,nspin = arrays['dHksp'].shape

  arrays['pksp'] = np.zeros_like(arrays['dHksp'])

  for ispin in range(nspin):
    for ik in range(nktot):
      for l in range(3):
        arrays['pksp'][ik,l,:,:,ispin] = arrays['dHksp'][ik,l,:,:,ispin].dot(arrays['v_k'][ik,:,:,ispin])

  vec_cross = np.ascontiguousarray(np.conj(np.swapaxes(arrays['v_k'],1,2)))

  for ispin in range(nspin):
    for ik in range(nktot):
      for l in range(3):
        arrays['pksp'][ik,l,:,:,ispin] = vec_cross[ik,:,:,ispin].dot(arrays['pksp'][ik,l,:,:,ispin])
