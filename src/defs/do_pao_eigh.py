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

def do_pao_eigh ( data_controller ):
  from communication import gather_scatter
  from numpy.linalg import eigh
  from mpi4py import MPI

  rank = MPI.COMM_WORLD.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  snktot,nawf,_,nspin = arrays['Hksp'].shape

  arrays['E_k'] = np.zeros((snktot,nawf,nspin), dtype=float)
  arrays['v_k'] = np.zeros((snktot,nawf,nawf,nspin), dtype=complex)

  for ispin in range(nspin):
    for n in range(snktot):
      arrays['E_k'][n,:,ispin],arrays['v_k'][n,:,:,ispin] = eigh(arrays['Hksp'][n,:,:,ispin], UPLO='U')
