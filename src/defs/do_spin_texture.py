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

import os

def do_spin_texture ( data_controller ):
  import numpy as np
  from mpi4py import MPI
  from communication import gather_full

  rank = MPI.COMM_WORLD.Get_rank()

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  nspin = attributes['nspin']
  nktot = attributes['nkpnts']
  fermi_up,fermi_dw = attributes['fermi_up'],attributes['fermi_dw']
  nawf,nk1,nk2,nk3 = attributes['nawf'],attributes['nk1'],attributes['nk2'],attributes['nk3']

  E_k_full = gather_full(arrays['E_k'], attributes['npool'])

  for ispin in range(nspin):
    ind_plot = np.zeros(nawf, dtype=int)
    icount = None
    if rank == 0:
      icount = 0
      for ib in range(nawf):
        E_k_min = np.amin(E_k_full[:,ib,ispin])
        E_k_max = np.amax(E_k_full[:,ib,ispin])
        btwUp = (E_k_min < fermi_up and E_k_max > fermi_up)
        btwDwn = (E_k_min < fermi_dw and E_k_max > fermi_dw)
        btwUaD = (E_k_min > fermi_dw and E_k_max < fermi_up)
        if btwUp or btwDwn or btwUaD:
          ind_plot[icount] = ib
          icount += 1

    icount = MPI.COMM_WORLD.bcast(icount)

    Sj = arrays['Sj']
    snktot = arrays['v_k'].shape[0]
    sktxtaux = np.zeros((snktot,3,nawf,nawf),dtype=complex)

    # Compute matrix elements of the spin operator
    for ik in range(snktot):
      for ispin in range(nspin):
        for l in range(3):
          sktxtaux[ik,l,:,:] = np.conj(arrays['v_k'][ik,:,:,ispin].T).dot(Sj[l,:,:]).dot(arrays['v_k'][ik,:,:,ispin])

    sktxt = gather_full(sktxtaux, attributes['npool'])
    sktxtaux = None

    if rank == 0:
      sktxt = np.reshape(sktxt, (nk1,nk2,nk3,3,nawf,nawf), order='C')

## FILE OUTPUT
##      for ib in range(icount):
##        np.savez(os.path.join(attributes['inputpath'],'spin_text_band_'+str(ib)), spinband = sktxt[:,:,:,:,ind_plot[ib],ind_plot[ib]])

    sktxt = None

  E_k_full = None
