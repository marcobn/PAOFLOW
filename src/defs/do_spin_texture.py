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


### In serious need of update
def do_spin_texture ( data_controller ):
  import os
  import numpy as np
  from mpi4py import MPI
  from .communication import gather_full

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  fermi_up,fermi_dw = attributes['fermi_up'],attributes['fermi_dw']
  nawf,nk1,nk2,nk3 = attributes['nawf'],attributes['nk1'],attributes['nk2'],attributes['nk3']
  E_k_full = gather_full(arrays['E_k'], attributes['npool'])
 
  ind_plot = []
  icount = None
  if rank == 0:
    icount = 0
    for ib in range(nawf):
      E_k_min = np.amin(E_k_full[:,ib,0])
      E_k_max = np.amax(E_k_full[:,ib,0])
      btwUp = (E_k_min < fermi_up and E_k_max > fermi_up)
      btwDwn = (E_k_min < fermi_dw and E_k_max > fermi_dw)
      btwUaD = (E_k_min > fermi_dw and E_k_max < fermi_up)
      if btwUp or btwDwn or btwUaD:
        ind_plot.append(ib)
        icount += 1

  icount = comm.bcast(icount)
  ind_plot = comm.bcast(ind_plot)

  Sj = arrays['Sj']
  snktot = arrays['v_k'].shape[0]
  sktxtaux = np.zeros((snktot,3,nawf,nawf), dtype=complex)

  # Compute matrix elements of the spin operator
  for ik in range(snktot):
      for l in range(3):
        sktxtaux[ik,l,:,:] = np.conj(arrays['v_k'][ik,:,:,0].T).dot(Sj[l,:,:]).dot(arrays['v_k'][ik,:,:,0])

  sktxtaux = np.take(np.diagonal(sktxtaux,axis1=2,axis2=3), ind_plot, axis=2)
  sktxt = gather_full(np.ascontiguousarray(sktxtaux), attributes['npool'])
  sktxtaux = None

  if rank == 0:
    if 'kq' in arrays and E_k_full.shape[0] == arrays['kq'].shape[1]:
      f=open(os.path.join(attributes['opath'],'spin-texture-bands'+'.dat'),'w')
      for ik in range(E_k_full.shape[0]):
        for ib in range(icount):
          idx=ind_plot[ib]
          f.write('\t'.join(['%d'%ik]+['% 5.8f'%E_k_full[ik,idx]]+['% 5.8f'%j for j in sktxt[ik,:,ib].real])+'\n')
        f.write("\n")
      f.close()
    else:
      sktxt = np.reshape(sktxt, (nk1,nk2,nk3,3,icount), order='C')
      for ib in range(icount):
        np.savez(os.path.join(attributes['opath'],'spin_text_band_'+str(ib)), spinband=sktxt[:,:,:,:,ib])


  sktxt = None
  E_k_full = None
