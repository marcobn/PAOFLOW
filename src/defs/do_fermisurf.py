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


def do_fermisurf ( data_controller ):
  import numpy as np
  from mpi4py import MPI
  from os.path import join
  from .communication import gather_full

  rank = MPI.COMM_WORLD.Get_rank()

  arry,attr = data_controller.data_dicts()

  #maximum number of bands crossing fermi surface
###### PAR!!!!!
  E_k_full = gather_full(arry['E_k'], attr['npool'])

  nbndx_plot = 10
  nawf = attr['nawf']
  nktot = attr['nkpnts']
  fermi_up,fermi_dw = attr['fermi_up'],attr['fermi_dw']
  nk1,nk2,nk3 = attr['nk1'],attr['nk2'],attr['nk3']

  if rank == 0:
    E_k_rs = np.reshape(E_k_full, (nk1,nk2,nk3,nawf,attr['nspin']))

  for ispin in range(attr['nspin']):

    icount = 0
    eigband = (np.zeros((nk1,nk2,nk3,nbndx_plot),dtype=float) if rank==0 else None)

    if rank == 0:

      Efermi = 0.0
      ind_plot = np.zeros(nbndx_plot)

      for ib in range(nawf):
        E_k_min = np.amin(E_k_full[:,ib,ispin])
        E_k_max = np.amax(E_k_full[:,ib,ispin])
        btwUp = (E_k_min < fermi_up and E_k_max > fermi_up)
        btwDwn = (E_k_min < fermi_dw and E_k_max > fermi_dw)
        btwUaD = (E_k_min > fermi_dw and E_k_max < fermi_up)
        if btwUp or btwDwn or btwUaD:
          if ( icount >= nbndx_plot ):
            print('Too many bands contributing')
            MPI.COMM_WORLD.Abort()
          eigband[:,:,:,icount] = E_k_rs[:,:,:,ib,ispin]
          ind_plot[icount] = ib
          icount += 1

    feig = 'FermiSurf_%d.bxsf'%ispin
    data_controller.write_bxsf(feig, eigband, icount)

    for ib in range(icount):
      np.savez(join(attr['opath'],'Fermi_surf_band_%d_%d'%(ib,ispin)), nameband=eigband[:,:,:,ib])

  E_k_full = E_k_rs = None
