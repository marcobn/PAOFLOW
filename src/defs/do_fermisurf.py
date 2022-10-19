#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
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


def do_fermisurf ( data_controller ):
  import numpy as np
  from mpi4py import MPI
  from os.path import join
  from .communication import gather_full

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arry,attr = data_controller.data_dicts()

  #maximum number of bands crossing fermi surface
  ###### PARALLELIZATION
  E_kf = gather_full(arry['E_k'], attr['npool'])

  if rank == 0:
    if attr['verbose']:
      print('Writing bxsf file for Fermi Surface')

    nawf,nktot = attr['nawf'],attr['nkpnts']
    nk1,nk2,nk3 = attr['nk1'],attr['nk2'],attr['nk3']
    fermi_up,fermi_dw = attr['fermi_up'],attr['fermi_dw']

    E_ks = np.reshape(E_kf, (nk1,nk2,nk3,nawf,attr['nspin']))

    for ispin in range(attr['nspin']):

      ind_plot = []
      eigband = []

      for ib in range(nawf):
        E_k_min = np.amin(E_kf[:,ib,ispin])
        E_k_max = np.amax(E_kf[:,ib,ispin])
        btwUp = E_k_min < fermi_up and E_k_max > fermi_up
        btwDwn = E_k_min < fermi_dw and E_k_max > fermi_dw
        btwUaD = E_k_min > fermi_dw and E_k_max < fermi_up
        if btwUp or btwDwn or btwUaD:
          ind_plot.append(ib)
          eigband.append(E_ks[:,:,:,ib,ispin])

      feig = 'FermiSurf_%d.bxsf'%ispin
      eigband = np.array(eigband)
      data_controller.write_bxsf(feig, np.moveaxis(eigband,0,3), len(ind_plot), indices=ind_plot)

      for i,ib in enumerate(eigband):
        np.savez(join(attr['opath'],'Fermi_surf_band_%d_%d'%(ind_plot[i],ispin)), nameband=ib)

  comm.Barrier()
  E_kf = E_ks = None
