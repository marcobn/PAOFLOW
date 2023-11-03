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

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def do_dos ( data_controller, emin, emax, ne, delta ):

  arry,attr = data_controller.data_dicts()
  bnd = attr['bnd']
  netot = attr['nkpnts']*bnd
  emax = np.amin(np.array([attr['shift'], emax]))
  arry['dos'] = np.empty((ne,), dtype=float)
  # DOS calculation with gaussian smearing
  ene = np.linspace(emin, emax, ne)

  if rank == 0 and attr['verbose']:
    print('Writing DoS Files')

  for ispin in range(attr['nspin']):

    dosaux = np.zeros((ne), order="C")

    E_k = arry['E_k'][:,:bnd,ispin]

    for n in range(ne):
      dosaux[n] = np.sum(np.exp(-((ene[n]-E_k)/delta)**2))

    dos = np.zeros((ne), dtype=float) if rank == 0 else None

    comm.Reduce(dosaux,dos,op=MPI.SUM)
    dosaux = None

    if rank == 0:
      dos *= float(bnd)/(float(netot)*np.sqrt(np.pi)*delta)
      arry['dos'] = dos
    fdos = 'dos_%s.dat'%str(ispin)
    data_controller.write_file_row_col(fdos, ene, dos)
    data_controller.broadcast_single_array('dos', dtype=float)
    #return dos if rank==0 else None

def do_dos_adaptive ( data_controller, emin, emax, ne ):
  from .smearing import gaussian, metpax

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arry,attr = data_controller.data_dicts()

  # DOS calculation with adaptive smearing
  ene = np.linspace(emin, emax, ne)
  arry['dosdk'] = np.empty((ne,), dtype=float)

  bnd = attr['bnd']
  netot = attr['nkpnts']*bnd

  if rank == 0 and attr['verbose']:
    print('Writing Adaptive DoS Files')

  for ispin in range(attr['nspin']):

    E_k = arry['E_k'][:,:bnd,ispin].reshape(arry['E_k'].shape[0]*bnd)
    delta = np.ravel(arry['deltakp'][:,:bnd,ispin], order='C')

    dosaux = np.zeros((ne), dtype=float)

    for n in range(ne):
      if attr['smearing'] == 'gauss':
        # adaptive Gaussian smearing
        dosaux[n] = np.sum(gaussian(ene[n],E_k,delta))

      elif attr['smearing'] == 'm-p':
        # adaptive Methfessel and Paxton smearing
        dosaux[n] = np.sum(metpax(ene[n],E_k,delta))

    dos = np.zeros((ne), dtype=float) if rank==0 else None
    comm.Reduce(dosaux, dos, op=MPI.SUM)
    dosaux = None

    if rank == 0:
      dos *= float(bnd)/netot
      arry['dosdk'] = dos
    fdosdk = 'dosdk_%s.dat'%str(ispin)
    data_controller.write_file_row_col(fdosdk, ene,dos)
    data_controller.broadcast_single_array('dosdk', dtype=float)

