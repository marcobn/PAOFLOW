#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
#  M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
#  PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
#  Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np
from mpi4py import MPI

def build_Pn ( nawf, nkpnts, nspin, U ):
  Pn = np.zeros((nkpnts,U.shape[0]), dtype=float)
  for ispin in range(nspin):
    for ik in range(nkpnts):
      UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
      Pn[ik] = np.real(np.sum(np.conj(UU)*UU,axis=0))

  return np.mean(Pn,axis=0)


def do_projectability ( data_controller ):

  #----------------------
  # Building the Projectability
  #----------------------
  rank = MPI.COMM_WORLD.Get_rank()

  arry,attr = data_controller.data_dicts()

  pthr,shift = attr['pthr'],attr['shift']

  if rank != 0:
    attr['shift'] = None
  else:
    Pn = build_Pn(attr['nawf'], attr['nkpnts'], attr['nspin'], arry['U'])

    if attr['verbose']:
      print('Projectability vector ', Pn)

    # Count the number of projectable states
    bnd = len(np.where(Pn>attr['pthr'])[0])
    Pn = None

    if bnd == 0:
      raise Exception('No projectable bands!')

    if bnd >= arry['my_eigsmat'].shape[0]:
      print('\nWARNING: Number of projectable states is equal to the number of bands.\n\tIncrease nbnd in nscf calculation to include the required Null space, to where the states with bad projectability are shifted.\n')

    attr['bnd'] = bnd

    attr['shift'] = (np.amin(arry['my_eigsmat'][bnd,:,:]) if shift=='auto' else shift)

    if attr['verbose']:
      print('# of bands with good projectability > {} = {}'.format(attr['pthr'],bnd))
      if attr['bnd'] < attr['nbnds']:
        print('Range of suggested shift ', np.amin(arry['my_eigsmat'][bnd,:,:]), ' , ', np.amax(arry['my_eigsmat'][bnd,:,:]))

  # Broadcast 
  data_controller.broadcast_attribute('bnd')
  data_controller.broadcast_attribute('shift')
