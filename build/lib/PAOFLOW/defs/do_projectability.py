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

def build_Pn ( nawf, nbnds, nkpnts, nspin, U ):
  Pn = 0.
  for ispin in range(nspin):
    for ik in range(nkpnts):
      UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
      Pn += np.real(np.sum(np.conj(UU)*UU,axis=0))/nkpnts/nspin
  return Pn


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
    Pn = build_Pn(attr['nawf'], attr['nbnds'], attr['nkpnts'], attr['nspin'], arry['U'])

    if attr['verbose']:
      print('Projectability vector ', Pn)

    # Check projectability and decide bnd
    bnd = 0
    for n in range(attr['nbnds']):
      if Pn[n] > attr['pthr']:
        bnd += 1

    Pn = None
    attr['bnd'] = maxbnd = bnd
    warn_txt = 'WARNING: All bands meet the projectability threshold. Consider increasing nbnd in QE.'
    if bnd == attr['nawf']:
      maxbnd = bnd-1
      print(warn_txt)

    if 'shift' not in attr or attr['shift']=='auto':
      if maxbnd >= arry['my_eigsmat'].shape[0]:
        maxbnd = arry['my_eigsmat'].shape[0] - 1
        print(warn_txt)
      shift_v = np.amin(arry['my_eigsmat'][maxbnd,:,:])
      attr['shift'] = (shift_v if shift=='auto' else shift)

    if attr['verbose']:
      print('# of bands with good projectability > {} = {}'.format(attr['pthr'],bnd))
    if attr['verbose'] and bnd < attr['nbnds']:
      print('Range of suggested shift ', np.amin(arry['my_eigsmat'][maxbnd,:,:]), ' , ', np.amax(arry['my_eigsmat'][maxbnd,:,:]))

  # Broadcast 
  data_controller.broadcast_attribute('bnd')
  data_controller.broadcast_attribute('shift')
