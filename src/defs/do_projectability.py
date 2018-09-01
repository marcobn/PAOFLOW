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
#
import numpy as np
from mpi4py import MPI

def build_Pn ( nawf, nbnds, nkpnts, nspin, U ):
    Pn = 0.0
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

    arrays,attributes = data_controller.data_dicts()

    pthr = attributes['pthr']
    shift = attributes['shift']

    bnd = None
    if rank != 0:
        attributes['shift'] = None
    else:
        Pn = build_Pn(attributes['nawf'], attributes['nbnds'], attributes['nkpnts'], attributes['nspin'], arrays['U'])

        if attributes['verbose']:
            print('Projectability vector ',Pn)


        # Check projectability and decide bnd
        bnd = 0
        for n in range(attributes['nbnds']):
            if Pn[n] > attributes['pthr']:
                bnd += 1
        Pn = None
        attributes['bnd'] = bnd
        if 'shift' not in attributes or attributes['shift']=='auto':
            attributes['shift'] = (np.amin(arrays['my_eigsmat'][bnd,:,:]) if shift=='auto' else shift)

        if attributes['verbose']:
            print('# of bands with good projectability > {} = {}'.format(attributes['pthr'],bnd))
        if attributes['verbose'] and bnd < attributes['nbnds']:
            print('Range of suggested shift ', np.amin(arrays['my_eigsmat'][bnd,:,:]), ' , ', np.amax(arrays['my_eigsmat'][bnd,:,:]))


    # Broadcast 
    data_controller.broadcast_single_attribute('bnd')
    data_controller.broadcast_single_attribute('shift')
