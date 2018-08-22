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
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    snawf,nk1,nk2,nk3,nspin = arrays['Hksp'].shape
    nawf = attributes['nawf']
    nktot = nk1*nk2*nk3

    arrays['Hksp'] = np.reshape(arrays['Hksp'], (snawf,nktot,nspin))

##### PARALLELIZATION
    if rank == 0: print(arrays['Hksp'].shape)
    arrays['Hksp'] = gather_scatter(arrays['Hksp'], 1, attributes['npool'])
    if rank == 0: print(arrays['Hksp'].shape)

    snktot = arrays['Hksp'].shape[1]

    arrays['Hksp'] = np.reshape(np.moveaxis(arrays['Hksp'], 0, 1), (snktot,nawf,nawf,nspin), order='C')

    E_kaux = np.zeros((snktot,nawf,nspin), dtype=float)
    v_kaux = np.zeros((snktot,nawf,nawf,nspin), dtype=complex)

    for ispin in range(nspin):
        E_kaux[:,:,ispin], v_kaux[:,:,:,ispin] = diagonalize_Hksp(arrays['Hksp'][:,:,:,ispin])

    return(E_kaux, v_kaux)

def diagonalize_Hksp ( Hksp ):
    from numpy.linalg import eigh

    nawf = Hksp.shape[1]
    snktot = Hksp.shape[0]

    ekp = np.zeros((snktot,nawf), dtype=float)
    ekv = np.zeros((snktot,nawf,nawf), dtype=complex)

    for n in range(snktot):
        eigval,eigvec = eigh(Hksp[n,:,:], UPLO='U')
        ekp[n,:] = np.real(eigval)
        ekv[n,:,:] = eigvec

    return(ekp, ekv)
