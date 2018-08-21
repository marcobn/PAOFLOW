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


def do_adaptive_smearing ( data_controller ):
    from numpy.linalg import norm
    import numpy as np

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    #----------------------
    # adaptive smearing as in Yates et al. Phys. Rev. B 75, 195121 (2007).
    #----------------------

    a_vectors = arrays['a_vectors']

    nawf = attributes['nawf']
    nspin = attributes['nspin']
    nkpnts = attributes['nkpnts']
    npks = arrays['pksp'].shape[0]

    deltakp = np.zeros((npks,nawf,nspin), dtype=float)

    diag = np.diag_indices(nawf)

    omega = attributes['alat']**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))
    dk = (8.*np.pi**3/omega/(nkpnts))**(1./3.)

    if smearing == 'gauss':
        afac = 0.7
    elif smearing == 'm-p':
        afac = 1.0        

    pksaux = np.ascontiguousarray(arrays['pksp'][:,:,diag[0],diag[1]])

    deltakp = np.zeros((npks,nawf,nspin),dtype=float)
    deltakp2 = np.zeros((npks,nawf,nawf,nspin),dtype=float)

    for n in range(nawf):
        deltakp[:,n] = norm(np.real(pksaux[:,:,n]), axis=1)
        for m in range(nawf):
            deltakp2[:,n,m,:] = norm(pksaux[:,:,n,:]-pksaux[:,:,m,:], axis=1)

    pksaux = None
    deltakp *= afac*dk
    deltakp2 *= afac*dk

    arrays['deltakp'] = deltakp
    arrays['deltakp2'] = deltakp2
