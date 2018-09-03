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

# Fourier interpolation on extended grid (zero padding)
def do_double_grid ( data_controller ):
    import numpy as np
    from mpi4py import MPI
    from zero_pad import zero_pad
    from scipy import fftpack as FFT

    rank = MPI.COMM_WORLD.Get_rank()

    arrays,attributes = data_controller.data_dicts()

    snawf,nk1,nk2,nk3,nspin = arrays['HRs'].shape
    nk1p = attributes['nfft1']
    nk2p = attributes['nfft2']
    nk3p = attributes['nfft3']
    nfft1 = nk1p-nk1
    nfft2 = nk2p-nk2
    nfft3 = nk3p-nk3

    # Extended R to k (with zero padding)
    arrays['Hksp']  = np.empty((arrays['HRs'].shape[0],nk1p,nk2p,nk3p,nspin), dtype=complex)

    for ispin in range(nspin):
        for n in range(arrays['HRs'].shape[0]):
            arrays['Hksp'][n,:,:,:,ispin] = FFT.fftn(zero_pad(arrays['HRs'][n,:,:,:,ispin],
                                                      nk1,nk2,nk3,nfft1,nfft2,nfft3))

    attributes['nk1'] = nk1p
    attributes['nk2'] = nk2p
    attributes['nk3'] = nk3p
    attributes['nkpnts'] = nk1p*nk2p*nk3p
