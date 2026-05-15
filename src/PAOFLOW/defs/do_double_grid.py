# Fourier interpolation on extended grid (zero padding)
def do_double_grid(data_controller):
    import numpy as np
    from mpi4py import MPI
    from scipy import fftpack as FFT

    from .communication import scatter_full
    from .zero_pad import zero_pad

    rank = MPI.COMM_WORLD.Get_rank()

    arrays, attr = data_controller.data_dicts()

    HRs = None
    if rank == 0:
        nawf, nk1, nk2, nk3 = attr['nawf'], attr['nk1'], attr['nk2'], attr['nk3']
        HRs = np.reshape(arrays['HRs'], (nawf**2, nk1, nk2, nk3, attr['nspin']))
    HRs = scatter_full(HRs, attr['npool'])

    snawf, nk1, nk2, nk3, nspin = HRs.shape
    nk1p = attr['nfft1']
    nk2p = attr['nfft2']
    nk3p = attr['nfft3']
    nfft1 = nk1p - nk1
    nfft2 = nk2p - nk2
    nfft3 = nk3p - nk3

    # Extended R to k (with zero padding)
    arrays['Hksp'] = np.empty((HRs.shape[0], nk1p, nk2p, nk3p, nspin), dtype=complex)

    for ispin in range(nspin):
        for n in range(HRs.shape[0]):
            arrays['Hksp'][n, :, :, :, ispin] = FFT.fftn(
                zero_pad(HRs[n, :, :, :, ispin], nk1, nk2, nk3, nfft1, nfft2, nfft3)
            )

    attr['nk1'] = nk1p
    attr['nk2'] = nk2p
    attr['nk3'] = nk3p
    attr['nkpnts'] = nk1p * nk2p * nk3p
