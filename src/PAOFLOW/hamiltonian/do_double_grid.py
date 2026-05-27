# Fourier interpolation on extended grid (zero padding)
def do_double_grid(data_controller):
    """Interpolate the Hamiltonian onto a finer k-grid via zero-padding in real space.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required array: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``).
        Required attributes: ``nawf``, ``nk1``, ``nk2``, ``nk3``, ``nspin``,
        ``nfft1``, ``nfft2``, ``nfft3``, ``npool``.

    Returns
    -------
    None
        Adds or updates the following entries in ``data_controller.data_arrays``
        and ``data_controller.data_attributes``:

        - ``Hksp`` : np.ndarray, shape ``(snawf, nfft1, nfft2, nfft3, nspin)``,
          complex — the k-space Hamiltonian on the extended, interpolated grid,
          distributed over MPI pools.

        Updates attributes: ``nk1 = nfft1``, ``nk2 = nfft2``, ``nk3 = nfft3``,
        ``nkpnts = nfft1 * nfft2 * nfft3``.

    Notes
    -----
    Fourier interpolation is achieved by inserting zeros in the frequency domain
    before the inverse transform.  Specifically, the real-space Hamiltonian
    :math:`H(\\mathbf{R})` (``HRs``) is zero-padded from
    ``(nk1, nk2, nk3)`` to ``(nfft1, nfft2, nfft3)`` using :func:`zero_pad`,
    which correctly preserves Hermitian symmetry (:math:`H(-\\mathbf{R}) =
    H^\\dagger(\\mathbf{R})`) for both even and odd grid sizes.  A forward FFT
    then yields the interpolated :math:`H(\\mathbf{k})` on the dense grid.

    The reshaped ``HRs`` array (``nawf**2`` rows) is scattered across MPI pools
    before processing; only rank 0 performs the initial reshape.
    """
    import numpy as np
    from mpi4py import MPI
    from scipy import fftpack as FFT

    from ..utils.communication import scatter_full
    from ..utils.zero_pad import zero_pad

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
