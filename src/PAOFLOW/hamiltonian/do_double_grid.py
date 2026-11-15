# Fourier interpolation on extended grid (zero padding or Wigner-Seitz images)
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..DataController import DataController


def do_double_grid(data_controller: DataController, interpolation: str = 'zero_padding') -> None:
    """Interpolate the Hamiltonian onto a finer k-grid (zero padding or Wigner-Seitz images).

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required array: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``), and
        ``a_vectors`` for the Wigner-Seitz route.
        Required attributes: ``nawf``, ``nk1``, ``nk2``, ``nk3``, ``nspin``,
        ``nfft1``, ``nfft2``, ``nfft3``, ``npool``.
    interpolation : {'zero_padding', 'wigner_seitz'}, optional
        How the real-space grid is mapped to lattice vectors before the forward FFT.
        ``'zero_padding'`` (default) pads the parallelepiped of grid vectors with zeros
        (:func:`zero_pad`); ``'wigner_seitz'`` places every grid vector at its shortest image in
        the supercell lattice (:mod:`PAOFLOW.hamiltonian.wigner_seitz`). Both cost one FFT per
        orbital pair and reproduce the DFT mesh exactly.

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

    Raises
    ------
    ValueError
        For an unknown ``interpolation``, or (Wigner-Seitz) when the fine grid cannot hold every
        image at its own lattice vector (the gradients would then belong to another interpolant).

    Notes
    -----
    Fourier interpolation is achieved by inserting zeros in the frequency domain
    before the inverse transform.  Specifically, the real-space Hamiltonian
    :math:`H(\\mathbf{R})` (``HRs``) is zero-padded from
    ``(nk1, nk2, nk3)`` to ``(nfft1, nfft2, nfft3)`` using :func:`zero_pad`,
    which correctly preserves Hermitian symmetry (:math:`H(-\\mathbf{R}) =
    H^\\dagger(\\mathbf{R})`) for both even and odd grid sizes.  A forward FFT
    then yields the interpolated :math:`H(\\mathbf{k})` on the dense grid.

    The Wigner-Seitz route evaluates

    .. math::

        H(\\mathbf k_j) = \\sum_{\\mathbf n} w_{\\mathbf n}\\,
            e^{-2\\pi i\\,\\mathbf k_j\\cdot\\mathbf n}\\,\\mathtt{HRs}[\\mathbf n \\bmod N]

    over the Wigner-Seitz images :math:`\\mathbf n` with their degeneracy weights. On the fine grid
    :math:`\\mathbf k_j = j/M` the phase depends on :math:`\\mathbf n \\bmod M` only, so the sum is
    the forward FFT of the weighted hoppings placed at :math:`\\mathbf n \\bmod M`.

    The reshaped ``HRs`` array (``nawf**2`` rows) is scattered across MPI pools
    before processing; only rank 0 performs the initial reshape.
    """
    import numpy as np
    from mpi4py import MPI
    from scipy import fftpack as FFT

    from ..utils.communication import scatter_full
    from ..utils.zero_pad import zero_pad

    rank = MPI.COMM_WORLD.Get_rank()

    if interpolation not in ('zero_padding', 'wigner_seitz'):
        raise ValueError(
            "interpolation must be 'zero_padding' or 'wigner_seitz', got %r" % (interpolation,)
        )

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

    if interpolation == 'wigner_seitz':
        from .wigner_seitz import (
            check_fine_grid_holds_images,
            fine_grid_placement,
            wigner_seitz_fine_grid_array,
            wigner_seitz_images,
        )

        images = wigner_seitz_images((nk1, nk2, nk3), np.asarray(arrays['a_vectors'], dtype=float))
        check_fine_grid_holds_images(images, (nk1p, nk2p, nk3p))
        placement = fine_grid_placement(images, (nk1p, nk2p, nk3p))

    # Extended R to k (zero padding or Wigner-Seitz images)
    arrays['Hksp'] = np.empty((HRs.shape[0], nk1p, nk2p, nk3p, nspin), dtype=complex)

    for ispin in range(nspin):
        for n in range(HRs.shape[0]):
            if interpolation == 'wigner_seitz':
                padded = wigner_seitz_fine_grid_array(HRs[n, :, :, :, ispin], images, placement)
            else:
                padded = zero_pad(HRs[n, :, :, :, ispin], nk1, nk2, nk3, nfft1, nfft2, nfft3)
            arrays['Hksp'][n, :, :, :, ispin] = FFT.fftn(padded)

    attr['nk1'] = nk1p
    attr['nk2'] = nk2p
    attr['nk3'] = nk3p
    attr['nkpnts'] = nk1p * nk2p * nk3p
