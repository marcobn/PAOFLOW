import numpy as np
from mpi4py import MPI

from ..utils.communication import gather_scatter

# initialize parallel execution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scipy import fftpack as FFT


def energy_and_velocity_degenerate_groups(
    velocity_groups: list[np.ndarray], energy_groups: list[np.ndarray]
) -> list[np.ndarray]:
    """Bands degenerate in both the energy and the velocity component, at one k-point.

    Parameters
    ----------
    velocity_groups : list of ndarray of int
        Groups of consecutive bands with equal diagonal velocity (from :func:`get_degeneracies`).
    energy_groups : list of ndarray of int
        Groups of consecutive bands with equal energy (``degen`` of :func:`pao_eigh`).

    Returns
    -------
    list of ndarray of int
        The intersections with at least two bands, as consecutive index arrays.

    Notes
    -----
    Degenerate perturbation theory for :math:`\\partial^2 E/\\partial k_i\\partial k_j` diagonalises
    the curvature in a subspace of states with the same energy and the same velocity. Two
    non-degenerate bands with equal velocities (both zero at a band extremum or on a mirror plane)
    are not such a subspace; diagonalising them together reassigned their curvatures by size.
    """
    energy_sets = [set(int(b) for b in group) for group in energy_groups]
    groups = []
    for velocity_group in velocity_groups:
        members = set(int(b) for b in velocity_group)
        for energy_set in energy_sets:
            common = sorted(members & energy_set)
            if len(common) >= 2:
                groups.append(np.array(common, dtype=int))
    return groups


IJ_PAIRS = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]], dtype=int)
"""The six unique Cartesian pairs ``xx, yy, zz, xy, xz, yz`` of a symmetric tensor."""


def do_d2Hd2k_ij(
    Hksp: np.ndarray,
    Dnm: np.ndarray,
    Rfft: np.ndarray,
    alat: float,
    npool: int,
    ipol: int,
    jpol: int,
) -> np.ndarray:
    """Compute one second-order derivative :math:`d^2H/dk_i dk_j` of the k-space Hamiltonian.

    Parameters
    ----------
    Hksp : np.ndarray, shape ``(snawf, nk1, nk2, nk3, nspin)``
        Distributed real-space Hamiltonian in the FFT representation, where
        each element already contains the factor :math:`i \\cdot a_{\\text{lat}}`
        (i.e. ``HR * 1j * alat``).
    Dnm : np.ndarray, shape ``(snawf, 3)``
        Cartesian factors, one per distributed real-space element ``n``,
        entering the three additional terms of the second derivative
        (see Notes).
    Rfft : np.ndarray, shape ``(nk1, nk2, nk3, 3)``
        Real-space grid vectors used as multiplication factors in FFT-based
        gradient computation.
    alat : float
        Lattice constant in Bohr radii, used to scale the real-space Hamiltonian
        prior to the FFT.
    npool : int
        Number of MPI pools used for the :func:`gather_scatter` redistribution.
    ipol, jpol : int
        Cartesian directions of the derivative.

    Returns
    -------
    d2Hksp : np.ndarray, shape ``(nawf, nawf, nkpnts, nspin)``
        :math:`d^2 H / dk_i dk_j` in the PAO basis on this rank's k-points.

    Notes
    -----
    The second derivative is obtained via the FFT convolution

    .. math::

        d^2 H(\\mathbf{k}) / dk_i dk_j
            = \\mathcal{F}\\left[ R_i R_j \\cdot H(\\mathbf{R}) \\cdot i \\cdot a_{\\text{lat}} \\right]
            + D_i D_j \\cdot \\mathcal{F}\\left[ H(\\mathbf{R}) \\cdot i / a_{\\text{lat}} \\right]
            + D_i \\cdot \\mathcal{F}\\left[ R_j \\cdot H(\\mathbf{R}) \\cdot i \\right]
            + D_j \\cdot \\mathcal{F}\\left[ R_i \\cdot H(\\mathbf{R}) \\cdot i \\right]

    where :math:`R_i` is the *i*-th Cartesian component of the real-space
    lattice vector grid ``Rfft`` and :math:`D_i` is ``Dnm[:, i]``.  The four
    terms are the expansion of
    :math:`\\mathcal{F}[(a_{\\text{lat}} R_i + D_i)(a_{\\text{lat}} R_j + D_j)
    \\cdot H(\\mathbf{R}) \\cdot i / a_{\\text{lat}}]`, i.e.
    :math:`-(a_{\\text{lat}} R_i + D_i)(a_{\\text{lat}} R_j + D_j)` times
    :math:`H(\\mathbf{R})` under the transform, the form the sparse
    backend assembles bond by bond.  Projection onto the Bloch states and the
    curvature itself are in
    :func:`~PAOFLOW.spectrum.do_band_curvature.band_curvature_k_ij`.
    """
    # ----------------------
    # Compute the gradient of the k-space Hamiltonian
    # ----------------------
    Rfft = np.transpose(Rfft, (3, 0, 1, 2))

    num_n, nk1, nk2, nk3, nspin = Hksp.shape

    comm.Barrier()
    ########################################
    ### real space grid replaces k space ###
    ########################################
    d2Hksp = np.zeros((num_n, nk1, nk2, nk3, nspin), dtype=complex, order='C')

    RIJ = Rfft[ipol] * Rfft[jpol]

    for ispin in range(d2Hksp.shape[4]):
        for n in range(d2Hksp.shape[0]):
            # because of the way this is coded...Hksp is actually HR*1.0j*alat
            d2Hksp[n, :, :, :, ispin] = (
                FFT.fftn(RIJ * Hksp[n, :, :, :, ispin] * 1.0j * alat)
                + Dnm[n, ipol] * Dnm[n, jpol] * FFT.fftn(Hksp[n, :, :, :, ispin] * 1.0j / alat)
                + Dnm[n, ipol] * FFT.fftn(Rfft[jpol] * Hksp[n, :, :, :, ispin] * 1.0j)
                + Dnm[n, jpol] * FFT.fftn(Rfft[ipol] * Hksp[n, :, :, :, ispin] * 1.0j)
            )

    # gather the arrays into flattened dHk
    d2Hksp = np.reshape(d2Hksp, (num_n, nk1 * nk2 * nk3, nspin), order='C')
    d2Hksp = gather_scatter(d2Hksp, 1, npool)
    nawf = int(np.sqrt(d2Hksp.shape[0]))

    d2Hksp = np.reshape(d2Hksp, (nawf, nawf, d2Hksp.shape[1], nspin), order='C')
    comm.Barrier()
    return d2Hksp
