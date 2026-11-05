import numpy as np
from mpi4py import MPI

from ..utils.communication import load_balancing
from ..projection.do_atwfc_proj import calc_atwfc_k, calc_gkspace, fft_allwfc_G2R, ortho_atwfc_k
from ..writers.write2xsf import write2xsf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_density(data_controller, nr1, nr2, nr3):
    """Compute the real-space electron density and write it to an XSF file.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``basis``, ``v_k`` (shape ``(nkpnts, nawf, bnd, nspin)``),
        ``E_k`` (shape ``(nkpnts, bnd, nspin)``).
        Required attributes: ``nspin``, ``nkpnts``, ``bnd``, ``omega``,
        ``outputdir``, ``verbose``.
    nr1 : int
        Number of real-space grid points along the first lattice vector.
    nr2 : int
        Number of real-space grid points along the second lattice vector.
    nr3 : int
        Number of real-space grid points along the third lattice vector.

    Returns
    -------
    None
        For each spin channel ``ispin``, writes an XSF file
        ``{outputdir}/density_{ispin}.xsf`` containing the real-space
        electron density :math:`\\rho(\\mathbf{r})`.

    Notes
    -----
    The electron density is accumulated as

    .. math::

        \\rho(\\mathbf{r}) = \\sum_{n\\mathbf{k},\\, E_{n\\mathbf{k}} \\leq 0}
            \\frac{2}{N_k \\Omega}
            \\left| \\sum_\\mu c^\\mu_{n\\mathbf{k}}\\, \\phi^\\mu(\\mathbf{r}) \\right|^2

    where :math:`c^\\mu_{n\\mathbf{k}}` are the eigenvector coefficients in the
    PAO basis, :math:`\\phi^\\mu` are the real-space atomic wavefunctions
    (evaluated on the ``(nr1, nr2, nr3)`` grid), :math:`N_k` is the total
    number of k-points, and :math:`\\Omega` is the unit-cell volume.  The
    sum runs over all occupied states (eigenvalues :math:`\\leq 0`).

    Work is distributed across MPI ranks via :func:`load_balancing`.
    An MPI reduction collects contributions from all ranks on rank 0 before
    writing.  If ``verbose`` is enabled, the total charge is printed.
    """
    arry, attr = data_controller.data_dicts()

    # Calculation of the electron density
    if rank == 0 and attr['verbose']:
        print('Writing density files')

    rhoaux = np.zeros((nr1, nr2, nr3, attr['nspin']), dtype=complex, order='C')

    ini_ik, end_ik = load_balancing(comm.Get_size(), rank, attr['nkpnts'])

    for ispin in range(attr['nspin']):
        for ik in range(ini_ik, end_ik):
            accumulate_density_k(
                rhoaux[:, :, :, ispin],
                data_controller,
                ik,
                arry['E_k'][ik - ini_ik, :, ispin],
                arry['v_k'][ik - ini_ik, :, :, ispin],
                nr1,
                nr2,
                nr3,
            )

    write_density(data_controller, rhoaux)


def accumulate_density_k(rho, data_controller, ik, E, v_k, nr1, nr2, nr3) -> None:
    """Add the occupied states of DFT k-point ``ik`` to the density ``rho``, in place.

    Parameters
    ----------
    rho : np.ndarray, shape ``(nr1, nr2, nr3)``, complex
        Accumulator (one spin channel).
    data_controller : DataController
        Supplies ``basis`` (internal projections), the plane-wave basis of the
        DFT k-points, ``nkpnts``, ``bnd`` and ``omega``.
    ik : int
        Index of the k-point in the DFT run: the PAO mesh must be the DFT
        grid, uninterpolated, as the dense pipeline has always assumed.
    E : np.ndarray, shape ``(nbands,)``
        Eigenvalues at this k-point; states with ``E <= 0`` are occupied.
    v_k : np.ndarray, shape ``(nawf, nbands)``
        Eigenvectors (columns), at least the lowest ``bnd``.
    """
    arry, attr = data_controller.data_dicts()
    basis = arry['basis']
    eps = 1.0e-5
    gkspace = calc_gkspace(data_controller, ik, gamma_only=False)
    atwfcgk = calc_atwfc_k(basis, gkspace)
    oatwfcgk = ortho_atwfc_k(atwfcgk)
    atwfcr = fft_allwfc_G2R(oatwfcgk, gkspace, nr1, nr2, nr3, attr['omega'])
    for nb in range(attr['bnd']):
        if E[nb] <= 0.0 + eps:
            tmp = np.tensordot(v_k[:, nb], atwfcr[:, :, :, :], axes=(0, 0))
            rho += 2 * np.conj(tmp) * tmp / attr['nkpnts'] * attr['omega'] / (nr1 * nr2 * nr3)


def write_density(data_controller, rhoaux) -> None:
    """Reduce the per-rank densities and write one XSF file per spin channel.

    Parameters
    ----------
    rhoaux : np.ndarray, shape ``(nr1, nr2, nr3, nspin)``, complex
        This rank's :func:`accumulate_density_k` sums.

    Notes
    -----
    All spin channels are reduced at once (the reduction used to free the
    accumulator after the first channel, so ``nspin = 2`` failed).
    """
    arry, attr = data_controller.data_dicts()
    rho = np.zeros(rhoaux.shape, dtype=complex, order='C') if rank == 0 else None
    comm.Reduce(rhoaux, rho, op=MPI.SUM)
    rhoaux = None

    if rank == 0:
        for ispin in range(attr['nspin']):
            fdensity = attr['outputdir'] + '/density_%s.xsf' % str(ispin)
            write2xsf(data_controller, filename=fdensity, data=np.real(rho[:, :, :, ispin]))
        if attr['verbose']:
            print('Total charge = ', np.real(np.sum(rho)).round(3))
