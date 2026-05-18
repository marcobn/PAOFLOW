import numpy as np
from mpi4py import MPI

from .communication import load_balancing
from .do_atwfc_proj import calc_atwfc_k, calc_gkspace, fft_allwfc_G2R, ortho_atwfc_k
from .write2xsf import write2xsf

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

    basis = arry['basis']
    eps = 1.0e-5
    for ispin in range(attr['nspin']):
        for ik in range(ini_ik, end_ik):
            gkspace = calc_gkspace(data_controller, ik, gamma_only=False)
            atwfcgk = calc_atwfc_k(basis, gkspace)
            oatwfcgk = ortho_atwfc_k(atwfcgk)
            atwfcr = fft_allwfc_G2R(oatwfcgk, gkspace, nr1, nr2, nr3, attr['omega'])
            for nb in range(attr['bnd']):
                if arry['E_k'][ik - ini_ik, nb, ispin] <= 0.0 + eps:
                    tmp = np.tensordot(
                        arry['v_k'][ik - ini_ik, :, nb, ispin], atwfcr[:, :, :, :], axes=(0, 0)
                    )
                    rhoaux[:, :, :, ispin] += (
                        2 * np.conj(tmp) * tmp / attr['nkpnts'] * attr['omega'] / (nr1 * nr2 * nr3)
                    )

        rho = (
            np.zeros((nr1, nr2, nr3, attr['nspin']), dtype=complex, order='C')
            if rank == 0
            else None
        )

        comm.Reduce(rhoaux, rho, op=MPI.SUM)
        rhoaux = None

        if rank == 0:
            fdensity = attr['outputdir'] + '/density_%s.xsf' % str(ispin)
            write2xsf(data_controller, filename=fdensity, data=np.real(rho[:, :, :, ispin]))
    if rank == 0:
        if attr['verbose']:
            print('Total charge = ', np.real(np.sum(rho)).round(3))
