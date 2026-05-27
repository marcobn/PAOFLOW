import numpy as np
from mpi4py import MPI

from .smearing import intmetpax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def E_Fermi(Hksp, data_controller, parallel=False):
    """Find the Fermi energy using a bisection bracketing algorithm.

    Parameters
    ----------
    Hksp : np.ndarray, shape ``(nawf, nawf, snktot, nspin)`` or ``(nawf, nawf, nk1, nk2, nk3, nspin)``
        k-space Hamiltonian distributed over MPI pools.  The array is
        reshaped internally to ``(nawf, nawf, snktot, nspin)`` before
        diagonalisation.
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required attributes: ``insulator``, ``nkpnts``, ``bnd``, ``nelec``,
        ``dftSO``.
    parallel : bool, optional
        If ``True``, an MPI reduction is performed across ranks to compute
        the global Fermi energy.  Default is ``False``.

    Returns
    -------
    float
        Fermi energy in eV.

    Notes
    -----
    For insulators the Fermi energy is set to the maximum eigenvalue of the
    highest occupied band, accounting for spin–orbit coupling via ``dftSO``.

    For metals, eigenvalues at each local k-point are first computed by
    diagonalising ``Hksp``.  The Fermi energy is then located by bisection:
    upper and lower bounds :math:`E_{\\text{up}}` and :math:`E_{\\text{lw}}` are
    established such that the integrated occupation at :math:`E_{\\text{up}}`
    exceeds ``nelec`` and at :math:`E_{\\text{lw}}` is below ``nelec``.  The
    midpoint is accepted when

    .. math::

        \\left| N(E_F) - N_{\\text{elec}} \\right| < \\epsilon

    where :math:`N(E)` is the smeared electron count computed by
    :func:`intmetpax` with a fixed Gaussian broadening of 0.01 eV, and
    :math:`\\epsilon = 10^{-10}`.  A maximum of 100 iterations is performed.
    """
    # Calculate the Fermi energy using a braketing algorithm

    arry, attr = data_controller.data_dicts()

    insulator = attr['insulator']
    nktot, nbnd = attr['nkpnts'], attr['bnd']
    nelec, dftSO = attr['nelec'], attr['dftSO']
    nawf, _, snktot, nspin = Hksp.shape
    eig = np.zeros((nawf, snktot, nspin))
    Hksp = Hksp.reshape((nawf, nawf, snktot, nspin), order='C')

    for ispin in range(nspin):
        for ik in range(snktot):
            eig[:, ik, ispin] = np.linalg.eigvalsh(Hksp[:, :, ik, ispin])

    if insulator:
        Efr = np.amax(eig[(nelec - 1 if dftSO else nelec // 2 - 1)])
        if parallel:
            Efm = np.zeros((1), dtype=float) if rank == 0 else None
            comm.Reduce(Efr, Efm, op=MPI.MAX)
            return comm.bcast(Efm[0] if rank == 0 else None)
        else:
            return Efr

    else:
        Elw = 1.0e8
        Eup = -1.0e8
        eps = 1.0e-10
        degauss = 0.01

        nmbnd = nbnd - 1 if nbnd == nawf else nbnd
        for ispin in range(nspin):
            for kp in range(snktot):
                Elw = min(Elw, eig[0, kp, ispin])
                Eup = max(Elw, eig[nmbnd, kp, ispin])

        Eup = Eup + 2 * degauss
        Elw = Elw - 2 * degauss

        # bisection method
        fac = 1 if dftSO else 2
        sumkup_aux = fac * np.sum(intmetpax(eig[:nbnd, :, :], Eup, degauss))
        sumklw_aux = fac * np.sum(intmetpax(eig[:nbnd, :, :], Elw, degauss))

        if parallel:
            sumkup = np.zeros((1), dtype=float) if rank == 0 else None
            sumklw = np.zeros((1), dtype=float) if rank == 0 else None
            comm.Reduce(sumkup_aux, sumkup, op=MPI.SUM)
            comm.Reduce(sumklw_aux, sumklw, op=MPI.SUM)
            sumkup = comm.bcast(sumkup[0] / nktot if rank == 0 else None)
            sumklw = comm.bcast(sumklw[0] / nktot if rank == 0 else None)
        else:
            sumkup = sumkup_aux / nktot
            sumklw = sumklw_aux / nktot

        if (sumkup - nelec) < -eps or (sumklw - nelec) > eps:
            if rank == 0:
                print('Error: cannot bracket Ef')

        maxiter = 100
        for i in range(maxiter):
            Ef = (Eup + Elw) / 2
            sumkmid_aux = fac * np.sum(intmetpax(eig[:, :, :], Ef, degauss))
            if parallel:
                sumkmid = np.zeros((1,), dtype=float) if rank == 0 else None
                comm.Reduce(sumkmid_aux, sumkmid, op=MPI.SUM)
                sumkmid = comm.bcast(sumkmid[0] / nktot if rank == 0 else None)
            else:
                sumkmid = sumkmid_aux / nktot

            if np.abs(sumkmid - nelec) < eps:
                break
            elif sumkmid - nelec < -eps:
                Elw = Ef
            else:
                Eup = Ef

        return Ef
