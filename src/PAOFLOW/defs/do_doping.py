import numpy as np
import scipy.integrate
import scipy.optimize
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def _fd_criterion_gen(threshold):
    """Return a root-finding criterion function for the Fermi-Dirac tail.

    The returned callable evaluates ``f(x) - threshold``, where
    ``f(x) = 1 / (exp(x) + 1)`` is the Fermi-Dirac function.  Finding
    its root gives the reduced energy ``x = (\\varepsilon - \\mu) / k_BT``
    beyond which the occupation drops below ``threshold``.  This is used
    to determine the energy window over which the Fermi-Dirac function is
    non-negligible.

    Parameters
    ----------
    threshold : float
        Occupation level at the desired tail cut-off (e.g. ``1e-8``).

    Returns
    -------
    callable
        A scalar function ``_fd_criterion(x)`` whose root is the reduced
        energy corresponding to ``threshold`` occupation.
    """

    def _fd_criterion(x):
        """Fermi-Dirac value minus threshold; root gives the tail cut-off."""
        return 1.0 / (np.exp(x) + 1.0) - threshold

    return _fd_criterion


def FD(ene, mu, temp):
    """Evaluate the Fermi-Dirac occupation function.

    Parameters
    ----------
    ene : np.ndarray
        Energy grid (eV).
    mu : float
        Chemical potential (eV).
    temp : float
        Temperature in Kelvin.  When ``0.0``, the step function is used.

    Returns
    -------
    np.ndarray
        Fermi-Dirac occupation :math:`f(\\varepsilon) = [e^{(\\varepsilon-\\mu)/k_BT}+1]^{-1}`
        clamped to ``[0, 1]``.
    """
    _FD_THRESHOLD = 1e-8
    _FD_XMAX = scipy.optimize.newton(_fd_criterion_gen(_FD_THRESHOLD), 0.0)

    temp_ev = temp * 8.617332478e-5
    if temp == 0.0:
        dela = ene - mu
        nruter = np.where(dela < 0.0, 1.0, 0.0)
        nruter[np.isclose(dela, 0.0)] = 0.5
    else:
        x = (ene - mu) / temp_ev
        nruter = np.where(x < 0.0, 1.0, 0.0)
        indices = np.logical_and(x > -_FD_XMAX, x < _FD_XMAX)
        nruter[indices] = 1.0 / (np.exp(x[indices]) + 1.0)
    return nruter


def calc_N(data_controller, ene, dos, mu, temp, dosweight=2.0):
    """Integrate the DOS times the Fermi-Dirac function to get the electron count.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_attributes``.
        Required attribute: ``core_electrons`` (int).
    ene : np.ndarray
        Energy grid (eV).
    dos : np.ndarray, shape ``(ne,)``
        Density of states on ``ene``.
    mu : float
        Trial chemical potential (eV).
    temp : float
        Temperature (K).
    dosweight : float, optional
        Spin degeneracy weight applied to the integral (default ``2.0``).

    Returns
    -------
    float
        Electron count minus ``core_electrons`` minus the integrated DOS
        (negative of the total electron number minus ``core_electrons``).
    """
    arry, attr = data_controller.data_dicts()
    core_electrons = attr['core_electrons']

    if temp == 0.0:
        occ = np.where(ene < mu, 1.0, 0.0)
        occ[ene == mu] = 0.5
    else:
        occ = FD(ene, mu, temp)
    dos_occ = dos * occ
    return -dosweight * scipy.integrate.simpson(dos_occ, ene) - core_electrons


def solve_for_mu(
    data_controller, ene, dos, N0, temp, refine=False, try_center=False, dosweight=2.0
):
    """Find the chemical potential that yields a target electron count.

    Parameters
    ----------
    data_controller : DataController
        Passed directly to :func:`calc_N`.
    ene : np.ndarray
        Energy grid (eV).
    dos : np.ndarray, shape ``(ne,)``
        Density of states on ``ene``.
    N0 : float
        Target electron count (including doping offset).
    temp : float
        Temperature (K).
    refine : bool, optional
        When ``True``, perform a bounded scalar minimisation of
        :math:`|N(\\mu) - N_0|` around the best grid point (default ``False``).
    try_center : bool, optional
        When ``True`` and ``mu`` falls inside a band gap, try to place
        :math:`\\mu` at the gap centre (default ``False``).
    dosweight : float, optional
        Spin degeneracy weight forwarded to :func:`calc_N` (default ``2.0``).

    Returns
    -------
    float
        Chemical potential :math:`\\mu` (eV) that satisfies :math:`N(\\mu) = N_0`
        (computed on rank 0 only; all other ranks return ``None``).

    Notes
    -----
    The function first scans the energy grid for the point that minimises
    :math:`|N(\\mu) - N_0|`, then optionally refines using
    ``scipy.optimize.minimize_scalar`` with ``method='bounded'``.
    """
    _FD_THRESHOLD_GAP = 1e-3
    _FD_XMAX_GAP = scipy.optimize.newton(_fd_criterion_gen(_FD_THRESHOLD_GAP), 0.0)

    dela = np.empty_like(ene)

    if rank == 0:
        for i, e in enumerate(ene):
            dela[i] = (calc_N(data_controller, ene, dos, e, temp, dosweight)) + N0

        dela = np.abs(dela)
        pos = dela.argmin()
        mu = ene[pos]
        center = False
        ########################################
        # checking if dos is zero takes care not to include band gaps in the integral calculation
        #######################################

        if dos[pos] == 0.0:
            lpos = -1
            hpos = -1
            for i in range(pos, -1, -1):
                if dos[i] != 0.0:
                    lpos = i
                    break
            for i in range(pos, dos.size):
                if dos[i] != 0.0:
                    hpos = i
                    break
            if -1 in (lpos, hpos):
                raise ValueError('mu0 lies outside the range of band energies')
            hene = ene[hpos]
            lene = ene[lpos]
            if try_center and min(hene - mu, mu - lene) >= _FD_XMAX_GAP * temp / 2.0:
                pos = int(round(0.5 * (lpos + hpos)))
                mu = ene[pos]
                center = True
        if refine:
            if center:
                mu = 0.5 * (lene + hene)
            else:
                residual = calc_N(data_controller, ene, dos, mu, temp, dosweight) + N0
                if np.isclose(residual, 0):
                    lpos = pos
                    hpos = pos
                elif residual > 0:
                    lpos = pos
                    hpos = min(pos + 1, ene.size - 1)
                else:
                    lpos = max(0, pos - 1)
                    hpos = pos
                if hpos != lpos:
                    lmu = ene[lpos]
                    hmu = ene[hpos]

                    def calc_abs_residual(muarg):
                        return abs(calc_N(data_controller, ene, dos, muarg, temp, dosweight) + N0)

                    result = scipy.optimize.minimize_scalar(
                        calc_abs_residual, bounds=(lmu, hmu), method='bounded'
                    )
                    mu = result.x
        return mu


def do_doping(data_controller, temps, ene, fname):
    """Compute the temperature-dependent chemical potential for a doped system.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``dos`` (fixed-smearing DOS) or ``dosdk``
        (adaptive-smearing DOS), selected by ``attr['smearing']``.
        Required attributes: ``smearing``, ``doping_conc``, ``nelec``,
        ``omega``.
    temps : array_like of float
        Temperatures in Kelvin at which :math:`\\mu(T)` is evaluated.
    ene : np.ndarray
        Energy grid (eV) over which the DOS is defined.
    fname : str
        Base name for the output file.  The sign and magnitude of
        ``doping_conc`` are appended to produce e.g. ``fname_n1e18.dat``.

    Returns
    -------
    None
        Writes a two-column file ``{fname}{sign}{|doping|}.dat`` via
        :meth:`DataController.write_file_row_col` with columns
        temperature and :math:`\\mu(T)`.

    Notes
    -----
    The target electron number is
    :math:`N = N_{\\rm elec} - n_d \\Omega`
    where :math:`\\Omega` is the unit-cell volume and :math:`n_d` is the
    carrier concentration in :math:`\\text{cm}^{-3}`.  For each temperature
    :func:`solve_for_mu` is called on rank 0 and the result broadcast.
    """
    arry, attr = data_controller.data_dicts()
    omega_conv = 1.481847093e-25

    if attr['smearing'] is None:
        dos = arry['dos']
    else:
        dos = arry['dosdk']

    doping = attr['doping_conc']
    nelec, omega = attr['nelec'], attr['omega'] * omega_conv

    nT = len(temps)
    mu = np.empty(nT)
    N = nelec - doping * omega

    for iT, temp in enumerate(temps):
        if rank == 0:
            mu[iT] = solve_for_mu(data_controller, ene, dos, N, temp, refine=True, try_center=True)

        mu[iT] = comm.bcast(mu[iT], root=0)

    fdope = '%s%s%s.dat' % (fname, 'n' if doping < 0 else 'p', np.abs(doping))
    data_controller.write_file_row_col(fdope, temps, mu)
