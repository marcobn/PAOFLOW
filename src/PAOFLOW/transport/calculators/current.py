from __future__ import annotations

import numpy as np
from scipy.special import expit

from PAOFLOW.transport.calculators.transmittance import interpolate_transmittance


def build_bias_grid(vmin: float, vmax: float, nv: int) -> np.ndarray:
    """
    Construct a linear bias voltage grid.

    Parameters
    ----------
    `vmin` : float
        Minimum bias value.
    `vmax` : float
        Maximum bias value.
    `nv` : int
        Number of bias points.

    Returns
    -------
    `vgrid` : ndarray
        Bias voltage values.
    """
    return np.linspace(vmin, vmax, nv)


def fermi_dirac(E: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    Compute the Fermi-Dirac occupation.

    Parameters
    ----------
    `E` : ndarray
        Energy values.
    `mu` : float
        Chemical potential.
    `sigma` : float
        Broadening factor (eV), :math:`k_B T`.

    Returns
    -------
    `f` : ndarray
        Fermi-Dirac occupations, 1 below `mu` and 0 above it.

    Notes
    -----
    .. math::
        f(E) = \frac{1}{e^{(E-\mu)/\sigma} + 1},

    evaluated as the logistic function of :math:`-(E-\mu)/\sigma`, which does not overflow.
    Before 15 November 2026 this returned :math:`1 - f`, and ``compute_current_vs_bias``
    returned the current with the opposite sign.
    """
    return expit(-(np.asarray(E) - mu) / sigma)


def locate(xx: np.ndarray, x: float) -> int:
    """
    Locate the index `j` in `xx` such that `xx[j] <= x < xx[j+1]`.

    Parameters
    ----------
    `xx` : ndarray
        Monotonic array (ascending or descending) of values.
    `x` : float
        Value to locate within the array.

    Returns
    -------
    `j` : int
        Index satisfying `xx[j] <= x < xx[j+1]`.

    Raises
    ------
    ValueError
        If the value is outside the bounds of the array.
    """
    n = len(xx)
    jl = 0
    ju = n
    is_ascending = xx[-1] > xx[0]

    if (x < xx[0] and is_ascending) or (x > xx[0] and not is_ascending):
        raise ValueError('Value is outside the bounds of the array.')
    if (x >= xx[-1] and is_ascending) or (x <= xx[-1] and not is_ascending):
        raise ValueError('Value is outside the bounds of the array.')

    while ju - jl > 1:
        jm = (ju + jl) // 2
        if (x > xx[jm]) == is_ascending:
            jl = jm
        else:
            ju = jm

    if jl == n - 1:
        raise ValueError('Located index out of valid bounds.')

    return jl


def compute_current_vs_bias(
    egrid: np.ndarray,
    transm: np.ndarray,
    vgrid: np.ndarray,
    mu_L: float,
    mu_R: float,
    sigma: float,
) -> np.ndarray:
    r"""
    Compute current I(V) as a function of bias using Landauer formula.

    Parameters
    ----------
    `egrid` : ndarray
        Energy grid.
    `transm` : ndarray
        Transmittance on the energy grid.
    `vgrid` : ndarray
        Bias voltages.
    `mu_L` : float
        Left chemical potential coefficient.
    `mu_R` : float
        Right chemical potential coefficient.
    `sigma` : float
        Broadening parameter (smearing width, in eV).

    Returns
    -------
    `currents` : ndarray
        Current at each bias voltage.

    Notes
    -----
    Implements:

    .. math::
        I(V) = \int dE \; T(E) [f(E - \mu_L) - f(E - \mu_R)]

    The integration mesh is refined to resolve the energy window around the chemical potentials, which
    extends ``10 * sigma + 3 dE`` beyond them (clipped to the energy grid).

    The current is positive for :math:`\mu_L > \mu_R` (electrons flow from left to right), in units
    of :math:`2e/h` times eV, the same convention as ``Transport.compute_current_self_consistent``.
    Before 15 November 2026 ``fermi_dirac`` returned :math:`1 - f` and the sign was the opposite.
    """
    ne = len(egrid)
    de_old = (egrid[-1] - egrid[0]) / (ne - 1)
    currents = np.zeros_like(vgrid)

    for iv, V in enumerate(vgrid):
        muL_v = mu_L * V
        muR_v = mu_R * V

        # The Fermi window extends several sigma beyond both chemical potentials: the former margin of
        # sigma + 3 dE kept only tanh(sigma / (2 sigma)) of the current on fine energy grids (dE << sigma).
        # The window is clipped to the energy grid instead of skipping the bias point.
        margin = 10.0 * sigma + 3 * de_old
        low = max(min(muL_v, muR_v) - margin, egrid[0])
        high = min(max(muL_v, muR_v) + margin, egrid[-1] - 0.5 * de_old)
        try:
            i_start = locate(egrid, low)
            i_end = locate(egrid, high)
        except ValueError:
            continue

        ndim = i_end - i_start + 1
        if ndim < 2:
            continue

        if ndim % 2 == 0:
            i_end -= 1
            ndim -= 1

        de = (egrid[i_end] - egrid[i_start]) / (ndim - 1)
        ndiv = max(1, int(round(de / (2 * sigma))))

        egrid_new, transm_new = interpolate_transmittance(egrid, transm, i_start, i_end, ndiv)
        fL = fermi_dirac(egrid_new, muL_v, sigma)
        fR = fermi_dirac(egrid_new, muR_v, sigma)

        de_new = egrid_new[1] - egrid_new[0]

        integral = 0.0
        for i in range(len(egrid_new) - 1):
            fL_i = fL[i]
            fL_ip1 = fL[i + 1]
            fR_i = fR[i]
            fR_ip1 = fR[i + 1]

            fdiff_i = fL_i - fR_i
            fdiff_ip1 = fL_ip1 - fR_ip1

            t_i = transm_new[i]
            t_ip1 = transm_new[i + 1]

            integral += (
                fdiff_i * t_i * de_new / 3.0
                + fdiff_ip1 * t_ip1 * de_new / 3.0
                + fdiff_i * t_ip1 * de_new / 6.0
                + fdiff_ip1 * t_i * de_new / 6.0
            )

        currents[iv] = integral

    return currents
