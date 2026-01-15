from typing import Tuple

import numpy as np
from PAOFLOW.transport.calculators.transmittance import interpolate_transmittance
from PAOFLOW.transport.utils.locate import locate


def read_transmittance(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the transmittance data file.

    Parameters
    ----------
    `file_path` : str
        Path to the file containing energy and transmittance values.

    Returns
    -------
    `egrid` : ndarray
        Array of energy values.
    `transm` : ndarray
        Corresponding transmittance values.
    """
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1]


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
    """
    Compute the Fermi-Dirac distribution.

    Parameters
    ----------
    `E` : ndarray
        Energy values.
    `mu` : float
        Chemical potential.
    `sigma` : float
        Broadening factor (eV).

    Returns
    -------
    `f` : ndarray
        Fermi-Dirac values.
    """
    return 1.0 / (np.exp(-(E - mu) / sigma) + 1.0)


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

    The integration mesh is refined to resolve the energy window around the chemical potentials.
    """
    ne = len(egrid)
    de_old = (egrid[-1] - egrid[0]) / (ne - 1)
    currents = np.zeros_like(vgrid)

    for iv, V in enumerate(vgrid):
        muL_v = mu_L * V
        muR_v = mu_R * V

        try:
            i_start = locate(egrid, min(muL_v, muR_v) - sigma - 3 * de_old)
            i_end = locate(egrid, max(muL_v, muR_v) + sigma + 3 * de_old)
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

        egrid_new, transm_new = interpolate_transmittance(
            egrid, transm, i_start, i_end, ndiv
        )
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
