"""Reference formulas that serve as baselines for superconductivity comparisons.

The moments of an Eliashberg function, the McMillan and Allen-Dynes critical temperatures
(the formulas of :func:`PAOFLOW.elphon.eph_kq.mcmillan_allen_dynes_tc`), their inversion
(the coupling a measured Tc implies), the BCS gap and its universal temperature dependence,
and the strong-coupling correction to the gap ratio.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

KB_EV = 8.617333262e-5  # Boltzmann constant (eV/K)

#: The BCS gap ratio :math:`2\\Delta(0)/k_B T_c = 2\\pi e^{-\\gamma}`.
BCS_GAP_RATIO = 3.5277


def a2f_moments(omega_ev: ArrayLike, a2f: ArrayLike) -> dict[str, float]:
    """:math:`\\lambda`, :math:`\\omega_{\\log}` and :math:`\\omega_2` of an Eliashberg function.

    .. math::

        \\lambda = 2\\int \\frac{\\alpha^2F(\\omega)}{\\omega}d\\omega,\\quad
        \\omega_{\\log} = \\exp\\Big[\\frac{2}{\\lambda}\\int \\frac{\\alpha^2F}{\\omega}\\ln\\omega\\,d\\omega\\Big],\\quad
        \\omega_2 = \\Big[\\frac{2}{\\lambda}\\int \\alpha^2F\\,\\omega\\,d\\omega\\Big]^{1/2}

    Parameters
    ----------
    omega_ev, a2f : array_like
        Frequencies (eV, positive entries used) and the function.

    Returns
    -------
    dict
        ``lambda``, ``omega_log_ev``, ``omega_2_ev``.
    """
    w = np.asarray(omega_ev, dtype=float)
    f = np.asarray(a2f, dtype=float)
    keep = w > 0.0
    w, f = w[keep], f[keep]
    density = 2.0 * f / w
    lam = float(np.trapezoid(density, w))
    if lam <= 0.0:
        return {'lambda': lam, 'omega_log_ev': np.nan, 'omega_2_ev': np.nan}
    omega_log = float(np.exp(np.trapezoid(density * np.log(w), w) / lam))
    omega_2 = float(np.sqrt(np.trapezoid(density * w**2, w) / lam))
    return {'lambda': lam, 'omega_log_ev': omega_log, 'omega_2_ev': omega_2}


def cumulative_lambda(omega_ev: ArrayLike, a2f: ArrayLike) -> NDArray[np.float64]:
    """:math:`\\lambda(\\omega) = 2\\int_0^\\omega \\alpha^2F(\\omega')/\\omega'\\,d\\omega'` on the given grid."""
    w = np.asarray(omega_ev, dtype=float)
    f = np.asarray(a2f, dtype=float)
    density = np.where(w > 0.0, 2.0 * f / np.where(w > 0.0, w, 1.0), 0.0)
    out = np.zeros_like(w)
    out[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(w))
    return out


def mcmillan_tc(lam: float, omega_log_ev: float, mu_star: float = 0.1) -> float:
    """McMillan's Tc (K) with the Allen-Dynes prefactor :math:`\\omega_{\\log}/1.2`."""
    from ...elphon.eph_kq import mcmillan_allen_dynes_tc

    return mcmillan_allen_dynes_tc(lam, omega_log_ev, omega_log_ev, mu_star)['Tc_mcmillan_K']


def allen_dynes_tc(
    lam: float, omega_log_ev: float, omega_2_ev: float, mu_star: float = 0.1
) -> float:
    """Allen-Dynes Tc (K) with the strong-coupling and shape corrections :math:`f_1 f_2`."""
    from ...elphon.eph_kq import mcmillan_allen_dynes_tc

    return mcmillan_allen_dynes_tc(lam, omega_log_ev, omega_2_ev, mu_star)['Tc_allen_dynes_K']


def lambda_from_tc(
    tc_k: float,
    omega_log_ev: float,
    mu_star: float = 0.1,
    omega_2_ev: float | None = None,
    lambda_max: float = 10.0,
) -> float:
    """The coupling :math:`\\lambda` that gives ``tc_k`` with McMillan (or Allen-Dynes when ``omega_2_ev`` is given).

    Raises
    ------
    ValueError
        If no coupling below ``lambda_max`` reaches ``tc_k``.
    """
    from scipy.optimize import brentq

    def residual(lam: float) -> float:
        tc = (
            mcmillan_tc(lam, omega_log_ev, mu_star)
            if omega_2_ev is None
            else allen_dynes_tc(lam, omega_log_ev, omega_2_ev, mu_star)
        )
        return tc - tc_k

    low = mu_star / (1.0 - 0.62 * mu_star) + 1e-6  # where the exponent's denominator turns positive
    if residual(lambda_max) < 0.0:
        raise ValueError('Tc = %g K needs lambda > %g' % (tc_k, lambda_max))
    return float(brentq(residual, low, lambda_max, xtol=1e-8))


def bcs_gap_zero_mev(tc_k: float) -> float:
    """Weak-coupling BCS gap :math:`\\Delta(0) = 1.764\\,k_B T_c` (meV)."""
    return 0.5 * BCS_GAP_RATIO * KB_EV * tc_k * 1.0e3


def bcs_gap_vs_t(
    temps_k: ArrayLike, tc_k: float, gap0_mev: float | None = None
) -> NDArray[np.float64]:
    """The universal BCS gap :math:`\\Delta(T)` (meV) scaled to ``gap0_mev`` and ``tc_k``.

    Solves the BCS gap equation in reduced units for every temperature below Tc:

    .. math::

        \\int_0^\\infty d\\varepsilon\\Big[\\frac{\\tanh(E/2k_BT)}{E} -
        \\frac{\\tanh(\\varepsilon/2k_BT_c)}{\\varepsilon}\\Big] = 0,\\qquad
        E = \\sqrt{\\varepsilon^2 + \\Delta^2}.

    Parameters
    ----------
    temps_k : array_like
    tc_k : float
    gap0_mev : float, optional
        Zero-temperature gap the curve is scaled to (default: the BCS value
        :math:`1.764\\,k_B T_c`), e.g. the model's or the measured one.

    Returns
    -------
    ndarray
        :math:`\\Delta(T)` in meV, zero above Tc.
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq

    temps = np.atleast_1d(np.asarray(temps_k, dtype=float))
    gap0 = bcs_gap_zero_mev(tc_k) if gap0_mev is None else float(gap0_mev)
    bcs0 = 0.5 * BCS_GAP_RATIO  # Delta(0) / k_B Tc
    cutoff = 400.0  # in units of k_B Tc

    def equation(delta: float, t: float) -> float:
        def integrand(x: float) -> float:
            e = np.sqrt(x * x + delta * delta)
            return np.tanh(e / (2.0 * t)) / e - np.tanh(x / 2.0) / x if x > 0.0 else 0.0

        return quad(integrand, 1e-9, cutoff, limit=400)[0]

    out = np.zeros_like(temps)
    for i, temp in enumerate(temps):
        t = temp / tc_k
        if t <= 0.0:
            out[i] = 1.0
        elif t < 1.0:
            out[i] = brentq(equation, 1e-9, 1.2 * bcs0, args=(t,), xtol=1e-9) / bcs0
    return out * gap0


def strong_coupling_gap_ratio(tc_k: float, omega_log_ev: float) -> float:
    """:math:`2\\Delta(0)/k_BT_c = 3.53[1 + 12.5(T_c/\\omega_{\\log})^2\\ln(\\omega_{\\log}/2T_c)]` (Carbotte 1990)."""
    x = KB_EV * tc_k / omega_log_ev
    return float(3.53 * (1.0 + 12.5 * x * x * np.log(1.0 / (2.0 * x))))
