"""Trivial models and fits that serve as baselines for transport comparisons.

A deviation between a calculation and a measurement means nothing alone (playbook rule 6);
these are the one- or two-parameter models it is quoted against: the Bloch-Gruneisen
resistivity of a metal, a power law :math:`y \\propto x^{p}`, a linear fit, the residual
resistivity that reconciles a phonon-limited model with a sample by Matthiessen's rule, and
Matthiessen's combination of mobilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _window(x: NDArray, window: tuple[float, float] | None) -> NDArray[np.bool_]:
    if window is None:
        return np.ones_like(x, dtype=bool)
    return (x >= window[0]) & (x <= window[1])


def bloch_gruneisen_integral(y: float, n: int = 5) -> float:
    """:math:`J_n(y) = \\int_0^y x^n e^x/(e^x-1)^2\\,dx` (``n = 5``: phonon scattering in a metal)."""
    from scipy.integrate import quad

    if y <= 0.0:
        return 0.0

    def integrand(x: float) -> float:
        if x < 1e-8:
            return x ** (n - 2)
        ex = np.exp(-x)
        return x**n * ex / (1.0 - ex) ** 2

    value, _ = quad(integrand, 0.0, float(y), limit=200)
    return float(value)


def bloch_gruneisen(temps_k: ArrayLike, theta_k: float, n: int = 5) -> NDArray[np.float64]:
    """Shape :math:`(T/\\Theta)^n J_n(\\Theta/T)` of the Bloch-Gruneisen resistivity.

    Parameters
    ----------
    temps_k : array_like
    theta_k : float
        The Debye (Bloch-Gruneisen) temperature.
    n : int, optional
        Exponent (5: electron-phonon in simple metals; 3: s-d scattering; 2: e-e).

    Returns
    -------
    ndarray
        Dimensionless; linear in ``T`` above ``theta_k`` (:math:`\\to T/((n-1)\\Theta)`),
        :math:`\\propto T^n` far below it. :math:`\\rho(T) = \\rho_0 + A\\,\\mathrm{shape}(T)`.
    """
    temps = np.atleast_1d(np.asarray(temps_k, dtype=float))
    out = np.zeros_like(temps)
    for i, t in enumerate(temps):
        if t > 0.0:
            out[i] = (t / theta_k) ** n * bloch_gruneisen_integral(theta_k / t, n)
    return out


def _linear_bg(temps: NDArray, rho: NDArray, theta: float, n: int, rho0: float | None) -> tuple:
    shape = bloch_gruneisen(temps, theta, n)
    if rho0 is None:
        design = np.column_stack([np.ones_like(temps), shape])
        (r0, a), *_ = np.linalg.lstsq(design, rho, rcond=None)
    else:
        r0 = float(rho0)
        a = float(np.dot(shape, rho - r0) / np.dot(shape, shape)) if np.any(shape) else 0.0
    model = r0 + a * shape
    return float(r0), float(a), float(np.sqrt(np.mean((rho - model) ** 2))), model


def fit_bloch_gruneisen(
    temps_k: ArrayLike,
    rho: ArrayLike,
    theta_k: float | None = None,
    n: int = 5,
    rho0: float | None = None,
    theta_range: tuple[float, float] = (30.0, 1500.0),
) -> dict:
    """Fit :math:`\\rho(T) = \\rho_0 + A\\,(T/\\Theta)^n J_n(\\Theta/T)` to a resistivity curve.

    Parameters
    ----------
    temps_k, rho : array_like
        The curve (any resistivity unit; the fit is linear in ``rho``).
    theta_k : float, optional
        Fix the Debye temperature (otherwise it is fitted: a coarse scan over ``theta_range``
        followed by a local refinement).
    n : int, optional
        Exponent.
    rho0 : float, optional
        Fix the residual resistivity (``0.0`` for a phonon-limited model).
    theta_range : (min, max), optional
        Scan range of the Debye temperature.

    Returns
    -------
    dict
        ``rho0``, ``amplitude``, ``theta_k``, ``n``, ``rms`` and ``curve`` (the fit at ``temps_k``).
    """
    temps = np.asarray(temps_k, dtype=float)
    values = np.asarray(rho, dtype=float)
    if temps.size < 3:
        raise ValueError('at least three points are needed')
    if theta_k is not None:
        r0, a, rms, model = _linear_bg(temps, values, float(theta_k), n, rho0)
        return {
            'rho0': r0,
            'amplitude': a,
            'theta_k': float(theta_k),
            'n': n,
            'rms': rms,
            'curve': model,
        }
    from scipy.optimize import minimize_scalar

    grid = np.geomspace(theta_range[0], theta_range[1], 60)
    scan = [_linear_bg(temps, values, th, n, rho0)[2] for th in grid]
    i = int(np.argmin(scan))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    result = minimize_scalar(
        lambda th: _linear_bg(temps, values, th, n, rho0)[2], bounds=(lo, hi), method='bounded'
    )
    theta = float(result.x)
    r0, a, rms, model = _linear_bg(temps, values, theta, n, rho0)
    return {'rho0': r0, 'amplitude': a, 'theta_k': theta, 'n': n, 'rms': rms, 'curve': model}


def fit_power_law(x: ArrayLike, y: ArrayLike, window: tuple[float, float] | None = None) -> dict:
    """Least-squares :math:`\\ln y = \\ln c + p\\ln x` over the points with ``x`` in ``window``.

    Returns
    -------
    dict
        ``exponent`` (:math:`p`), ``prefactor`` (:math:`c`), ``rms_log10`` and ``n``.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = _window(xs, window) & (xs > 0) & (ys > 0)
    if np.count_nonzero(keep) < 2:
        raise ValueError('at least two positive points are needed for a power law')
    p, lnc = np.polyfit(np.log(xs[keep]), np.log(ys[keep]), 1)
    residual = np.log10(ys[keep]) - np.log10(np.exp(lnc) * xs[keep] ** p)
    return {
        'exponent': float(p),
        'prefactor': float(np.exp(lnc)),
        'rms_log10': float(np.sqrt(np.mean(residual**2))),
        'n': int(keep.sum()),
    }


def fit_linear(x: ArrayLike, y: ArrayLike, window: tuple[float, float] | None = None) -> dict:
    """Least-squares line :math:`y = a + b\\,x` over the points with ``x`` in ``window``."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = _window(xs, window)
    if np.count_nonzero(keep) < 2:
        raise ValueError('at least two points are needed for a line')
    b, a = np.polyfit(xs[keep], ys[keep], 1)
    rms = float(np.sqrt(np.mean((ys[keep] - a - b * xs[keep]) ** 2)))
    return {'slope': float(b), 'intercept': float(a), 'rms': rms, 'n': int(keep.sum())}


def matthiessen_offset(
    temps_exp: ArrayLike,
    rho_exp: ArrayLike,
    temps_model: ArrayLike,
    rho_model: ArrayLike,
    window: tuple[float, float] | None = None,
) -> dict:
    """The residual resistivity that best reconciles a phonon-limited model with a sample.

    The model is interpolated at the experimental temperatures inside its range; the offset
    is the mean of ``rho_exp - rho_model`` over ``window`` (Matthiessen's rule). Quote it as
    a *diagnostic* next to the unshifted deviation, never as a fitted model.

    Returns
    -------
    dict
        ``rho0`` (the offset), ``rms_before``, ``rms_after`` (same unit as the input),
        ``slope_ratio`` (experimental over model dρ/dT in the window) and ``n``.
    """
    te = np.asarray(temps_exp, dtype=float)
    re = np.asarray(rho_exp, dtype=float)
    tm = np.asarray(temps_model, dtype=float)
    rm = np.asarray(rho_model, dtype=float)
    order = np.argsort(tm)
    tm, rm = tm[order], rm[order]
    keep = _window(te, window) & (te >= tm[0]) & (te <= tm[-1])
    if np.count_nonzero(keep) < 1:
        raise ValueError('no experimental temperature inside the model range and the window')
    model_at = np.interp(te[keep], tm, rm)
    diff = re[keep] - model_at
    rho0 = float(np.mean(diff))
    out = {
        'rho0': rho0,
        'rms_before': float(np.sqrt(np.mean(diff**2))),
        'rms_after': float(np.sqrt(np.mean((diff - rho0) ** 2))),
        'n': int(keep.sum()),
        'slope_ratio': np.nan,
    }
    if keep.sum() >= 2:
        be = np.polyfit(te[keep], re[keep], 1)[0]
        bm = np.polyfit(te[keep], model_at, 1)[0]
        out['slope_ratio'] = float(be / bm) if bm != 0.0 else np.nan
    return out


def matthiessen_mobility(*mobilities: ArrayLike) -> NDArray[np.float64]:
    """:math:`1/\\mu = \\sum_i 1/\\mu_i` (e.g. phonon-limited + ionised-impurity mobilities)."""
    inverse = sum(1.0 / np.asarray(m, dtype=float) for m in mobilities)
    return 1.0 / inverse
