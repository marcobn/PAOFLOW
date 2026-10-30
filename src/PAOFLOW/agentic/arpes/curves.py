"""Dispersions, linewidths and self-energies from ARPES intensity maps (model or measured).

The functions take plain arrays, so they apply equally to the maps of
:mod:`PAOFLOW.arpes` (``intensity`` of shape ``(nk, nw)`` on momenta ``k`` and
energies ``w``) and to calibrated experimental images
(:mod:`PAOFLOW.agentic.arpes.images`).
The analysis follows the standard ARPES practice: momentum distribution curves (MDCs,
``I(k)`` at fixed energy) give the dressed dispersion :math:`k(\\omega)` and the momentum
width; with a bare band :math:`\\varepsilon_b(k)` these give the self-energy,

.. math::

    \\mathrm{Re}\\,\\Sigma(\\omega) = \\omega - \\varepsilon_b(k(\\omega)),\\qquad
    \\mathrm{Im}\\,\\Sigma(\\omega) =
    -\\tfrac{1}{2}\\,\\Delta k_{\\rm FWHM}(\\omega)\\,|v_b(k(\\omega))|,

and the slopes of the dressed and bare dispersions at the Fermi level give the
mass enhancement :math:`\\lambda = |v_b/v| - 1` (the kink).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _mask(x: NDArray, window: tuple[float, float] | None) -> NDArray[np.bool_]:
    if window is None:
        return np.ones_like(x, dtype=bool)
    return (x >= window[0]) & (x <= window[1])


def refine_peak(x: ArrayLike, y: ArrayLike, index: int) -> tuple[float, float]:
    """Parabolic refinement of a maximum at sample ``index``: ``(x_peak, y_peak)``.

    Falls back to the sample itself at the ends of the array or when the three points do
    not describe a maximum.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    i = int(index)
    if i <= 0 or i >= ys.size - 1:
        return float(xs[i]), float(ys[i])
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    denominator = y0 - 2.0 * y1 + y2
    if denominator >= 0.0:
        return float(xs[i]), float(ys[i])
    delta = float(np.clip(0.5 * (y0 - y2) / denominator, -1.0, 1.0))
    step = xs[i + 1] - xs[i] if delta > 0 else xs[i] - xs[i - 1]
    return float(xs[i] + delta * step), float(y1 - 0.25 * (y0 - y2) * delta)


def peak_position(x: ArrayLike, y: ArrayLike, window: tuple[float, float] | None = None) -> float:
    """Refined position of the maximum of ``y(x)`` inside ``window``."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    idx = np.flatnonzero(_mask(xs, window))
    if idx.size < 1:
        raise ValueError('empty window')
    j = int(idx[np.argmax(ys[idx])])
    return refine_peak(xs, ys, j)[0]


def _crossing(xs: NDArray, ys: NDArray, above: int, below: int, level: float) -> float:
    """``x`` where ``y`` crosses ``level`` between samples ``above`` and ``below``.

    Inverse quadratic interpolation through the three samples around the crossing, else linear.
    """
    linear = xs[below] + (level - ys[below]) * (xs[above] - xs[below]) / (ys[above] - ys[below])
    third = above - (below - above)  # the sample beyond ``above``, away from the crossing
    if 0 <= third < ys.size:
        idx = sorted((third, above, below))
        coefficients = np.polyfit(xs[idx], ys[idx] - level, 2)
        lo, hi = sorted((xs[above], xs[below]))
        roots = [
            r.real for r in np.roots(coefficients) if abs(r.imag) < 1e-12 and lo <= r.real <= hi
        ]
        if roots:
            return float(roots[0])
    return float(linear)


def half_maximum_width(
    x: ArrayLike, y: ArrayLike, window: tuple[float, float] | None = None, baseline: float = 0.0
) -> tuple[float, float, float]:
    """Full width at half maximum of the peak of ``y(x)`` in ``window``: ``(fwhm, x_low, x_high)``.

    The half level uses the parabolically refined peak height; each crossing is found by
    inverse quadratic interpolation through the three samples around it (linear if the
    parabola has no root there). The result is grid-limited: below about five samples per
    width use :func:`fit_lorentzian`. NaN when a side does not fall below the half maximum
    inside the window.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    idx = np.flatnonzero(_mask(xs, window))
    if idx.size < 3:
        return np.nan, np.nan, np.nan
    xs, ys = xs[idx], ys[idx]
    j = int(np.argmax(ys))
    half = baseline + 0.5 * (refine_peak(xs, ys, j)[1] - baseline)
    low = high = np.nan
    for i in range(j, 0, -1):
        if ys[i - 1] <= half:
            low = _crossing(xs, ys, i, i - 1, half)
            break
    for i in range(j, ys.size - 1):
        if ys[i + 1] <= half:
            high = _crossing(xs, ys, i, i + 1, half)
            break
    return float(high - low), float(low), float(high)


def lorentzian(
    x: ArrayLike, center: float, fwhm: float, amplitude: float, offset: float = 0.0
) -> NDArray[np.float64]:
    """``offset + amplitude * (fwhm/2)^2 / ((x - center)^2 + (fwhm/2)^2)``."""
    xs = np.asarray(x, dtype=float)
    g = 0.5 * fwhm
    return offset + amplitude * g * g / ((xs - center) ** 2 + g * g)


def fit_lorentzian(
    x: ArrayLike, y: ArrayLike, window: tuple[float, float] | None = None
) -> dict[str, float]:
    """Least-squares Lorentzian (plus a constant) through ``y(x)`` in ``window``.

    Returns
    -------
    dict
        ``center``, ``fwhm``, ``amplitude``, ``offset`` and ``rms``.
    """
    from scipy.optimize import curve_fit

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = _mask(xs, window)
    xs, ys = xs[keep], ys[keep]
    if xs.size < 5:
        raise ValueError('at least five points are needed for a Lorentzian fit')
    center = peak_position(xs, ys)
    fwhm = half_maximum_width(xs, ys, baseline=float(ys.min()))[0]
    span = float(xs.max() - xs.min())
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        fwhm = 0.1 * span
    p0 = [center, fwhm, float(ys.max() - ys.min()), float(ys.min())]
    bounds = ([xs.min(), 1e-12, 0.0, -np.inf], [xs.max(), 10.0 * span, np.inf, np.inf])
    params, _ = curve_fit(lorentzian, xs, ys, p0=p0, bounds=bounds, maxfev=20000)
    rms = float(np.sqrt(np.mean((ys - lorentzian(xs, *params)) ** 2)))
    return {
        'center': float(params[0]),
        'fwhm': float(params[1]),
        'amplitude': float(params[2]),
        'offset': float(params[3]),
        'rms': rms,
    }


def _curve_peaks(
    axis: NDArray,
    curves: NDArray,
    positions: NDArray,
    method: str,
    min_contrast: float,
    edge_margin: int,
) -> dict[str, NDArray[np.float64]]:
    """Peak position and width of every row of ``curves`` (``(n, len(axis))``)."""
    if method not in ('peak', 'lorentzian'):
        raise ValueError("method must be 'peak' or 'lorentzian'")
    reference = float(curves.max()) if curves.size else 0.0
    out_pos, out_center, out_width, out_amp = [], [], [], []
    for position, curve in zip(positions, curves):
        if curve.max() - curve.min() < min_contrast * reference:
            continue
        j = int(np.argmax(curve))
        if j < edge_margin or j >= curve.size - edge_margin:
            continue  # the maximum sits at the edge of the window: not a peak
        if method == 'lorentzian':
            fit = fit_lorentzian(axis, curve)
            center, width, amplitude = fit['center'], fit['fwhm'], fit['amplitude']
        else:
            center, amplitude = refine_peak(axis, curve, j)
            width = half_maximum_width(axis, curve, baseline=float(curve.min()))[0]
        out_pos.append(position)
        out_center.append(center)
        out_width.append(width)
        out_amp.append(amplitude)
    return {
        'position': np.asarray(out_pos, dtype=float),
        'center': np.asarray(out_center, dtype=float),
        'fwhm': np.asarray(out_width, dtype=float),
        'amplitude': np.asarray(out_amp, dtype=float),
    }


def mdc_dispersion(
    k: ArrayLike,
    w: ArrayLike,
    intensity: ArrayLike,
    energy_window: tuple[float, float] | None = None,
    k_window: tuple[float, float] | None = None,
    method: str = 'peak',
    min_contrast: float = 0.05,
    edge_margin: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Dressed dispersion from momentum distribution curves: the MDC peak at every energy.

    Parameters
    ----------
    k : array_like ``(nk,)``
        Momentum along the cut (1/Å, or a distance along the path).
    w : array_like ``(nw,)``
        Energies (eV, relative to E_F).
    intensity : array_like ``(nk, nw)``
    energy_window, k_window : (min, max), optional
        Energies analysed, and the momentum range searched.
    method : {'peak', 'lorentzian'}
        Parabolic refinement of the maximum with the half-maximum width, or a Lorentzian fit.
    min_contrast : float, optional
        MDCs with a peak-to-base contrast below this fraction of the map maximum are skipped.
    edge_margin : int, optional
        MDCs whose maximum sits within this many samples of the ``k_window`` edge are skipped.

    Returns
    -------
    dict
        ``energy_ev``, ``k`` (peak momentum), ``fwhm_k`` and ``amplitude``, one entry per
        energy with a resolved peak.
    """
    ks = np.asarray(k, dtype=float)
    ws = np.asarray(w, dtype=float)
    data = np.asarray(intensity, dtype=float)
    if data.shape != (ks.size, ws.size):
        raise ValueError('intensity must have shape (nk, nw) = %s' % ((ks.size, ws.size),))
    kmask, wmask = _mask(ks, k_window), _mask(ws, energy_window)
    peaks = _curve_peaks(
        ks[kmask], data[kmask][:, wmask].T, ws[wmask], method, min_contrast, edge_margin
    )
    return {
        'energy_ev': peaks['position'],
        'k': peaks['center'],
        'fwhm_k': peaks['fwhm'],
        'amplitude': peaks['amplitude'],
    }


def edc_dispersion(
    k: ArrayLike,
    w: ArrayLike,
    intensity: ArrayLike,
    k_window: tuple[float, float] | None = None,
    energy_window: tuple[float, float] | None = None,
    method: str = 'peak',
    min_contrast: float = 0.05,
    edge_margin: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Dispersion from energy distribution curves: the EDC peak at every momentum.

    Same arguments as :func:`mdc_dispersion`; returns ``k``, ``energy_ev`` (peak energy),
    ``fwhm_ev`` and ``amplitude``. EDC widths include the Fermi cutoff and the energy
    resolution; MDCs are the better tool for the self-energy of a steep band.
    """
    ks = np.asarray(k, dtype=float)
    ws = np.asarray(w, dtype=float)
    data = np.asarray(intensity, dtype=float)
    if data.shape != (ks.size, ws.size):
        raise ValueError('intensity must have shape (nk, nw) = %s' % ((ks.size, ws.size),))
    kmask, wmask = _mask(ks, k_window), _mask(ws, energy_window)
    peaks = _curve_peaks(
        ws[wmask], data[kmask][:, wmask], ks[kmask], method, min_contrast, edge_margin
    )
    return {
        'k': peaks['position'],
        'energy_ev': peaks['center'],
        'fwhm_ev': peaks['fwhm'],
        'amplitude': peaks['amplitude'],
    }


def fermi_momentum(energy_ev: ArrayLike, k: ArrayLike) -> float:
    """Momentum where a dispersion ``k(E)`` crosses ``E = 0`` (NaN if it does not)."""
    es = np.asarray(energy_ev, dtype=float)
    ks = np.asarray(k, dtype=float)
    if es.size < 2 or not (es.min() <= 0.0 <= es.max()):
        return np.nan
    order = np.argsort(es)
    return float(np.interp(0.0, es[order], ks[order]))


def linear_band(
    energy_ev: ArrayLike, k: ArrayLike, window_ev: tuple[float, float] | None = None
) -> dict[str, float]:
    """Line :math:`E = v\\,(k - k_F)` through the dispersion points with energies in ``window_ev``.

    Returns
    -------
    dict
        ``velocity`` (eV·Å per unit of ``k``), ``k_fermi``, ``rms`` and ``n``.
    """
    es = np.asarray(energy_ev, dtype=float)
    ks = np.asarray(k, dtype=float)
    keep = _mask(es, window_ev) & np.isfinite(ks)
    if np.count_nonzero(keep) < 2:
        raise ValueError('at least two dispersion points are needed in the window')
    v, e0 = np.polyfit(ks[keep], es[keep], 1)
    rms = float(np.sqrt(np.mean((es[keep] - e0 - v * ks[keep]) ** 2)))
    return {'velocity': float(v), 'k_fermi': float(-e0 / v), 'rms': rms, 'n': int(keep.sum())}


def self_energy_from_dispersion(
    energy_ev: ArrayLike,
    k_dressed: ArrayLike,
    fwhm_k: ArrayLike,
    bare: Callable[[NDArray], NDArray] | tuple[ArrayLike, ArrayLike],
) -> dict[str, NDArray[np.float64]]:
    """Self-energy of a band from its MDC dispersion and momentum widths against a bare band.

    Parameters
    ----------
    energy_ev : array_like ``(n,)``
        Energies of the MDCs.
    k_dressed : array_like ``(n,)``
        Peak momenta.
    fwhm_k : array_like ``(n,)``
        Momentum full widths at half maximum.
    bare : callable or (k_bare, e_bare)
        The bare band :math:`\\varepsilon_b(k)`: a function of ``k``, or a tabulated
        dispersion (``k`` increasing) interpolated linearly.

    Returns
    -------
    dict
        ``w_ev``, ``re_sigma_ev`` (:math:`\\omega - \\varepsilon_b(k(\\omega))`), ``im_sigma_ev``
        (:math:`-\\tfrac{1}{2}\\Delta k\\,|v_b|`, negative), ``bare_energy_ev`` and
        ``bare_velocity`` at the dressed momenta.
    """
    w = np.asarray(energy_ev, dtype=float)
    kd = np.asarray(k_dressed, dtype=float)
    dk = np.asarray(fwhm_k, dtype=float)
    if callable(bare):
        e_b = np.asarray(bare(kd), dtype=float)
        h = 1e-4 * (np.max(np.abs(kd)) + 1e-12)
        v_b = (np.asarray(bare(kd + h), dtype=float) - np.asarray(bare(kd - h), dtype=float)) / (
            2 * h
        )
    else:
        kb = np.asarray(bare[0], dtype=float)
        eb = np.asarray(bare[1], dtype=float)
        order = np.argsort(kb)
        kb, eb = kb[order], eb[order]
        e_b = np.interp(kd, kb, eb)
        v_b = np.interp(kd, kb, np.gradient(eb, kb))
    return {
        'w_ev': w,
        're_sigma_ev': w - e_b,
        'im_sigma_ev': -0.5 * dk * np.abs(v_b),
        'bare_energy_ev': e_b,
        'bare_velocity': v_b,
    }


def kink_energy(w_ev: ArrayLike, re_sigma_ev: ArrayLike, side: str = 'occupied') -> float:
    """Energy of the extremum of ``Re Sigma`` (where the dispersion kinks).

    Parameters
    ----------
    w_ev, re_sigma_ev : array_like
    side : {'occupied', 'empty', 'both'}
        Restrict to ``w < 0``, ``w > 0``, or search everywhere.
    """
    w = np.asarray(w_ev, dtype=float)
    re = np.asarray(re_sigma_ev, dtype=float)
    keep = {'occupied': w < 0.0, 'empty': w > 0.0, 'both': np.ones_like(w, dtype=bool)}[side]
    if not np.any(keep):
        return np.nan
    idx = np.flatnonzero(keep)
    return float(w[idx[np.argmax(np.abs(re[idx]))]])


def mass_enhancement_from_velocities(v_bare: float, v_dressed: float) -> float:
    """:math:`\\lambda = |v_b / v| - 1` from the bare and dressed Fermi velocities."""
    return float(abs(v_bare / v_dressed) - 1.0)
