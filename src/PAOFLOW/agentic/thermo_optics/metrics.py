"""Quantitative comparison of model and experimental curves.

Every number reported by the workflow comes from one of these functions, so its
definition is fixed:

* :func:`curve_deviation` -- deviation statistics in a stated space (``log10`` for
  quantities spanning decades, ``linear`` or ``relative`` otherwise);
* :func:`best_shift` -- the rigid abscissa shift (e.g. an energy offset) that
  minimises the deviation, with the deviation before and after;
* :func:`threshold_crossing` -- where a curve crosses a level (absorption onsets);
* :func:`arrhenius_fit` -- activation energy of ``y ∝ exp(-E_a/k_B T)``;
* :func:`match_conditions` -- pairs experimental and model conditions (e.g.
  temperatures) within a tolerance, reporting the mismatch.

A comparison is meaningful only against a reference: quote the same statistic
for a baseline (a published calculation of the same data, a trivial model, or the
spread between independent measurements).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

KB_EV = 8.617333262e-5


def interpolate(
    x: ArrayLike, y: ArrayLike, x_query: ArrayLike, log_y: bool = False
) -> NDArray[np.float64]:
    """Linear interpolation (in ``log10 y`` if ``log_y``); NaN outside the data range."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if log_y:
        good = y > 0
        x, y = x[good], np.log10(y[good])
    values = np.interp(np.asarray(x_query, dtype=float), x, y, left=np.nan, right=np.nan)
    return 10.0**values if log_y else values


def curve_deviation(
    x_exp: ArrayLike,
    y_exp: ArrayLike,
    x_model: ArrayLike,
    y_model: ArrayLike,
    space: str = 'log10',
    window: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Deviation of a model curve from experimental points.

    Parameters
    ----------
    x_exp, y_exp : array_like
        Experimental points.
    x_model, y_model : array_like
        Model curve (interpolated to the experimental abscissae).
    space : {'log10', 'linear', 'relative'}, optional
        ``log10``: residual ``log10(y_model / y_exp)`` (dex); ``linear``:
        ``y_model - y_exp``; ``relative``: ``(y_model - y_exp) / y_exp``.
    window : tuple of float, optional
        Use only experimental points with ``window[0] <= x <= window[1]``.

    Returns
    -------
    dict
        ``n`` (points compared), ``bias`` (mean residual), ``rms``, ``median``,
        ``max_abs``, and for ``log10`` also ``median_ratio`` (``10**median``).
    """
    x_exp = np.asarray(x_exp, dtype=float)
    y_exp = np.asarray(y_exp, dtype=float)
    keep = np.isfinite(x_exp) & np.isfinite(y_exp)
    if window is not None:
        keep &= (x_exp >= window[0]) & (x_exp <= window[1])
    if space == 'log10':
        keep &= y_exp > 0
    model = interpolate(x_model, y_model, x_exp[keep], log_y=(space == 'log10'))
    exp = y_exp[keep]
    finite = np.isfinite(model)
    model, exp = model[finite], exp[finite]
    if space == 'log10':
        positive = model > 0
        residual = np.log10(model[positive] / exp[positive])
    elif space == 'linear':
        residual = model - exp
    elif space == 'relative':
        residual = (model - exp) / exp
    else:
        raise ValueError("space must be 'log10', 'linear' or 'relative'")
    if residual.size == 0:
        return {'n': 0, 'bias': np.nan, 'rms': np.nan, 'median': np.nan, 'max_abs': np.nan}
    out = {
        'n': int(residual.size), 'bias': float(residual.mean()),
        'rms': float(np.sqrt(np.mean(residual**2))), 'median': float(np.median(residual)),
        'max_abs': float(np.max(np.abs(residual))),
    }  # fmt: skip
    if space == 'log10':
        out['median_ratio'] = float(10.0 ** out['median'])
    return out


def best_shift(
    x_exp: ArrayLike,
    y_exp: ArrayLike,
    x_model: ArrayLike,
    y_model: ArrayLike,
    shifts: ArrayLike,
    space: str = 'log10',
    window: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Rigid abscissa shift of the model that minimises the rms deviation.

    The model is evaluated at ``x - shift`` (a positive shift moves the model to
    larger x).  Report it next to the unshifted deviation: a shift is a
    *diagnostic* (e.g. a band-gap error), never a silent fit.

    Returns
    -------
    dict
        ``shift`` (best), ``rms_unshifted``, ``rms_shifted``, ``n``.
    """
    x_model = np.asarray(x_model, dtype=float)
    shifts = np.asarray(shifts, dtype=float)
    scores = []
    for shift in shifts:
        scores.append(curve_deviation(x_exp, y_exp, x_model + shift, y_model, space, window)['rms'])
    scores = np.asarray(scores)
    base = curve_deviation(x_exp, y_exp, x_model, y_model, space, window)
    if not np.any(np.isfinite(scores)):  # no shift brings the model onto the experimental points
        return {
            'shift': np.nan,
            'rms_unshifted': base['rms'],
            'rms_shifted': np.nan,
            'n': base['n'],
        }
    k = int(np.nanargmin(scores))
    return {'shift': float(shifts[k]), 'rms_unshifted': base['rms'], 'rms_shifted': float(scores[k]),
            'n': base['n']}  # fmt: skip


def threshold_crossing(x: ArrayLike, y: ArrayLike, level: float, log_y: bool = False) -> float:
    """Smallest x where a curve first rises through ``level`` (NaN if never)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    target = np.log10(np.clip(y, 1e-300, None)) if log_y else y
    level_t = np.log10(level) if log_y else level
    above = np.nonzero(target >= level_t)[0]
    if above.size == 0 or above[0] == 0:
        return float(x[above[0]]) if above.size else np.nan
    i = above[0]
    return float(np.interp(level_t, [target[i - 1], target[i]], [x[i - 1], x[i]]))


def arrhenius_fit(temps_k: ArrayLike, values: ArrayLike) -> dict[str, float]:
    """Fit ``values = A exp(-E_a / k_B T)``; returns ``activation_ev``, ``prefactor``, ``rms_log``."""
    temps = np.asarray(temps_k, dtype=float)
    values = np.asarray(values, dtype=float)
    good = values > 0
    slope, intercept = np.polyfit(1.0 / (KB_EV * temps[good]), np.log(values[good]), 1)
    fitted = slope / (KB_EV * temps[good]) + intercept
    return {'activation_ev': float(-slope), 'prefactor': float(np.exp(intercept)),
            'rms_log': float(np.sqrt(np.mean((np.log(values[good]) - fitted) ** 2)))}  # fmt: skip


def match_conditions(
    experiment: Sequence[float], model: Sequence[float], tolerance: float
) -> list[tuple[float, float | None, float]]:
    """Pair each experimental condition with the nearest model one.

    Returns ``(exp, model or None, |difference|)``; ``None`` when the nearest model
    value is farther than ``tolerance`` (the comparison then needs a new run or an
    interpolation, stated as such).
    """
    model = np.asarray(model, dtype=float)
    out = []
    for value in experiment:
        k = int(np.argmin(np.abs(model - value)))
        difference = float(abs(model[k] - value))
        out.append((float(value), float(model[k]) if difference <= tolerance else None, difference))
    return out


def interpolate_in_condition(
    conditions: ArrayLike,
    curves: ArrayLike,
    target: float,
    log_y: bool = False,
    transform: Callable[[NDArray], NDArray] | None = None,
) -> NDArray[np.float64]:
    """Interpolate a family of curves ``(ncond, nx)`` to one condition value.

    ``transform`` maps the condition before the linear interpolation (e.g.
    ``lambda t: 1 / t`` for Arrhenius-like temperature dependences, with
    ``log_y=True``).  The result is an interpolated *model*, to be labelled as such.
    """
    conditions = np.asarray(conditions, dtype=float)
    curves = np.asarray(curves, dtype=float)
    axis = transform(conditions) if transform is not None else conditions
    value = transform(np.asarray(target, dtype=float)) if transform is not None else target
    order = np.argsort(axis)
    axis, curves = axis[order], curves[order]
    if log_y:
        curves = np.log10(np.clip(curves, 1e-300, None))
    out = np.array([np.interp(value, axis, curves[:, j]) for j in range(curves.shape[1])])
    return 10.0**out if log_y else out
