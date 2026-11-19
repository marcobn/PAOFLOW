"""Temperature scans of SSCHA runs: the stability temperature of a harmonically unstable lattice.

Near a second-order displacive transition the softest mode of the free-energy Hessian follows the mean-field
(Landau) form

.. math::

    \\omega^2(T) = a\\,(T - T_s),

so the signed square :math:`\\omega|\\omega|` (negative for imaginary frequencies) is linear in :math:`T` and its
zero is the stability temperature :math:`T_s` (Bianco, Errea, Paulatto, Calandra and Mauri, Phys. Rev. B 96, 014111
(2017)). Linear behaviour holds only close to :math:`T_s` (far from it the soft mode saturates): the bracketing pair
gives the estimate, a local fit over the points around it its consistency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .run import SSCHARun


def scan_table(
    runs: Sequence[SSCHARun], fine_mesh: tuple[int, int, int] | None = None
) -> list[dict]:
    """One row per run: temperature, electrons, stage, Hessian verdict and softest mode.

    Parameters
    ----------
    runs : sequence of SSCHARun
        The runs of the scan (same supercell and harmonic support).
    fine_mesh : tuple of int, optional
        Also the softest mode of the exported Hessian force constants on this q-mesh.

    Returns
    -------
    list of dict
        ``temperature``, ``electronic_temperature``, ``stage``, ``stable``, ``lowest_cm1`` and ``lowest_q`` (supercell
        grid; ``None`` before the Hessian), ``fine_lowest_cm1`` and ``fine_lowest_q`` (with ``fine_mesh``), sorted by
        temperature.
    """
    rows = []
    for run in sorted(runs, key=lambda item: item.temperature):
        lowest = run.lowest_hessian()
        row = dict(
            temperature=run.temperature, electronic_temperature=run.electronic_temperature,
            stage=run.status()['stage'], stable=None if run.hessian() is None else bool(run.hessian()['stable']),
            lowest_cm1=None if lowest is None else lowest[1], lowest_q=None if lowest is None else lowest[0],
            fine_lowest_cm1=None, fine_lowest_q=None,
        )  # fmt: skip
        fc = run.force_constants('hessian')
        if fine_mesh is not None and fc is not None:
            from .dispersion import grid_minimum

            q, w = grid_minimum(fc, fine_mesh)
            row.update(fine_lowest_cm1=w, fine_lowest_q=q.tolist())
        rows.append(row)
    return rows


def signed_square(frequencies_cm1: Sequence[float]) -> np.ndarray:
    """:math:`\\omega|\\omega|` (cm^-2): negative for imaginary frequencies, linear in T near the transition."""
    w = np.asarray(frequencies_cm1, dtype=float)
    return w * np.abs(w)


def stability_temperature(temperatures: Sequence[float], lowest_cm1: Sequence[float]) -> dict:
    """Stability temperature from the softest Hessian mode at several temperatures.

    Parameters
    ----------
    temperatures : sequence of float
        Ionic temperatures (K), at least two.
    lowest_cm1 : sequence of float
        Softest free-energy Hessian frequency at each temperature (cm^-1, imaginary as negative).

    Returns
    -------
    dict
        ``t_stability`` (K: zero of :math:`\\omega|\\omega|` interpolated linearly in the bracketing pair, or the fit
        when no pair brackets it), ``bracket`` (``(T_unstable, T_stable)`` or ``None``), ``t_fit`` and ``slope`` /
        ``rms`` (least-squares :math:`\\omega^2 = a (T - T_s)` over ``fit_temperatures``: the bracketing pair and one
        neighbour on each side, or the three points nearest to the sign change; cm^-2/K, cm^-2), ``extrapolated``
        (no bracket: the estimate lies outside the computed temperatures) and ``direction`` (``'stabilised'`` when
        the lattice becomes stable on heating, ``'destabilised'`` otherwise).

    Raises
    ------
    ValueError
        For fewer than two temperatures, or when the fit has no zero (flat signed square).
    """
    order = np.argsort(np.asarray(temperatures, dtype=float))
    t = np.asarray(temperatures, dtype=float)[order]
    w2 = signed_square(np.asarray(lowest_cm1, dtype=float)[order])
    if t.size < 2:
        raise ValueError('the stability temperature needs at least two temperatures')
    bracket, crossing = None, None
    for i in range(t.size - 1):
        if (w2[i] < 0.0) != (w2[i + 1] < 0.0):
            bracket = (
                (float(t[i]), float(t[i + 1])) if w2[i] < 0.0 else (float(t[i + 1]), float(t[i]))
            )
            crossing = i
            break
    if crossing is not None:  # the bracketing pair and one neighbour on each side
        window = np.arange(max(crossing - 1, 0), min(crossing + 3, t.size))
    else:  # the three temperatures nearest to the side where the sign would change
        window = np.argsort(np.abs(w2))[:3]
    window = np.sort(window)
    slope, intercept = np.polyfit(t[window], w2[window], 1)
    if slope == 0.0:
        raise ValueError('flat softest mode: no stability temperature')
    t_fit = -intercept / slope
    rms = float(np.sqrt(np.mean((slope * t[window] + intercept - w2[window]) ** 2)))
    t_stability = t_fit
    if crossing is not None:
        i = crossing
        t_stability = t[i] - w2[i] * (t[i + 1] - t[i]) / (w2[i + 1] - w2[i])
    return dict(
        t_stability=float(t_stability), bracket=bracket, t_fit=float(t_fit), slope=float(slope), rms=rms,
        fit_temperatures=[float(x) for x in t[window]], extrapolated=bracket is None,
        direction='stabilised' if slope > 0 else 'destabilised',
    )  # fmt: skip


def next_temperature(
    temperatures: Sequence[float], lowest_cm1: Sequence[float], tolerance_k: float = 100.0
) -> float | None:
    """The next temperature of a stability scan, or ``None`` when the bracket is narrower than ``tolerance_k``.

    Parameters
    ----------
    temperatures, lowest_cm1 : sequence of float
        The scan so far (see :func:`stability_temperature`).
    tolerance_k : float, optional
        Target width of the bracket (K).

    Returns
    -------
    float or None
        With a bracket: the linear estimate of :math:`T_s`, moved to at least a quarter of the bracket from its
        ends (a safeguarded secant step). Without one: the fit's :math:`T_s`, limited to twice the computed range
        beyond its nearest end and rounded to 10 K.
    """
    result = stability_temperature(temperatures, lowest_cm1)
    t = np.sort(np.asarray(temperatures, dtype=float))
    if result['bracket'] is not None:
        low, high = sorted(result['bracket'])
        if high - low <= tolerance_k:
            return None
        quarter = 0.25 * (high - low)
        guess = min(max(result['t_stability'], low + quarter), high - quarter)
    else:
        span = max(t[-1] - t[0], tolerance_k)
        guess = min(max(result['t_fit'], t[0] - 2.0 * span), t[-1] + 2.0 * span)
        guess = max(guess, 10.0)
    return float(np.round(guess, -1))
