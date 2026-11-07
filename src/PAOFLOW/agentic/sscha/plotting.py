"""Figures of SSCHA runs: convergence, temperature scans, dispersions (harmonic vs anharmonic).

Conventions of the thermo-optics toolkit (:mod:`PAOFLOW.agentic.thermo_optics.plotting`): temperature is a single-hue
ordinal ramp (cold light, hot dark; at most five per panel), the harmonic reference is a dashed neutral-gray line,
computed points are markers, fits are thin lines, text stays in ink colours. :func:`save_figure` writes a vector PDF
and a 600 dpi PNG.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..thermo_optics.plotting import (
    INK,
    REFERENCE_THEORY,
    TEMPERATURE_RAMP,
    panel_label,
    publication_style,
    save_figure,
    temperature_colors,
)
from .run import SSCHARun
from .scan import signed_square

__all__ = ['plot_convergence', 'plot_dispersions', 'plot_scan', 'publication_style', 'save_figure']


def _frame(axis) -> None:
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    axis.tick_params(top=False, right=False)


def plot_convergence(run: SSCHARun, axes=None):
    """Free energy and lowest auxiliary frequencies per population of one run.

    Parameters
    ----------
    run : SSCHARun
        The run.
    axes : pair of matplotlib axes, optional
        Target axes (created side by side otherwise).

    Returns
    -------
    figure, axes
    """
    import matplotlib.pyplot as plt

    history = run.history()
    if axes is None:
        figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
    else:
        figure = axes[0].figure
    p = np.array([row['population'] for row in history])
    f = np.array([row['free_energy_ry'] for row in history]) * 1.0e3
    df = np.array([row['free_energy_error_ry'] for row in history]) * 1.0e3
    color = TEMPERATURE_RAMP[2]
    axes[0].errorbar(p, f - f[-1], yerr=df, fmt='o', color=color, ms=4, capsize=2, lw=1.0)
    axes[0].axhline(0.0, color=INK['grid'], lw=0.8, zorder=0)
    axes[0].set_xlabel('population')
    axes[0].set_ylabel(r'$F - F_\mathrm{last}$ (mRy)')
    lowest = np.array([row['lowest_cm1'][0] for row in history])
    axes[1].plot(p, lowest, 'o-', color=color, ms=4, lw=1.0, label='lowest auxiliary mode')
    axes[1].set_xlabel('population')
    axes[1].set_ylabel(r'frequency (cm$^{-1}$)')
    for row in history:
        if row['converged']:
            axes[1].annotate('converged', (row['population'], row['lowest_cm1'][0]), textcoords='offset points',
                             xytext=(0, 6), ha='center', fontsize=6.5, color=INK['secondary'])  # fmt: skip
    for axis, label in zip(axes, ('(a)', '(b)')):
        _frame(axis)
        axis.set_xticks(p)
        panel_label(axis, label)
    axes[1].set_title('%g K' % run.temperature, loc='right', fontsize=7, color=INK['secondary'])
    return figure, axes


def plot_scan(rows: Sequence[Mapping], result: Mapping | None = None, experiment: Mapping | None = None,
              axis=None, fine: bool = False):  # fmt: skip
    """Signed square of the softest Hessian mode against temperature, the fit and the stability temperature.

    Parameters
    ----------
    rows : sequence of dict
        :func:`~PAOFLOW.agentic.sscha.scan.scan_table` rows with a Hessian.
    result : dict, optional
        :func:`~PAOFLOW.agentic.sscha.scan.stability_temperature` of the same rows (line and :math:`T_s`).
    experiment : dict, optional
        ``{'temperature': K, 'label': str}`` (and optional ``'low'``, ``'high'`` for a range): a measured
        transition or stability temperature, drawn as a gray band or line.
    axis : matplotlib axis, optional
        Target axis.
    fine : bool, optional
        Plot the fine-mesh softest mode (``fine_lowest_cm1``) instead of the supercell-grid one.

    Returns
    -------
    figure, axis
    """
    import matplotlib.pyplot as plt

    key = 'fine_lowest_cm1' if fine else 'lowest_cm1'
    points = [row for row in rows if row.get(key) is not None]
    if axis is None:
        figure, axis = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    else:
        figure = axis.figure
    t = np.array([row['temperature'] for row in points])
    w2 = signed_square([row[key] for row in points])
    axis.axhline(0.0, color=INK['secondary'], lw=0.7, zorder=1)
    if experiment:
        low, high = experiment.get('low'), experiment.get('high')
        if low is not None and high is not None:
            axis.axvspan(low, high, color=INK['grid'], zorder=0, lw=0)
        axis.axvline(experiment['temperature'], color=REFERENCE_THEORY, ls=(0, (4, 3)), lw=1.0,
                     label=experiment.get('label', 'experiment'))  # fmt: skip
    if result:
        fitted = result.get('fit_temperatures') or list(t)
        span = np.linspace(min(min(fitted), result['t_fit']), max(max(fitted), result['t_fit']))
        axis.plot(span, result['slope'] * (span - result['t_fit']), color=TEMPERATURE_RAMP[1], lw=1.0,
                  label=r'local fit $\omega^2 = a(T - T_s)$')  # fmt: skip
        axis.axvline(result['t_stability'], color=TEMPERATURE_RAMP[4], lw=1.0,
                     label=r'$T_s$ = %.0f K%s' % (result['t_stability'],
                                                   ' (extrapolated)' if result['extrapolated'] else ''))  # fmt: skip
    axis.plot(t, w2, 'o', color=TEMPERATURE_RAMP[3], ms=5, label='SSCHA Hessian', zorder=3)
    axis.set_xlabel('temperature (K)')
    axis.set_ylabel(r'$\omega|\omega|$ of the softest mode (cm$^{-2}$)')
    axis.legend(loc='best')
    _frame(axis)
    return figure, axis


def plot_dispersions(curves: Mapping[float, Mapping], reference: Mapping | None = None, axis=None,
                     reference_label: str = 'harmonic'):  # fmt: skip
    """Dispersions at several temperatures against a harmonic reference.

    Parameters
    ----------
    curves : dict
        ``{temperature: dispersion}`` with :func:`~PAOFLOW.agentic.sscha.dispersion.dispersion` results on the
        same path (at most five temperatures).
    reference : dict, optional
        The harmonic dispersion on the same path (dashed gray).
    axis : matplotlib axis, optional
        Target axis.
    reference_label : str, optional
        Legend entry of the reference.

    Returns
    -------
    figure, axis
    """
    import matplotlib.pyplot as plt

    if axis is None:
        figure, axis = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    else:
        figure = axis.figure
    first = reference if reference is not None else next(iter(curves.values()))
    for tick in first['ticks'][1:-1]:
        axis.axvline(tick, color=INK['grid'], lw=0.7, zorder=0)
    axis.axhline(0.0, color=INK['secondary'], lw=0.7, zorder=1)
    if reference is not None:
        axis.plot(reference['distance'], reference['frequencies_cm1'], color=REFERENCE_THEORY, lw=0.9,
                  ls=(0, (4, 3)))  # fmt: skip
        axis.plot([], [], color=REFERENCE_THEORY, lw=0.9, ls=(0, (4, 3)), label=reference_label)
    colors = temperature_colors(list(curves))
    for temperature in sorted(curves):
        curve = curves[temperature]
        axis.plot(
            curve['distance'], curve['frequencies_cm1'], color=colors[float(temperature)], lw=1.2
        )
        axis.plot(
            [], [], color=colors[float(temperature)], lw=1.2, label='SSCHA %g K' % temperature
        )
    axis.set_xticks(first['ticks'])
    axis.set_xticklabels(
        [r'$\Gamma$' if label in ('G', 'Gamma', 'Γ') else label for label in first['labels']]
    )
    axis.set_xlim(first['ticks'][0], first['ticks'][-1])
    axis.set_ylabel(r'frequency (cm$^{-1}$; imaginary < 0)')
    axis.legend(loc='best')
    _frame(axis)
    return figure, axis
