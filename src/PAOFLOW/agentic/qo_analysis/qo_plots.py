"""Figures for quantum-oscillation comparisons (matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#17becf',
           '#bcbd22', '#7f7f7f']  # fmt: skip
GREEK_SYMBOLS = [('epsilon', 'ε'), ('kappa', 'κ'), ('sigma', 'σ'), ('gamma', 'γ'), ('delta', 'δ'),
                 ('alpha', 'α'), ('beta', 'β'), ('mu', 'μ'), ('eta', 'η'), ('zeta', 'ζ'), (' deg', '°')]  # fmt: skip


def _pyplot():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def with_greek_symbols(text: str) -> str:
    """Replace Greek letter names (``beta1``) by symbols (``β1``) for plot labels."""
    for name, symbol in GREEK_SYMBOLS:
        text = text.replace(name, symbol)
    return text


def group_colors(groups, user_colors: dict | None = None) -> dict[str, str]:
    """Colour per degenerate group, user choices first."""
    user_colors = user_colors or {}
    ordered = sorted({str(group) for group in groups})
    return {
        group: user_colors.get(group, PALETTE[i % len(PALETTE)]) for i, group in enumerate(ordered)
    }


def _measured_peaks(axes, peaks: pd.DataFrame, weak_threshold: float) -> None:
    weak, strong = peaks.rel_mag < weak_threshold, peaks.rel_mag >= weak_threshold
    axes.scatter(peaks.calc_angle[weak], peaks.freq_T[weak], s=10, facecolors='none', edgecolors='0.75',
                 lw=0.6, label=f'exp. peak, weak (< {weak_threshold:.2f})', zorder=1)  # fmt: skip
    axes.scatter(peaks.calc_angle[strong], peaks.freq_T[strong], s=25 + 160 * peaks.rel_mag[strong],
                 facecolors='none', edgecolors='k', lw=1.1, label='exp. peak (size = FFT magnitude)',
                 zorder=3)  # fmt: skip


def _branch_labels(axes, assignments: pd.DataFrame, colors: dict[str, str]) -> None:
    for entry in assignments.itertuples():
        text = with_greek_symbols(str(entry.label))
        group = '' if pd.isna(entry.group) else str(entry.group)
        if str(entry.status) == 'unassigned' or not group:
            axes.text(entry.text_angle_deg, entry.text_freq_T, text, fontsize=8, style='italic', color='0.35',
                      ha='center', zorder=6,
                      bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))  # fmt: skip
            continue
        color = colors.get(group, 'k')
        axes.annotate(text, xy=(entry.arrow_angle_deg, entry.arrow_freq_T),
                      xytext=(entry.text_angle_deg, entry.text_freq_T), fontsize=12, fontweight='bold',
                      color=color, ha='center', va='center', zorder=6,
                      bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, lw=0.8, alpha=0.85),
                      arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0, shrinkA=2, shrinkB=1))  # fmt: skip


def plot_sweep_overlay(
    orbits: pd.DataFrame,
    peaks: pd.DataFrame,
    path: str | Path,
    sweep_angle: str = 'polar',
    weak_threshold: float = 0.15,
    colors: dict | None = None,
    labels: dict | None = None,
    assignments: pd.DataFrame | None = None,
    title: str = '',
    axis_label: str | None = None,
    guide_angles: tuple[float, ...] = (0, 45, 90),
) -> None:
    """Calculated orbits (colour = group, size ~ curvature amplitude) over the measured sweep peaks.

    Returns
    -------
    None
        Writes the figure to ``path``.
    """
    plt = _pyplot()
    colors = group_colors(orbits.group, colors)
    labels = labels or {}
    largest = orbits.amp_factor.quantile(0.98) if len(orbits) else 1.0
    figure, axes = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    _measured_peaks(axes, peaks, weak_threshold)
    for group, members in orbits.groupby('group'):
        axes.scatter(members[f'{sweep_angle}_deg'], members.freq_T,
                     s=4 + 26 * np.clip(members.amp_factor / largest, 0, 1), color=colors[str(group)],
                     alpha=0.75, lw=0, zorder=2, label=labels.get(str(group), f'calc. {group}'))  # fmt: skip
    if assignments is not None and len(assignments):
        _branch_labels(axes, assignments, colors)
    for angle in guide_angles:
        axes.axvline(angle, color='0.85', lw=0.8, zorder=0)
    low, high = orbits[f'{sweep_angle}_deg'].min(), orbits[f'{sweep_angle}_deg'].max()
    axes.set(yscale='log', ylim=(5, 1e4), xlim=(low - 2, high + 2), ylabel='F (T)',
             xlabel=axis_label or f'{sweep_angle} angle (deg)')  # fmt: skip
    axes.set_title(title, fontsize=11)
    axes.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_on_measured_map(
    orbits: pd.DataFrame,
    image: str | Path,
    frequency_axis: str,
    frequency_pixels: tuple[float, float],
    frequency_values: tuple[float, float],
    angle_pixels: tuple[float, float],
    angle_values: tuple[float, float],
    path: str | Path,
    sweep_angle: str = 'polar',
    log_frequency: bool = True,
    colors: dict | None = None,
    angle_scale: float = 1.0,
    angle_offset: float = 0.0,
) -> None:
    """Draw the calculated orbits on the original experimental FFT map (same calibration as digitising)."""
    plt = _pyplot()
    from PIL import Image

    colors = group_colors(orbits.group, colors)
    f_pix, f_val = np.asarray(frequency_pixels, float), np.asarray(frequency_values, float)
    a_pix, a_val = np.asarray(angle_pixels, float), np.asarray(angle_values, float)
    frequencies = orbits.freq_T.to_numpy()
    measured_angle = (orbits[f'{sweep_angle}_deg'].to_numpy() - angle_offset) / angle_scale
    if log_frequency:
        f_coord = f_pix[0] + (np.log10(frequencies) - np.log10(f_val[0])) * (
            f_pix[1] - f_pix[0]
        ) / (np.log10(f_val[1]) - np.log10(f_val[0]))
    else:
        f_coord = f_pix[0] + (frequencies - f_val[0]) * (f_pix[1] - f_pix[0]) / (
            f_val[1] - f_val[0]
        )
    a_coord = a_pix[0] + (measured_angle - a_val[0]) * (a_pix[1] - a_pix[0]) / (a_val[1] - a_val[0])
    x, y = (f_coord, a_coord) if frequency_axis == 'x' else (a_coord, f_coord)
    inside = (frequencies > f_val.min()) & (frequencies < f_val.max())
    figure, axes = plt.subplots(figsize=(14, 6), dpi=150)
    axes.imshow(Image.open(image))
    for group in sorted({str(g) for g in orbits.group}):
        member = inside & (orbits.group.astype(str) == group).to_numpy()
        axes.scatter(x[member], y[member], s=14, color='white', lw=0)
        axes.scatter(x[member], y[member], s=6, color=colors[group], lw=0, label=f'calc. {group}')
    axes.axis('off')
    axes.legend(fontsize=8, loc='upper right', markerscale=4, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_network_orbits(
    direct_orbits: pd.DataFrame,
    network: pd.DataFrame,
    peaks: pd.DataFrame,
    unexplained: np.ndarray,
    matched: np.ndarray,
    path: str | Path,
    sweep_angle: str = 'polar',
    weak_threshold: float = 0.15,
    min_copies: int = 20,
    min_amplitude: float = 0.05,
    title: str = '',
    axis_label: str | None = None,
) -> None:
    """Breakdown-network orbits against the measured peaks that no direct orbit explains."""
    plt = _pyplot()
    robust = (network.copies >= min_copies) & (network.R_ref >= min_amplitude)
    weak_open = network.uses_open & ~robust & (network.copies >= 3)
    figure, axes = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    axes.scatter(direct_orbits[f'{sweep_angle}_deg'], direct_orbits.freq_T, s=3, color='0.82', lw=0,
                 label='direct orbits')  # fmt: skip
    for open_flag, color, text in ((False, '#1f77b4', 'network orbit, closed pieces only'),
                                   (True, '#d62728', 'network orbit using open-sheet pieces')):  # fmt: skip
        member = robust & (network.uses_open == open_flag)
        axes.scatter(network.angle[member], network.freq_T[member], s=8 + 60 * network.R_ref[member],
                     color=color, alpha=0.75, lw=0, label=f'{text} (size ~ R)')  # fmt: skip
    axes.scatter(network.angle[weak_open], network.freq_T[weak_open], s=6, marker='s', facecolors='none',
                 edgecolors='#d62728', lw=0.5, alpha=0.6, label='open-sheet network orbit, below threshold')  # fmt: skip
    strong = peaks.rel_mag >= weak_threshold
    axes.scatter(peaks.calc_angle[strong], peaks.freq_T[strong], s=25 + 160 * peaks.rel_mag[strong],
                 facecolors='none', edgecolors='0.45', lw=0.9, label='exp. peak')  # fmt: skip
    axes.scatter(peaks.calc_angle[unexplained], peaks.freq_T[unexplained],
                 s=25 + 160 * peaks.rel_mag[unexplained], facecolors='none', edgecolors='k', lw=1.4,
                 label='exp. peak, no direct orbit')  # fmt: skip
    axes.scatter(peaks.calc_angle[matched], peaks.freq_T[matched], marker='x', s=45, color='k', lw=1.5,
                 zorder=5, label='... matched by a network orbit')  # fmt: skip
    axes.set(yscale='log', ylim=(10, 1e4), xlim=(peaks.calc_angle.min() - 2, peaks.calc_angle.max() + 2),
             ylabel='F (T)', xlabel=axis_label or f'{sweep_angle} angle (deg)')  # fmt: skip
    axes.set_title(title, fontsize=11)
    axes.legend(fontsize=7, loc='lower right', framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_run_comparison(
    panels: list[tuple[str, pd.DataFrame, pd.DataFrame, str, str]],
    peaks: pd.DataFrame | None,
    path: str | Path,
    sweep_angle: str = 'polar',
    weak_threshold: float = 0.15,
) -> None:
    """One panel per reference band: reference orbits (grey) vs this run (red), with measured peaks."""
    plt = _pyplot()
    figure, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 6.5), dpi=130, sharey=True,
                                squeeze=False)  # fmt: skip
    for axis, (title, reference, other, reference_label, other_label) in zip(axes[0], panels):
        if peaks is not None:
            strong = peaks.loc[peaks.rel_mag >= weak_threshold]
            axis.scatter(strong.calc_angle, strong.freq_T, s=20 + 120 * strong.rel_mag, facecolors='none',
                         edgecolors='k', lw=0.9, zorder=1, label='exp. peak')  # fmt: skip
        axis.scatter(reference[f'{sweep_angle}_deg'], reference.freq_T, s=10, color='0.6', lw=0,
                     label=reference_label)  # fmt: skip
        if len(other):
            axis.scatter(
                other[f'{sweep_angle}_deg'],
                other.freq_T,
                s=5,
                color='#d62728',
                lw=0,
                label=other_label,
            )
        else:
            axis.text(
                0.5,
                0.05,
                f'{other_label}: no data',
                transform=axis.transAxes,
                ha='center',
                color='#d62728',
            )
        axis.set(yscale='log', ylim=(5, 1e4), title=title, xlabel=f'{sweep_angle} angle (deg)')
        axis.legend(fontsize=7, loc='lower right')
    axes[0][0].set_ylabel('F (T)')
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
