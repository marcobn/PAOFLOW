"""Publication figures for model-vs-experiment comparisons.

Conventions (one encoding per job):

* **Temperature** (an ordered magnitude) is a single-hue ordinal ramp, light ->
  dark for cold -> hot (:data:`TEMPERATURE_RAMP`, validated: monotone lightness,
  adjacent steps >= 0.06 apart, light end >= 2:1 on white).  At most five
  temperatures per panel; more go to separate panels.
* **Experiment** is drawn as markers, **this model** as solid lines, a **reference
  calculation** (published theory) as dashed neutral-gray lines; every series is
  named in a legend, and up to four are labelled directly.
* Axes and grids are recessive; text stays in ink colours, never series colours.

:func:`save_figure` writes a vector PDF and a 600 dpi PNG.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

#: Ordinal blue ramp (steps 250, 350, 450, 550, 650), cold -> hot.
TEMPERATURE_RAMP = ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281')
#: Second ordinal ramp (orange), for a second temperature-encoded family in one figure.
TEMPERATURE_RAMP_ALT = ('#ef9064', '#e9703f', '#cf5a26', '#a9461c', '#823513')
INK = {'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#8a8984', 'grid': '#e4e3df'}
REFERENCE_THEORY = '#52514e'


def temperature_colors(
    temperatures: Sequence[float], ramp: Sequence[str] = TEMPERATURE_RAMP
) -> dict:
    """Colour of each temperature: evenly spaced ramp steps, cold -> hot.

    Raises
    ------
    ValueError
        For more temperatures than ramp steps (facet the figure instead).
    """
    temps = sorted(set(float(t) for t in temperatures))
    if len(temps) > len(ramp):
        raise ValueError(
            '%d temperatures but %d ramp steps: split them into panels' % (len(temps), len(ramp))
        )
    if len(temps) == 1:
        return {temps[0]: ramp[len(ramp) // 2]}
    steps = np.round(np.linspace(0, len(ramp) - 1, len(temps))).astype(int)
    return {t: ramp[s] for t, s in zip(temps, steps)}


@contextmanager
def publication_style(width_in: float = 3.4, height_in: float = 2.6, base_size: float = 8.0):
    """Matplotlib style for journal figures (single column 3.4 in, double 7.0 in)."""
    import matplotlib as mpl

    params = {
        'figure.figsize': (width_in, height_in), 'figure.dpi': 150, 'font.size': base_size,
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans', 'axes.labelsize': base_size, 'axes.titlesize': base_size,
        'axes.linewidth': 0.6, 'axes.edgecolor': INK['secondary'], 'axes.labelcolor': INK['primary'],
        'xtick.color': INK['secondary'], 'ytick.color': INK['secondary'],
        'xtick.labelcolor': INK['primary'], 'ytick.labelcolor': INK['primary'],
        'xtick.labelsize': base_size - 1, 'ytick.labelsize': base_size - 1,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6, 'lines.linewidth': 1.3,
        'lines.markersize': 4.0, 'legend.fontsize': base_size - 1, 'legend.frameon': False,
        'legend.handlelength': 1.8, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.grid': False,
    }  # fmt: skip
    with mpl.rc_context(params):
        yield


def save_figure(figure, stem: str | Path, dpi: int = 600) -> list[Path]:
    """Save ``<stem>.pdf`` (vector) and ``<stem>.png`` (``dpi``); returns the paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix('.pdf'), stem.with_suffix('.png')]
    figure.savefig(paths[0])
    figure.savefig(paths[1], dpi=dpi)
    return paths


def label_line(axis, x: float, y: float, text: str, color: str = INK['primary'], **kwargs) -> None:
    """Direct label next to a curve end, in ink colour (identity comes from the line)."""
    kwargs.setdefault('fontsize', 6.5)
    kwargs.setdefault('va', 'center')
    kwargs.setdefault('xytext', (2, 0))
    axis.annotate(text, (x, y), textcoords='offset points', color=color, **kwargs)


def panel_label(axis, text: str, inside: bool = False) -> None:
    """``(a)``, ``(b)`` ... above the upper-left corner of the frame (or just inside it)."""
    x, y, va = (0.02, 0.97, 'top') if inside else (0.0, 1.02, 'bottom')
    axis.text(x, y, text, transform=axis.transAxes, ha='left', va=va, fontweight='bold',
              color=INK['primary'])  # fmt: skip


def provenance_note(figure, text: str, chars_per_inch: float = 21.0) -> None:
    """Small provenance note under a figure, wrapped to the figure width."""
    import textwrap

    width = max(40, int(figure.get_figwidth() * chars_per_inch))
    figure.text(0.01, -0.02, textwrap.fill(text, width), ha='left', va='top', fontsize=5.5,
                color=INK['secondary'], linespacing=1.25)  # fmt: skip
