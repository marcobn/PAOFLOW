"""Figure of the surface spectral function of ``Transport.compute_surface_bands``.

Used by :meth:`PAOFLOW.GPAO.GPAO.plot_surface_bands`. The reader is separate from the drawing so
that other viewers can load the same files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: path labels of the band-path generator that are Gamma
GAMMA_LABELS = ('gG', 'G', 'Gamma', 'GAMMA')


@dataclass
class SurfaceSpectrum:
    """Surface spectral function on a k-path, as written by ``compute_surface_bands``.

    Attributes
    ----------
    spectral : ndarray, shape ``(ne, nk)``
        :math:`A(k, E)` (states/eV), one row per energy.
    energy : ndarray, shape ``(ne,)``
        :math:`E - E_F` (eV).
    kdist : ndarray, shape ``(nk,)``
        Distance along the path in the surface plane (``2 pi / alat``).
    ticks : list of int
        Path indices of the high-symmetry points.
    labels : list of str
        Their labels, as the band-path generator writes them (``gG`` for Gamma).
    """

    spectral: NDArray[np.float64]
    energy: NDArray[np.float64]
    kdist: NDArray[np.float64]
    ticks: list[int]
    labels: list[str]


def read_surface_bands(path: str = '.', postfix: str = '') -> SurfaceSpectrum:
    """Read ``surfband{postfix}.dat`` and its two axis files.

    Parameters
    ----------
    path : str, optional
        Output directory of the transport run (``configure_outputs(output_dir=...)``).
    postfix : str, optional
        ``configure_outputs(postfix=...)`` of the run.

    Returns
    -------
    SurfaceSpectrum
        The map and its axes.

    Raises
    ------
    FileNotFoundError
        When one of the three files is missing.
    ValueError
        When the map does not match its axes.
    """
    spectral = np.atleast_2d(np.loadtxt(os.path.join(path, f'surfband{postfix}.dat')))
    energy = np.atleast_1d(np.loadtxt(os.path.join(path, f'surfband_egrid{postfix}.dat')))
    kdist: list[float] = []
    ticks: list[int] = []
    labels: list[str] = []
    with open(os.path.join(path, f'surfband_kpath{postfix}.dat')) as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].startswith('#'):
                continue
            kdist.append(float(parts[1]))
            if len(parts) > 2:
                ticks.append(int(parts[0]))
                labels.append(parts[2])
    if spectral.shape != (energy.size, len(kdist)):
        raise ValueError(
            f'surfband{postfix}.dat has shape {spectral.shape}, the axes give '
            f'({energy.size}, {len(kdist)})'
        )
    return SurfaceSpectrum(spectral, energy, np.asarray(kdist), ticks, labels)


def surface_label(label: str, labels: dict[str, str] | None = None) -> str:
    """Display form of a path label: an overbar, as for every point of the surface zone.

    Parameters
    ----------
    label : str
        Label of the band-path generator (``gG``, ``K``, ``M``, ...).
    labels : dict, optional
        Display forms that replace the default, keyed by the generator's label.

    Returns
    -------
    str
        Matplotlib mathtext, ``$\\overline{\\Gamma}$`` for Gamma.
    """
    if labels is not None and label in labels:
        return labels[label]
    symbol = r'\Gamma' if label in GAMMA_LABELS else r'\mathrm{%s}' % label
    return r'$\overline{%s}$' % symbol


def plot_surface_spectrum(
    spectrum: SurfaceSpectrum,
    scale: str = 'log',
    clip_percentile: float = 99.0,
    cmap: str = 'inferno',
    labels: dict[str, str] | None = None,
    y_lim: tuple[float, float] | None = None,
    title: str | None = None,
    filename: str | None = None,
    ax: Any = None,
) -> Any:
    """Draw :math:`A(k, E)` as a map along the surface path.

    Parameters
    ----------
    spectrum : SurfaceSpectrum
        The map and its axes (:func:`read_surface_bands`).
    scale : {'log', 'linear'}, optional
        Colour scale. ``'log'`` shows the bulk continuum and the much sharper surface states
        together; ``'linear'`` clips at ``clip_percentile`` and normalises to 1.
    clip_percentile : float, optional
        Percentile of ``A`` that saturates the linear scale.
    cmap : str, optional
        Matplotlib colour map (perceptually uniform by default, readable in grayscale).
    labels : dict, optional
        Display forms of the path labels (:func:`surface_label`).
    y_lim : tuple of float, optional
        Energy window (eV).
    title : str, optional
    filename : str, optional
        Save the figure there and close it; ``None`` shows it. Ignored with ``ax``.
    ax : matplotlib Axes, optional
        Draw into this axis (of a larger figure), which is then neither saved nor shown.

    Returns
    -------
    matplotlib Axes
        The axis of the map.

    Raises
    ------
    ValueError
        For an unknown ``scale``.

    Notes
    -----
    In surface mode the conductor Green's function carries only the left-lead self-energy, so

    .. math::

        A(k, E) = -\\frac{1}{\\pi}\\,\\mathrm{Im}\\,\\mathrm{Tr}\\,G_s(k, E + i\\delta)

    is the bulk band structure projected on the surface plus the surface states. The log scale
    starts at ``max(A) * 1e-4`` (or the smallest positive value, if larger).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    values = np.asarray(spectrum.spectral, dtype=float)
    if scale == 'log':
        positive = values[values > 0.0]
        top = float(positive.max()) if positive.size else 1.0
        floor = max(float(positive.min()) if positive.size else top, top * 1.0e-4)
        values = np.clip(values, floor, None)
        norm = LogNorm(vmin=floor, vmax=top)
        bar_label = r'$A(k, E)$ (states/eV)'
    elif scale == 'linear':
        top = float(np.percentile(values, clip_percentile)) or 1.0
        values = np.clip(values / top, 0.0, 1.0)
        norm = Normalize(vmin=0.0, vmax=1.0)
        bar_label = r'$A(k, E)$ / $A_{%g\%%}$' % clip_percentile
    else:
        raise ValueError("scale must be 'log' or 'linear', got %r" % scale)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.pcolormesh(
        spectrum.kdist, spectrum.energy, values, cmap=cmap, norm=norm, shading='auto'
    )
    fig.colorbar(image, ax=ax, label=bar_label)

    tick_positions = [spectrum.kdist[index] for index in spectrum.ticks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([surface_label(label, labels) for label in spectrum.labels])
    for position in tick_positions[1:-1]:
        ax.axvline(position, color='w', lw=0.5, alpha=0.5)
    ax.axhline(0.0, color='w', ls='--', lw=0.8, alpha=0.8)
    ax.set_xlim(spectrum.kdist[0], spectrum.kdist[-1])
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)')
    if title is not None:
        ax.set_title(title)

    if own_figure:
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename, dpi=200)
            plt.close(fig)
    return ax
