"""Figures of the non-linear and coupled optical responses (used by :class:`PAOFLOW.GPAO.GPAO`).

Every function draws on a fresh figure, optionally saves it (``filename``) and shows it. Colours: the
categorical slots of the PAOFLOW report figures in a fixed order (validated for adjacent-line
contrast under colour-vision deficiency), the total in primary ink; colour follows the entity (a
kernel level or an SHG term keeps its hue in every figure).
"""

from __future__ import annotations

import os
import re

import numpy as np
from numpy.typing import NDArray

#: categorical slots 1-8 (fixed order) and the ink of totals and text
SLOTS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e1e0d9'

#: response key -> (symbol, unit of a 3D crystal, unit of a 2D sheet)
NONLINEAR_UNITS = {
    'shift': (r'$\sigma_{\rm shift}$', r'$\mu$A/V$^2$', r'nm $\mu$A/V$^2$'),
    'shiftm': (r'$\sigma^{M}_{\rm shift}$', r'$\mu$A/V$^2$', r'nm $\mu$A/V$^2$'),
    'injectc': (r'$\eta_C$', r'A V$^{-2}$ s$^{-1}$', r'nm A V$^{-2}$ s$^{-1}$'),
    'injectm': (r'$\eta_M$', r'A V$^{-2}$ s$^{-1}$', r'nm A V$^{-2}$ s$^{-1}$'),
    'shg': (r'$\chi^{(2)}$', 'pm/V', r'pm$^2$/V'),
    'pcr': (r'Re $\sigma^{(2)}(0)$', r'$\mu$A/V$^2$', r'nm $\mu$A/V$^2$'),
    'pci': (r'Im $\sigma^{(2)}(0)$', r'$\mu$A/V$^2$', r'nm $\mu$A/V$^2$'),
}

#: SHG term groups of plot_shg_terms (grouped=True): label -> member terms (SHG_TERMS names)
SHG_TERM_GROUPS = (
    ('interband, resonant at 2ω', ('interband_2w',)),
    ('interband, resonant at ω', ('interband_w',)),
    ('intraband, resonant at 2ω', ('intraband_2w_derivative', 'intraband_2w_velocity')),
    (
        'intraband, resonant at ω',
        ('intraband_w_derivative', 'intraband_w_velocity', 'intraband_w_todd'),
    ),
)

#: density-matrix channels (metals; PAOFLOW.response.density_matrix_response.CHANNELS) -> legend
#: label of the SHG split and of the zero-frequency photocurrent (with its scaling in tau)
CHANNEL_LABELS = {
    'fermi_sea_inter': ('Fermi sea, interband', 'Fermi sea, interband (shift; τ⁰)'),
    'fermi_sea_intra': ('Fermi sea, populations', 'Fermi sea, populations (injection; ∝ τ)'),
    'fermi_surface_mixed': (
        'Fermi surface, interband',
        'Fermi surface, interband (Berry-curvature dipole; ∝ τ)',
    ),
    'fermi_surface_drude': ('Fermi surface, Drude', 'Fermi surface, Drude (∝ τ²)'),
}

#: kernel level -> (legend label, colour); the hues of the GaAs report figures
LEVEL_STYLE = {
    'none': ('independent particles', SLOTS[0]),
    'hartree': ('+ local fields (Hartree, G ≠ 0)', SLOTS[1]),
    'hartree+alda': ('+ ALDA', SLOTS[2]),
    'hartree+alda+bootstrap': ('+ bootstrap', SLOTS[3]),
    'hartree+alda+lrc': ('+ LRC', SLOTS[4]),
}


def component_files(path: str, prefix: str, spin: int | None = None) -> dict[str, str]:
    """Files ``<prefix>_<abc>[_<spin>].dat`` in ``path``, keyed by the component label.

    Parameters
    ----------
    path : str
        Output directory of the run.
    prefix : str
        File prefix (``shift``, ``shgr``, ``shgtr_interband_2w``, ``cepsi_ipa`` ...).
    spin : int, optional
        Spin channel suffix; ``None`` takes the files without one.

    Returns
    -------
    dict
        Component label (``'xyz'``, ``'xx'`` ...) -> file path, sorted by label.
    """
    tail = '' if spin is None else '_%d' % int(spin)
    pattern = re.compile(r'^%s_([xyz]{2,3})%s\.dat$' % (re.escape(prefix), re.escape(tail)))
    found = {}
    for name in sorted(os.listdir(path)):
        match = pattern.match(name)
        if match:
            found[match.group(1)] = os.path.join(path, name)
    return found


def _style_axes(ax) -> None:
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _finish(fig, title: str | None, filename: str | None) -> None:
    import matplotlib.pyplot as plt

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_nonlinear_spectra(
    energy: NDArray[np.float64],
    curves: dict[str, NDArray[np.float64]],
    y_label: str,
    title: str | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    filename: str | None = None,
) -> None:
    """Overlay spectra (one per tensor component) on one axis.

    Parameters
    ----------
    energy : ndarray
        Photon energy (eV).
    curves : dict
        Legend label -> values on ``energy``.
    y_label : str
    title, x_lim, y_lim, filename : optional

    Returns
    -------
    None
        Shows the figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0.0, color='#c3c2b7', lw=0.8)
    for i, (label, values) in enumerate(curves.items()):
        ax.plot(energy, values, color=SLOTS[i % len(SLOTS)], lw=1.5, label=label)
    ax.set_xlabel(r'photon energy $\hbar\omega$ (eV)')
    ax.set_ylabel(y_label)
    ax.set_xlim(*(x_lim or (energy.min(), energy.max())))
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    _style_axes(ax)
    if len(curves) > 1:
        ax.legend(frameon=False)
    _finish(fig, title, filename)


def plot_shg_terms(
    energy: NDArray[np.float64],
    total: NDArray[np.complex128],
    terms: dict[str, NDArray[np.complex128]],
    unit: str = 'pm/V',
    grouped: bool = True,
    component: str = 'xyz',
    title: str | None = None,
    x_lim: tuple[float, float] | None = None,
    filename: str | None = None,
) -> None:
    """Re and Im of an SHG component and of its terms (``shg_terms=True`` runs).

    Parameters
    ----------
    energy : ndarray
        Fundamental photon energy (eV).
    total : ndarray, complex
    terms : dict
        Term name of :data:`~PAOFLOW.response.nonlinear_optics.SHG_TERMS` -> complex values, or the
        density-matrix channels of a metal (:data:`CHANNEL_LABELS`, always drawn one by one).
    unit : str, optional
    grouped : bool, optional
        Four curves (interband and intraband, resonant at ω or 2ω), as in the PAOFLOW reports;
        ``False`` draws the seven terms.
    component : str, optional
        Label of the component, for the panel titles.
    title, x_lim, filename : optional

    Returns
    -------
    None
        Shows the figure.
    """
    import matplotlib.pyplot as plt

    if grouped and any(name in terms for _, names in SHG_TERM_GROUPS for name in names):
        series = [
            (label, sum(terms[name] for name in names if name in terms))
            for label, names in SHG_TERM_GROUPS
        ]
    else:
        series = [
            (CHANNEL_LABELS.get(name, (name.replace('_', ' '),))[0], values)
            for name, values in terms.items()
        ]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True)
    for ax, part, name in ((axes[0], np.real, 'Re'), (axes[1], np.imag, 'Im')):
        ax.axhline(0.0, color='#c3c2b7', lw=0.8)
        for i, (label, values) in enumerate(series):
            ax.plot(energy, part(values), color=SLOTS[i % len(SLOTS)], lw=1.4, label=label)
        ax.plot(energy, part(total), color=INK, lw=2.0, label='total')
        ax.set_ylabel(r'%s $\chi^{(2)}_{%s}$ (%s)' % (name, component, unit))
        _style_axes(ax)
    axes[1].set_xlabel(r'fundamental photon energy $\hbar\omega$ (eV)')
    axes[1].set_xlim(*(x_lim or (energy.min(), energy.max())))
    axes[0].legend(frameon=False, fontsize=8)
    _finish(fig, title, filename)


def plot_photocurrent_channels(
    energy: NDArray[np.float64],
    total: NDArray[np.complex128],
    channels: dict[str, NDArray[np.complex128]],
    unit: str = r'$\mu$A/V$^2$',
    component: str = 'xyz',
    relaxation: str | None = None,
    title: str | None = None,
    x_lim: tuple[float, float] | None = None,
    filename: str | None = None,
) -> None:
    r"""Zero-frequency photocurrent of a metal split by origin: linear (Re) and circular (Im) panels.

    Parameters
    ----------
    energy : ndarray
        Photon energy (eV).
    total : ndarray, complex
        :math:`\sigma^{abc}(0;\omega,-\omega)`; :math:`J = {\rm Re}\,\sigma\,{\rm Re}[E^bE^{c*}]
        - {\rm Im}\,\sigma\,{\rm Im}[E^bE^{c*}]`.
    channels : dict
        Channel of :data:`CHANNEL_LABELS` -> complex values (they sum to ``total``).
    unit : str, optional
    component : str, optional
    relaxation : str, optional
        Legend note on the intraband relaxation (e.g. ``'τ = 20 fs'``).
    title, x_lim, filename : optional

    Returns
    -------
    None
        Shows the figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True)
    panels = ((axes[0], np.real, 'Re', 'linear'), (axes[1], np.imag, 'Im', 'circular'))
    for ax, part, name, polarisation in panels:
        ax.axhline(0.0, color='#c3c2b7', lw=0.8)
        for i, (channel, values) in enumerate(channels.items()):
            label = CHANNEL_LABELS.get(channel, (None, channel.replace('_', ' ')))[1]
            ax.plot(energy, part(values), color=SLOTS[i % len(SLOTS)], lw=1.4, label=label)
        ax.plot(energy, part(total), color=INK, lw=2.0, label='total')
        ax.set_ylabel(r'%s $\sigma_{%s}$ (%s), %s' % (name, component, unit, polarisation))
        _style_axes(ax)
    axes[1].set_xlabel(r'photon energy $\hbar\omega$ (eV)')
    axes[1].set_xlim(*(x_lim or (energy.min(), energy.max())))
    axes[0].legend(frameon=False, fontsize=8, title=relaxation, title_fontsize=8)
    _finish(fig, title, filename)


def plot_coupled_levels(
    curves: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    y_label: str,
    alphas: dict[str, float] | None = None,
    title: str | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    filename: str | None = None,
) -> None:
    """Overlay a response at the kernel levels of ``coupled_optical_response``.

    Parameters
    ----------
    curves : dict
        Kernel level (``'none'``, ``'hartree'``, ...) -> (energy, values).
    y_label : str
    alphas : dict, optional
        ``'hartree+alda+lrc'`` / ``'hartree+alda+bootstrap'`` -> :math:`\\alpha`, shown in the legend.
    title, x_lim, y_lim, filename : optional

    Returns
    -------
    None
        Shows the figure. Spectra with fewer than six energies are drawn with markers.
    """
    import matplotlib.pyplot as plt

    alphas = alphas or {}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axhline(0.0, color='#c3c2b7', lw=0.8)
    for level, (energy, values) in curves.items():
        label, color = LEVEL_STYLE.get(level, (level, INK2))
        if level in alphas and np.isfinite(alphas[level]):
            label += ', α = %.3f' % alphas[level]
        marker = 'o' if len(energy) < 6 else None
        ax.plot(energy, values, color=color, lw=1.5, marker=marker, ms=5, label=label)
    ax.set_xlabel(r'photon energy $\hbar\omega$ (eV)')
    ax.set_ylabel(y_label)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, title, filename)


def plot_molecular_beta(
    panels: list[tuple[str, NDArray[np.float64], dict[str, NDArray[np.float64]]]],
    title: str | None = None,
    x_lim: tuple[float, float] | None = None,
    filename: str | None = None,
) -> None:
    """Panels of molecular hyperpolarisabilities against the photon energy.

    Parameters
    ----------
    panels : list
        ``(y_label, energy, {legend label: values})`` per panel.
    title, x_lim, filename : optional

    Returns
    -------
    None
        Shows the figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(panels), 1, figsize=(6.4, 3.2 * len(panels)), squeeze=False)
    for ax, (y_label, energy, curves) in zip(axes[:, 0], panels):
        for i, (label, values) in enumerate(curves.items()):
            ax.plot(energy, values, color=SLOTS[i % len(SLOTS)], lw=1.5, label=label)
        ax.set_ylabel(y_label)
        ax.set_xlim(*(x_lim or (energy.min(), energy.max())))
        _style_axes(ax)
        if len(curves) > 1:
            ax.legend(frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel(r'photon energy $\hbar\omega$ (eV)')
    _finish(fig, title, filename)


__all__ = [
    'LEVEL_STYLE',
    'NONLINEAR_UNITS',
    'SHG_TERM_GROUPS',
    'SLOTS',
    'component_files',
    'plot_coupled_levels',
    'plot_molecular_beta',
    'plot_nonlinear_spectra',
    'plot_shg_terms',
]
