from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any


def plot_dos(es, dos, title, x_lim, y_lim, vertical, col, x_label=None, y_label=None):
    """ """

    fig = plt.figure()

    tit = 'DoS' if title is None else title
    fig.suptitle(tit)

    ax = fig.add_subplot(111)

    if vertical:
        ax.plot(dos, es, color=col)
    else:
        ax.plot(es, dos, color=col)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    elif vertical:
        ax.set_xlim(0, ax.get_xlim()[1])
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    elif not vertical:
        ax.set_ylim(0, ax.get_ylim()[1])

    el = 'Energy (eV)' if x_label is None else x_label
    dl = 'electrons/eV' if y_label is None else y_label
    xl = el if not vertical else dl
    yl = dl if not vertical else el

    ax.set_xlabel(xl, fontsize=12)
    ax.set_ylabel(yl, fontsize=12)

    plt.show()


def plot_pdos(es, dos, title, x_lim, y_lim, vertical, cols, labels, legend):
    """ """
    import numpy as np

    if labels is None:
        labels = list(range(len(dos)))
    else:
        if len(labels) != len(dos):
            raise Exception('Must provide one label for each pdos file')

    if cols is None or isinstance(cols, str):
        cols = [cols] * len(dos)
    else:
        cols = np.array(cols)
        cs = cols.shape
        if len(cs) == 1:
            cols = [cols] * len(dos)
        elif cs[0] != len(labels):
            raise Exception('Must provide one color for each pdos file')

    fig = plt.figure()

    tit = 'PDoS' if title is None else title
    fig.suptitle(tit)

    ax = fig.add_subplot(111)

    if vertical:
        for i, d in enumerate(dos):
            ax.plot(d, es, color=cols[i], label=labels[i])
    else:
        for i, d in enumerate(dos):
            ax.plot(es, d, color=cols[i], label=labels[i])
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    elif vertical:
        ax.set_xlim(0, ax.get_xlim()[1])
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    elif not vertical:
        ax.set_ylim(0, ax.get_ylim()[1])

    el = 'Energy (eV)'
    dl = 'electrons/eV'
    xl = el if not vertical else dl
    yl = dl if not vertical else el

    ax.set_xlabel(xl, fontsize=12)
    ax.set_ylabel(yl, fontsize=12)

    if legend:
        ax.legend()

    plt.show()


def normalize_weights(w: np.ndarray) -> np.ndarray:
    if np.nanmin(w) >= 0 and np.nanmax(w) <= 1:
        return w.copy()
    lo, hi = np.nanpercentile(w, [1, 99])
    if hi - lo > 0:
        return np.clip((w - lo) / (hi - lo), 0, 1)
    return np.zeros_like(w)


def plot_weighted_bands(
    outputdir, bands, sym_points, title, cbar_label, label, filename, y_lim, col
):
    """ """

    hline_style = {
        'linestyle': '--',
        'linewidth': 1,
        'color': 'blue',
    }  # horizontal line style
    vline_style = {
        'linestyle': '-',
        'linewidth': 1,
        'color': 'gray',
    }  # horizontal line style

    w_norm = normalize_weights(bands['site_weight'].to_numpy())

    fig = plt.figure()

    tit = '' if title is None else title
    fig.suptitle(tit)

    ax = fig.add_subplot(111)

    sizes = 8 + 7 * w_norm  # marker size scaling
    sc = ax.scatter(
        bands['kindex'],
        bands['eigenvalue'],
        s=sizes,
        c=w_norm,
        cmap='jet',
        alpha=0.8,
        edgecolors='none',
    )

    if cbar_label is None:
        fig.colorbar(sc, ax=ax, label='Weight')
    else:
        fig.colorbar(sc, ax=ax, label=cbar_label)

    ax.hlines(0.0, sym_points[0][0], sym_points[0][len(sym_points[0]) - 1], **hline_style)

    if y_lim is None:
        y_lim = ax.get_ylim()
    ax.set_xlim(0, bands.shape[1])
    ax.set_ylim(*y_lim)
    if sym_points is None:
        ax.xaxis.set_visible(False)
    else:
        ax.set_xticks(sym_points[0])
        ax.set_xticklabels(sym_points[1])
        ax.vlines(sym_points[0], y_lim[0], y_lim[1], **vline_style)
    if label is None:
        label = r'$\epsilon$($\mathbf{k}$) (eV)'

    ax.set_ylabel(label, fontsize=12)

    if filename is not None:
        if outputdir is None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(outputdir + filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_bands(bands, sym_points, title, label, y_lim, col, labels=None, legend=True):
    """Plot one or more band structures for comparison.

    Arguments:
      bands: ndarray (nbands, nkpts) or list of such arrays.
      col: single color or list of colors, one per dataset.
      labels: optional list of legend labels, one per dataset.
      legend: show legend when labels are provided (default True).
    """
    import numpy as np

    # --- normalise to list-of-arrays ---
    if isinstance(bands, np.ndarray):
        bands_list = [bands]
    else:
        bands_list = list(bands)
    n_sets = len(bands_list)

    # --- normalise colours ---
    default_cols = [
        'black',
        'tab:red',
        'tab:blue',
        'tab:green',
        'tab:orange',
        'tab:purple',
        'tab:brown',
    ]
    if col is None:
        cols = [default_cols[i % len(default_cols)] for i in range(n_sets)]
    elif isinstance(col, (str, tuple)):
        if n_sets == 1:
            cols = [col]
        else:
            cols = [default_cols[i % len(default_cols)] for i in range(n_sets)]
            cols[0] = col  # keep user colour for first dataset
    else:
        cols = list(col)

    # --- normalise labels ---
    if labels is None:
        labels = [None] * n_sets

    fig = plt.figure()

    tit = 'Band Structure' if title is None else title
    fig.suptitle(tit)

    ax = fig.add_subplot(111)

    for idx, (bset, c, lbl) in enumerate(zip(bands_list, cols, labels)):
        for j, b in enumerate(bset):
            ax.plot(b, color=c, label=lbl if j == 0 else None)

    ref = bands_list[0]
    if y_lim is None:
        y_lim = ax.get_ylim()
    ax.set_xlim(0, ref.shape[1])
    ax.set_ylim(*y_lim)
    if sym_points is None:
        ax.xaxis.set_visible(False)
    else:
        ax.set_xticks(sym_points[0])
        ax.set_xticklabels(sym_points[1])
        ax.vlines(sym_points[0], y_lim[0], y_lim[1], color='gray')
    if label is None:
        label = r'$\epsilon$($\mathbf{k}$) (eV)'
    ax.set_ylabel(label, fontsize=12)

    if legend and any(l is not None for l in labels):
        ax.legend()

    plt.show()


def plot_dos_beside_bands(
    es, dos, bands, sym_points, title, band_label, x_lim, y_lim, col, dos_ticks
):
    """ """
    from matplotlib import gridspec

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[5, 1])

    tit = 'Band Structure and DoS' if title is None else title
    fig.suptitle(tit)

    ax_b = fig.add_subplot(spec[0])
    ax_d = fig.add_subplot(spec[1])

    for b in bands:
        ax_b.plot(b, color=col)
    if y_lim is None:
        y_lim = ax_b.get_ylim()
    ax_b.set_xlim(0, bands.shape[1] - 1)
    ax_b.set_ylim(*y_lim)
    if sym_points is None:
        ax_b.xaxis.set_visible(False)
    else:
        ax_b.set_xticks(sym_points[0])
        ax_b.set_xticklabels(sym_points[1])
        ax_b.vlines(sym_points[0], y_lim[0], y_lim[1], color='gray')
    if band_label is None:
        band_label = r'$\epsilon$($\mathbf{k}$) (eV)'
    ax_b.set_ylabel(band_label, fontsize=12)

    ax_d.plot(dos, es, color=col)
    if x_lim is not None:
        ax_d.set_xlim(*x_lim)
    else:
        ax_d.set_xlim(0, ax_d.get_xlim()[1])
    if y_lim is not None:
        ax_d.set_ylim(*y_lim)
    if not dos_ticks:
        ax_d.yaxis.set_visible(False)
        ax_d.xaxis.set_visible(False)
        plt.tight_layout()

    plt.show()


def plot_berry_under_bands(
    berry,
    bands,
    sym_points,
    title,
    band_label,
    berry_label,
    x_lim,
    y_lim,
    col,
    dos_ticks,
):
    """ """
    from matplotlib import gridspec

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])

    tit = 'Band Structure and Berry Phase' if title is None else title
    fig.suptitle(tit)

    ax_ba = fig.add_subplot(spec[0])
    ax_be = fig.add_subplot(spec[1])

    ax_be.plot(berry, color=col)
    for b in bands:
        ax_ba.plot(b, color=col)
    if y_lim is None:
        y_lim = ax_ba.get_ylim()
    ax_be.set_xlim(0, bands.shape[1] - 1)
    ax_ba.set_xlim(0, bands.shape[1] - 1)
    ax_ba.set_ylim(*y_lim)
    if sym_points is None:
        ax_be.xaxis.set_visible(False)
        ax_ba.xaxis.set_visible(False)
    else:
        tlim = ax_be.get_ylim()
        ax_be.set_ylim(*tlim)
        ax_be.set_xticks(sym_points[0])
        ax_be.set_xticklabels(sym_points[1])
        ax_be.vlines(sym_points[0], tlim[0], tlim[1], color='gray')
        ax_ba.set_xticks(sym_points[0])
        ax_ba.set_xticklabels(sym_points[1])
        ax_ba.vlines(sym_points[0], y_lim[0], y_lim[1], color='gray')

    if berry_label is None:
        berry_label = r'$\Omega$($\mathbf{k}$)'
    if band_label is None:
        band_label = r'$\epsilon$($\mathbf{k}$) (eV)'
    ax_be.set_ylabel(berry_label, fontsize=12)
    ax_ba.set_ylabel(band_label, fontsize=12)

    plt.show()
    quit()


def plot_tensor(
    enes, tensors, eles, title, x_lim, y_lim, x_lab, y_lab, col, legend, min_zero=False
):
    """ """
    import numpy as np

    fig = plt.figure()

    if title is None:
        raise ValueError("'title' cannot be None in plot_tensor")
    fig.suptitle(title)

    ax = fig.add_subplot(111)

    lmap = {0: 'x', 1: 'y', 2: 'z'}
    lkey = lambda a, b: lmap[a] + lmap[b]
    if len(eles) == 0:
        tval = np.empty(tensors.shape[0], dtype=float)
        for i, v in enumerate(tensors):
            tval[i] = np.sum([v[j, j] for j in range(3)]) / 3
        col = col if type(col) is str else col[0]
        ax.plot(enes, tval, color=col, label='Avg.')
    else:
        if type(col) is str:
            for e in eles:
                ax.plot(enes, tensors[:, e[0], e[1]], color=col, label=lkey(*e))
        elif len(col) >= len(eles):
            for i, e in enumerate(eles):
                ax.plot(enes, tensors[:, e[0], e[1]], color=col[i], label=lkey(*e))
        else:
            for e in eles:
                ax.plot(enes, tensors[:, e[0], e[1]], label=lkey(*e))

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    elif min_zero:
        ax.set_ylim(0, ax.get_ylim()[1])

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

    if legend:
        ax.legend()

    plt.show()


def plot_shc_tensor(
    enes, shc, title, x_lim, y_lim, x_lab, y_lab, cols, labels, legend, legend_outside=False
):
    """ """

    fig = plt.figure()

    if title is None:
        raise ValueError("'title' cannot be None in plot_tensor")
    fig.suptitle(title)

    ax = fig.add_subplot(111)

    if len(cols) >= len(shc):
        for i, s in enumerate(shc):
            ax.plot(enes, s, color=cols[i], label=labels[i])
    else:
        raise Exception('Dimensions of colors are incorrect. Blame GPAO.py')

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

    if legend:
        if legend_outside:
            # Shrink the axes and place the legend in a panel on the right
            # so it does not overlap the curves.
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            ax.legend(
                loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False
            )
        else:
            ax.legend()

    plt.show()


def plot_optical(curves, title, x_lim, y_lim, x_label, y_label, cols=None, legend=True):
    """Overlay an arbitrary selection of optical spectra on a single axis.

    This is the generic renderer behind the user-facing optical-property
    selection (dielectric function, refractive index, absorption,
    reflectivity, optical conductivity and emissivity). Each curve may carry
    its own abscissa, so spectra sampled on the photon-energy grid and the
    total-emissivity-versus-temperature curve can both be drawn through the
    same entry point.

    Arguments:
      curves (list): Sequence of ``(x, y, label)`` tuples, one per spectrum.
        ``x`` and ``y`` are 1D arrays of equal length; ``label`` is the legend
        text (may be ``None``).
      title (str): Figure title (defaults to ``'Optical properties'``).
      x_lim (tuple): ``(x_min, x_max)`` axis limits, or ``None``.
      y_lim (tuple): ``(y_min, y_max)`` axis limits, or ``None``.
      x_label (str): X-axis label (defaults to ``'Energy (eV)'``).
      y_label (str): Y-axis label (defaults to ``'Optical response'``).
      cols (str/tuple or list): A single color applied to every curve, or a
        list of colors (one per curve). ``None`` lets matplotlib cycle.
      legend (bool): Show the legend when any curve carries a label.
    """
    fig = plt.figure()
    fig.suptitle('Optical properties' if title is None else title)

    ax = fig.add_subplot(111)

    ncurves = len(curves)
    if cols is None or isinstance(cols, str) or isinstance(cols, tuple):
        cols = [cols] * ncurves
    elif len(cols) < ncurves:
        cols = list(cols) + [None] * (ncurves - len(cols))

    for i, (x, y, label) in enumerate(curves):
        ax.plot(x, y, color=cols[i], label=label)

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    ax.set_xlabel('Energy (eV)' if x_label is None else x_label, fontsize=12)
    ax.set_ylabel('Optical response' if y_label is None else y_label, fontsize=12)

    if legend and any(label is not None for _, _, label in curves):
        ax.legend()

    plt.show()


def plot_color_swatch(rgb01, hexstr=None, title=None, label=None):
    """Display a solid swatch of the perceived visible color of a material.

    Arguments:
      rgb01 (sequence): sRGB color components in [0, 1].
      hexstr (str): Optional hex string annotated on the swatch (e.g. '#rrggbb').
      title (str): Figure title (defaults to 'Perceived color').
      label (str): Optional text drawn above the hex value (e.g. the material).
    """
    rgb01 = tuple(float(c) for c in rgb01)

    fig = plt.figure(figsize=(3.0, 3.0))
    fig.suptitle('Perceived color' if title is None else title)
    ax = fig.add_subplot(111)
    ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=rgb01, edgecolor='black'))

    # Choose readable text color from the swatch luminance.
    luminance = 0.2126 * rgb01[0] + 0.7152 * rgb01[1] + 0.0722 * rgb01[2]
    text_col = 'black' if luminance > 0.5 else 'white'
    annotation = []
    if label is not None:
        annotation.append(label)
    if hexstr is not None:
        annotation.append(hexstr)
    if annotation:
        ax.text(
            0.5,
            0.5,
            '\n'.join(annotation),
            color=text_col,
            ha='center',
            va='center',
            fontsize=13,
            transform=ax.transAxes,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    plt.show()


def plot_phonons(
    distances,
    frequencies,
    ticks=None,
    dos=None,
    title=None,
    y_lim=None,
    col='black',
    units='THz',
    filename=None,
):
    """Plot a phonon dispersion, optionally with a side density-of-states panel.

    Arguments:
      distances (ndarray): 1D array of cumulative path distances.
      frequencies (ndarray): 2D array (nq, nbranch) of phonon frequencies.
      ticks (tuple): Optional (positions, labels) for high-symmetry points.
      dos (tuple): Optional (frequency, dos) arrays for a side DOS panel.
      title (str): Plot title.
      y_lim (tuple): Frequency axis limits (y_min, y_max).
      col (str or tuple): Line colour.
      units (str): Frequency unit string for the axis label.
      filename (str): If given, save the figure to this path.
    """
    from matplotlib import gridspec

    distances = np.asarray(distances)
    frequencies = np.asarray(frequencies)

    fig = plt.figure()
    tit = 'Phonon Dispersion' if title is None else title
    fig.suptitle(tit)

    if dos is not None:
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[5, 1])
        ax_b = fig.add_subplot(spec[0])
        ax_d = fig.add_subplot(spec[1])
    else:
        ax_b = fig.add_subplot(111)
        ax_d = None

    for branch in frequencies.T:
        ax_b.plot(distances, branch, color=col)

    ax_b.axhline(0.0, color='gray', linewidth=0.8, linestyle='--')

    if y_lim is None:
        y_lim = ax_b.get_ylim()
    ax_b.set_xlim(distances[0], distances[-1])
    ax_b.set_ylim(*y_lim)

    if ticks is not None:
        positions, labels = ticks
        ax_b.set_xticks(positions)
        ax_b.set_xticklabels(labels)
        ax_b.vlines(positions, y_lim[0], y_lim[1], color='gray')
    else:
        ax_b.set_xlabel('Wave vector', fontsize=12)

    ax_b.set_ylabel('Frequency (%s)' % units, fontsize=12)

    if ax_d is not None:
        dos_freq, dos_val = dos
        ax_d.plot(dos_val, dos_freq, color=col)
        ax_d.set_ylim(*y_lim)
        ax_d.set_xlim(0, ax_d.get_xlim()[1])
        ax_d.yaxis.set_visible(False)
        ax_d.set_xlabel('DOS', fontsize=12)
        plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def plot_gruneisen_band(
    distances,
    gruneisen,
    ticks=None,
    title=None,
    y_lim=None,
    col='black',
    filename=None,
):
    """Plot the mode Grueneisen parameters along a q-path (dispersion style).

    Arguments:
      distances (ndarray): 1D array of cumulative path distances.
      gruneisen (ndarray): 2D array (nq, nbranch) of mode Grueneisen parameters.
      ticks (tuple): Optional (positions, labels) for high-symmetry points.
      title (str): Plot title.
      y_lim (tuple): Grueneisen-axis limits (y_min, y_max).
      col (str or tuple): Line colour.
      filename (str): If given, save the figure to this path.
    """
    distances = np.asarray(distances)
    gruneisen = np.asarray(gruneisen)

    fig = plt.figure()
    tit = 'Mode Gr\u00fcneisen parameters' if title is None else title
    fig.suptitle(tit)
    ax = fig.add_subplot(111)

    for branch in gruneisen.T:
        ax.plot(distances, branch, color=col)

    ax.axhline(0.0, color='gray', linewidth=0.8, linestyle='--')

    if y_lim is None:
        y_lim = ax.get_ylim()
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylim(*y_lim)

    if ticks is not None:
        positions, labels = ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.vlines(positions, y_lim[0], y_lim[1], color='gray')
    else:
        ax.set_xlabel('Wave vector', fontsize=12)

    ax.set_ylabel('Gr\u00fcneisen parameter ' + r'$\gamma_{q\nu}$', fontsize=12)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def plot_phonon_thermal(
    temperatures,
    free_energy,
    entropy,
    heat_capacity,
    title=None,
    filename=None,
):
    """Plot the harmonic thermal properties as a function of temperature.

    Arguments:
      temperatures (ndarray): 1D array of temperatures (K).
      free_energy (ndarray): Helmholtz free energy (kJ/mol).
      entropy (ndarray): Entropy (J/K/mol).
      heat_capacity (ndarray): Constant-volume heat capacity (J/K/mol).
      title (str): Plot title.
      filename (str): If given, save the figure to this path.
    """
    temperatures = np.asarray(temperatures)

    fig, ax = plt.subplots()
    tit = 'Thermal Properties' if title is None else title
    fig.suptitle(tit)

    ax.plot(temperatures, free_energy, color='tab:blue', label='Free energy (kJ/mol)')
    ax.plot(temperatures, entropy, color='tab:orange', label='Entropy (J/K/mol)')
    ax.plot(temperatures, heat_capacity, color='tab:green', label=r'$C_v$ (J/K/mol)')

    ax.set_xlim(temperatures[0], temperatures[-1])
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Thermal properties', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def plot_qha(
    temperatures,
    volume=None,
    thermal_expansion=None,
    bulk_modulus=None,
    heat_capacity=None,
    gruneisen=None,
    ev=None,
    title=None,
    filename=None,
):
    """Plot the quasi-harmonic quantities as a function of temperature.

    Each supplied quantity is drawn in its own panel; ``None`` panels are
    skipped.  Pass ``ev=(volumes, energies)`` to include the static E-V curve.

    Arguments:
      temperatures (ndarray): 1D array of temperatures (K).
      volume (ndarray): Equilibrium volume V(T) (Angstrom^3).
      thermal_expansion (ndarray): Volumetric thermal expansion alpha(T) (1/K).
      bulk_modulus (ndarray): Isothermal bulk modulus B(T) (GPa).
      heat_capacity (ndarray): Constant-pressure heat capacity Cp(T) (J/K/mol).
      gruneisen (ndarray): Thermodynamic Gruneisen parameter gamma(T).
      ev (tuple): Optional (volumes, energies) static E-V data (Angstrom^3, eV).
      title (str): Overall figure title.
      filename (str): If given, save the figure to this path.
    """
    temperatures = np.asarray(temperatures)

    panels = []
    if ev is not None:
        panels.append(('E-V', ev, 'Volume (Ang$^3$)', 'Energy (eV)'))
    if volume is not None:
        panels.append(('V(T)', volume, 'Temperature (K)', 'Volume (Ang$^3$)'))
    if thermal_expansion is not None:
        panels.append(('alpha(T)', thermal_expansion, 'Temperature (K)', r'$\alpha$ (K$^{-1}$)'))
    if bulk_modulus is not None:
        panels.append(('B(T)', bulk_modulus, 'Temperature (K)', 'Bulk modulus (GPa)'))
    if heat_capacity is not None:
        panels.append(('Cp(T)', heat_capacity, 'Temperature (K)', r'$C_p$ (J/K/mol)'))
    if gruneisen is not None:
        panels.append(('gamma(T)', gruneisen, 'Temperature (K)', r'$\gamma$'))

    n = len(panels)
    if n == 0:
        raise ValueError('plot_qha requires at least one quantity to plot.')

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    fig.suptitle('Quasi-Harmonic Approximation' if title is None else title)
    axes = np.atleast_1d(axes).ravel()

    for ax, (tag, ydata, xlabel, ylabel) in zip(axes, panels):
        ydata = np.asarray(ydata)
        if tag == 'E-V':
            xdata = np.asarray(ev[0])
            ydata = np.asarray(ev[1])
            ax.plot(xdata, ydata, 'o-', color='tab:purple')
        else:
            ax.plot(temperatures, ydata, color='tab:blue')
            ax.set_xlim(temperatures[0], temperatures[-1])
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def plot_ir_spectrum(
    frequencies,
    intensities,
    modes=None,
    title=None,
    x_lim=None,
    col='black',
    units='cm-1',
    filename=None,
):
    """Plot a broadened infrared spectrum, optionally with the mode sticks.

    Arguments:
      frequencies (ndarray): 1D array of frequencies for the broadened curve.
      intensities (ndarray): 1D array of broadened intensities.
      modes (tuple): Optional (mode_freq, mode_intensity) arrays drawn as
        vertical sticks at the discrete mode positions.
      title (str): Plot title.
      x_lim (tuple): Frequency axis limits (x_min, x_max).
      col (str or tuple): Line colour.
      units (str): Frequency unit string for the axis label.
      filename (str): If given, save the figure to this path.
    """
    frequencies = np.asarray(frequencies)
    intensities = np.asarray(intensities)

    fig, ax = plt.subplots()
    tit = 'Infrared Spectrum' if title is None else title
    fig.suptitle(tit)

    ax.plot(frequencies, intensities, color=col)

    if modes is not None:
        mode_freq, mode_int = np.asarray(modes[0]), np.asarray(modes[1])
        ax.vlines(mode_freq, 0.0, mode_int, color='tab:red', linewidth=1.0)

    if x_lim is None:
        x_lim = (frequencies[0], frequencies[-1])
    ax.set_xlim(*x_lim)
    ax.set_ylim(0.0, ax.get_ylim()[1])

    ax.set_xlabel('Frequency (%s)' % units, fontsize=12)
    ax.set_ylabel('IR intensity (arb. units)', fontsize=12)
    ax.grid(alpha=0.3)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def plot_raman_spectrum(
    frequencies,
    intensities,
    modes=None,
    title=None,
    x_lim=None,
    col='black',
    units='cm-1',
    filename=None,
):
    """Plot a broadened Raman spectrum, optionally with the mode sticks.

    Arguments:
      frequencies (ndarray): 1D array of frequencies for the broadened curve.
      intensities (ndarray): 1D array of broadened intensities.
      modes (tuple): Optional (mode_freq, mode_intensity) arrays drawn as
        vertical sticks at the discrete mode positions.
      title (str): Plot title.
      x_lim (tuple): Frequency axis limits (x_min, x_max).
      col (str or tuple): Line colour.
      units (str): Frequency unit string for the axis label.
      filename (str): If given, save the figure to this path.
    """
    frequencies = np.asarray(frequencies)
    intensities = np.asarray(intensities)

    fig, ax = plt.subplots()
    tit = 'Raman Spectrum' if title is None else title
    fig.suptitle(tit)

    ax.plot(frequencies, intensities, color=col)

    if modes is not None:
        mode_freq, mode_int = np.asarray(modes[0]), np.asarray(modes[1])
        ax.vlines(mode_freq, 0.0, mode_int, color='tab:blue', linewidth=1.0)

    if x_lim is None:
        x_lim = (frequencies[0], frequencies[-1])
    ax.set_xlim(*x_lim)
    ax.set_ylim(0.0, ax.get_ylim()[1])

    ax.set_xlabel('Frequency (%s)' % units, fontsize=12)
    ax.set_ylabel('Raman intensity (arb. units)', fontsize=12)
    ax.grid(alpha=0.3)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


# Ordinal blue ramp (light -> dark with increasing temperature) and the three
# categorical slots used for the gap estimates.
_ME_T_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
_ME_SERIES = ['#2a78d6', '#eb6834', '#1baf7a']


def plot_migdal_eliashberg(
    data: Mapping[str, Any],
    temps: Sequence[float] | None = None,
    title: str | None = None,
    filename: str | None = None,
    real_axis_max_mev: float = 60.0,
) -> None:
    """Plot the isotropic Migdal-Eliashberg results versus temperature.

    Parameters
    ----------
    data : mapping
        Contents of ``migdal_eliashberg.npz`` written by
        :func:`PAOFLOW.elphon.migdal_eliashberg.write_me_outputs`.
    temps : sequence of float, optional
        Temperatures (K) drawn in the frequency-resolved panels; defaults to up
        to four temperatures with a non-zero gap.
    title : str, optional
        Overall figure title.
    filename : str, optional
        If given, the figure is also saved to this path.
    real_axis_max_mev : float, optional
        Upper frequency (meV) of the real-axis ``Delta(w)`` panels (default 60),
        capped at the end of the real-axis grid (``wscut``).

    Returns
    -------
    None
        Shows the figure (and writes ``filename``).

    Notes
    -----
    Panels: ``Delta(i w_n)`` and ``Z(i w_n)``; ``Re`` / ``Im Delta(w)`` on the
    real axis (analytic continuation solid, Pade dashed); the quasiparticle
    DOS; ``Delta(T)`` from the lowest Matsubara frequency and the two real-axis
    gap edges; and the largest eigenvalue of the linearised kernel with
    ``Tc``.  Temperatures use one blue ramp (light to dark with increasing T);
    the three gap estimates use distinct colours and markers.
    """
    from matplotlib.lines import Line2D

    from ..elphon.migdal_eliashberg import temperature_tag

    temperatures = np.asarray(data['temps'])
    gap0_mev = np.asarray(data['gap0_imag']) * 1e3
    if temps is None:
        gapped = temperatures[gap0_mev > 0.0]
        picks = np.linspace(0, gapped.size - 1, min(4, gapped.size)).round().astype(int)
        temps = gapped[np.unique(picks)] if gapped.size else []
    temps = [t for t in temps if 'imag_' + temperature_tag(t) in data]
    ramp_last = len(_ME_T_RAMP) - 1
    colours = [
        _ME_T_RAMP[int(round(i * ramp_last / max(len(temps) - 1, 1)))] for i in range(len(temps))
    ]
    w_mev = np.asarray(data['w_real']) * 1e3
    edges_mev = np.concatenate([np.asarray(data['gap_acon']), np.asarray(data['gap_pade'])]) * 1e3
    if np.any(np.isfinite(edges_mev) & (edges_mev > 0)):
        edge_mev = np.nanmax(edges_mev)
    else:
        edge_mev = gap0_mev.max()
    w_max = min(w_mev[-1], real_axis_max_mev)

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    fig.suptitle('Isotropic Migdal-Eliashberg' if title is None else title)
    ax_gap_n, ax_z_n, ax_re, ax_im, ax_qdos, ax_gap_t, ax_rho, ax_legend = axes.ravel()

    for t, colour in zip(temps, colours):
        tag = temperature_tag(t)
        wn, Zn, deltan = np.asarray(data['imag_' + tag]).T
        ax_gap_n.plot(wn * 1e3, deltan * 1e3, color=colour, lw=1.5)
        ax_z_n.plot(wn * 1e3, Zn, color=colour, lw=1.5)
        for key, style in (('acon', '-'), ('pade', '--')):
            if key + '_' + tag in data:
                re_delta, im_delta = np.asarray(data[key + '_' + tag])[:, 2:4].T * 1e3
                ax_re.plot(w_mev, re_delta, color=colour, ls=style, lw=1.5)
                ax_im.plot(w_mev, im_delta, color=colour, ls=style, lw=1.5)
        if 'qdos_' + tag in data:
            ax_qdos.plot(w_mev, data['qdos_' + tag], color=colour, lw=1.5)
    ax_gap_n.set(
        xlabel=r'$\omega_n$ (meV)', ylabel=r'$\Delta(i\omega_n)$ (meV)', title='Imaginary axis: gap'
    )
    ax_z_n.set(
        xlabel=r'$\omega_n$ (meV)',
        ylabel=r'$Z(i\omega_n)$',
        title='Imaginary axis: renormalisation',
    )
    for ax, label in ((ax_re, r'Re $\Delta(\omega)$ (meV)'), (ax_im, r'Im $\Delta(\omega)$ (meV)')):
        ax.set(
            xlabel=r'$\omega$ (meV)',
            ylabel=label,
            xlim=(0.0, w_max),
            title='Real axis: ' + label.split(' (')[0],
        )
        visible = [line.get_ydata()[w_mev <= w_max] for line in ax.get_lines()]
        if visible:  # scale y to the visible frequency window
            low = min(0.0, min(y.min() for y in visible))
            high = max(y.max() for y in visible)
            ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
        ax.axhline(0.0, color='0.6', lw=0.8)
    ax_qdos.set(
        xlabel=r'$\omega$ (meV)',
        ylabel=r'$N_S(\omega)/N_F$',
        xlim=(0.0, 4.0 * edge_mev if edge_mev > 0 else w_max),
        title='Quasiparticle DOS',
    )
    ax_qdos.set_ylim(0.0, min(ax_qdos.get_ylim()[1], 8.0))

    estimates = (
        ('gap0_imag', r'$\Delta(i\omega_0)$', 'o'),
        ('gap_pade', 'gap edge, Pade', 's'),
        ('gap_acon', 'gap edge, analytic cont.', '^'),
    )
    for (key, label, marker), colour in zip(estimates, _ME_SERIES):
        gap_mev = np.asarray(data[key]) * 1e3
        finite = np.isfinite(gap_mev)
        ax_gap_t.plot(
            temperatures[finite], gap_mev[finite], color=colour, marker=marker, ms=5, lw=1.2,
            label=label,
        )  # fmt: skip
    tc_gap = float(data['Tc_gap']) if 'Tc_gap' in data else float('nan')
    gap_title = r'Gap vs $T$'
    if np.isfinite(tc_gap):
        gap_title += r'  ($T_c \approx %.2f$ K)' % tc_gap
    ax_gap_t.set(xlabel='Temperature (K)', ylabel=r'$\Delta$ (meV)', title=gap_title)
    ax_gap_t.set_ylim(bottom=0.0)
    ax_gap_t.legend(frameon=False, fontsize=9)

    if 'max_eigenvalue' in data:
        linear_temps = np.asarray(data['lin_temps'])
        rho = np.asarray(data['max_eigenvalue'])
        ax_rho.plot(linear_temps, rho, color=_ME_SERIES[0], marker='o', ms=5, lw=1.2)
        ax_rho.axhline(1.0, color='0.5', ls='--', lw=1.0)
        tc_linear = float(data['Tc_linear'])
        if np.isfinite(tc_linear):
            ax_rho.axvline(tc_linear, color='0.5', ls=':', lw=1.0)
            ax_rho.annotate(
                r'$T_c = %.2f$ K' % tc_linear,
                (tc_linear, 1.0),
                xytext=(6, 8),
                textcoords='offset points',
            )
        ax_rho.set(
            xlabel='Temperature (K)',
            ylabel=r'max eigenvalue $\rho$',
            title='Linearised Migdal-Eliashberg kernel',
        )
    else:
        ax_rho.set_visible(False)

    for ax in axes.ravel()[:-1]:
        ax.grid(alpha=0.3)
    ax_legend.axis('off')
    handles = [Line2D([], [], color=colour, lw=2) for colour in colours]
    handles += [Line2D([], [], color='0.3', ls='-'), Line2D([], [], color='0.3', ls='--')]
    ax_legend.legend(
        handles,
        ['T = %g K' % t for t in temps] + ['analytic continuation', 'Pade'],
        loc='center',
        frameon=False,
        title='Frequency panels',
    )

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_migdal_eliashberg_aniso(
    data: Mapping[str, Any],
    temps: Sequence[float] | None = None,
    iso_data: Mapping[str, Any] | None = None,
    title: str | None = None,
    filename: str | None = None,
    distribution_scale: float = 3.0e-3,
) -> None:
    """Plot the anisotropic Migdal-Eliashberg results versus temperature.

    Parameters
    ----------
    data : mapping
        Contents of ``migdal_eliashberg_aniso.npz`` written by
        :func:`PAOFLOW.elphon.anisotropic_eliashberg.write_me_aniso_outputs`.
    temps : sequence of float, optional
        Temperatures (K) drawn in the distribution and quasiparticle-DOS
        panels; defaults to up to four temperatures with a non-zero gap.
    iso_data : mapping, optional
        Contents of the isotropic ``migdal_eliashberg.npz``
        (:func:`PAOFLOW.elphon.migdal_eliashberg.write_me_outputs`); its
        ``Delta(i w_0)`` is drawn for comparison.
    title : str, optional
        Overall figure title.
    filename : str, optional
        If given, the figure is also saved to this path.
    distribution_scale : float, optional
        Horizontal scale (K per 1/eV^2) of the gap distributions in the
        ``Delta(T)`` panel; the default 3e-3 is that of EPW tutorial 04.

    Returns
    -------
    None
        Shows the figure (and writes ``filename``).

    Notes
    -----
    Panels: the distribution of ``Delta_nk(i w_0)`` at every temperature,
    drawn as in EPW tutorial 04 (``plot_gap0.gnu``): EPW's
    ``gap_distribution_FS``
    (:func:`~PAOFLOW.elphon.anisotropic_eliashberg.epw_gap_distribution`),
    filled from ``T`` to ``T + distribution_scale * rho(Delta)``, with the
    isotropic gap when given; the gap distributions and the quasiparticle DOS at the
    chosen temperatures; the distribution of ``lambda_nk``; ``Delta_nk`` versus
    ``lambda_nk`` at the lowest temperature; and the largest eigenvalue of the
    linearised kernel.
    """
    temperatures = np.asarray(data['temps'])
    gap0_mev = np.asarray(data['gap0']) * 1e3  # (ntemps, nstates)
    lambda_nk = np.asarray(data['lambda_nk'])
    gapped = temperatures[gap0_mev.max(axis=1) > 0.0]
    if temps is None:
        picks = np.linspace(0, gapped.size - 1, min(4, gapped.size)).round().astype(int)
        temps = gapped[np.unique(picks)] if gapped.size else []
    picked = [int(np.argmin(np.abs(temperatures - t))) for t in temps]
    ramp_last = len(_ME_T_RAMP) - 1
    colours = [
        _ME_T_RAMP[int(round(i * ramp_last / max(len(picked) - 1, 1)))] for i in range(len(picked))
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle('Anisotropic Migdal-Eliashberg' if title is None else title)
    ax_gap_t, ax_dist, ax_qdos, ax_lambda, ax_corr, ax_rho = axes.ravel()

    from ..elphon.anisotropic_eliashberg import epw_gap_distribution

    weights = np.asarray(data['weight'])
    for it, T in enumerate(temperatures):
        distribution = epw_gap_distribution(gap0_mev[it] * 1e-3, weights)
        if distribution is None:
            continue
        grid, _, raw = distribution
        width = T + distribution_scale * raw
        ax_gap_t.fill_betweenx(grid * 1e3, T, width, color=_ME_SERIES[0], alpha=0.2, lw=0)
        ax_gap_t.plot(width, grid * 1e3, color=_ME_SERIES[0], lw=1.0)
    if iso_data is not None:
        ax_gap_t.plot(
            np.asarray(iso_data['temps']), np.asarray(iso_data['gap0_imag']) * 1e3,
            color=_ME_SERIES[2], ls='--', lw=2, label='isotropic',
        )  # fmt: skip
        ax_gap_t.legend(frameon=False, fontsize=9, loc='lower left')
    mu_star = float(data['mu_star']) if 'mu_star' in data else float('nan')
    ax_gap_t.text(
        0.97, 0.95, r'FSR ($\mu^*_c = %.2g$)' % mu_star, transform=ax_gap_t.transAxes,
        ha='right', va='top',
    )  # fmt: skip
    t_step = np.min(np.diff(temperatures)) if temperatures.size > 1 else 5.0
    ax_gap_t.set_xlim(0.0, temperatures.max() + 1.5 * t_step)
    gap_title = r'$\Delta_{n\mathbf{k}}(i\omega_0)$ vs $T$'
    tc_gap = float(data['Tc_gap']) if 'Tc_gap' in data else float('nan')
    if np.isfinite(tc_gap):
        gap_title += r'  ($T_c \approx %.1f$ K)' % tc_gap
    ax_gap_t.set(xlabel='Temperature (K)', ylabel=r'$\Delta_{n\mathbf{k}}$ (meV)', title=gap_title)
    ax_gap_t.set_ylim(bottom=0.0)

    gap_grid = np.asarray(data['gap_grid_mev'])
    w_mev = np.asarray(data['w_real']) * 1e3
    qdos = np.asarray(data['qdos'])
    for it, colour in zip(picked, colours):
        label = 'T = %g K' % temperatures[it]
        ax_dist.plot(gap_grid, data['gap0_distribution'][it], color=colour, lw=1.5, label=label)
        if np.all(np.isfinite(qdos[it])):
            ax_qdos.plot(w_mev, qdos[it], color=colour, lw=1.5, label=label)
    ax_dist.set(
        xlabel=r'$\Delta_{n\mathbf{k}}(i\omega_0)$ (meV)', ylabel=r'$\rho(\Delta)$ (1/meV)',
        title='Gap distribution',
    )  # fmt: skip
    ax_dist.set_ylim(bottom=0.0)
    ax_dist.legend(frameon=False, fontsize=9)
    gap_top = float(np.nanmax(gap0_mev)) if gap0_mev.size else 1.0
    ax_qdos.set(
        xlabel=r'$\omega$ (meV)', ylabel=r'$N_S(\omega)/N_F$', title='Quasiparticle DOS (Pade)',
        xlim=(0.0, min(w_mev[-1], 2.5 * gap_top)),
    )  # fmt: skip
    ax_qdos.set_ylim(0.0, min(ax_qdos.get_ylim()[1], 6.0))
    ax_qdos.legend(frameon=False, fontsize=9)

    ax_lambda.plot(data['lambda_grid'], data['lambda_distribution'], color=_ME_SERIES[0], lw=1.5)
    ax_lambda.axvline(float(data['lambda']), color='0.5', ls=':', lw=1.0)
    ax_lambda.annotate(
        r'$\lambda = %.3f$' % float(data['lambda']), (float(data['lambda']), 0.0),
        xytext=(6, 12), textcoords='offset points',
    )  # fmt: skip
    ax_lambda.set(
        xlabel=r'$\lambda_{n\mathbf{k}}$', ylabel=r'$\rho(\lambda_{n\mathbf{k}})$',
        title='Coupling distribution',
    )  # fmt: skip
    ax_lambda.set_ylim(bottom=0.0)

    if gap0_mev.size:
        ax_corr.scatter(lambda_nk, gap0_mev[0], color=_ME_SERIES[0], s=10, edgecolors='none')
    ax_corr.set(
        xlabel=r'$\lambda_{n\mathbf{k}}$', ylabel=r'$\Delta_{n\mathbf{k}}(i\omega_0)$ (meV)',
        title=r'Gap vs coupling at $T = %g$ K' % temperatures[0],
    )  # fmt: skip

    if 'max_eigenvalue' in data:
        linear_temps = np.asarray(data['lin_temps'])
        rho = np.asarray(data['max_eigenvalue'])
        ax_rho.plot(linear_temps, rho, color=_ME_SERIES[0], marker='o', ms=5, lw=1.2)
        ax_rho.axhline(1.0, color='0.5', ls='--', lw=1.0)
        tc_linear = float(data['Tc_linear'])
        if np.isfinite(tc_linear):
            ax_rho.axvline(tc_linear, color='0.5', ls=':', lw=1.0)
            ax_rho.annotate(
                r'$T_c = %.2f$ K' % tc_linear, (tc_linear, 1.0), xytext=(6, 8),
                textcoords='offset points',
            )  # fmt: skip
        ax_rho.set(
            xlabel='Temperature (K)', ylabel=r'max eigenvalue $\rho$',
            title='Linearised anisotropic kernel',
        )  # fmt: skip
    else:
        ax_rho.set_visible(False)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_phonon_assisted_absorption(
    data: Mapping[str, Any],
    eta_ev: float = 0.05,
    temperature: float | None = None,
    epw_indabs_file: str | None = None,
    title: str | None = None,
    filename: str | None = None,
) -> None:
    """Plot the phonon-assisted and direct absorption versus photon energy.

    Parameters
    ----------
    data : mapping
        Contents of ``absorption.npz`` written by
        :func:`PAOFLOW.elphon.phonon_assisted_absorption.write_absorption_outputs`.
    eta_ev : float, optional
        Intermediate-state broadening (eV) to show; the closest computed one is
        used (default 0.05).
    temperature : float, optional
        Temperature (K) of the left panel; defaults to the first one.
    epw_indabs_file : str, optional
        EPW ``epsilon2_indabs_<T>K.dat`` overlaid (dashed) on the phonon-assisted
        curve of the left panel, at the same broadening column.
    title : str, optional
        Overall figure title.
    filename : str, optional
        If given, the figure is also saved to this path.

    Returns
    -------
    None
        Shows the figure (and writes ``filename``).

    Notes
    -----
    Left: polarisation-averaged ``Im eps`` (direct, phonon-assisted and total,
    Gaussian delta) on a log scale.  Right: the absorption coefficient (cm^-1)
    for every temperature (one blue ramp, light to dark with increasing T).
    The indirect and direct gaps of the dense grid are marked.
    """
    omega = np.asarray(data['omega_ev'])
    etas = np.asarray(data['etas_ev'])
    temps = np.atleast_1d(np.asarray(data['temps_k']))
    ieta = int(np.argmin(np.abs(etas - eta_ev)))
    itemp = 0 if temperature is None else int(np.argmin(np.abs(temps - temperature)))
    indirect = np.asarray(data['eps2_indirect']).mean(axis=2)  # (nT, neta, nw)
    direct = np.asarray(data['eps2_direct']).mean(axis=1)  # (nT, nw)
    alpha = np.asarray(data['alpha_cm'])  # (nT, neta, nw)

    fig, (ax_eps, ax_alpha) = plt.subplots(1, 2, figsize=(10, 4))
    floor = 1.0e-6
    curves = [
        (direct[itemp], 'direct', _ME_SERIES[0]),
        (indirect[itemp, ieta], 'phonon-assisted', _ME_SERIES[1]),
        (direct[itemp] + indirect[itemp, ieta], 'total', _ME_SERIES[2]),
    ]
    for values, label, color in curves:
        ax_eps.plot(omega, np.clip(values, floor, None), color=color, lw=2, label=label)
    if epw_indabs_file is not None:
        epw = np.loadtxt(epw_indabs_file)
        column = min(ieta + 1, epw.shape[1] - 1)
        ax_eps.plot(
            epw[:, 0], np.clip(epw[:, column], floor, None), color=_ME_SERIES[1], lw=1.5,
            ls='--', label='phonon-assisted (EPW)',
        )  # fmt: skip
    ax_eps.set_yscale('log')
    ax_eps.set_ylim(bottom=max(floor, 1.0e-5 * np.max(indirect[itemp, ieta])))
    ax_eps.set_xlabel('Photon energy (eV)')
    ax_eps.set_ylabel(r'Im $\varepsilon(\omega)$')
    ax_eps.set_title(r'T = %g K, $\eta$ = %g eV' % (temps[itemp], etas[ieta]), fontsize=10)

    ramp_last = len(_ME_T_RAMP) - 1
    colors = [
        _ME_T_RAMP[int(round(i * ramp_last / max(len(temps) - 1, 1)))] for i in range(len(temps))
    ]
    for it, temp in enumerate(temps):
        ax_alpha.plot(
            omega, np.clip(alpha[it, ieta], floor, None), color=colors[it], lw=2,
            label='%g K' % temp,
        )  # fmt: skip
    ax_alpha.set_yscale('log')
    ax_alpha.set_ylim(bottom=max(1.0e-1, 1.0e-6 * np.max(alpha[:, ieta])))
    ax_alpha.set_xlabel('Photon energy (eV)')
    ax_alpha.set_ylabel(r'$\alpha$ (cm$^{-1}$)')
    ax_alpha.set_title(
        r'direct + phonon-assisted, $n_r$ = %s' % np.round(np.mean(data['refractive_index']), 2)
        if 'refractive_index' in data
        else 'direct + phonon-assisted',
        fontsize=10,
    )

    for ax in (ax_eps, ax_alpha):
        for key, label in (('indirect_gap_ev', r'$E_g^{ind}$'), ('direct_gap_ev', r'$E_g^{dir}$')):
            if key in data and omega[0] <= float(data[key]) <= omega[-1]:
                ax.axvline(float(data[key]), color='#52514e', lw=1, ls=':')
                ax.text(
                    float(data[key]), 1.0, ' ' + label, transform=ax.get_xaxis_transform(),
                    va='top', ha='left', color='#52514e', fontsize=9,
                )  # fmt: skip
        ax.set_xlim(omega[0], omega[-1])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9, loc='lower right')
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_thermal_emissivity(
    data: Mapping[str, Any],
    temps: Sequence[float] | None = None,
    title: str | None = None,
    filename: str | None = None,
) -> None:
    """Plot the spectral and total hemispherical emissivity versus temperature.

    Parameters
    ----------
    data : mapping
        Contents of ``emissivity.npz`` written by
        :func:`PAOFLOW.elphon.phonon_assisted_absorption.write_emissivity_outputs`.
    temps : sequence of float, optional
        Temperatures (K) of the spectral panel; defaults to up to four evenly
        spaced ones.
    title : str, optional
        Overall figure title.
    filename : str, optional
        If given, the figure is also saved to this path.

    Returns
    -------
    None
        Shows the figure (and writes ``filename``).

    Notes
    -----
    Left: spectral hemispherical emissivity of the slab (solid) and of the
    opaque half-space (dashed) at each temperature (one blue ramp, light to dark
    with increasing T).  Right: the Planck-weighted total emissivity of the slab
    and of the half-space versus T; the lowest Planck coverage of the
    photon-energy grid is noted.
    """
    omega = np.asarray(data['omega_ev'])
    all_temps = np.atleast_1d(np.asarray(data['temps_k']))
    if temps is None:
        picks = np.unique(np.round(np.linspace(0, all_temps.size - 1, min(4, all_temps.size))))
        index = picks.astype(int)
    else:
        index = np.array([int(np.argmin(np.abs(all_temps - t))) for t in temps])
    thickness_um = float(data['thickness_m']) * 1.0e6

    from matplotlib.colors import LinearSegmentedColormap

    fig, (ax_w, ax_t) = plt.subplots(1, 2, figsize=(10, 4))
    t_ramp = LinearSegmentedColormap.from_list('temperature', _ME_T_RAMP)
    for rank_t, it in enumerate(index):
        color = t_ramp(rank_t / max(len(index) - 1, 1))
        ax_w.plot(
            omega, data['hemispherical_slab'][it], color=color, lw=2,
            label='%g K' % all_temps[it],
        )  # fmt: skip
        ax_w.plot(omega, data['hemispherical_opaque'][it], color=color, lw=1, ls='--')
    ax_w.plot([], [], color='#52514e', lw=1, ls='--', label='opaque half-space')
    ax_w.set_xscale('log')
    ax_w.set_xlim(omega[0], omega[-1])
    ax_w.set_ylim(0.0, 1.0)
    ax_w.set_xlabel('Photon energy (eV)')
    ax_w.set_ylabel(r'Hemispherical emissivity $\varepsilon(\omega, T)$')
    ax_w.set_title('slab, d = %g µm' % thickness_um, fontsize=10)

    ax_t.plot(
        all_temps, data['total_slab'], color=_ME_SERIES[0], lw=2, marker='o', ms=8,
        label='slab, d = %g µm' % thickness_um,
    )  # fmt: skip
    ax_t.plot(
        all_temps, data['total_opaque'], color=_ME_SERIES[1], lw=2, marker='s', ms=8,
        ls='--', label='opaque half-space',
    )  # fmt: skip
    ax_t.set_ylim(0.0, 1.0)
    ax_t.set_xlabel('Temperature (K)')
    ax_t.set_ylabel(r'Total hemispherical emissivity $\varepsilon(T)$')
    ax_t.set_title(
        'Planck coverage of the grid ≥ %.3f' % float(np.min(data['planck_coverage'])), fontsize=10
    )
    ax_w.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax_t.legend(frameon=False, fontsize=9, loc='lower right')
    for ax in (ax_w, ax_t):
        ax.grid(alpha=0.3)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
