import numpy as np
from matplotlib import pyplot as plt


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
