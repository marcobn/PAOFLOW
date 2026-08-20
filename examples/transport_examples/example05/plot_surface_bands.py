"""Plot the Bi2Se3(0001) surface spectral function written by ``main.py``.

Reads the three files produced by ``Transport.compute_surface_bands()``:

    surfband_surf.dat        (ne x nk) spectral map A(k, E)
    surfband_egrid_surf.dat  energy axis, one value per row
    surfband_kpath_surf.dat  k axis: index, distance, high-symmetry label

and renders the heatmap in the style of the WannierTools Bi2Se3 figure. Note
that ``.gitignore`` ignores ``examples/**/*.png``, so the output has to be
force-added or copied out if it is going into a manuscript.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUTPUT_DIR = './output/paoflow'
POSTFIX = '_surf'

# Sequential map built from the group palette (deep blue / vermillion / amber).
# Lightness decreases monotonically from white, so it survives grayscale print.
SPECTRAL_CMAP = LinearSegmentedColormap.from_list(
    'paoflow_spectral', ['#FFFFFF', '#FFC20A', '#D55E00', '#004488']
)

LABEL_MAP = {
    'gG': r'$\overline{\Gamma}$',
    'K': r'$\overline{\mathrm{K}}$',
    'M': r'$\overline{\mathrm{M}}$',
}


def read_kpath(path):
    """Return (kdist, ticks, labels) from a surfband_kpath file."""
    kdist, ticks, labels = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            index, dist = int(parts[0]), float(parts[1])
            kdist.append(dist)
            if len(parts) >= 3:
                ticks.append(index)
                labels.append(LABEL_MAP.get(parts[2], parts[2]))
    return np.array(kdist), ticks, labels


def main():
    spectral = np.loadtxt(os.path.join(OUTPUT_DIR, f'surfband{POSTFIX}.dat'))
    egrid = np.loadtxt(os.path.join(OUTPUT_DIR, f'surfband_egrid{POSTFIX}.dat'))
    kdist, ticks, labels = read_kpath(os.path.join(OUTPUT_DIR, f'surfband_kpath{POSTFIX}.dat'))

    print('spectral map:', spectral.shape, '(ne x nk)')

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Log scale keeps both the bulk continuum and the sharp Dirac cone legible:
    # the surface state is one state per k against a continuum of hundreds, so a
    # linear scale washes it out completely.
    log_spectral = np.log10(np.abs(spectral) + 1e-6)

    # Clip the colour scale to percentiles rather than to the full range. The
    # broadened tails of A(k, E) reach far below anything physically meaningful,
    # and letting them set vmin pushes the whole map into the midtones.
    vmin, vmax = np.percentile(log_spectral, [60.0, 99.8])

    image = ax.imshow(
        log_spectral,
        origin='lower',
        aspect='auto',
        extent=[kdist[0], kdist[-1], egrid[0], egrid[-1]],
        cmap=SPECTRAL_CMAP,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks([kdist[t] for t in ticks])
    ax.set_xticklabels(labels)
    for t in ticks[1:-1]:
        ax.axvline(kdist[t], color='0.4', lw=0.6, alpha=0.8)
    ax.axhline(0.0, color='0.2', ls='--', lw=0.8, alpha=0.8)

    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_title(r'Bi$_2$Se$_3$(0001) surface spectral function')
    fig.colorbar(image, ax=ax, label=r'$\log_{10} A(k,E)$')
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, f'surfband{POSTFIX}.png')
    fig.savefig(out, dpi=200)
    print('wrote', out)


if __name__ == '__main__':
    main()
