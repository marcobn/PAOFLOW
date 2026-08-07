"""Plot the surface-projected band structure written by ``main.py``.

Reads the three files produced by ``Transport.compute_surface_bands()``:

    surfband_surf.dat        (ne x nk) spectral map A(k, E)
    surfband_egrid_surf.dat  energy axis, one value per row
    surfband_kpath_surf.dat  k axis: index, distance, high-symmetry label

and renders the heatmap in the style of Fig. 10 of the PAOFLOW paper.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = './output/paoflow'
POSTFIX = '_surf'

# Display names for the path labels emitted by the PAOFLOW band-path generator.
LABEL_MAP = {
    'gG': r'$\Gamma$',
    'gS': r'$\Sigma$',
    'gD': r'$\Delta$',
    'gL': r'$\Lambda$',
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

    fig, ax = plt.subplots(figsize=(6, 5))

    # Log scale keeps both the bulk continuum and the sharp surface states legible.
    image = ax.imshow(
        np.log10(np.abs(spectral) + 1e-6),
        origin='lower',
        aspect='auto',
        extent=[kdist[0], kdist[-1], egrid[0], egrid[-1]],
        cmap='inferno',
    )

    ax.set_xticks([kdist[t] for t in ticks])
    ax.set_xticklabels(labels)
    for t in ticks[1:-1]:
        ax.axvline(kdist[t], color='w', lw=0.5, alpha=0.5)
    ax.axhline(0.0, color='w', ls='--', lw=0.8, alpha=0.8)

    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_title('Fe(001) surface-projected band structure')
    fig.colorbar(image, ax=ax, label=r'$\log_{10} A(k,E)$')
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, f'surfband{POSTFIX}.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)
    plt.show()


if __name__ == '__main__':
    main()
