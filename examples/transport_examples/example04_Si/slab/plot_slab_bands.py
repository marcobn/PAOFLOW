"""Plot the surface-projected slab bands written by ``main.py``.

Reads ``site-projected-bands_0.dat`` (``k_index  E  weight``, ordered
band-major) and renders it in the style of the upper panel of Fig. 10: every
eigenvalue as a dot, coloured by how much of the state sits on the two
outermost atomic planes.

Pair it with ``../plot_surface_bands.py`` (lower panel) -- same surface BZ, same
Gamma-Xbar, same energy reference (E_F = 0).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = './output/paoflow'
EMIN, EMAX = -13.0, 2.0


def main():
    raw = np.loadtxt(os.path.join(OUTPUT_DIR, 'site-projected-bands_0.dat'))
    kidx = raw[:, 0].astype(int)
    energy = raw[:, 1]
    weight = raw[:, 2]

    nkpi = kidx.max() + 1
    nbnd = raw.shape[0] // nkpi
    print(f'slab bands: {nbnd} bands x {nkpi} k-points')

    # do_site_projected_bands masks with complex(1,1), so the reported weights
    # are 2x the true |v|^2. Normalizing to the maximum removes the factor and
    # puts the colour axis on 0..1 as in the paper.
    weight = weight / weight.max()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the most surface-localized states last so they are not buried under
    # the dense bulk-like subbands.
    order = np.argsort(weight)
    image = ax.scatter(
        kidx[order],
        energy[order],
        c=weight[order],
        s=3.0,
        cmap='jet',
        vmin=0.0,
        vmax=1.0,
        linewidths=0,
    )

    ax.set_xlim(0, nkpi - 1)
    ax.set_ylim(EMIN, EMAX)
    ax.set_xticks([0, nkpi - 1])
    ax.set_xticklabels([r'$\Gamma$', 'X'])
    ax.axhline(0.0, color='k', ls='--', lw=0.8)

    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_title('Si(001) slab bands projected on the 2 outermost planes')
    fig.colorbar(image, ax=ax, label='surface weight (normalized)')
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'slab_projected_bands.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)
    if os.environ.get('DISPLAY'):
        plt.show()


if __name__ == '__main__':
    main()
