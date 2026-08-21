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

# Match ../plot_surface_bands.py exactly so the two panels stack.
EMIN, EMAX = -12.5, 2.0

# Rigid shift applied to the slab eigenvalues, in eV.
#
# The two panels of Fig. 10 do NOT share an energy zero as computed. This slab
# is metallic (dangling bonds + smearing), so QE writes a real fermi_energy and
# PAOFLOW puts E = 0 at the slab's Fermi level. The bulk run in the parent
# directory is an insulator run with occupations='fixed', where QE writes
# <fermi_energy> equal to <highestOccupiedLevel> -- so its E = 0 is the bulk
# valence-band MAXIMUM, not a Fermi level. Stacking the panels without aligning
# puts the bulk zero below the entire in-gap surface band, which is why the
# lower panel reads as a semiconductor and the upper one as a metal.
#
# To align: run both, read the bottom of the deepest bulk-like band at Gamma-bar
# off each panel, and set ALIGN_SHIFT to the difference (slab -> bulk). The deep
# valence states are bulk-like in the slab interior and are the cleanest common
# reference; the surface-derived states are exactly what differs and must not be
# used. Leave at 0.0 to plot the slab on its own Fermi level.
ALIGN_SHIFT = 0.0

LABEL_MAP = {
    'gG': r'$\overline{\Gamma}$',
    'X': r'$\overline{\mathrm{X}}$',
    'M': r'$\overline{\mathrm{M}}$',
    'Z': r'$\overline{\mathrm{Z}}$',
    'R': r'$\overline{\mathrm{R}}$',
    'A': r'$\overline{\mathrm{A}}$',
}


def read_kpath_labels(path):
    """Return (ticks, labels) from the label block at the top of kpath_points.txt.

    The file opens with one ``"<label> <npoints>"`` record per path segment,
    where the number is how many k-points that segment contributes before the
    next label, followed by the k-point coordinates.
    """
    ticks, labels, index = [], [], 0
    for line in open(path):
        parts = line.split()
        if len(parts) != 2:
            continue
        labels.append(parts[0])
        ticks.append(index)
        index += int(parts[1])
    return ticks, labels


def main():
    raw = np.loadtxt(os.path.join(OUTPUT_DIR, 'site-projected-bands_0.dat'))
    kidx = raw[:, 0].astype(int)
    energy = raw[:, 1] + ALIGN_SHIFT
    weight = raw[:, 2]

    nkpi = kidx.max() + 1
    nbnd = raw.shape[0] // nkpi
    print(f'slab bands: {nbnd} bands x {nkpi} k-points')

    # Read the tick labels from kpath_points.txt rather than assuming the path.
    # This is not defensive padding: bands() silently ignores band_path unless
    # high_sym_points is passed too, so an unlabelled axis will happily present
    # the full default TET path as if it were Gamma-bar -> X-bar.
    ticks, labels = read_kpath_labels(os.path.join(OUTPUT_DIR, 'kpath_points.txt'))
    if labels != ['gG', 'X']:
        print(f'  WARNING: path is {"-".join(labels)}, not gG-X.')
        print('  bands() dropped your band_path -- pass high_sym_points as well.')
        print('  The panel will not line up with ../plot_surface_bands.py.')

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
    ax.set_xticks([t for t in ticks if t < nkpi])
    ax.set_xticklabels([LABEL_MAP.get(l, l) for t, l in zip(ticks, labels) if t < nkpi])
    for t in ticks[1:-1]:
        if t < nkpi:
            ax.axvline(t, color='0.6', lw=0.5)
    ax.axhline(0.0, color='k', ls='--', lw=0.8)

    ax.set_ylabel(r'$E - E_F$ (eV)' if ALIGN_SHIFT == 0.0 else r'$E - E_\mathrm{VBM}$ (eV)')
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
