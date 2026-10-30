#!/usr/bin/env python3
"""Phonon dispersion and DOS from the ph.x dynamical matrices (q2r.x + matdyn.x in PAOFLOW).

    python phonons.py [--units cm-1|THz] [--mesh 24 24 24] [--sigma 3.0] [--npoints 51]

Reads <prefix>.dyn0 (the q-grid and its irreducible q) and <prefix>.dyn1 ... dynN of the
phonon directory of system.py, Fourier transforms them to the real-space force constants
(output/<prefix>.fc, QE's flfrc format) and interpolates the dynamical matrix along the
high-symmetry path of the lattice and on a q-mesh for the DOS (Wigner-Seitz atom-pair images,
acoustic sum rule as matdyn.x asr = 'simple').  For a polar crystal (ph.x with epsil) the
long-range dipole term is subtracted before the transform and added back at every q (LO-TO
splitting), and the plain interpolation is drawn alongside for comparison.

Outputs (output/): <prefix>.fc, <fname>_band.dat, <fname>_band.labels, <fname>_dos.dat
[+ the _plain counterparts] and <fname>_phonons.png.
"""

import argparse
import os

import numpy as np

from PAOFLOW.elphon.workflows import load_system, phonon_dispersion

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = getattr(CONFIG, 'PHONONS', {})
NAME = getattr(CONFIG, 'NAME', SYSTEM.prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--units', choices=('cm-1', 'THz'), default=DEFAULTS.get('units', 'cm-1'))
    parser.add_argument(
        '--mesh', type=int, nargs=3, default=DEFAULTS.get('mesh', (24, 24, 24)),
        help='q-mesh of the DOS',
    )  # fmt: skip
    parser.add_argument(
        '--sigma', type=float, default=DEFAULTS.get('sigma', 3.0),
        help='Gaussian broadening of the DOS, in --units',
    )  # fmt: skip
    parser.add_argument('--npoints', type=int, default=51, help='points per line of the path')
    parser.add_argument('--no-plot', action='store_true', help='write the files only')
    args = parser.parse_args()
    out = phonon_dispersion(
        SYSTEM, OUTPUT, units=args.units, mesh=args.mesh, sigma=args.sigma, npoints=args.npoints
    )
    result, scale, units = out['result'], out['scale'], args.units
    dyn = result['dynamical_matrices']
    print(
        'q2r + matdyn: %s.dyn0-%d, q-grid %s, %d atoms'
        % (SYSTEM.prefix, len(dyn['dyn_paths']), 'x'.join(map(str, dyn['qgrid'])), dyn['nat'])
    )
    frequencies = result['dispersion']['frequency_thz'] * scale
    print(
        '  frequencies along the path: %.2f to %.2f %s'
        % (frequencies.min(), frequencies.max(), units)
    )
    if out['polar']:
        model = result['long_range']
        print(
            '  polar crystal: epsilon_inf = %.3f, Z* = %s (charge neutrality imposed)'
            % (model.epsilon[0, 0], ', '.join('%.3f' % z for z in model.born_charges[:, 0, 0]))
        )
        gamma = np.sort(result['dispersion']['frequency_thz'][0]) * scale
        gamma_plain = np.sort(out['plain']['dispersion']['frequency_thz'][0]) * scale
        print(
            '  Gamma (limit along the first path segment): TO %.1f, LO %.1f %s; '
            'plain interpolation: %.1f, %.1f'
            % (gamma[-2], gamma[-1], units, gamma_plain[-2], gamma_plain[-1])
        )
    for path in result['files']:
        print('  wrote %s' % path)
    if not args.no_plot:
        plot(out)


def plot(out):
    """<fname>_phonons.png: the dispersion and DOS (with the plain interpolation when polar)."""
    figure = os.path.join(OUTPUT, out['fname'] + '_phonons.png')
    if not out['polar']:
        from PAOFLOW import GPAO

        stem = os.path.join(OUTPUT, out['fname'])
        GPAO.GPAO().plot_phonons(
            stem + '_band.dat',
            dos_file=stem + '_dos.dat',
            labels_file=stem + '_band.labels',
            title='%s phonons (ph.x, q2r + matdyn)' % NAME,
            units=out['units'],
            filename=figure,
        )
        print('  wrote %s' % figure)
        return
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scale, units = out['scale'], out['units']
    disp, plain, dos = (
        out['result']['dispersion'],
        out['plain']['dispersion'],
        out['result']['dos'],
    )
    fig, (ax, ax_dos) = plt.subplots(
        1, 2, figsize=(7.2, 4.0), gridspec_kw={'width_ratios': [3, 1]}, sharey=True,
        constrained_layout=True,
    )  # fmt: skip
    x, ticks, labels = disp['distance'], disp['tick_distance'], disp['tick_label']
    ax.plot(x, plain['frequency_thz'] * scale, color='#D55E00', lw=1.2, ls='--',
            label='plain interpolation')  # fmt: skip
    ax.plot(x, disp['frequency_thz'] * scale, color='#0072B2', lw=1.6, label='with the dipole term')
    handles, names = ax.get_legend_handles_labels()
    nbranch = disp['frequency_thz'].shape[1]
    ax.legend(handles[::nbranch], names[::nbranch], frameon=False, loc='center',
              bbox_to_anchor=(0.5, 0.68))  # fmt: skip
    for t in ticks:
        ax.axvline(t, color='0.8', lw=0.8, zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel('frequency (%s)' % units)
    ax.set_title('%s phonons: LO-TO splitting from ε∞ and Z*' % NAME, fontsize=10)
    if dos is not None:
        ax_dos.plot(dos['dos'], dos['frequency_thz'] * scale, color='#0072B2', lw=1.4)
        ax_dos.set_xlabel('DOS (states/%s)' % units)
        ax_dos.set_xlim(left=0)
    for a in (ax, ax_dos):
        a.spines[['top', 'right']].set_visible(False)
    fig.savefig(figure, dpi=200)
    print('  wrote %s' % figure)


if __name__ == '__main__':
    main()
