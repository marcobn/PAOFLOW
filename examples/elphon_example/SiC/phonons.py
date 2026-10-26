#!/usr/bin/env python3
"""Phonon dispersion and DOS of 3C-SiC with the LO-TO splitting (q2r.x + matdyn.x in PAOFLOW).

    python phonons.py [--units cm-1|THz] [--mesh 24 24 24] [--sigma 3.0] [--npoints 51]

Run it after ph.x (phonon/ph.in: ldisp, epsil = .true., fildyn = 'sic.dyn'), which writes
sic.dyn0 (the 6x6x6 q-grid and its 16 irreducible q) and sic.dyn1 ... sic.dyn16 (one star each;
sic.dyn1 also holds the dielectric tensor and the Born effective charges).

A polar crystal needs the non-analytic dipole term: as q2r.x with zasr = 'simple', the
long-range force constants of the rigid-ion model (epsilon_inf, Z*, Ewald sum) are subtracted
from the coarse dynamical matrices before the transform to real space, and as matdyn.x they are
added back at every q of the path and of the DOS mesh, with the direction-dependent limit at
Gamma (LO-TO splitting).  The same run without the model (long_range=False) shows what the
plain Fourier interpolation misses: the LO branch collapses onto the TO near Gamma.

Outputs (output/phonons): sic.fc (flfrc with the tensors, readable by matdyn.x), sic_band.dat,
sic_band.labels, sic_dos.dat, their *_plain counterparts, and sic_phonons.png.
"""

import argparse
import os
import warnings

import numpy as np

from PAOFLOW.phonon.qe_force_constants import THZ_TO_CM1, q2r_matdyn

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = 'sic'
PHONON_DIR = os.path.join(HERE, 'phonon')
OUTPUTDIR = os.path.join(HERE, 'output', 'phonons')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--units', choices=('cm-1', 'THz'), default='cm-1')
    parser.add_argument('--mesh', type=int, nargs=3, default=(24, 24, 24), help='q-mesh of the DOS')
    parser.add_argument('--sigma', type=float, default=3.0, help='DOS broadening, in --units')
    parser.add_argument('--npoints', type=int, default=51, help='points per line of the path')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    sigma_thz = args.sigma / THZ_TO_CM1 if args.units == 'cm-1' else args.sigma
    common = dict(npoints=args.npoints, mesh=args.mesh, sigma_thz=sigma_thz, units=args.units)
    with_model = q2r_matdyn(
        PREFIX, PHONON_DIR, outdir=OUTPUTDIR, fname=PREFIX, force_constants_file=PREFIX + '.fc', **common
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # the plain interpolation warns about the missing LO-TO term
        plain = q2r_matdyn(PREFIX, PHONON_DIR, outdir=OUTPUTDIR, fname=PREFIX + '_plain', long_range=False, **common)
    model, dyn = with_model['long_range'], with_model['dynamical_matrices']
    scale = THZ_TO_CM1 if args.units == 'cm-1' else 1.0
    print('q2r + matdyn: %s.dyn0-%d, q-grid %s' % (PREFIX, len(dyn['dyn_paths']), 'x'.join(map(str, dyn['qgrid']))))
    print('  epsilon_inf = %.3f, Z*(Si) = %.3f, Z*(C) = %.3f (charge neutrality imposed)'
          % (model.epsilon[0, 0], model.born_charges[0, 0, 0], model.born_charges[1, 0, 0]))
    gamma = np.sort(with_model['dispersion']['frequency_thz'][0]) * scale
    gamma_plain = np.sort(plain['dispersion']['frequency_thz'][0]) * scale
    print('  Gamma (limit along the first path segment): TO %.1f, LO %.1f %s; plain interpolation: %.1f, %.1f'
          % (gamma[-2], gamma[-1], args.units, gamma_plain[-2], gamma_plain[-1]))
    for path in with_model['files']:
        print('  wrote %s' % path)
    if not args.no_plot:
        plot(with_model, plain, args.units, scale)


def plot(with_model, plain, units, scale):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    disp, disp_plain, dos = with_model['dispersion'], plain['dispersion'], with_model['dos']
    fig, (ax, ax_dos) = plt.subplots(
        1, 2, figsize=(7.2, 4.0), gridspec_kw={'width_ratios': [3, 1]}, sharey=True, constrained_layout=True
    )
    x, ticks, labels = disp['distance'], disp['tick_distance'], disp['tick_label']
    ax.plot(x, disp_plain['frequency_thz'] * scale, color='#D55E00', lw=1.2, ls='--', label='plain interpolation')
    ax.plot(x, disp['frequency_thz'] * scale, color='#0072B2', lw=1.6, label='with the dipole term')
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[::6], names[::6], frameon=False, loc='center', bbox_to_anchor=(0.5, 0.68))
    for t in ticks:
        ax.axvline(t, color='0.8', lw=0.8, zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel('frequency (%s)' % units)
    ax.set_title('3C-SiC phonons: LO-TO splitting from ε∞ and Z*', fontsize=10)
    if dos is not None:
        ax_dos.plot(dos['dos'], dos['frequency_thz'] * scale, color='#0072B2', lw=1.4)
        ax_dos.set_xlabel('DOS (states/%s)' % units)
        ax_dos.set_xlim(left=0)
    for a in (ax, ax_dos):
        a.spines[['top', 'right']].set_visible(False)
    path = os.path.join(OUTPUTDIR, PREFIX + '_phonons.png')
    fig.savefig(path, dpi=200)
    print('  wrote %s' % path)


if __name__ == '__main__':
    main()
