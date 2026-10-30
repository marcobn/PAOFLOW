#!/usr/bin/env python3
"""The long-range coupling of a polar crystal: |g(k = Gamma, q)| along a direction, with/without the model.

    mpirun -np N python long_range_check.py [--nk 24] [--quadrupoles ../coupling/quadrupole.fmt]

After the scf, ph.x (epsil, fildvscf) and the nscf, the coarse coupling is computed by
PAOFLOW from the dvscf files, the analytic dipole vertex of the rigid-ion model (epsilon_inf,
Z*) is subtracted on the coarse grid and added back at every dense q, and the band-resolved
|g| at k = Gamma is evaluated for q along the direction on the dense --nk grid.  The LO mode
diverges as 1/|q| (Fröhlich); the plain interpolation of the full coupling cannot represent
that and underestimates the coupling near Gamma.  With --quadrupoles the dipole + quadrupole
model is added to the comparison.

Outputs (output/): g_path.dat (|q|, mode-group sums of |g|^2 over the valence bands, with and
without the model, the analytic dipole term alone [and the quadrupole model]) and
g_long_range.png.
"""

import argparse
import os

import numpy as np

from PAOFLOW.elphon.workflows import is_root, load_system, long_range_path

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = getattr(CONFIG, 'COUPLING', {})
NAME = getattr(CONFIG, 'NAME', SYSTEM.prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--nk', type=int, default=DEFAULTS.get('nk', 24),
        help='dense grid of the path points (multiples of 1/nk)',
    )  # fmt: skip
    parser.add_argument(
        '--direction', type=float, nargs=3, default=DEFAULTS.get('direction', (1.0, 0.0, 0.0)),
        help='Cartesian direction of q (2 pi / alat)',
    )  # fmt: skip
    parser.add_argument(
        '--quadrupoles', default=SYSTEM.quadrupoles_path, metavar='FILE',
        help='quadrupole.fmt of quadrupoles.py (default: the one of system.py)',
    )  # fmt: skip
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    SYSTEM.check_inputs()
    electrons = SYSTEM.electrons(outputdir=OUTPUT)
    out = long_range_path(
        SYSTEM, electrons, nk=args.nk, direction=args.direction, quadrupoles=args.quadrupoles
    )
    if not is_root():
        return
    os.makedirs(OUTPUT, exist_ok=True)
    rows, model = out['rows'], out['model']
    path = os.path.join(OUTPUT, 'g_path.dat')
    np.savetxt(path, rows, fmt='%12.5e', header=out['header'])
    print(
        'Long-range coupling of %s (k = Gamma, q along %s, dense %d^3):'
        % (NAME, tuple(args.direction), args.nk)
    )
    print(
        '  epsilon_inf %.3f, Z* %s, omega_LO at the first q %.1f meV'
        % (
            model.epsilon[0, 0],
            ' / '.join('%.3f' % z for z in model.born_charges[:, 0, 0]),
            out['omega_lo_first_mev'],
        )
    )
    quad = out['has_quadrupoles']
    print(
        '  |q|     |g_LO|^2 model   plain   dipole-only%s  (meV^2)'
        % ('   dipole+quadrupole LO / TO / acoustic (model LO/TO/ac)' if quad else '')
    )
    for r in rows[:4]:
        extra = ''
        if quad:
            extra = '   %10.1f / %8.1f / %8.1f  (%8.1f / %8.1f / %8.1f)' % (
                r[8], r[9], r[10], r[1], r[2], r[3],
            )  # fmt: skip
        print('  %.4f  %12.1f %12.1f %12.1f%s' % (r[0], r[1], r[4], r[7], extra))
    print('  wrote %s' % path)
    if not args.no_plot:
        plot(rows)


def plot(rows):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    q = rows[:, 0]
    ax.plot(q, rows[:, 7], color='0.55', lw=1.0, ls=':', label='dipole term alone')
    ax.plot(q, rows[:, 4], color='#D55E00', lw=1.4, ls='--', marker='s', ms=4,
            label='plain interpolation')  # fmt: skip
    ax.plot(q, rows[:, 1], color='#0072B2', lw=1.6, marker='o', ms=4, label='with the dipole term')
    if rows.shape[1] > 8:
        ax.plot(q, rows[:, 8], color='#009E73', lw=1.6, marker='^', ms=4,
                label='dipole + quadrupole terms')  # fmt: skip
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('|q| (2π/a)')
    ax.set_ylabel('Σ$_{mn}$ |g$_{LO}$|² (meV²), k = Γ')
    ax.set_title('%s: Fröhlich 1/|q| coupling of the LO mode' % NAME, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    path = os.path.join(OUTPUT, 'g_long_range.png')
    fig.savefig(path, dpi=200)
    print('  wrote %s' % path)


if __name__ == '__main__':
    main()
