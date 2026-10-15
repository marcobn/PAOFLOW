#!/usr/bin/env python3
"""Phonon dispersion and DOS of Si from the ph.x dynamical matrices (q2r.x + matdyn.x).

    python q2r_matdyn.py [--units cm-1|THz] [--mesh 24 24 24] [--sigma 3.0] [--npoints 51]

Run it in this directory after ph.x (ph.in: ldisp = .true., fildyn = 'Si2.dyn'),
which writes Si2.dyn0 (the 3x3x3 q-grid and its 4 irreducible q-points) and
Si2.dyn1 ... Si2.dyn4 (the dynamical matrices of each q-star, 27 q in all).

It does in one step what QE does with q2r.x and matdyn.x:

  1. q2r:    Fourier transform of the dynamical matrices to the real-space
             interatomic force constants of the 3x3x3 supercell, written as
             output/Si2.fc in QE's flfrc format (readable by matdyn.x);
  2. matdyn: Wigner-Seitz interpolation of the dynamical matrix (atom-pair
             images, simple acoustic sum rule, as matdyn.x asr = 'simple')
             along the high-symmetry path of the fcc lattice and on a q-mesh
             for the density of states.

Outputs (output/): Si2.fc, si_band.dat (path distance, frequencies),
si_band.labels (high-symmetry ticks), si_dos.dat (frequency, DOS) and the
figure si_phonons.png, drawn with GPAO.plot_phonons as in paoflow-gen's phonon
workflow.  Si is non-polar: no LO-TO splitting term is needed (a warning is
printed for dynamical matrices with Born effective charges).
"""

import argparse
import os

from PAOFLOW import GPAO
from PAOFLOW.phonon.qe_force_constants import THZ_TO_CM1, q2r_matdyn

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = 'Si2'  # fildyn of ph.x without the .dynN suffix
OUTPUTDIR = os.path.join(HERE, 'output')
FNAME = 'si'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--units', choices=('cm-1', 'THz'), default='cm-1', help='frequency unit')
    parser.add_argument('--mesh', type=int, nargs=3, default=(24, 24, 24), help='q-mesh of the DOS')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian broadening of the DOS, in --units (default 3 cm-1)')
    parser.add_argument('--npoints', type=int, default=51, help='points per line of the path')
    parser.add_argument('--no-plot', action='store_true', help='write the files only')
    args = parser.parse_args()

    sigma_thz = args.sigma / THZ_TO_CM1 if args.units == 'cm-1' else args.sigma
    result = q2r_matdyn(
        PREFIX, HERE, outdir=OUTPUTDIR, fname=FNAME, force_constants_file=PREFIX + '.fc',
        npoints=args.npoints, mesh=args.mesh, sigma_thz=sigma_thz, units=args.units,
    )

    dyn = result['dynamical_matrices']
    scale = THZ_TO_CM1 if args.units == 'cm-1' else 1.0
    frequencies = result['dispersion']['frequency_thz'] * scale
    print('q2r + matdyn: %s.dyn0-%d, q-grid %s, %d atoms'
          % (PREFIX, len(dyn['dyn_paths']), 'x'.join(map(str, dyn['qgrid'])), dyn['nat']))
    print('  frequencies along the path: %.2f to %.2f %s'
          % (frequencies.min(), frequencies.max(), args.units))
    for path in result['files']:
        print('  wrote %s' % path)

    if not args.no_plot:
        figure = os.path.join(OUTPUTDIR, FNAME + '_phonons.png')
        GPAO.GPAO().plot_phonons(
            os.path.join(OUTPUTDIR, FNAME + '_band.dat'),
            dos_file=os.path.join(OUTPUTDIR, FNAME + '_dos.dat'),
            labels_file=os.path.join(OUTPUTDIR, FNAME + '_band.labels'),
            title='Si phonons (ph.x 3x3x3, q2r + matdyn)',
            units=args.units, filename=figure,
        )
        print('  wrote %s' % figure)


if __name__ == '__main__':
    main()
