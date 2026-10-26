#!/usr/bin/env python3
"""Dynamical quadrupoles of 3C-SiC from single-q ph.x density responses (no ABINIT needed).

    python quadrupoles.py --write-inputs      # phonon/quadrupoles/q*/ph.in and run.sh
    (cd phonon/quadrupoles && bash run.sh)    # twelve single-q ph.x runs with fildrho (~1 min each; needs the scf wfc files)
    python quadrupoles.py --extract           # fit Z*_el (check) and Q, write quadrupole.fmt

The electronic charge induced by a monochromatic displacement has the expansion
-Omega <n~> e^{iq.tau} eps_L(q) = -i q.Z_el - 1/2 q.Q.q + O(q^3): the imaginary part is the
electronic Born charge (a check against the Gamma-point DFPT, Z* - Z_ion), the real part at
order q^2 the dynamical quadrupole (Royo and Stengel, PRX 9, 021050 (2019); for Si Brunin et
al., PRL 125, 136601 (2020) give |Q_{x,yz}| = 13.66 e bohr^2; in zincblende SiC the Si and C
quadrupoles differ in magnitude and sign).  Six directions and two
magnitudes determine the 18 components per atom and the q -> 0 extrapolation.  Set
QUADRUPOLES = 'quadrupole.fmt' in mobility.py / coupling.py to include the term in the coupling.
Needs phonon/ph.in with epsil = .true. (dielectric tensor) and the scf save of phonon/.
"""

import argparse
import glob
import os
import shutil

import numpy as np

from PAOFLOW.elphon.dfpt_io import read_nscf_plane_wave_setup
from PAOFLOW.elphon.quadrupoles import (
    DIRECTIONS,
    quadrupoles_from_runs,
    single_q_phonon_inputs,
    write_quadrupole_fmt,
)
from PAOFLOW.inputs.read_upf import UPF
from PAOFLOW.phonon.qe_force_constants import read_dynamical_matrix_file

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = 'sic'
PHONON_DIR = os.path.join(HERE, 'phonon')
SAVEDIR = os.path.join(PHONON_DIR, PREFIX + '.save')
QDIR = os.path.join(PHONON_DIR, 'quadrupoles')
MAGNITUDES = (0.02, 0.04)  # |q| in 2 pi / alat
PSEUDO_DIR = HERE


def write_inputs():
    runs = single_q_phonon_inputs(PREFIX, MAGNITUDES, DIRECTIONS, outdir='./')
    os.makedirs(QDIR, exist_ok=True)
    lines = ['#!/bin/bash', '# single-q ph.x runs for the quadrupole expansion (run from this directory)', 'set -e',
             'NP=${NP:-8}', 'for d in q*/; do', '  cd $d', "  rm -rf _ph0; mkdir -p %s.save" % PREFIX,
             "  cp ../../%s.save/charge-density.dat ../../%s.save/data-file-schema.xml ../../%s.save/wfc*.dat %s.save/"
             % (PREFIX, PREFIX, PREFIX, PREFIX),
             "  cp ../../%s.save/*.upf ../../%s.save/*.UPF %s.save/ 2>/dev/null || true" % (PREFIX, PREFIX, PREFIX),
             '  echo "$(date +%T) $d"; mpirun -np $NP ph.x -nk $NP -in ph.in > ph.out 2>&1', '  cd ..', 'done', 'echo done']
    with open(os.path.join(QDIR, 'run.sh'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    for i, (name, q, text) in enumerate(runs):
        d = os.path.join(QDIR, 'q%02d' % (i + 1))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ph.in'), 'w') as fh:
            fh.write(text.replace("pseudo_dir = ''", ''))
        np.savetxt(os.path.join(d, 'q.txt'), q[None], header='q (2 pi / alat), Cartesian')
    print('wrote %d ph.x inputs under %s (run.sh runs them; each needs the scf save of phonon/)' % (len(runs), QDIR))


def extract():
    setup = read_nscf_plane_wave_setup(SAVEDIR)
    gamma_dyn = sorted(glob.glob(os.path.join(PHONON_DIR, PREFIX + '.dyn1')))
    if not gamma_dyn:
        raise SystemExit('phonon/%s.dyn1 (the Gamma dynamical matrix with epsil = .true.) is needed' % PREFIX)
    gamma = read_dynamical_matrix_file(gamma_dyn[0])
    if gamma['epsilon'] is None:
        raise SystemExit('the Gamma dynamical matrix has no dielectric tensor: rerun ph.x with epsil = .true.')
    alat, omega = setup['alat'], setup['omega']
    tau_cart = setup['tau_alat'] * alat
    zion = np.array([UPF(os.path.join(SAVEDIR, setup['pseudo_files'][sp])).val for sp in setup['atom_species']])
    runs = []
    for d in sorted(glob.glob(os.path.join(QDIR, 'q*'))):
        if os.path.isfile(os.path.join(d, 'q.txt')):
            runs.append((np.loadtxt(os.path.join(d, 'q.txt')), d))
    if len(runs) < 6:
        raise SystemExit('found %d finished runs under %s; six directions are needed' % (len(runs), QDIR))
    fit = quadrupoles_from_runs(
        runs, PREFIX, setup['fft_grid'], tau_cart, alat, omega, gamma['epsilon'], zion, zstar_ref=gamma['born_charges']
    )
    print('Born charges fitted from the q-expansion vs the Gamma DFPT: max |diff| = %.3f e (density sign %+.0f)'
          % (np.abs(fit['zstar'] - gamma['born_charges']).max(), fit['sign']))
    quad = fit['quadrupoles']
    for mag, q in sorted(fit['quadrupoles_by_magnitude'].items()):
        print('  |q| = %.3f: Q[Si]_{x,yz} = %.3f, Q[C]_{x,yz} = %.3f e bohr^2; largest other component %.3f'
              % (mag, q[0, 0, 1, 2], q[1, 0, 1, 2], np.abs(q - np.where(np.abs(q) > 0.5 * np.abs(q).max(), q, 0.0)).max()))
    print('extrapolated q -> 0: Q[Si]_{x,yz} = %.3f, Q[C]_{x,yz} = %.3f e bohr^2'
          % (quad[0, 0, 1, 2], quad[1, 0, 1, 2]))
    path = write_quadrupole_fmt(os.path.join(HERE, 'quadrupole.fmt'), quad)
    print('wrote %s (set QUADRUPOLES = %r in mobility.py / coupling.py)' % (path, os.path.basename(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--write-inputs', action='store_true')
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    if args.write_inputs:
        write_inputs()
    if args.extract:
        extract()
    if not (args.write_inputs or args.extract):
        parser.print_help()


if __name__ == '__main__':
    main()
