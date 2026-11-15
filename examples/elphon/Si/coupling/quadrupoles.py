#!/usr/bin/env python3
"""Dynamical quadrupoles from single-q ph.x density responses (no ABINIT needed).

    python quadrupoles.py --write-inputs      # <phonon dir>/quadrupoles/q*/ph.in and run.sh
    (cd <phonon dir>/quadrupoles && NP=8 bash run.sh)   # twelve single-q ph.x runs with fildrho
    python quadrupoles.py --extract           # fit Z* (check) and Q, write quadrupole.fmt

The electronic charge induced by a monochromatic displacement has the expansion
-Omega <n~> e^{iq.tau} eps_L(q) = -i q.Z_el - 1/2 q.Q.q + O(q^3): the imaginary part is the
electronic Born charge (a check against the Gamma-point DFPT, Z* - Z_ion), the real part at
order q^2 the dynamical quadrupole (Royo and Stengel, PRX 9, 021050 (2019); Brunin et al.,
PRL 125, 136601 (2020)).  Six directions and two magnitudes determine the 18 components per
atom and the q -> 0 extrapolation.  Set quadrupoles='coupling/quadrupole.fmt' in system.py to
include the term in the long-range model of every workflow.

Needs ph.x run with epsil = .true. (dielectric tensor) and the scf save of the phonon
directory (with its wfc files).
"""

import argparse
import os

import numpy as np

from PAOFLOW.elphon.workflows import load_system, quadrupole_inputs, quadrupoles

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
MAGNITUDES = (0.02, 0.04)  # |q| in 2 pi / alat


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--write-inputs', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument(
        '--magnitudes', type=float, nargs='+', default=MAGNITUDES, help='|q| (2 pi / alat)'
    )
    parser.add_argument(
        '--output', default=os.path.join(HERE, 'quadrupole.fmt'), help='quadrupole.fmt to write'
    )
    args = parser.parse_args()
    if args.write_inputs:
        runs = quadrupole_inputs(SYSTEM, magnitudes=args.magnitudes)
        print(
            'wrote %d ph.x inputs under %s (run.sh runs them; each needs the scf save of %s)'
            % (len(runs), os.path.dirname(runs[0]), SYSTEM.phonon_path)
        )
    if args.extract:
        fit = quadrupoles(SYSTEM, args.output)
        gamma = fit['gamma']
        print(
            'Born charges fitted from the q-expansion vs the Gamma DFPT: max |diff| = %.3f e '
            '(density sign %+.0f)'
            % (np.abs(fit['zstar'] - gamma['born_charges']).max(), fit['sign'])
        )
        quad = fit['quadrupoles']
        for mag, q in sorted(fit['quadrupoles_by_magnitude'].items()):
            print('  |q| = %.3f: Q[atom 1]_{x,yz} = %.3f e bohr' % (mag, q[0, 0, 1, 2]))
        print(
            'extrapolated q -> 0: '
            + ', '.join(
                'Q[atom %d]_{x,yz} = %.3f' % (i + 1, quad[i, 0, 1, 2]) for i in range(quad.shape[0])
            )
            + ' e bohr'
        )
        print(
            "wrote %s (set quadrupoles='coupling/%s' in system.py)"
            % (fit['path'], os.path.basename(fit['path']))
        )
    if not (args.write_inputs or args.extract):
        parser.print_help()


if __name__ == '__main__':
    main()
