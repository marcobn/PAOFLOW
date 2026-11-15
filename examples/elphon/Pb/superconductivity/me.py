#!/usr/bin/env python3
"""Superconducting gap and Tc from the isotropic Migdal-Eliashberg equations.

    python me.py                          # gap on the imaginary and real axes, Delta(T), Tc
    python me.py --a2f <prefix>.a2f       # the same on EPW's own alpha^2F
    python plot_me.py

Post-processing of the alpha^2F written by eliashberg.py (output/eliashberg.npz).  For every
temperature output/me/ receives, in EPW's formats, <prefix>.imag_iso_<T> (Matsubara axis),
<prefix>.pade_iso_<T> and <prefix>.acon_iso_<T> (real axis from Padé and from the analytic
continuation) and <prefix>.qdos_iso_<T> (quasiparticle DOS).  gap_vs_T.dat collects Delta(T)
from the lowest Matsubara frequency and the two real-axis gap edges, max_eigenvalue.dat the
largest eigenvalue of the linearised kernel (1 at Tc), and migdal_eliashberg.npz all of it.
"""

import argparse
import os

from PAOFLOW.elphon.workflows import load_system, migdal_eliashberg

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = CONFIG.SUPERCONDUCTIVITY


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--a2f', help="EPW '<prefix>.a2f' to use instead of output/eliashberg.npz")
    parser.add_argument(
        '--temps', type=float, nargs='+', default=DEFAULTS['me_temps'], help='temperatures (K)'
    )
    parser.add_argument(
        '--linear-temps', type=float, nargs=3, default=DEFAULTS.get('me_linear_temps'),
        metavar=('TMIN', 'TMAX', 'NSTEMP'), help='linearised-kernel temperatures',
    )  # fmt: skip
    parser.add_argument('--mu-star', type=float, default=DEFAULTS.get('mu_star', 0.1))
    parser.add_argument(
        '--wscut', type=float, default=DEFAULTS.get('wscut', 0.1), help='Matsubara cutoff (eV)'
    )
    parser.add_argument(
        '--degaussq', type=float, default=DEFAULTS.get('degaussq_mev', 0.15),
        help='phonon smearing of alpha^2F (meV)',
    )  # fmt: skip
    parser.add_argument(
        '--degaussq-linear', type=float, default=DEFAULTS.get('degaussq_linear_mev', 0.5),
        help='phonon smearing for the linearised kernel (meV)',
    )  # fmt: skip
    parser.add_argument(
        '--npade', type=float, default=DEFAULTS.get('npade', 90.0),
        help='%% of Matsubara points for Padé',
    )  # fmt: skip
    parser.add_argument('--no-pade', action='store_true', help='skip the Padé output')
    parser.add_argument('--no-acon', action='store_true', help='skip the analytic continuation')
    parser.add_argument('--no-linear', action='store_true', help='skip the linearised kernel')
    args = parser.parse_args()
    migdal_eliashberg(
        os.path.join(OUTPUT, 'eliashberg.npz'),
        os.path.join(OUTPUT, 'me'),
        SYSTEM.prefix,
        temps=args.temps,
        mu_star=args.mu_star,
        wscut=args.wscut,
        degaussq_mev=args.degaussq,
        degaussq_linear_mev=args.degaussq_linear,
        npade=args.npade,
        pade=not args.no_pade,
        acon=not args.no_acon,
        linear_temps=None if args.no_linear else args.linear_temps,
        epw_a2f=args.a2f,
    )


if __name__ == '__main__':
    main()
