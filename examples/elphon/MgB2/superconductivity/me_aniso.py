#!/usr/bin/env python3
"""Superconducting gaps and Tc from the anisotropic (Fermi-surface-restricted) Migdal-Eliashberg equations.

    python me_aniso.py
    python me_aniso.py --linear-temps 20 60 9    # also Tc from the linearised kernel
    python plot_me.py

Post-processing of the Fermi-surface coupling written by eliashberg.py --fs-coupling
(output/fs_coupling.npz; use equal dense grids, --nk N --nq N, otherwise the equations split
into independent sublattice problems).  For every temperature output/me_aniso/ receives, in
EPW's formats, <prefix>.imag_aniso_<T> (Delta_nk and Z_nk on the Matsubara axis), the gap
distributions, <prefix>.imag_aniso_gap_FS_<T>, <prefix>.pade_aniso_<T> and <prefix>.qdos_<T>;
<prefix>.lambda_FS holds lambda_nk, gap_vs_T_aniso.dat the gap range versus T and
migdal_eliashberg_aniso.npz all of it.
"""

import argparse
import os

from PAOFLOW.elphon.workflows import load_system, migdal_eliashberg_aniso

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = CONFIG.SUPERCONDUCTIVITY


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--temps', type=float, nargs='+', default=DEFAULTS.get('me_aniso_temps', DEFAULTS['me_temps']),
        help='temperatures (K)',
    )  # fmt: skip
    parser.add_argument(
        '--linear-temps', type=float, nargs=3, metavar=('TMIN', 'TMAX', 'NSTEMP'),
        help='also compute the linearised-kernel eigenvalue on these temperatures',
    )  # fmt: skip
    parser.add_argument('--mu-star', type=float, default=DEFAULTS.get('mu_star', 0.1))
    parser.add_argument(
        '--wscut', type=float, default=DEFAULTS.get('wscut_aniso', 0.5), help='Matsubara cutoff (eV)'
    )  # fmt: skip
    parser.add_argument(
        '--npade', type=float, default=DEFAULTS.get('npade', 90.0),
        help='%% of Matsubara points for Padé',
    )  # fmt: skip
    parser.add_argument('--no-pade', action='store_true', help='skip the real axis')
    args = parser.parse_args()
    migdal_eliashberg_aniso(
        os.path.join(OUTPUT, 'fs_coupling.npz'),
        os.path.join(OUTPUT, 'me_aniso'),
        SYSTEM.prefix,
        temps=args.temps,
        mu_star=args.mu_star,
        wscut=args.wscut,
        npade=args.npade,
        pade=not args.no_pade,
        linear_temps=args.linear_temps,
    )


if __name__ == '__main__':
    main()
