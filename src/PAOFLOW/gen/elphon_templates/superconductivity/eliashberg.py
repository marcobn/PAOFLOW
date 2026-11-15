#!/usr/bin/env python3
"""Eliashberg function alpha^2F, lambda, omega_log and Tc from the coarse electron-phonon coupling.

    mpirun -np N python eliashberg.py                       # dense k and q grids of system.py
    mpirun -np N python eliashberg.py --nk 48 --nq 24       # other dense grids (nk % nq == 0)
    mpirun -np N python eliashberg.py --coarse-q            # dense k only, q on the coarse grid
    mpirun -np N python eliashberg.py --transport           # also alpha^2F_tr (transport/resistivity.py)
    mpirun -np N python eliashberg.py --fs-coupling         # also the Fermi-surface coupling (me_aniso.py)
    mpirun -np N python eliashberg.py --adaptive-smearing 0.7 --sigma-ev 0.001

PAOFLOW builds the PAO electronic structure on the coarse-grid nscf save, computes the coarse
coupling from the ph.x outputs (or reads EPW's .epb files) and interpolates the electrons, the
phonons and the vertex g(R_e, R_p) to the dense grids in the PAO gauge.  The coarse vertex is
cached in the cache directory of system.py and shared by every workflow.

Outputs (output/): alpha2F.dat, eliashberg.npz [alpha2F_tr.dat, fs_coupling.npz].
"""

import argparse
import os

from PAOFLOW.elphon.workflows import eliashberg, load_system, smearing_label, write_eliashberg

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = CONFIG.SUPERCONDUCTIVITY


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--nk', type=int, default=DEFAULTS['nk'], help='dense k-grid (default %(default)s)'
    )
    parser.add_argument(
        '--nq', type=int, default=DEFAULTS['nq'], help='dense q-grid (default %(default)s)'
    )
    parser.add_argument('--coarse-q', action='store_true', help='keep q on the coarse grid')
    parser.add_argument(
        '--sigma-ev', type=float, default=DEFAULTS.get('sigma_ev', 0.05),
        help='Fermi-surface smearing in eV (the floor with --adaptive-smearing)',
    )  # fmt: skip
    parser.add_argument(
        '--adaptive-smearing', type=float, default=DEFAULTS.get('adaptive_smearing'), metavar='A',
        help='adaptive Fermi-surface smearing prefactor a (Yates; 0.7 for Gaussians): '
             'width a |v_nk| dk per dense state, floored at --sigma-ev',
    )  # fmt: skip
    parser.add_argument('--mu-star', type=float, default=DEFAULTS.get('mu_star', 0.1))
    parser.add_argument(
        '--transport', action=argparse.BooleanOptionalAction, default=DEFAULTS.get('transport', False),
        help='also alpha^2F_tr, lambda_tr and the plasma frequency (transport/resistivity.py)',
    )  # fmt: skip
    parser.add_argument(
        '--fs-coupling', action=argparse.BooleanOptionalAction,
        default=DEFAULTS.get('fs_coupling', False),
        help='store the Fermi-surface pair coupling for me_aniso.py (dense q, nk == nq)',
    )  # fmt: skip
    parser.add_argument(
        '--fsthick-ev', type=float, default=DEFAULTS.get('fsthick_ev'),
        help='Fermi window of the Fermi-surface states (default 4 sigma)',
    )  # fmt: skip
    args = parser.parse_args()
    SYSTEM.check_inputs()
    electrons = SYSTEM.electrons(outputdir=OUTPUT)
    fs_coupling = args.fs_coupling and not args.coarse_q  # needs the dense-q loop
    out = eliashberg(
        SYSTEM, electrons, nk=args.nk, nq=args.nq, sigma_ev=args.sigma_ev, mu_star=args.mu_star,
        coarse_q=args.coarse_q, transport=args.transport, fs_coupling=fs_coupling,
        fsthick_ev=args.fsthick_ev, adaptive_smearing=args.adaptive_smearing,
    )  # fmt: skip
    grids = 'k %d^3, q %s' % (
        args.nk, 'coarse %s' % 'x'.join(map(str, SYSTEM.qgrid)) if args.coarse_q else '%d^3' % args.nq,
    )  # fmt: skip
    label = '%s, %s' % (grids, smearing_label(args.sigma_ev, args.adaptive_smearing))
    write_eliashberg(out, OUTPUT, mu_star=args.mu_star, label=label)


if __name__ == '__main__':
    main()
