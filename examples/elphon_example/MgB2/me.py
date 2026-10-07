#!/usr/bin/env python3
"""MgB2 superconducting gap and Tc from the isotropic Migdal-Eliashberg equations.

Post-processing of the alpha^2F written by main.py (output/eliashberg.npz), with
the Eliashberg settings of the MgB2 exercise of EPW tutorial 04 (see README.md):

    python me.py                        # gap on the imaginary and real axes, Delta(T), Tc
    python me.py --a2f epw/mgb2.a2f     # the same on EPW's own alpha^2F
    python plot_me.py

For every temperature output/me/ receives, in EPW's formats, mgb2.imag_iso_<T>
(Matsubara axis), mgb2.pade_iso_<T> and mgb2.acon_iso_<T> (real axis from Pade
and from the analytic continuation) and mgb2.qdos_iso_<T> (quasiparticle DOS).
gap_vs_T.dat collects Delta(T) from the lowest Matsubara frequency and the two
real-axis gap edges, max_eigenvalue.dat the largest eigenvalue of the
linearised kernel (1 at Tc), and migdal_eliashberg.npz all of it.

The isotropic equations average the two MgB2 gaps (sigma and pi bands) into
one; me_aniso.py solves the anisotropic equations, whose Tc is about twice as high.
"""

import argparse
import os

import numpy as np

from PAOFLOW.elphon.migdal_eliashberg import (
    a2f_from_epw,
    a2f_from_npz,
    linearized_eigenvalues,
    matsubara_lambda,
    migdal_eliashberg_iso,
    write_me_outputs,
)

# ----------------------------------------------------------------------- #
# Configuration (EPW tutorial 04, MgB2)                                   #
# ----------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'output', 'eliashberg.npz')  # written by main.py
MEDIR = os.path.join(HERE, 'output', 'me')
PREFIX = 'mgb2'

MU_STAR = 0.1  # Coulomb pseudopotential (EPW muc)
WSCUT = 0.5  # Matsubara cutoff in eV (EPW wscut; phonons reach 100 meV)
DEGAUSSQ = 0.15  # phonon smearing of alpha^2F in meV for the ME equations (EPW degaussq)
DEGAUSSQ_LINEAR = 0.5  # ... for the linearised kernel, as EPW's tc_linear step
NPADE = 90  # percentage of Matsubara points in the Pade approximant (EPW npade)
TEMPS = [float(t) for t in range(5, 31)]  # K, 5-30 K in steps of 1 K
LINEAR_TEMPS = (5.0, 30.0, 26)  # Tmin, Tmax (K), nstemp of the linearised kernel


def main() -> None:
    """Solve the Migdal-Eliashberg equations on the alpha^2F and write output/me/."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--a2f', help="EPW '<prefix>.a2f' to use instead of output/eliashberg.npz")
    parser.add_argument('--temps', type=float, nargs='+', default=TEMPS, help='temperatures (K)')
    parser.add_argument(
        '--linear-temps',
        type=float,
        nargs=3,
        default=LINEAR_TEMPS,
        metavar=('TMIN', 'TMAX', 'NSTEMP'),
        help='linearised-kernel temperatures',
    )
    parser.add_argument('--mu-star', type=float, default=MU_STAR)
    parser.add_argument('--wscut', type=float, default=WSCUT, help='Matsubara cutoff (eV)')
    parser.add_argument('--degaussq', type=float, default=DEGAUSSQ, help='phonon smearing (meV)')
    parser.add_argument(
        '--degaussq-linear',
        type=float,
        default=DEGAUSSQ_LINEAR,
        help='phonon smearing for the linearised kernel (meV)',
    )
    parser.add_argument(
        '--npade', type=float, default=NPADE, help='%% of Matsubara points for Pade'
    )
    parser.add_argument('--no-pade', action='store_true', help='skip the Pade output')
    parser.add_argument('--no-acon', action='store_true', help='skip the analytic continuation')
    parser.add_argument('--no-linear', action='store_true', help='skip the linearised kernel')
    args = parser.parse_args()

    if args.a2f:  # EPW's alpha^2F is used as written (its own degaussq)
        omega, a2F = a2f_from_epw(args.a2f)
        omega_lin, a2F_lin = omega, a2F
        source = args.a2f
    else:
        if not os.path.isfile(NPZ):
            raise SystemExit('%s not found; run main.py first.' % NPZ)
        omega, a2F = a2f_from_npz(NPZ, args.degaussq * 1e-3)
        omega_lin, a2F_lin = a2f_from_npz(NPZ, args.degaussq_linear * 1e-3)
        source = NPZ
    lam = matsubara_lambda(omega, a2F, 1.0, 0)[0]  # 2 int a2F/w dw
    print(
        'alpha^2F from %s  (lambda = %.4f, mu* = %.2f, wscut = %.3f eV)'
        % (source, lam, args.mu_star, args.wscut)
    )

    print('Isotropic Migdal-Eliashberg equations:')
    res = migdal_eliashberg_iso(
        omega,
        a2F,
        args.temps,
        args.mu_star,
        args.wscut,
        npade=args.npade,
        lpade=not args.no_pade,
        lacon=not args.no_acon,
        verbose=True,
    )
    lin = None
    if not args.no_linear:
        tmin, tmax, nst = args.linear_temps
        lin = linearized_eigenvalues(
            omega_lin, a2F_lin, np.linspace(tmin, tmax, int(nst)), args.mu_star, args.wscut
        )
    write_me_outputs(res, MEDIR, PREFIX, lin)

    print('Tc from Delta^2(T) -> 0     : %.3f K' % res['Tc_gap'])
    if lin is not None:
        print('Tc from the linearised kernel: %.3f K' % lin['Tc_linear'])
    if not res['converged'].all():
        print('WARNING: not converged at T =', res['temps'][~res['converged']])
    print('Wrote %s' % MEDIR)


if __name__ == '__main__':
    main()
