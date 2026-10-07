#!/usr/bin/env python3
"""MgB2 superconducting gaps and Tc from the anisotropic Migdal-Eliashberg equations.

Post-processing of the Fermi-surface coupling written by main.py
(output/fs_coupling.npz), with the Fermi-surface-restricted settings of the
MgB2 anisotropic exercise of EPW tutorial 04 (see README.md):

    python me_aniso.py
    python me_aniso.py --linear-temps 20 60 9    # also Tc from the linearised kernel
    python plot_me_aniso.py

For every temperature output/me_aniso/ receives, in EPW's formats,
mgb2.imag_aniso_<T> (Delta_nk and Z_nk on the Matsubara axis),
mgb2.imag_aniso_gap0_<T> and mgb2.pade_aniso_gap0_<T> (gap distributions),
mgb2.imag_aniso_gap_FS_<T> (gap of every Fermi-surface state),
mgb2.pade_aniso_<T> and mgb2.qdos_<T>.  mgb2.lambda_FS and mgb2.lambda_k_pairs
hold lambda_nk, gap_vs_T_aniso.dat the gap range versus T, and
migdal_eliashberg_aniso.npz all of it.

The sigma and pi Fermi sheets of MgB2 couple differently to the B bond-stretching
phonons, so the solution has two gaps (README.md: about 2 and 5-8 meV at 5 K)
and a Tc about twice the isotropic one.  Use equal dense grids in main.py
(--nk N --nq N): with a coarser q-grid the equations split into independent
sublattice problems.
"""

import argparse
import os

import numpy as np

from PAOFLOW.elphon.fermi_surface_coupling import read_fs_coupling
from PAOFLOW.elphon.anisotropic_eliashberg import (
    coupling_strength,
    linearized_eigenvalues_aniso,
    migdal_eliashberg_aniso,
    write_me_aniso_outputs,
)

# ----------------------------------------------------------------------- #
# Configuration (EPW tutorial 04, MgB2, anisotropic FSR)                  #
# ----------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
FS_COUPLING = os.path.join(HERE, 'output', 'fs_coupling.npz')  # written by main.py
MEDIR = os.path.join(HERE, 'output', 'me_aniso')
PREFIX = 'mgb2'

MU_STAR = 0.1  # Coulomb pseudopotential (EPW muc)
WSCUT = 0.5  # Matsubara cutoff in eV (EPW wscut)
NPADE = 90  # percentage of Matsubara points in the Pade approximants (EPW npade)
TEMPS = np.linspace(5.0, 45.0, 9)  # K (EPW temps = 5 45, nstemp = 9)


def main() -> None:
    """Solve the anisotropic Migdal-Eliashberg equations and write output/me_aniso/."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--temps', type=float, nargs='+', default=TEMPS, help='temperatures (K)')
    parser.add_argument(
        '--linear-temps',
        type=float,
        nargs=3,
        metavar=('TMIN', 'TMAX', 'NSTEMP'),
        help='also compute the linearised-kernel eigenvalue on these temperatures',
    )
    parser.add_argument('--mu-star', type=float, default=MU_STAR)
    parser.add_argument('--wscut', type=float, default=WSCUT, help='Matsubara cutoff (eV)')
    parser.add_argument(
        '--npade', type=float, default=NPADE, help='%% of Matsubara points for Pade'
    )
    parser.add_argument('--no-pade', action='store_true', help='skip the real axis')
    args = parser.parse_args()

    if not os.path.isfile(FS_COUPLING):
        raise SystemExit('%s not found; run main.py (without --iso-only) first.' % FS_COUPLING)
    coupling = read_fs_coupling(FS_COUPLING)
    strength = coupling_strength(coupling)
    print(
        'Fermi-surface coupling: %d irreducible states (k %d^3, q %d^3, fsthick %.2f eV)'
        % (coupling['band'].size, coupling['nk_dense'], coupling['nq_dense'],
           coupling['fsthick_ev'])
    )  # fmt: skip
    print(
        '  lambda = %.4f   lambda_nk = %.3f .. %.3f   N_F = %.4f states/eV/spin'
        % (strength['lambda'], strength['lambda_nk'].min(), strength['lambda_nk'].max(),
           strength['dos_ef'])
    )  # fmt: skip

    print('Anisotropic Migdal-Eliashberg equations (mu* = %.2f, wscut = %.2f eV):'
          % (args.mu_star, args.wscut))  # fmt: skip
    result = migdal_eliashberg_aniso(
        coupling,
        args.temps,
        args.mu_star,
        args.wscut,
        npade=args.npade,
        lpade=not args.no_pade,
        verbose=True,
    )
    linear = None
    if args.linear_temps is not None:
        tmin, tmax, nstemp = args.linear_temps
        linear = linearized_eigenvalues_aniso(
            coupling, np.linspace(tmin, tmax, int(nstemp)), args.mu_star, args.wscut
        )
    write_me_aniso_outputs(result, coupling, MEDIR, PREFIX, linear)

    print('Tc from Delta_max^2(T) -> 0    : %.2f K' % result['Tc_gap'])
    if linear is not None:
        print('Tc from the linearised kernel : %.2f K' % linear['Tc_linear'])
    if not result['converged'].all():
        print('WARNING: not converged at T =', result['temps'][~result['converged']])
    print('Wrote %s' % MEDIR)


if __name__ == '__main__':
    main()
