#!/usr/bin/env python3
"""Pb superconductivity with PAOFLOW on EPW's coarse electron-phonon coupling.

The QE/EPW steps follow EPW tutorial 04 (see README.md).  This script then
builds the PAO electronic structure on the EPW nscf save and interpolates EPW's
coarse coupling (``epw/pb.epb*``) to dense k and q grids in the PAO gauge:

    mpirun -np 8 python main.py                 # dense k and q (default)
    mpirun -np 8 python main.py --coarse-q      # dense k only, coarse 6^3 q

Reference (EPW itself, 48^3 k / 24^3 q): lambda = 1.1515, w_log = 4.44 meV,
Tc(Allen-Dynes) = 4.76 K.
"""

import argparse
import os

import numpy as np
from mpi4py import MPI

from PAOFLOW import PAOFLOW
from PAOFLOW.elphon.do_pao_eph import eliashberg_from_qe_coupling
from PAOFLOW.elphon.do_pao_eph_dense_q import eliashberg_dense_q
from PAOFLOW.elphon.elph_bloch import RY_TO_EV, read_nscf

# ----------------------------------------------------------------------- #
# Configuration                                                           #
# ----------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
EPW_DIR = os.path.join(HERE, 'epw')  # EPW outdir: pb.epb*, pb.ukk, pb.save
SAVEDIR = os.path.join(EPW_DIR, 'pb.save')  # the nscf save EPW read
BASISDIR = os.path.join(HERE, 'BASIS_PS')  # paoflow-genbasis-ps --pseudo Pb.upf --out BASIS_PS
OUTPUTDIR = os.path.join(HERE, 'output')

COARSE_GRID = (6, 6, 6)  # nscf k-grid == ph.x/EPW q-grid
MASSES_AMU = [207.2]
NELEC = 14  # valence electrons of Pb.upf (5d10 6s2 6p2)
PTHR = 0.95  # projectability threshold
NK_DENSE = 48  # dense electron grid (EPW: nkf = 48)
NQ_DENSE = 24  # dense phonon grid (EPW: nqf = 24); NK_DENSE % NQ_DENSE == 0
SIGMA_EV = 0.05  # Fermi-surface smearing (EPW: degaussw)
MU_STAR = 0.1

KB_EV = 8.617333262e-5


def pao_electronic_structure():
    """PAO projections and Hamiltonian on the EPW nscf save."""
    pf = PAOFLOW.PAOFLOW(
        workpath=HERE, outputdir=OUTPUTDIR, savedir=SAVEDIR, save_overlaps=False, verbose=False
    )
    pf.projections(basispath=BASISDIR, configuration='standard')
    pf.projectability(pthr=PTHR)
    # The projections A_k = <phi|psi_k> rotate EPW's matrix elements to the PAO
    # gauge; pao_hamiltonian deletes them, so keep a copy first.
    projections = pf.data_controller.full_projections()[:, :, :, 0].copy()
    # The explicit K_POINTS list already spans the whole Brillouin zone.
    pf.pao_hamiltonian(expand_wedge=False)
    return projections, pf.data_controller.data_arrays['HRs'], read_nscf(SAVEDIR)


def eliashberg(projections, HRs, nscf, nk_dense, nq_dense, sigma_ev, coarse_q):
    """Isotropic Eliashberg properties from EPW's coarse coupling."""
    common = dict(
        source='epw',
        masses_amu=MASSES_AMU,
        nk_dense=nk_dense,
        sigmas_ry=[sigma_ev / RY_TO_EV],
        nelec=NELEC,
        mu_star=MU_STAR,
    )
    lattice = (nscf['kpts_cryst'], nscf['bg'], nscf['at'])
    if coarse_q:
        # q stays on EPW's coarse grid; q-points and phonons come from the .epb files.
        return eliashberg_from_qe_coupling(
            projections, HRs, *lattice, EPW_DIR, None, COARSE_GRID, None, **common
        )
    # Both k and q interpolated; the dense q-grid is folded to its irreducible wedge.
    return eliashberg_dense_q(
        projections, HRs, *lattice, EPW_DIR, COARSE_GRID, None, None, COARSE_GRID, None,
        nq_dense=nq_dense, sym_rots=nscf['s_cryst'], tau_cryst=nscf['tau_cryst'],
        species=nscf['atom_names'], **common,
    )


def report(out, label):
    """Print the Eliashberg summary and write alpha2F.dat / eliashberg.npz."""
    print('PAOFLOW on EPW coupling (%s):' % label)
    print('  lambda   = %.4f' % out['lambda'])
    print('  w_log    = %.3f meV' % (out['omega_log'] * 1.0e3))
    print('  Tc (McM) = %.2f K   Tc (AD) = %.2f K   (mu* = %.2f)'
          % (out['Tc_mcmillan'], out['Tc_allen_dynes'], MU_STAR))
    os.makedirs(OUTPUTDIR, exist_ok=True)
    np.savetxt(
        os.path.join(OUTPUTDIR, 'alpha2F.dat'),
        np.column_stack([out['omega'] * 1.0e3, out['a2F']]),
        header='omega(meV)  alpha^2F  (lambda=%.4f, w_log=%.2fK, Tc_McM=%.3fK, Tc_AD=%.3fK, mu*=%.2f)'
        % (out['lambda'], out['omega_log'] / KB_EV, out['Tc_mcmillan'], out['Tc_allen_dynes'], MU_STAR),
    )
    np.savez(os.path.join(OUTPUTDIR, 'eliashberg.npz'), **out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--coarse-q', action='store_true', help='keep q on the coarse EPW grid')
    parser.add_argument('--nk', type=int, default=NK_DENSE, help='dense k-grid (default %(default)s)')
    parser.add_argument('--nq', type=int, default=NQ_DENSE, help='dense q-grid (default %(default)s)')
    parser.add_argument('--sigma-ev', type=float, default=SIGMA_EV, help='smearing in eV')
    args = parser.parse_args()

    projections, HRs, nscf = pao_electronic_structure()
    out = eliashberg(projections, HRs, nscf, args.nk, args.nq, args.sigma_ev, args.coarse_q)

    if MPI.COMM_WORLD.Get_rank() == 0:
        grids = 'k %d^3, q %s' % (args.nk, 'coarse 6^3' if args.coarse_q else '%d^3' % args.nq)
        report(out, '%s, sigma %.3f eV' % (grids, args.sigma_ev))


if __name__ == '__main__':
    main()
