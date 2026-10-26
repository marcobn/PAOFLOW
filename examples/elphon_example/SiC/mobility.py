#!/usr/bin/env python3
"""Phonon-limited carrier mobility of 3C-SiC with the Fröhlich (polar) coupling (SERTA / MRTA).

    mpirun -np N python mobility.py [--nk 24 --nq 24] [--temps 200 300 400] [--window-ev 0.3]

State-resolved scattering rates of the electrons near the conduction-band minimum (X valley)
and of the holes near Gamma on dense k and q grids, from the coarse DFPT coupling of
phonon/ (computed by PAOFLOW, no EPW) with the long-range dipole vertex subtracted on the
coarse grid and restored at every dense q (LongRangeModel.from_coupling), and the dipole
force constants likewise (LO-TO splitting).  The polar LO scattering dominates the electron
mobility of SiC at room temperature.  Outputs go to output/mobility (mobility.dat,
scattering_rates.npz: see examples/elphon_example/Si/mobility.py for the full option set).
"""

import argparse
import os

import numpy as np

from PAOFLOW.elphon.carrier_scattering import (
    carrier_scattering_dense_q,
    rta_transport,
    write_mobility_outputs,
)
from PAOFLOW.elphon.elph_bloch import atom_masses
from PAOFLOW.elphon.long_range import LongRangeModel
from PAOFLOW.elphon.pao_electrons import pao_hamiltonian_from_save
from PAOFLOW.elphon.wavefunction_symmetry import stores_irreducible_wedge

HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(HERE, 'nscf', 'sic.save')
PHONON_DIR = os.path.join(HERE, 'phonon')
BASISDIR = os.path.join(HERE, 'BASIS_PS')
OUTPUTDIR = os.path.join(HERE, 'output')
COARSE_GRID = (6, 6, 6)
QGRID = (6, 6, 6)
MASSES_AMU = [28.0855, 12.0107]  # mass of each species (amu), ATOMIC_SPECIES order
NELEC = 8
PTHR = 0.90
NK_DENSE = 36  # the X-valley electrons need fine grids: 24^3 is far from converged, 48^3 and up for numbers
NQ_DENSE = 36
TEMPS_K = [200.0, 300.0, 400.0]
WINDOW_EV = 0.3
SMEARING_EV = 0.01
LONG_RANGE = True  # subtract and restore the dipole (Fröhlich) term; False reproduces the plain interpolation


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--nk', type=int, default=NK_DENSE)
    parser.add_argument('--nq', type=int, default=NQ_DENSE)
    parser.add_argument('--temps', type=float, nargs='+', default=TEMPS_K)
    parser.add_argument('--doping', type=float, default=0.0, help='net doping (cm^-3)')
    parser.add_argument('--window-ev', type=float, default=WINDOW_EV)
    parser.add_argument('--smearing-ev', type=float, default=SMEARING_EV,
                        help='delta smearing (eV); the floor of the widths with --adaptive-smearing')
    parser.add_argument('--adaptive-smearing', type=float, default=None, metavar='A',
                        help='adaptive smearing prefactor a (Yates; 0.7 for Gaussians): width a |v| dq per final state, floored at --smearing-ev; default fixed smearing')
    parser.add_argument('--no-long-range', action='store_true', help='plain interpolation (for comparison)')
    args = parser.parse_args()
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    el = pao_hamiltonian_from_save(
        SAVEDIR, BASISDIR, 'standard', PTHR, expand_wedge=False,
        unfold_kpoints=stores_irreducible_wedge(SAVEDIR), workpath=HERE, outputdir='output',
    )  # fmt: skip
    nscf = el.nscf
    use_model = LONG_RANGE and not args.no_long_range
    model = LongRangeModel.from_coupling(PHONON_DIR, SAVEDIR) if use_model else None
    out = carrier_scattering_dense_q(
        el.projections, el.HRs, nscf['kpts_cryst'], nscf['bg'], nscf['at'], nscf['alat'], nscf['omega'],
        PHONON_DIR, QGRID, COARSE_GRID,
        coupling_source='dfpt', save_dir=SAVEDIR, long_range=model, allow_missing_long_range=not use_model,
        masses_amu=atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names']), nelec=NELEC,
        nk_dense=args.nk, nq_dense=args.nq, temps_k=args.temps, doping_cm3=args.doping,
        window_ev=args.window_ev, smearing_ev=args.smearing_ev,
        adaptive_smearing=args.adaptive_smearing, min_smearing_ev=args.smearing_ev,
        sym_rots=nscf['s_cryst'], tau_cryst=nscf['tau_cryst'], species=nscf['atom_names'],
        orbital_positions=nscf['orbital_positions'],
    )  # fmt: skip
    if rank == 0:
        tag = 'mobility' if use_model else 'mobility_plain'
        if args.adaptive_smearing is not None:
            tag += '_adaptive'
        smearing = ('%.3f eV' % args.smearing_ev if args.adaptive_smearing is None else
                    'adaptive a = %.2f (floor %.1f meV)' % (args.adaptive_smearing, 1e3 * args.smearing_ev))
        print('PAOFLOW carrier transport of 3C-SiC (k %d^3, q %d^3, window %.2f eV, smearing %s, %s):'
              % (args.nk, args.nq, args.window_ev, smearing,
                 'Fröhlich term subtracted and restored' if use_model else 'plain interpolation'))
        print('  gap %.3f eV (DFT), doping %.2e cm^-3' % (out['cbm_ev'] - out['vbm_ev'], args.doping))
        serta, mrta = rta_transport(out, 'serta'), rta_transport(out, 'mrta')
        print('    T (K)   mobility e / h (cm^2/Vs):  SERTA              MRTA')
        for it, temp in enumerate(out['temps_k']):
            print('  %7.1f   %28.1f / %-8.1f %8.1f / %-8.1f' % (
                temp, serta['mobility_e'][it], serta['mobility_h'][it], mrta['mobility_e'][it], mrta['mobility_h'][it]))
        for path in write_mobility_outputs(out, os.path.join(OUTPUTDIR, tag)):
            print('  wrote %s' % path)


if __name__ == '__main__':
    main()
