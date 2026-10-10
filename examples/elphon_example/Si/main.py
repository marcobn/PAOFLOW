#!/usr/bin/env python3
"""PAOFLOW phonon-assisted optical absorption on EPW's coarse coupling (examples/elphon_example/Si).

Second-order (photon + phonon) absorption of an indirect-gap semiconductor, as
EPW's ``lindabs`` (indabs.f90; Noffsinger et al., PRL 108, 167402 (2012); EPW
tutorial 06), with the coupling, the electrons, their velocity matrix elements
and the phonons all interpolated in the PAO basis.  The direct (vertical)
absorption is computed alongside.

The coarse electron-phonon matrix elements come from EPW (``epbwrite = .true.``),
exactly as for the Eliashberg workflow.  paoflow-gen wrote the QE/EPW inputs in
two run directories:

    phonon/scf.in, phonon/ph.in               pw.x scf + ph.x DFPT (irreducible q)
    epw/nscf.in, epw/epw.in, epw/write_ukk.py   pw.x nscf (full k list) + epw.x

Run them in this order:

    cd phonon
    mpirun -np N pw.x -in scf.in > scf.out
    mpirun -np N ph.x -in ph.in  > ph.out
    python3 /path/to/q-e/EPW/bin/pp.py      # collects dvscf/patterns/dyn into save/
    cd ../epw
    mkdir -p Si2.save          # the nscf starts from the scf charge density
    cp ../phonon/Si2.save/{charge-density.dat,data-file-schema.xml} Si2.save/
    mpirun -np N pw.x -in nscf.in > nscf.out
    python3 write_ukk.py                    # Si2.ukk (+ empty .bvec/.mmn stubs)
    mpirun -np N epw.x -nk N -in epw.in > epw.out   # writes Si2.epb*, one file per pool
    cd ..
    mpirun -np N python main.py      # PAO interpolation -> Im eps(omega), alpha(omega)
    python plot.py

Outputs (OUTPUTDIR), one set per temperature, in EPW's layout:
epsilon2_indabs_<T>K.dat and epsilon2_indabs_lorenz<T>K.dat (photon energy and
the phonon-assisted Im eps for EPW's nine intermediate-state broadenings eta),
epsilon2_dirabs_<T>K.dat (direct Im eps), alpha_<T>K.dat (absorption
coefficient, cm^-1) and absorption.npz.

The Fermi level sits mid-gap (EPW: efermi_read + fermi_energy) unless
FERMI_ENERGY_EV is set.  The band energies are the PAO (DFT) ones: the
absorption edge follows the DFT indirect gap.  NONLOCAL_VELOCITY adds the
non-local pseudopotential term of the velocity operator (norm-conserving
pseudopotentials only); it changes the magnitude of Im eps, not the edge.
"""

import argparse
import os
import sys

from mpi4py import MPI

from PAOFLOW import PAOFLOW
from PAOFLOW.elphon.elph_bloch import atom_masses, pao_orbital_positions, read_nscf
from PAOFLOW.elphon.phonon_assisted_absorption import (
    phonon_assisted_absorption_dense_q,
    write_absorption_outputs,
)

# ----------------------------------------------------------------------- #
# Configuration  (edit freely -- masses / NELEC are system-specific)      #
# ----------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = 'Si2'
EPW_DIR = os.path.join(HERE, 'epw')  # EPW outdir: <prefix>.epb*, <prefix>.ukk, <prefix>.save
SAVEDIR = os.path.join(HERE, 'epw/Si2.save')  # the nscf save EPW read
BASISDIR = os.path.join(HERE, 'BASIS_PS')  # paoflow-genbasis-ps --pseudo <upf> --out <BASISDIR>
OUTPUTDIR = os.path.join(HERE, 'output')

COARSE_GRID = (6, 6, 6)  # nscf k-grid (EPW nk1..3)
QGRID = (3, 3, 3)  # ph.x / EPW q-grid (nq1..3); must divide COARSE_GRID
MASSES_AMU = [28.085]  # mass of each species (amu), ATOMIC_SPECIES order (EPW amass)
NELEC = 8  # valence electrons (locates the gap)
PTHR = 0.95  # projectability threshold
NK_DENSE = 12  # dense electron grid (EPW nkf1..3)
NQ_DENSE = 6  # dense phonon grid (EPW nqf1..3); NK_DENSE % NQ_DENSE == 0
OMEGA_EV = (0.05, 3.0, 0.05)  # photon energies (min, max, step): EPW omegamin, omegamax, omegastep
TEMPS_K = [300.0]  # temperatures (EPW temps)
DEGAUSS_EV = 0.05  # energy-conserving delta smearing (EPW degaussw)
FSTHICK_EV = 4.0  # states within FSTHICK_EV of the Fermi level (EPW fsthick)
FERMI_ENERGY_EV = None  # Fermi level on the PAO energy scale; None = mid-gap
REFRACTIVE_INDEX = 3.4  # constant n_r of alpha = omega Im eps / (n_r c)
NONLOCAL_VELOCITY = True  # non-local PP velocity term (norm-conserving PPs)


def pao_electronic_structure():
    """PAO projections, Hamiltonian and orbital centres on the EPW nscf save."""
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
    nscf = read_nscf(SAVEDIR)
    # Orbital centres: pair-resolved Wigner-Seitz interpolation and the
    # intersite position term of the velocity operator.
    nscf['orbital_positions'] = pao_orbital_positions(pf.data_controller, nscf['at'])
    return projections, pf.data_controller.data_arrays['HRs'], nscf, pf.data_controller


def optical_absorption(projections, HRs, nscf, data_controller, args):
    """Phonon-assisted + direct Im eps(omega) and alpha(omega) on the dense grids."""
    return phonon_assisted_absorption_dense_q(
        projections, HRs, nscf['kpts_cryst'], nscf['bg'], nscf['at'], nscf['alat'],
        nscf['omega'], EPW_DIR, QGRID, COARSE_GRID,
        masses_amu=atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names']),
        nelec=NELEC, nk_dense=args.nk, nq_dense=args.nq,
        omega_ev=(OMEGA_EV[0], args.omega_max, OMEGA_EV[2]), temps_k=args.temps,
        degauss_ev=args.degauss_ev, fsthick_ev=FSTHICK_EV, fermi_energy_ev=FERMI_ENERGY_EV,
        refractive_index=REFRACTIVE_INDEX,
        nonlocal_velocity=data_controller if args.nonlocal_velocity else None,
        sym_rots=nscf['s_cryst'], tau_cryst=nscf['tau_cryst'], species=nscf['atom_names'],
        orbital_positions=nscf['orbital_positions'],
    )


def report(out, args):
    """Print the gaps and write the spectra to OUTPUTDIR."""
    print('PAOFLOW phonon-assisted absorption (k %d^3, q %d^3, degauss %.3f eV%s):'
          % (args.nk, args.nq, args.degauss_ev, ', non-local velocity' if args.nonlocal_velocity else ''))
    print('  indirect gap = %.3f eV   direct gap = %.3f eV   E_F = %.3f eV'
          % (out['indirect_gap_ev'], out['direct_gap_ev'], out['fermi_energy_ev']))
    for path in write_absorption_outputs(out, OUTPUTDIR):
        print('  wrote %s' % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--nk', type=int, default=NK_DENSE, help='dense k-grid (default %(default)s)')
    parser.add_argument('--nq', type=int, default=NQ_DENSE, help='dense q-grid (default %(default)s)')
    parser.add_argument('--temps', type=float, nargs='+', default=TEMPS_K, help='temperatures (K)')
    parser.add_argument('--degauss-ev', type=float, default=DEGAUSS_EV, help='delta smearing (eV)')
    parser.add_argument('--omega-max', type=float, default=OMEGA_EV[1], help='max photon energy (eV)')
    parser.add_argument('--nonlocal-velocity', action=argparse.BooleanOptionalAction,
                        default=NONLOCAL_VELOCITY, help='non-local PP velocity term')
    args = parser.parse_args()
    if args.nk % args.nq:
        sys.exit('--nk (%d) must be a multiple of --nq (%d).' % (args.nk, args.nq))
    if not os.path.isdir(SAVEDIR):
        sys.exit('%s not found. Run the pw.x nscf in epw/ first.' % SAVEDIR)
    if not any(f.startswith(PREFIX + '.epb') for f in os.listdir(EPW_DIR)):
        sys.exit('No %s.epb* files in %s. Run epw.x with epbwrite = .true. first.' % (PREFIX, EPW_DIR))

    projections, HRs, nscf, data_controller = pao_electronic_structure()
    out = optical_absorption(projections, HRs, nscf, data_controller, args)

    # The result is identical on every rank; only rank 0 reports and writes files.
    if MPI.COMM_WORLD.Get_rank() == 0:
        report(out, args)


if __name__ == '__main__':
    main()
