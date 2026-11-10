#!/usr/bin/env python3
"""Finite-difference coupling: the displaced supercells of the crystal (coupling_source 'fd').

    python finite_differences.py --write     # fd/: supercell.in, disp-NNN.in, displacements.json, run_fd.sh
    (cd fd && NP=8 bash run_fd.sh)           # pw.x scf of the reference and every displaced cell
    python finite_differences.py --status    # how many of those runs have finished

The ideal supercell of FINITE_DIFFERENCES (system.py) is the unit cell of the finite
differences.  --write reads the scf save of the primitive cell (phonon/<prefix>.save, written by
run_qe.sh) and writes the reference supercell and its displaced copies: every atom along +-x,
+-y, +-z ('cartesian', 6 N cells) or random all-atom patterns and their mirrors ('random',
2 nconfig cells).  The Gamma phonons of the supercell are the primitive phonons on the
commensurate q-grid; they come from the forces of the same runs.

After the runs, SYSTEM (coupling_source='fd') reads the PAO Hamiltonian of every displaced cell
in the basis of the reference one: superconductivity/eliashberg.py (on one MPI rank) gives
alpha^2F, lambda and Tc, and with --transport the input of transport/resistivity.py.  Any
pseudopotential, DFT+U or hybrid functional that pw.x runs works here.

The supercell, the k-grid (SYSTEM.coarse_grid) and nelec of SYSTEM go together: edit them
together in system.py.
"""

import argparse
import os

from PAOFLOW.elphon.workflows import load_system
from PAOFLOW.elphon.workflows.finite_differences import (
    fd_runs_done,
    write_supercell_displacements,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
SETTINGS = CONFIG.FINITE_DIFFERENCES


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--write', action='store_true',
        help='pw.x inputs of the reference and displaced supercells, and run_fd.sh',
    )  # fmt: skip
    parser.add_argument('--status', action='store_true', help='finished pw.x runs')
    parser.add_argument(
        '--mode', choices=('cartesian', 'random'), default=SETTINGS['mode'],
        help="'cartesian': every atom along +-x, +-y, +-z; 'random': all-atom patterns",
    )  # fmt: skip
    parser.add_argument(
        '--distance', type=float, default=SETTINGS['distance'],
        help='displacement amplitude (bohr, default %(default)s)',
    )  # fmt: skip
    parser.add_argument(
        '--nconfig', type=int, default=SETTINGS.get('nconfig'),
        help="random configurations ('random' mode; default 3 N)",
    )  # fmt: skip
    parser.add_argument('--seed', type=int, default=SETTINGS.get('seed', 0))
    args = parser.parse_args()
    fd_dir = SYSTEM.coupling_dir
    if args.write:
        save = SYSTEM.path(SETTINGS['primitive_save'])
        if not os.path.isdir(save):
            raise SystemExit('%s not found: run the scf of run_qe.sh first.' % save)
        result = write_supercell_displacements(
            save, fd_dir, SETTINGS['supercell'], prefix=SETTINGS['prefix'],
            kgrid=SYSTEM.coarse_grid, distance=args.distance, mode=args.mode,
            nconfig=args.nconfig, seed=args.seed, nbnd=SETTINGS.get('nbnd'),
            pp_dir=SETTINGS.get('pseudo_dir'), configuration=SYSTEM.configuration,
        )  # fmt: skip
        print(
            'Wrote the reference supercell (%d atoms) and %d displaced cells (%s) in %s.'
            % (result['atoms'], result['cells'], result['mode'], fd_dir)
        )
        print('Next: (cd %s && NP=8 bash run_fd.sh)' % os.path.relpath(fd_dir))
    if args.status or not args.write:
        done, total = fd_runs_done(fd_dir)
        if not total:
            print('No displaced cells in %s yet: python finite_differences.py --write' % fd_dir)
        else:
            print('%d of %d pw.x runs finished in %s' % (done, total, fd_dir))


if __name__ == '__main__':
    # An error on one MPI rank stops every rank instead of hanging the job.
    from PAOFLOW.utils.mpi_guard import run_main

    run_main(main)
