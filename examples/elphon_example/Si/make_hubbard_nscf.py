#!/usr/bin/env python3
"""Write the pw.x nscf inputs that bring eACBN0 (DFT+U+V) electrons into the elphon workflow.

    python make_hubbard_nscf.py <eACBN0 scf.in> [--nk 6 6 6] [--nbnd 21] [--optics-nk 8 8 8]
                                [--optics-nbnd 60] [--out acbn0]

The eACBN0 loop (examples/acbn0_examples/Si, run on THIS pseudopotential and
lattice) leaves a pw.x input with the converged ``HUBBARD (ortho-atomic)`` card
and a save directory with the charge density and ``occup.txt``.  Neither ph.x
nor epw.x can use U+V or ortho-atomic projectors, so the coupling stays the PBE
one; PAOFLOW takes the band energies and eigenvectors from the eACBN0 PAO
Hamiltonian instead (``main.py --electrons-save``, ``mobility.py
--electrons-save``).  That needs two more pw.x nscf runs with the card:

  <out>/epw_nscf/nscf.in   the EPW k list (full grid, same NK as epw/nscf.in, NBND bands):
                           the PAO Hamiltonian that replaces the PBE one
  <out>/optics/nscf.in     an automatic grid with many bands for the extended-basis
                           dielectric_tensor run (optics_eps.py --savedir ...)

Both read the Hubbard scf save: copy charge-density.dat, data-file-schema.xml
AND occup.txt from the eACBN0 save into <out>/<run>/<prefix>.save/ (the script
prints the commands).  The nscf inputs repeat the HUBBARD card (QE reads U and V
from the input, the occupations from the save).
"""

import argparse
import os
import re

from PAOFLOW.gen.epw_inputs import hubbard_nscf_input, optics_nscf_input

HERE = os.path.dirname(os.path.abspath(__file__))


def prefix_of(text):
    """``prefix`` of a pw.x input (default 'pwscf')."""
    match = re.search(r"prefix\s*=\s*'([^']+)'", text)
    return match.group(1) if match else 'pwscf'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('template', help='pw.x input with the converged HUBBARD card')
    parser.add_argument('--nk', type=int, nargs=3, default=(6, 6, 6), metavar='N',
                        help='EPW coarse k-grid (COARSE_GRID of main.py)')
    parser.add_argument('--nbnd', type=int, default=21, help='bands of the EPW-grid nscf')
    parser.add_argument('--optics-nk', type=int, nargs=3, default=(8, 8, 8), metavar='N',
                        help='Monkhorst-Pack grid of the optics nscf')
    parser.add_argument('--optics-nbnd', type=int, default=60,
                        help='bands of the optics nscf (> orbitals of the extended basis)')
    parser.add_argument('--pseudo-dir', default='../../', help="pseudo_dir of the run directories")
    parser.add_argument('--out', default=os.path.join(HERE, 'acbn0'), help='parent of the run dirs')
    args = parser.parse_args()

    with open(args.template) as handle:
        template = handle.read()
    prefix = prefix_of(template)
    runs = {
        'epw_nscf': hubbard_nscf_input(template, prefix, tuple(args.nk), args.nbnd, args.pseudo_dir),
        'optics': optics_nscf_input(
            template, prefix, tuple(args.optics_nk), args.optics_nbnd, args.pseudo_dir
        ),
    }
    for run, text in runs.items():
        folder = os.path.join(args.out, run)
        os.makedirs(os.path.join(folder, prefix + '.save'), exist_ok=True)
        path = os.path.join(folder, 'nscf.in')
        with open(path, 'w') as handle:
            handle.write(text)
        print('wrote', path)
    save = os.path.dirname(os.path.abspath(args.template))
    print('\nBefore running them, copy the Hubbard scf save (occup.txt is required):')
    for run in runs:
        print('  cp <eACBN0 outdir>/%s.save/{charge-density.dat,data-file-schema.xml,occup.txt} %s/'
              % (prefix, os.path.join(args.out, run, prefix + '.save')))
    print('  (the eACBN0 template was %s)' % os.path.join(save, os.path.basename(args.template)))
    print('\nThen, e.g.:')
    print('  (cd %s/epw_nscf && mpirun -np 8 pw.x -in nscf.in > nscf.out)' % args.out)
    print('  (cd %s/optics && mpirun -np 8 pw.x -nk 8 -in nscf.in > nscf.out)' % args.out)
    print('  python optics_eps.py --savedir %s/optics/%s.save --outdir %s/optics/output'
          % (args.out, prefix, args.out))
    print('  mpirun -np 8 python main.py --electrons-save %s/epw_nscf/%s.save \\'
          % (args.out, prefix))
    print('      --emissivity --direct-dielectric %s/optics/output --varshni 4.73e-4 636' % args.out)


if __name__ == '__main__':
    main()
