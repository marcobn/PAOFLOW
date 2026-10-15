#!/usr/bin/env python3
"""Set the coarse k-mesh of the EPW run (nscf k-points list, epw.in, write_ukk.py, drivers).

    python kmesh.py NK1 NK2 NK3 [--no-drivers]

e.g. ``python kmesh.py 8 8 8``.  EPW needs the nscf on the full Gamma-centred
grid as an explicit list (``K_POINTS crystal``, all NK1*NK2*NK3 points, the
order of ``K_POINTS automatic``), and every file that refers to the grid has to
agree with it.  This script rewrites, in place:

  nscf.in       the K_POINTS crystal card (NK1*NK2*NK3 points, equal weights);
  epw.in        nk1, nk2, nk3;
  write_ukk.py  the number of k-points of the placeholder Si2.ukk;
  ../main.py, ../mobility.py
                COARSE_GRID, the real-space electron grid of PAOFLOW's
                electron-phonon vertex (skip with --no-drivers).

The q-grid (nq1..nq3 of epw.in, the ph.x grid of ../phonon/ph.in) must divide
the k-grid in every direction, so that k+q stays on the grid; the script stops
otherwise.  Afterwards rerun, in this directory, the nscf, write_ukk.py and
epw.x (see ../README.md).
"""

import argparse
import os
import re
import sys

from PAOFLOW.gen.epw_inputs import edit_pw_input, kpoints_card

HERE = os.path.dirname(os.path.abspath(__file__))
NSCF = os.path.join(HERE, 'nscf.in')
EPW = os.path.join(HERE, 'epw.in')
UKK_SCRIPT = os.path.join(HERE, 'write_ukk.py')
DRIVERS = [os.path.join(HERE, '..', name) for name in ('main.py', 'mobility.py')]


def epw_qgrid(text):
    """nq1, nq2, nq3 of epw.in."""
    values = {}
    for name in ('nq1', 'nq2', 'nq3'):
        match = re.search(r'\b%s\s*=\s*(\d+)' % name, text, flags=re.IGNORECASE)
        if match is None:
            sys.exit('%s not found in %s' % (name, EPW))
        values[name] = int(match.group(1))
    return values['nq1'], values['nq2'], values['nq3']


def replace_once(text, pattern, replacement, path):
    """Substitute one regex match, or stop if the pattern is missing."""
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count == 0:
        sys.exit('pattern %r not found in %s' % (pattern, path))
    return new_text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('nk', type=int, nargs=3, metavar=('NK1', 'NK2', 'NK3'), help='k-mesh')
    parser.add_argument('--no-drivers', action='store_true',
                        help='do not update COARSE_GRID in ../main.py and ../mobility.py')
    args = parser.parse_args()
    nk = tuple(args.nk)
    if min(nk) < 1:
        sys.exit('the mesh must be positive, got %s' % (nk,))

    with open(EPW) as handle:
        epw_text = handle.read()
    nq = epw_qgrid(epw_text)
    if any(k % q for k, q in zip(nk, nq)):
        sys.exit('the q-grid %s of epw.in does not divide the k-grid %s: k+q would leave the grid.'
                 % ('x'.join(map(str, nq)), 'x'.join(map(str, nk))))

    with open(NSCF) as handle:
        nscf_text = handle.read()
    with open(NSCF, 'w') as handle:
        handle.write(edit_pw_input(nscf_text, kpoints=kpoints_card(nk)))
    with open(EPW, 'w') as handle:
        handle.write(edit_pw_input(
            epw_text, {'inputepw': {'nk1': str(nk[0]), 'nk2': str(nk[1]), 'nk3': str(nk[2])}}
        ))  # fmt: skip
    with open(UKK_SCRIPT) as handle:
        ukk_text = handle.read()
    ukk_text = replace_once(ukk_text, r'nk_total=[^,)]+', 'nk_total=%d * %d * %d' % nk, UKK_SCRIPT)
    with open(UKK_SCRIPT, 'w') as handle:
        handle.write(ukk_text)
    edited = [NSCF, EPW, UKK_SCRIPT]

    if not args.no_drivers:
        for path in DRIVERS:
            if not os.path.isfile(path):
                continue
            with open(path) as handle:
                text = handle.read()
            text = replace_once(text, r'^COARSE_GRID = \([^)]*\)', 'COARSE_GRID = %r' % (nk,), path)
            with open(path, 'w') as handle:
                handle.write(text)
            edited.append(os.path.normpath(path))

    print('Coarse k-mesh %s (%d k-points; q-grid %s):'
          % ('x'.join(map(str, nk)), nk[0] * nk[1] * nk[2], 'x'.join(map(str, nq))))
    for path in edited:
        print('  updated %s' % path)
    print('Now rerun here: pw.x -in nscf.in; python write_ukk.py; epw.x -in epw.in')


if __name__ == '__main__':
    main()
