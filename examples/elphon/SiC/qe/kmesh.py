#!/usr/bin/env python3
"""Set the coarse k-mesh (nscf K_POINTS card, epw.in, write_ukk.py and system.py).

    python kmesh.py NK1 NK2 NK3 [--explicit]

PAOFLOW's native coupling reads the nscf on the irreducible wedge of the Gamma-centred grid
(K_POINTS automatic, the default) or on the full grid as an explicit list; EPW needs the
explicit list (K_POINTS crystal, all NK1*NK2*NK3 points, --explicit).  Every file that refers
to the grid has to agree with it.  This script rewrites, in place: nscf/nscf.in (the K_POINTS
card), nscf/epw.in (nk1..3) and nscf/write_ukk.py (the number of k-points) when they exist,
and coarse_grid of ../system.py.  The q-grid of phonon/ph.in must divide the k-grid in every
direction (k+q stays on the grid).  Afterwards rerun the nscf (and, for EPW, write_ukk.py and
epw.x).
"""

import argparse
import os
import re
import sys

from PAOFLOW.gen.epw_inputs import edit_pw_input, kpoints_card

HERE = os.path.dirname(os.path.abspath(__file__))
NSCF = os.path.join(HERE, 'nscf', 'nscf.in')
EPW = os.path.join(HERE, 'nscf', 'epw.in')
UKK_SCRIPT = os.path.join(HERE, 'nscf', 'write_ukk.py')
PH = os.path.join(HERE, 'phonon', 'ph.in')
SYSTEM = os.path.join(os.path.dirname(HERE), 'system.py')


def ph_qgrid(text):
    """nq1, nq2, nq3 of ph.in."""
    values = []
    for name in ('nq1', 'nq2', 'nq3'):
        match = re.search(r'\b%s\s*=\s*(\d+)' % name, text, flags=re.IGNORECASE)
        if match is None:
            sys.exit('%s not found in %s' % (name, PH))
        values.append(int(match.group(1)))
    return tuple(values)


def replace_once(text, pattern, replacement, path):
    """Substitute one regex match, or stop if the pattern is missing."""
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count == 0:
        sys.exit('pattern %r not found in %s' % (pattern, path))
    return new_text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('nk', type=int, nargs=3, metavar=('NK1', 'NK2', 'NK3'), help='k-mesh')
    parser.add_argument(
        '--explicit', action='store_true',
        help='write the explicit K_POINTS crystal list (required by EPW) instead of '
             'K_POINTS automatic (irreducible wedge)',
    )  # fmt: skip
    args = parser.parse_args()
    nk = tuple(args.nk)
    if min(nk) < 1:
        sys.exit('the mesh must be positive, got %s' % (nk,))
    with open(PH) as handle:
        nq = ph_qgrid(handle.read())
    if any(k % q for k, q in zip(nk, nq)):
        sys.exit(
            'the q-grid %s of ph.in does not divide the k-grid %s: k+q would leave the grid.'
            % ('x'.join(map(str, nq)), 'x'.join(map(str, nk)))
        )
    edited = []
    with open(NSCF) as handle:
        nscf_text = handle.read()
    card = kpoints_card(nk) if args.explicit else 'K_POINTS automatic\n  %d %d %d 0 0 0\n' % nk
    with open(NSCF, 'w') as handle:
        handle.write(edit_pw_input(nscf_text, kpoints=card))
    edited.append(NSCF)
    if os.path.isfile(EPW):
        with open(EPW) as handle:
            epw_text = handle.read()
        with open(EPW, 'w') as handle:
            handle.write(edit_pw_input(
                epw_text, {'inputepw': {'nk1': str(nk[0]), 'nk2': str(nk[1]), 'nk3': str(nk[2])}}
            ))  # fmt: skip
        edited.append(EPW)
    if os.path.isfile(UKK_SCRIPT):
        with open(UKK_SCRIPT) as handle:
            ukk_text = handle.read()
        ukk_text = replace_once(
            ukk_text, r'nk_total=[^,)]+', 'nk_total=%d * %d * %d' % nk, UKK_SCRIPT
        )
        with open(UKK_SCRIPT, 'w') as handle:
            handle.write(ukk_text)
        edited.append(UKK_SCRIPT)
    with open(SYSTEM) as handle:
        text = handle.read()
    text = replace_once(text, r'coarse_grid=\([^)]*\)', 'coarse_grid=%r' % (nk,), SYSTEM)
    with open(SYSTEM, 'w') as handle:
        handle.write(text)
    edited.append(SYSTEM)
    print(
        'Coarse k-mesh %s (%d k-points; q-grid %s):'
        % ('x'.join(map(str, nk)), nk[0] * nk[1] * nk[2], 'x'.join(map(str, nq)))
    )
    for path in edited:
        print('  updated %s' % path)
    if args.explicit:
        print('Now rerun in nscf/: pw.x -in nscf.in; python write_ukk.py; epw.x -in epw.in')
    else:
        print('Now rerun in nscf/: pw.x -in nscf.in (irreducible wedge; EPW needs --explicit)')


if __name__ == '__main__':
    main()
