#!/usr/bin/env python3
"""Spin susceptibility of Pd from the QE fixed-spin-moment ladder (fsm/M*.out).

Run the ladder first (from fsm/):  for f in M*.in; do mpirun -np 4 pw.x -in $f > ${f%.in}.out; done
For every M > 0 the susceptibility chi = dM/dh (mu_B per eV per cell, h the half Zeeman splitting)
is printed from the energy, E(M) - E(0) = M^2 / (2 chi), and from the splitting of the spin-up and
spin-down Fermi levels, h = (mu_up - mu_down) / 2. Compare with output/spin_susceptibility.dat.
"""

import glob
import os
import re

RY_EV = 13.605693122994
HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    text = open(path).read()
    energy = float(re.findall(r'!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry', text)[-1]) * RY_EV
    up, down = map(float, re.findall(r'spin up/dw Fermi energies are\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', text)[-1])
    moment = float(re.search(r'tot_magnetization\s*=\s*(-?\d+\.?\d*)', open(path[:-4] + '.in').read()).group(1))
    return moment, energy, up, down


rows = sorted(read(path) for path in glob.glob(os.path.join(HERE, 'fsm', 'M*.out')))
if not rows or rows[0][0] != 0.0:
    raise SystemExit('run the fixed-spin-moment ladder first (fsm/M0.000.in is the reference)')
e0 = rows[0][1]
print('# M (mu_B)   chi from E(M) (mu_B/eV)   chi from the Fermi splitting (mu_B/eV)')
for moment, energy, up, down in rows[1:]:
    print(f'{moment:8.3f}   {moment**2 / (2.0 * (energy - e0)):10.3f}   {moment / (0.5 * (up - down)):10.3f}')
