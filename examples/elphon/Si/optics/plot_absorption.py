#!/usr/bin/env python3
"""Plot the phonon-assisted and direct optical absorption (and the slab emissivity).

    python plot_absorption.py [--eta 0.05] [--temp 300]

Reads output/absorption.npz written by absorption.py: Im eps(omega) (direct, phonon-assisted
and total) and the absorption coefficient alpha(omega) for every temperature, with the
indirect and direct gaps marked.  When EPW's own epsilon2_indabs_<T>K.dat exists in the nscf
directory (an EPW run with lindabs = .true.), it is overlaid (dashed) for comparison.  The
figure is saved as output/absorption.png.  After absorption.py --emissivity a second figure
shows the spectral and total hemispherical emissivity versus temperature
(output/emissivity.png).
"""

import argparse
import os

import numpy as np

from PAOFLOW import GPAO
from PAOFLOW.elphon.workflows import load_system

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
NPZ = os.path.join(OUTPUT, 'absorption.npz')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--eta', type=float, default=0.05, help='broadening eta (eV)')
    parser.add_argument('--temp', type=float, default=None, help='temperature (K) of Im eps')
    args = parser.parse_args()
    if not os.path.isfile(NPZ):
        raise SystemExit('%s not found; run absorption.py first.' % NPZ)
    with np.load(NPZ) as data:
        temps = np.atleast_1d(data['temps_k'])
    temp = temps[0] if args.temp is None else temps[np.argmin(np.abs(temps - args.temp))]
    epw_file = os.path.join(SYSTEM.nscf_path, 'epsilon2_indabs_%.1fK.dat' % temp)
    GPAO.GPAO().plot_phonon_assisted_absorption(
        NPZ, eta_ev=args.eta, temperature=temp,
        epw_indabs_file=epw_file if os.path.isfile(epw_file) else None,
        filename=os.path.join(OUTPUT, 'absorption.png'),
    )  # fmt: skip
    emissivity_npz = os.path.join(OUTPUT, 'emissivity.npz')
    if os.path.isfile(emissivity_npz):
        GPAO.GPAO().plot_thermal_emissivity(
            emissivity_npz, filename=os.path.join(OUTPUT, 'emissivity.png')
        )


if __name__ == '__main__':
    main()
