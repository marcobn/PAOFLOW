#!/usr/bin/env python3
"""Plot the Migdal-Eliashberg results of me.py (output/me/migdal_eliashberg.npz).

python plot_me.py                  # up to four gapped temperatures in the frequency panels
python plot_me.py --temps 5 15 19  # chosen temperatures
"""

import argparse
import os

from PAOFLOW import GPAO

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIR = os.path.join(HERE, 'output', 'me')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--temps', type=float, nargs='+', help='temperatures (K) to show')
    args = parser.parse_args()
    npz = os.path.join(MEDIR, 'migdal_eliashberg.npz')
    if not os.path.isfile(npz):
        raise SystemExit('%s not found; run me.py first.' % npz)
    GPAO.GPAO().plot_migdal_eliashberg(
        npz,
        temps=args.temps,
        title='MgB2: isotropic Migdal-Eliashberg',
        filename=os.path.join(MEDIR, 'migdal_eliashberg.png'),
        real_axis_max_mev=150.0,  # phonons reach 100 meV
    )
