#!/usr/bin/env python3
"""Plot the anisotropic Migdal-Eliashberg results of me_aniso.py.

python plot_me_aniso.py                  # up to four gapped temperatures in the T panels
python plot_me_aniso.py --temps 5 20 35  # chosen temperatures

The isotropic gap of me.py (output/me/migdal_eliashberg.npz) is overlaid when present.
"""

import argparse
import os

from PAOFLOW import GPAO

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIR = os.path.join(HERE, 'output', 'me_aniso')
ISO_NPZ = os.path.join(HERE, 'output', 'me', 'migdal_eliashberg.npz')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--temps', type=float, nargs='+', help='temperatures (K) to show')
    args = parser.parse_args()
    npz = os.path.join(MEDIR, 'migdal_eliashberg_aniso.npz')
    if not os.path.isfile(npz):
        raise SystemExit('%s not found; run me_aniso.py first.' % npz)
    GPAO.GPAO().plot_migdal_eliashberg_aniso(
        npz,
        temps=args.temps,
        iso_npz_file=ISO_NPZ if os.path.isfile(ISO_NPZ) else None,
        title='MgB2: anisotropic Migdal-Eliashberg',
        filename=os.path.join(MEDIR, 'migdal_eliashberg_aniso.png'),
    )
