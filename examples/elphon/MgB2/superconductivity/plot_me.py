#!/usr/bin/env python3
"""Plot the Migdal-Eliashberg results of me.py and, when present, me_aniso.py.

    python plot_me.py                  # up to four gapped temperatures in the frequency panels
    python plot_me.py --temps 0.3 4.5  # chosen temperatures

Figures: output/me/migdal_eliashberg.png and output/me_aniso/migdal_eliashberg_aniso.png (the
isotropic gap is overlaid on the anisotropic figure).
"""

import argparse
import os

from PAOFLOW import GPAO
from PAOFLOW.elphon.workflows import load_system

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
NAME = getattr(CONFIG, 'NAME', CONFIG.SYSTEM.prefix)
DEFAULTS = CONFIG.SUPERCONDUCTIVITY

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--temps', type=float, nargs='+', help='temperatures (K) to show')
    parser.add_argument(
        '--real-axis-max-mev', type=float, default=DEFAULTS.get('real_axis_max_mev'),
        help='upper limit of the real-axis panels (meV); default from the phonon range',
    )  # fmt: skip
    args = parser.parse_args()
    iso = os.path.join(OUTPUT, 'me', 'migdal_eliashberg.npz')
    aniso = os.path.join(OUTPUT, 'me_aniso', 'migdal_eliashberg_aniso.npz')
    if not (os.path.isfile(iso) or os.path.isfile(aniso)):
        raise SystemExit('%s not found; run me.py (or me_aniso.py) first.' % iso)
    plotter = GPAO.GPAO()
    if os.path.isfile(iso):
        options = {}
        if args.real_axis_max_mev is not None:
            options['real_axis_max_mev'] = args.real_axis_max_mev
        plotter.plot_migdal_eliashberg(
            iso, temps=args.temps, title='%s: isotropic Migdal-Eliashberg' % NAME,
            filename=os.path.join(os.path.dirname(iso), 'migdal_eliashberg.png'), **options,
        )  # fmt: skip
    if os.path.isfile(aniso):
        plotter.plot_migdal_eliashberg_aniso(
            aniso, temps=args.temps, iso_npz_file=iso if os.path.isfile(iso) else None,
            title='%s: anisotropic Migdal-Eliashberg' % NAME,
            filename=os.path.join(os.path.dirname(aniso), 'migdal_eliashberg_aniso.png'),
        )  # fmt: skip
