#!/usr/bin/env python3
"""Recompute the thermal emissivity of a finished absorption run (no new kernel run).

    python emissivity.py --direct-dielectric ../qe/optics/output --direct-scissor thermal \\
        --outdir output_thermal [--source output] [--thickness-um 500] [--target-gap-ev 1.17]

Reads <source>/absorption.npz (written by absorption.py), assembles the dielectric function
with the external direct one and writes the emissivity files into --outdir, together with a
copy of absorption.npz so that the directory is a complete model folder (e.g. for the
thermo_optics analysis).  The source directory is never modified.

--direct-scissor sets the shift of the external direct dielectric function: same (the run's
scissor, as absorption.py does by default), thermal (only its temperature-dependent part, for
an optics run whose 0 K direct gap is already right, e.g. eACBN0) or none.  thermal needs the
0 K scissor: runs since October 2026 store it (scissor_0k_ev); for older ones give the 0 K
target gap with --target-gap-ev.
"""

import argparse
import os

from PAOFLOW.elphon.workflows import emissivity_from_absorption, load_system

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
DEFAULTS = CONFIG.OPTICS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', default=OUTPUT, help='output directory of absorption.py')
    parser.add_argument(
        '--direct-dielectric', required=True,
        help='output dir of the dielectric_tensor run (epsr/epsi_*.dat)',
    )  # fmt: skip
    parser.add_argument('--direct-scissor', choices=('same', 'thermal', 'none'), default='thermal')
    parser.add_argument(
        '--target-gap-ev', type=float, default=None,
        help='0 K target gap of the run scissor (older runs without scissor_0k_ev)',
    )  # fmt: skip
    parser.add_argument(
        '--thickness-um', type=float, default=DEFAULTS.get('thickness_um', 500.0),
        help='slab thickness (um)',
    )  # fmt: skip
    parser.add_argument(
        '--angles', type=float, nargs='+', default=DEFAULTS.get('angles', (0.0, 30.0, 60.0)),
        help='emission angles (deg)',
    )  # fmt: skip
    parser.add_argument('--outdir', required=True, help='new directory for the emissivity files')
    args = parser.parse_args()
    emissivity_from_absorption(
        args.source, args.outdir, direct_dielectric_dir=args.direct_dielectric,
        direct_scissor=args.direct_scissor, target_gap_ev=args.target_gap_ev,
        thickness_um=args.thickness_um, angles_deg=args.angles,
    )  # fmt: skip


if __name__ == '__main__':
    main()
