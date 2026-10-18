#!/usr/bin/env python3
"""Recompute the thermal emissivity of a finished absorption run (no new kernel run).

    python emissivity_from_absorption.py OUTPUT --direct-dielectric OPTICS_OUTPUT \\
        --direct-scissor thermal --outdir OUTPUT_THERMAL [--thickness-um 500] [--target-gap-ev 1.17]

Reads OUTPUT/absorption.npz (written by ``main.py``), assembles the dielectric
function with the external direct one (``thermal_emissivity``) and writes the
emissivity files into OUTDIR, together with a copy of ``absorption.npz`` so
that OUTDIR is a complete model folder (e.g. for the thermo_optics analysis).
OUTPUT itself is never modified.

``--direct-scissor`` sets the shift of the external direct dielectric function:
``same`` (the run's scissor, as ``main.py`` does by default), ``thermal`` (only
its temperature-dependent part, for an optics run whose 0 K direct gap is
already right, e.g. eACBN0) or ``none``.  ``thermal`` needs the 0 K scissor:
runs since October 2026 store it (``scissor_0k_ev``); for older ones give the
0 K target gap with ``--target-gap-ev`` (Delta(0) = target - Hamiltonian gap).
"""

import argparse
import os
import shutil

import numpy as np

from PAOFLOW.elphon.phonon_assisted_absorption import (
    read_direct_dielectric,
    thermal_emissivity,
    write_emissivity_outputs,
)


def load_results(path):
    """``absorption.npz`` as a dict (0-d arrays kept; strings as str)."""
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    for key, value in out.items():
        if value.dtype.kind in 'US':
            out[key] = str(value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output', help='output directory of main.py (absorption.npz)')
    parser.add_argument('--direct-dielectric', required=True,
                        help='output dir of the dielectric_tensor run (epsr/epsi_*.dat)')
    parser.add_argument('--direct-scissor', choices=('same', 'thermal', 'none'), default='thermal')
    parser.add_argument('--target-gap-ev', type=float, default=None,
                        help='0 K target gap of the run scissor (older runs without scissor_0k_ev)')
    parser.add_argument('--thickness-um', type=float, default=500.0, help='slab thickness (um)')
    parser.add_argument('--angles', type=float, nargs='+', default=(0.0, 30.0, 60.0),
                        help='emission angles (deg)')
    parser.add_argument('--outdir', required=True, help='new directory for the emissivity files')
    args = parser.parse_args()

    source = os.path.abspath(args.output)
    outdir = os.path.abspath(args.outdir)
    if outdir == source:
        raise SystemExit('--outdir must differ from the run output (it is never overwritten).')
    results = load_results(os.path.join(source, 'absorption.npz'))
    if args.direct_scissor == 'thermal' and 'scissor_0k_ev' not in results:
        if args.target_gap_ev is None:
            scissor = np.asarray(results['scissor_ev'], dtype=float)
            if np.any(scissor != 0.0) and str(results.get('scissor_mode', '')) != 'relative':
                raise SystemExit('this run has no scissor_0k_ev: give its 0 K target gap, --target-gap-ev.')
            results['scissor_0k_ev'] = 0.0  # no scissor, or the Varshni shift only
        else:
            results['scissor_0k_ev'] = args.target_gap_ev - float(results['indirect_gap_dft_ev'])
    direct = read_direct_dielectric(args.direct_dielectric)
    emissivity = thermal_emissivity(
        results, args.thickness_um * 1.0e-6, angles_deg=tuple(args.angles), direct_dielectric=direct,
        direct_scissor=args.direct_scissor,
    )  # fmt: skip
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(os.path.join(source, 'absorption.npz'), os.path.join(outdir, 'absorption.npz'))
    shifts = emissivity['direct_scissor_ev']
    print('direct eps from %s, shifted by %+.3f ... %+.3f eV (%s); f-sum ratio %.2f'
          % (args.direct_dielectric, shifts.min(), shifts.max(), args.direct_scissor,
             emissivity['f_sum_ratio'].mean()))  # fmt: skip
    print('    T (K)   total slab   total opaque')
    for it, temp in enumerate(emissivity['temps_k']):
        print('  %7.1f  %11.4f  %13.4f' % (temp, emissivity['total_slab'][it], emissivity['total_opaque'][it]))
    for path in write_emissivity_outputs(emissivity, outdir):
        print('  wrote %s' % path)


if __name__ == '__main__':
    main()
