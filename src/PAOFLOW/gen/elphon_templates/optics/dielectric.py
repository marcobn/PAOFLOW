#!/usr/bin/env python3
"""Direct dielectric function on the extended PAO basis (PAOFLOW's optical route).

    python dielectric.py                       # after the pw.x nscf of the optics directory
    python dielectric.py --interband-only      # metals: no Drude term (transport/resistivity.py)
    python dielectric.py --savedir <eACBN0 optics save> --outdir <dir>

Writes <outdir>/epsr_<c>.dat and epsi_<c>.dat, read by absorption.py --emissivity
--direct-dielectric <outdir> (semiconductors) or by resistivity.py --interband-dielectric
<outdir> (metals).  The extended basis needs more nscf bands than the electron-phonon run, so
it uses its own nscf (qe/optics); ph.x and the coupling are not rerun.  With absorption.py
--electrons-save (eACBN0 electrons) run it on the eACBN0 optics save, so that the direct
dielectric function and the band energies come from the same Hamiltonian.
"""

import argparse
import os

from PAOFLOW.elphon.workflows import direct_dielectric, load_system

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = CONFIG.OPTICS.get('dielectric', {})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--savedir', default=os.path.join(SYSTEM.optics_path, SYSTEM.prefix + '.save'),
        help='QE save of the optics nscf (default %(default)s)',
    )  # fmt: skip
    parser.add_argument(
        '--outdir', default=os.path.join(SYSTEM.optics_path, 'output'),
        help='output directory (default %(default)s)',
    )  # fmt: skip
    parser.add_argument(
        '--nfft', type=int, default=DEFAULTS.get('nfft', 24), help='interpolated k-grid per axis'
    )
    parser.add_argument('--emin', type=float, default=DEFAULTS.get('emin', 0.01))
    parser.add_argument('--emax', type=float, default=DEFAULTS.get('emax', 10.0))
    parser.add_argument('--ne', type=int, default=DEFAULTS.get('ne', 1000))
    parser.add_argument(
        '--delta', type=float, default=DEFAULTS.get('delta', 0.1), help='broadening (eV)'
    )
    parser.add_argument(
        '--interband-only', action='store_true', default=not DEFAULTS.get('intraband', True),
        help='no Drude term (the intraband part of a metal comes from alpha^2F_tr)',
    )  # fmt: skip
    parser.add_argument('--pthr', type=float, default=DEFAULTS.get('pthr', 0.95))
    args = parser.parse_args()
    direct_dielectric(
        SYSTEM, args.savedir, args.outdir, nfft=args.nfft, emin=args.emin, emax=args.emax,
        ne=args.ne, delta=args.delta, intraband=not args.interband_only, pthr=args.pthr,
        smearing=DEFAULTS.get('smearing'),
    )  # fmt: skip


if __name__ == '__main__':
    main()
