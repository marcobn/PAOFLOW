#!/usr/bin/env python3
"""Plot the phonon-limited carrier mobilities and scattering rates.

    python plot_mobility.py [--temperature T] [--experiment FILE]

Reads output/mobility.dat and carrier_scattering.npz written by mobility.py: the electron and
hole mobilities versus temperature (MRTA solid, SERTA dashed) and the MRTA rates 1/tau_nk of
the window states against the carrier energy from the band edge, at the temperature closest
to T.  The optional experiment file has three columns: T (K), electron and hole mobilities
(cm^2/(V s)), NaN for a missing value.  Saved as output/mobility.png.
"""

import argparse
import os

from PAOFLOW import GPAO

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--temperature', type=float, default=300.0, help='temperature (K) of the rates panel'
    )
    parser.add_argument(
        '--experiment', default=None, help='measured mobilities: T (K), electrons, holes (cm^2/Vs)'
    )
    parser.add_argument('--source', default=OUTPUT, help='output directory of mobility.py')
    args = parser.parse_args()
    mobility = os.path.join(args.source, 'mobility.dat')
    if not os.path.isfile(mobility):
        raise SystemExit('%s not found; run mobility.py first.' % mobility)
    scattering = os.path.join(args.source, 'carrier_scattering.npz')
    GPAO.GPAO().plot_carrier_mobility(
        mobility, scattering if os.path.isfile(scattering) else None,
        temperature_k=args.temperature, experiment_file=args.experiment,
        filename=os.path.join(args.source, 'mobility.png'),
    )  # fmt: skip


if __name__ == '__main__':
    main()
