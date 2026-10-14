#!/usr/bin/env python3
"""Plot the metal transport results of transport.py (examples/elphon_example/Pb).

    python plot_transport.py [--experiment rho_exp.dat] [--temps T ...]

Two figures in OUTPUTDIR/transport/:
  * metal_transport.png -- alpha^2F and the transport function alpha^2F_tr
    (lambda, lambda_tr, plasma frequency) and the resistivity rho(T), total and
    per scattering channel; --experiment overlays measured (T, rho) in
    micro-ohm cm;
  * emissivity.png -- spectral hemispherical emissivity at selected
    temperatures and the total (Planck-weighted) emissivity versus T.
"""

import argparse
import os

from PAOFLOW import GPAO

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTDIR = os.path.join(HERE, 'output')
TRANSPORT_DIR = os.path.join(OUTPUTDIR, 'transport')
EXPERIMENT = None  # optional file: T (K), rho (micro-ohm cm)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--experiment', default=EXPERIMENT, help='measured T (K), rho (uOhm cm)')
    parser.add_argument('--temps', type=float, nargs='+', default=None,
                        help='temperatures (K) of the spectral emissivity panel')
    args = parser.parse_args()
    eliashberg = os.path.join(OUTPUTDIR, 'eliashberg.npz')
    resistivity = os.path.join(TRANSPORT_DIR, 'resistivity.dat')
    if not os.path.isfile(resistivity):
        raise SystemExit('%s not found; run transport.py first.' % resistivity)
    plotter = GPAO.GPAO()
    plotter.plot_metal_transport(
        eliashberg, resistivity, experiment_file=args.experiment,
        filename=os.path.join(TRANSPORT_DIR, 'metal_transport.png'),
    )
    emissivity = os.path.join(TRANSPORT_DIR, 'emissivity.npz')
    if os.path.isfile(emissivity):
        plotter.plot_thermal_emissivity(
            emissivity, temps=args.temps, filename=os.path.join(TRANSPORT_DIR, 'emissivity.png')
        )


if __name__ == '__main__':
    main()
