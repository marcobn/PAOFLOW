#!/usr/bin/env python3
"""Plot the metal transport results of resistivity.py.

    python plot_resistivity.py [--experiment rho_exp.dat] [--temps T ...]

Two figures in output/:
  * metal_transport.png -- alpha^2F and the transport function alpha^2F_tr (lambda, lambda_tr,
    plasma frequency) and the resistivity rho(T), total and per scattering channel;
    --experiment overlays measured (T, rho) in micro-ohm cm;
  * emissivity.png -- spectral hemispherical emissivity at selected temperatures and the total
    (Planck-weighted) emissivity versus T.
"""

import argparse
import os

from PAOFLOW import GPAO

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
ELIASHBERG = os.path.join(os.path.dirname(HERE), 'superconductivity', 'output', 'eliashberg.npz')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--experiment', default=None, help='measured T (K), rho (uOhm cm)')
    parser.add_argument(
        '--temps', type=float, nargs='+', default=None,
        help='temperatures (K) of the spectral emissivity panel',
    )  # fmt: skip
    parser.add_argument('--eliashberg', default=ELIASHBERG, help='eliashberg.npz with alpha^2F_tr')
    args = parser.parse_args()
    resistivity = os.path.join(OUTPUT, 'resistivity.dat')
    if not os.path.isfile(resistivity):
        raise SystemExit('%s not found; run resistivity.py first.' % resistivity)
    plotter = GPAO.GPAO()
    plotter.plot_metal_transport(
        args.eliashberg, resistivity, experiment_file=args.experiment,
        filename=os.path.join(OUTPUT, 'metal_transport.png'),
    )  # fmt: skip
    emissivity = os.path.join(OUTPUT, 'emissivity.npz')
    if os.path.isfile(emissivity):
        plotter.plot_thermal_emissivity(
            emissivity, temps=args.temps, filename=os.path.join(OUTPUT, 'emissivity.png')
        )


if __name__ == '__main__':
    main()
