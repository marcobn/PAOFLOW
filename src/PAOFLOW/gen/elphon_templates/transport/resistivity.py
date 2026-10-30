#!/usr/bin/env python3
"""Metal resistivity, infrared optics and thermal emissivity from alpha^2F_tr.

    python resistivity.py [--temps T ...] [--rho0 0.5] [--b-ee 0.0] [--interband-dielectric DIR]

Reads ../superconductivity/output/eliashberg.npz of eliashberg.py --transport (the transport
Eliashberg function alpha^2F_tr, lambda_tr and the Fermi-surface plasma frequency) and
combines scattering channels by Matthiessen's rule:
  * electron-phonon: alpha^2F_tr (Allen): the DC rate 1/tau_tr(T) and, for the optics, the
    frequency-dependent rate 1/tau(w, T) with the mass enhancement lambda(w, T) of the
    extended Drude model;
  * residual (impurity) resistivity --rho0;
  * electron-electron (Fermi liquid): hbar/tau = b [(hbar w)^2 + (2 pi kT)^2], --b-ee.

Outputs (output/): resistivity.dat (T, rho total and per channel, micro-ohm cm) and the
thermal emissivity of the metal: eps_<T>K.dat (interband + extended Drude), emish_<T>K.dat,
emis_th<deg>_<T>K.dat, emist.dat, emissivity.npz.  The interband part comes from
--interband-dielectric DIR, the output of dielectric.py --interband-only on the extended
basis; without it eps_inter = 1.

The same channels drive PAOFLOW's Boltzmann transport, e.g.
    pf.transport(scattering_channels=[TransportSpectralChannel.from_npz(NPZ),
                                      ResidualResistivity(rho0_ohm_m, plasma_ev)])
"""

import argparse
import os

from PAOFLOW.elphon.workflows import load_system, metal_resistivity

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
DEFAULTS = CONFIG.TRANSPORT
ELIASHBERG = os.path.join(os.path.dirname(HERE), 'superconductivity', 'output', 'eliashberg.npz')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--eliashberg', default=ELIASHBERG, help='eliashberg.npz with alpha^2F_tr')
    parser.add_argument(
        '--temps', type=float, nargs='+',
        default=DEFAULTS.get('temps_k', [20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
        help='temperatures (K)',
    )  # fmt: skip
    parser.add_argument(
        '--rho0', type=float, default=DEFAULTS.get('rho0_uohm_cm', 0.0),
        help='residual resistivity (uOhm cm)',
    )  # fmt: skip
    parser.add_argument(
        '--b-ee', type=float, default=DEFAULTS.get('b_ee_per_ev', 0.0),
        help='Fermi-liquid coefficient (1/eV)',
    )  # fmt: skip
    parser.add_argument(
        '--interband-dielectric', default=DEFAULTS.get('interband_dielectric_dir'),
        help='output dir of a dielectric_tensor(intraband=False) run',
    )  # fmt: skip
    parser.add_argument(
        '--thickness-um', type=float, default=DEFAULTS.get('thickness_um', float('inf')),
        help='film thickness (um; metals are opaque)',
    )  # fmt: skip
    parser.add_argument(
        '--angles', type=float, nargs='+', default=DEFAULTS.get('angles', (0.0, 30.0, 60.0)),
        help='emission angles (deg)',
    )  # fmt: skip
    omega = DEFAULTS.get('omega_ev', (0.005, 2.0, 0.005))
    parser.add_argument('--omega-min', type=float, default=omega[0], help='min photon energy (eV)')
    parser.add_argument('--omega-max', type=float, default=omega[1], help='max photon energy (eV)')
    parser.add_argument(
        '--omega-step', type=float, default=omega[2], help='photon-energy step (eV)'
    )
    parser.add_argument(
        '--simple-drude', action='store_true', default=not DEFAULTS.get('extended_drude', True),
        help='DC rate instead of the extended Drude model',
    )  # fmt: skip
    args = parser.parse_args()
    metal_resistivity(
        args.eliashberg, OUTPUT, temps_k=args.temps, rho0_uohm_cm=args.rho0, b_ee_per_ev=args.b_ee,
        omega_ev=(args.omega_min, args.omega_max, args.omega_step), thickness_um=args.thickness_um,
        angles_deg=args.angles, interband_dielectric_dir=args.interband_dielectric,
        extended_drude=not args.simple_drude,
    )  # fmt: skip


if __name__ == '__main__':
    main()
