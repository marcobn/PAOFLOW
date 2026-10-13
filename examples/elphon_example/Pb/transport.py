#!/usr/bin/env python3
"""Metal resistivity, infrared optics and thermal emissivity from alpha^2F_tr (examples/elphon_example/Pb).

    python transport.py [--temps T ...] [--interband-dielectric DIR] [--thickness-um D]

Reads OUTPUTDIR/eliashberg.npz of `main.py --transport` (the transport
Eliashberg function alpha^2F_tr, lambda_tr and the Fermi-surface plasma
frequency) and combines scattering channels by Matthiessen's rule:

  * electron-phonon: alpha^2F_tr (Allen): the DC rate 1/tau_tr(T) and, for the
    optics, the frequency-dependent rate 1/tau(w, T) with the mass enhancement
    lambda(w, T) of the extended Drude model;
  * residual (impurity) resistivity RHO0_UOHM_CM;
  * electron-electron (Fermi liquid): hbar/tau = B_EE [(hbar w)^2 + (2 pi kT)^2].

Outputs (OUTPUTDIR/transport/): resistivity.dat (T, rho total and per channel,
micro-ohm cm) and the thermal emissivity of the metal: eps_<T>K.dat (interband
+ extended Drude), emish_<T>K.dat, emis_th<deg>_<T>K.dat, emist.dat,
emissivity.npz.  The interband part comes from --interband-dielectric DIR, the
output of a separate dielectric_tensor(..., intraband=False) run on the
extended basis (epsr/epsi_*.dat); without it eps_inter = 1.

The same channels drive PAOFLOW's Boltzmann transport, e.g.
    pf.transport(scattering_channels=[TransportSpectralChannel.from_npz(NPZ),
                                      ResidualResistivity(rho0_ohm_m, plasma_ev)])
"""

import argparse
import os

import numpy as np

from PAOFLOW.boltzmann.scattering_channels import (
    FermiLiquidRate,
    ResidualResistivity,
    TransportSpectralChannel,
)
from PAOFLOW.elphon.phonon_assisted_absorption import (
    read_direct_dielectric,
    write_emissivity_outputs,
)
from PAOFLOW.response.drude import dc_resistivity
from PAOFLOW.response.emissivity import metal_thermal_emissivity

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTDIR = os.path.join(HERE, 'output')
NPZ = os.path.join(OUTPUTDIR, 'eliashberg.npz')
RHO0_UOHM_CM = 0.0  # residual resistivity (micro-ohm cm)
B_EE_PER_EV = 0.0  # Fermi-liquid coefficient b (1/eV); 0 = no electron-electron term
TEMPS_K = [20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]  # temperatures (K)
OMEGA_EV = (0.005, 2.0, 0.005)  # photon energies (min, max, step) for the optics (eV)
THICKNESS_UM = float('inf')  # film thickness; metals are opaque
EMIS_ANGLES = (0.0, 30.0, 60.0)  # emission angles (deg) of the directional spectra
INTERBAND_DIELECTRIC_DIR = None  # dielectric_tensor(intraband=False) output (extended basis)
EXTENDED_DRUDE = True  # frequency-dependent rate + mass enhancement; False = DC rate


def scattering_channels(plasma_ev, rho0_uohm_cm, b_ee):
    """Electron-phonon (alpha^2F_tr) + residual + electron-electron channels."""
    channels = [TransportSpectralChannel.from_npz(NPZ)]
    if rho0_uohm_cm > 0.0:
        channels.append(ResidualResistivity(rho0_uohm_cm * 1.0e-8, plasma_ev))
    if b_ee > 0.0:
        channels.append(FermiLiquidRate(b_ee))
    return channels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--temps', type=float, nargs='+', default=TEMPS_K, help='temperatures (K)')
    parser.add_argument('--rho0', type=float, default=RHO0_UOHM_CM, help='residual resistivity (uOhm cm)')
    parser.add_argument('--b-ee', type=float, default=B_EE_PER_EV, help='Fermi-liquid coefficient (1/eV)')
    parser.add_argument('--interband-dielectric', default=INTERBAND_DIELECTRIC_DIR,
                        help='output dir of a dielectric_tensor(intraband=False) run')
    parser.add_argument('--thickness-um', type=float, default=THICKNESS_UM, help='film thickness (um)')
    parser.add_argument('--simple-drude', action='store_true', default=not EXTENDED_DRUDE,
                        help='DC rate instead of the extended Drude model')
    args = parser.parse_args()
    with np.load(NPZ) as data:
        if 'a2F_tr' not in data:
            raise SystemExit('%s has no alpha^2F_tr: run main.py --transport first.' % NPZ)
        plasma = float(data['omega_plasma_ev'])
        lambda_tr = float(data['lambda_tr'])
    channels = scattering_channels(plasma, args.rho0, args.b_ee)
    outdir = os.path.join(OUTPUTDIR, 'transport')
    os.makedirs(outdir, exist_ok=True)

    print('Metal transport from alpha^2F_tr: lambda_tr = %.3f, hbar omega_p = %.3f eV' % (lambda_tr, plasma))
    print('    T (K)   rho (uOhm cm)   ' + '   '.join('%s' % c.name for c in channels))
    rows = []
    for temp in args.temps:
        rho = dc_resistivity(plasma, channels, temp)
        parts = [rho['channels'][c.name][1] * 1.0e8 for c in channels]
        rows.append([temp, rho['rho_ohm_m'] * 1.0e8] + parts)
        print('  %7.1f  %14.4f   ' % (temp, rho['rho_ohm_m'] * 1.0e8) + '   '.join('%.4f' % p for p in parts))
    np.savetxt(os.path.join(outdir, 'resistivity.dat'), np.array(rows), fmt='%14.6f',
               header='T (K), rho total and per channel (uOhm cm): ' + ', '.join(c.name for c in channels))

    omin, omax, ostep = OMEGA_EV
    omega = np.arange(omin, omax + 0.5 * ostep, ostep)
    interband = read_direct_dielectric(args.interband_dielectric) if args.interband_dielectric else None
    emissivity = metal_thermal_emissivity(
        omega, args.temps, plasma, channels, interband=interband,
        thickness_m=args.thickness_um * 1.0e-6, angles_deg=EMIS_ANGLES,
        extended=not args.simple_drude,
    )
    print('Thermal emissivity (%s Drude, interband: %s):'
          % (emissivity['drude'], args.interband_dielectric or 'none'))
    print('    T (K)   total hemispherical   Planck coverage')
    for temp, total, cover in zip(emissivity['temps_k'], emissivity['total_opaque'],
                                  emissivity['planck_coverage']):
        print('  %7.1f  %20.5f  %16.4f' % (temp, total, cover))
    for path in write_emissivity_outputs(emissivity, outdir):
        print('  wrote %s' % path)


if __name__ == '__main__':
    main()
