#!/usr/bin/env python3
"""Phonon-limited carrier mobility (SERTA / MRTA) from the coarse electron-phonon coupling.

    mpirun -np N python mobility.py                          # dense grids and settings of system.py
    mpirun -np N python mobility.py --nk 60 --nq 60 --temps 300
    mpirun -np N python mobility.py --adaptive-smearing 0.7 --smearing-ev 0.001
    mpirun -np N python mobility.py --no-long-range           # plain interpolation (polar crystals)
    python plot_mobility.py

State-resolved electron-phonon scattering rates 1/tau_nk(T) of the states near the band edges
(semiconductors) or the Fermi level (metals), in the self-energy (SERTA) and
momentum-relaxation (MRTA) relaxation-time approximations, with the coupling, the electrons,
their velocities and the phonons interpolated in the PAO basis; then the k-resolved
conductivity, carrier densities, electron and hole mobilities and Seebeck coefficient.  The
long-range (Fröhlich, quadrupole) model of system.py is subtracted on the coarse grid and
restored at every dense q.  Converge nk = nq (mobilities typically need >= 48^3) together with
the smearing.

Carriers: --doping > 0 adds electrons (n type), < 0 holes (p type), 0 gives the intrinsic
carriers.  A scissor (--target-gap-ev, --varshni) corrects the carrier statistics only.
--electrons-save takes the band energies and eigenvectors from the PAO Hamiltonian of an
eACBN0 nscf in the same basis; --screening-eps applies the Poncé screening correction.

Outputs (output/): mobility.dat (T, chemical potential, n_e, n_h, SERTA and MRTA mobilities,
conductivities, Seebeck), rates_<T>K.dat, carrier_scattering.npz and scattering_rates.npz
(the transport-weighted rate versus energy for PAOFLOW's Boltzmann transport:
EnergyResolvedRate.from_npz(..., 'mrta') in pf.transport(scattering_channels=[...])).
"""

import argparse
import os

from PAOFLOW.elphon.workflows import (
    carrier_mobility,
    load_system,
    mobility_label,
    statistics_scissor,
    write_mobility,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = CONFIG.TRANSPORT


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--nk', type=int, default=DEFAULTS['nk'], help='dense k-grid (default %(default)s)'
    )
    parser.add_argument(
        '--nq', type=int, default=DEFAULTS['nq'], help='dense q-grid (default %(default)s)'
    )
    parser.add_argument(
        '--temps', type=float, nargs='+', default=DEFAULTS.get('temps_k', [300.0]),
        help='temperatures (K)',
    )  # fmt: skip
    parser.add_argument(
        '--doping', type=float, default=DEFAULTS.get('doping_cm3', 0.0),
        help='net doping (cm^-3; > 0 n type, < 0 p type, 0 intrinsic)',
    )  # fmt: skip
    parser.add_argument(
        '--window-ev', type=float, default=DEFAULTS.get('window_ev', 0.3),
        help='band-edge window (eV)',
    )  # fmt: skip
    parser.add_argument(
        '--smearing-ev', type=float, default=DEFAULTS.get('smearing_ev', 0.01),
        help='delta smearing (eV); the floor of the widths with --adaptive-smearing',
    )  # fmt: skip
    parser.add_argument(
        '--adaptive-smearing', type=float, default=DEFAULTS.get('adaptive_smearing'), metavar='A',
        help='adaptive smearing prefactor a (Yates; 0.7 for Gaussians): width a |v| dq per '
             'final state, floored at --smearing-ev; default fixed smearing',
    )  # fmt: skip
    parser.add_argument(
        '--target-gap-ev', type=float, default=DEFAULTS.get('target_gap_ev'),
        help='target indirect gap (eV) of the statistics scissor (0 K with --varshni)',
    )  # fmt: skip
    parser.add_argument(
        '--electrons-save', default=DEFAULTS.get('electrons_save'),
        help='QE save of an eACBN0 (DFT+U+V) nscf in the same PAO basis: its PAO '
             'Hamiltonian replaces the PBE one (band energies and eigenvectors)',
    )  # fmt: skip
    parser.add_argument(
        '--varshni', type=float, nargs=2, default=DEFAULTS.get('varshni'), metavar=('ALPHA', 'BETA'),
        help='temperature-dependent gap E0 - alpha T^2/(T + beta): around the target, '
             'or relative to the Hamiltonian gap without one (Si: 4.73e-4 636)',
    )  # fmt: skip
    parser.add_argument(
        '--screening-eps', type=float, default=DEFAULTS.get('screening_eps'),
        help='target dielectric constant of the Poncé screening correction of g (Si: 11.94)',
    )  # fmt: skip
    parser.add_argument(
        '--screening-eps-dft', type=float, default=DEFAULTS.get('screening_eps_dft'),
        help='DFT dielectric constant of the coupling (default: the DFPT tensor)',
    )  # fmt: skip
    parser.add_argument(
        '--no-long-range', action='store_true',
        help='plain interpolation of the full coupling (no dipole/quadrupole model)',
    )  # fmt: skip
    parser.add_argument(
        '--quadrupoles', default=SYSTEM.quadrupoles, metavar='FILE',
        help='quadrupole.fmt of coupling/quadrupoles.py (default: the one of system.py)',
    )  # fmt: skip
    parser.add_argument(
        '--tag', default=None,
        help='output subdirectory (default: output/, or output_plain/ with --no-long-range)',
    )  # fmt: skip
    args = parser.parse_args()
    SYSTEM.check_inputs()
    if args.no_long_range:
        SYSTEM.long_range = False
    SYSTEM.quadrupoles = args.quadrupoles
    electrons = SYSTEM.electrons(args.electrons_save, outputdir=OUTPUT)
    scissor_ev, target_gap_ev, mode = statistics_scissor(args.target_gap_ev, args.varshni)
    out = carrier_mobility(
        SYSTEM, electrons, nk=args.nk, nq=args.nq, temps_k=args.temps, doping_cm3=args.doping,
        window_ev=args.window_ev, smearing_ev=args.smearing_ev,
        adaptive_smearing=args.adaptive_smearing, scissor_ev=scissor_ev,
        target_gap_ev=target_gap_ev, scissor_mode=mode, screening_eps=args.screening_eps,
        screening_eps_dft=args.screening_eps_dft,
    )  # fmt: skip
    outdir = args.tag or ('output_plain' if args.no_long_range else OUTPUT)
    label = mobility_label(
        args.nk, args.nq, args.smearing_ev, args.adaptive_smearing, args.window_ev
    )
    if args.no_long_range:
        label += ', plain interpolation'
    write_mobility(out, os.path.join(HERE, outdir), label=label)


if __name__ == '__main__':
    main()
