#!/usr/bin/env python3
"""Phonon-assisted (indirect) + direct optical absorption from the coarse electron-phonon coupling.

    mpirun -np N python absorption.py                        # dense grids and settings of system.py
    mpirun -np N python absorption.py --temps 10 300 600      # several temperatures in one run
    mpirun -np N python absorption.py --nk 24 --nq 12         # denser grids (nk % nq == 0)
    mpirun -np N python absorption.py --adaptive-smearing 0.7 --degauss-ev 0.001
    mpirun -np N python absorption.py --emissivity --direct-dielectric ../qe/optics/output \\
        --omega-min 0.01 --omega-step 0.01 --degauss-ev 0.01 --temps 300 500 700 900 1100 1300 1500
    python plot_absorption.py

PAOFLOW interpolates the coarse coupling, the electrons with their velocity matrix elements and
the phonons in the PAO basis and evaluates the second-order (photon + phonon) absorption of
EPW's lindabs (Noffsinger et al. 2012) together with the direct (vertical) term.  Options:
a scissor of the band energies (--target-gap-ev, --gap-reference-save, --varshni), eACBN0
electrons (--electrons-save: the PAO Hamiltonian of a DFT+U+V nscf in the same basis replaces
the PBE one; the coupling stays the DFPT one), the Poncé screening correction of the coupling
(--screening-eps) and the thermal emissivity of a free-standing slab (--emissivity, with the
direct dielectric function of dielectric.py on the extended basis).

Outputs (output/, or --outdir NAME to keep earlier runs side by side): epsilon2_indabs_<T>K.dat,
epsilon2_dirabs_<T>K.dat, alpha_<T>K.dat in EPW's layout, absorption.npz and, with --emissivity,
emish_<T>K.dat, emis_th<deg>_<T>K.dat, eps_<T>K.dat, emist.dat, emissivity.npz.
"""

import argparse
import os

from PAOFLOW.elphon.phonon_assisted_absorption import read_direct_dielectric
from PAOFLOW.elphon.workflows import (
    absorption_label,
    load_system,
    phonon_assisted_absorption,
    scissor_settings,
    slab_emissivity,
    write_absorption,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = CONFIG.OPTICS


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
        '--degauss-ev', type=float, default=DEFAULTS.get('degauss_ev', 0.05),
        help='delta smearing (eV); the floor of the widths with --adaptive-smearing',
    )  # fmt: skip
    parser.add_argument(
        '--adaptive-smearing', type=float, default=DEFAULTS.get('adaptive_smearing'), metavar='A',
        help='adaptive smearing prefactor a (Yates; 0.7 for Gaussians): width '
             'a sqrt(|v_i(k) - v_j(k+q)|^2 dk^2 + |v_j(k+q)|^2 dq^2) per transition, '
             'floored at --degauss-ev; default fixed smearing',
    )  # fmt: skip
    omega = DEFAULTS.get('omega_ev', (0.05, 3.0, 0.05))
    parser.add_argument('--omega-min', type=float, default=omega[0], help='min photon energy (eV)')
    parser.add_argument('--omega-max', type=float, default=omega[1], help='max photon energy (eV)')
    parser.add_argument(
        '--omega-step', type=float, default=omega[2], help='photon-energy step (eV)'
    )
    parser.add_argument(
        '--fsthick-ev', type=float, default=DEFAULTS.get('fsthick_ev', 4.0),
        help='states within this window of the Fermi level (EPW fsthick)',
    )  # fmt: skip
    parser.add_argument(
        '--fermi-energy-ev', type=float, default=DEFAULTS.get('fermi_energy_ev'),
        help='Fermi level on the PAO energy scale (default mid-gap)',
    )  # fmt: skip
    parser.add_argument(
        '--position-dipoles', action=argparse.BooleanOptionalAction,
        default=DEFAULTS.get('position_dipoles', True),
        help='position dipoles of the PAO orbitals in the velocity',
    )  # fmt: skip
    parser.add_argument(
        '--position-dipole-level', choices=('full', 'intra-atomic'),
        default=DEFAULTS.get('position_dipole_level', 'full'),
        help='Löwdin-orthogonalised basis or on-site blocks only',
    )  # fmt: skip
    parser.add_argument(
        '--emissivity', action=argparse.BooleanOptionalAction,
        default=DEFAULTS.get('emissivity', False),
        help='also compute the thermal emissivity of a slab',
    )  # fmt: skip
    parser.add_argument(
        '--thickness-um', type=float, default=DEFAULTS.get('thickness_um', 500.0),
        help='slab thickness in um (default %(default)s)',
    )  # fmt: skip
    parser.add_argument(
        '--angles', type=float, nargs='+', default=DEFAULTS.get('angles', (0.0, 30.0, 60.0)),
        help='emission angles (deg) of the directional spectra',
    )  # fmt: skip
    parser.add_argument(
        '--kk-omega-max-ev', type=float, default=DEFAULTS.get('kk_omega_max_ev', 10.0),
        help='Kramers-Kronig range of the direct term (without --direct-dielectric)',
    )  # fmt: skip
    parser.add_argument(
        '--target-gap-ev', type=float, default=DEFAULTS.get('target_gap_ev'),
        help='target indirect gap (eV) of a scissor (0 K value with --varshni)',
    )  # fmt: skip
    parser.add_argument(
        '--gap-reference-save', default=DEFAULTS.get('gap_reference_save'),
        help='QE save of an eACBN0 (DFT+U+V) run whose PAO gap is the target',
    )  # fmt: skip
    parser.add_argument(
        '--refractive-index', type=float, default=DEFAULTS.get('refractive_index'),
        help='constant n_r of alpha (default: n(omega) of --direct-dielectric, else 3.4)',
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
        '--scissor-velocity', choices=('renormalized', 'ks'),
        default=DEFAULTS.get('scissor_velocity', 'renormalized'),
        help='velocities of the scissor-corrected Hamiltonian',
    )  # fmt: skip
    parser.add_argument(
        '--direct-scissor', choices=('same', 'thermal', 'none'),
        default=DEFAULTS.get('direct_scissor', 'same'),
        help='shift of the direct dielectric function: the run scissor, its thermal part only, '
             'or none (default %(default)s)',
    )  # fmt: skip
    parser.add_argument(
        '--direct-dielectric', default=DEFAULTS.get('direct_dielectric_dir'),
        help='output dir of an extended-basis dielectric_tensor run (epsr/epsi_*.dat)',
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
        '--outdir', default=OUTPUT,
        help='output directory (default output/; use another name to keep earlier runs)',
    )  # fmt: skip
    args = parser.parse_args()
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(HERE, args.outdir)
    SYSTEM.check_inputs()
    electrons = SYSTEM.electrons(args.electrons_save, outputdir=outdir)
    scissor_ev, target_gap_ev, mode = scissor_settings(
        SYSTEM, electrons, nk=args.nk, target_gap_ev=args.target_gap_ev,
        gap_reference_save=args.gap_reference_save, varshni=args.varshni,
        electrons_save=args.electrons_save,
    )  # fmt: skip
    direct = read_direct_dielectric(args.direct_dielectric) if args.direct_dielectric else None
    out = phonon_assisted_absorption(
        SYSTEM, electrons, nk=args.nk, nq=args.nq,
        omega_ev=(args.omega_min, args.omega_max, args.omega_step), temps_k=args.temps,
        degauss_ev=args.degauss_ev, adaptive_smearing=args.adaptive_smearing,
        fsthick_ev=args.fsthick_ev, fermi_energy_ev=args.fermi_energy_ev,
        refractive_index=args.refractive_index, direct_dielectric=direct,
        scissor_ev=scissor_ev, target_gap_ev=target_gap_ev, scissor_mode=mode,
        scissor_velocity=args.scissor_velocity, screening_eps=args.screening_eps,
        screening_eps_dft=args.screening_eps_dft, position_dipoles=args.position_dipoles,
        position_dipole_level=args.position_dipole_level, emissivity=args.emissivity,
        kk_omega_max_ev=args.kk_omega_max_ev,
    )  # fmt: skip
    label = absorption_label(
        args.nk, args.nq, args.degauss_ev, args.adaptive_smearing, args.position_dipoles,
        args.position_dipole_level,
    )  # fmt: skip
    write_absorption(out, outdir, label=label)
    if args.emissivity:
        slab_emissivity(
            out, outdir, thickness_um=args.thickness_um, angles_deg=args.angles,
            direct_dielectric=direct, direct_dielectric_label=args.direct_dielectric,
            direct_scissor=args.direct_scissor,
        )  # fmt: skip


if __name__ == '__main__':
    main()
