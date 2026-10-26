#!/usr/bin/env python3
"""PAOFLOW phonon-assisted optical absorption on the coarse electron-phonon coupling (examples/elphon_example/Si).

Second-order (photon + phonon) absorption of an indirect-gap semiconductor, as
EPW's ``lindabs`` (indabs.f90; Noffsinger et al., PRL 108, 167402 (2012); EPW
tutorial 06), with the coupling, the electrons, their velocity matrix elements
and the phonons all interpolated in the PAO basis.  The direct (vertical)
absorption is computed alongside.

The coarse electron-phonon matrix elements are computed by PAOFLOW from the
``ph.x`` outputs (COUPLING_SOURCE = 'dfpt'; EPW's ``.epb`` files are the alternative).
paoflow-gen wrote the QE inputs in two run directories:

    phonon/scf.in, phonon/ph.in     pw.x scf + ph.x DFPT (irreducible q, fildvscf)
    nscf/nscf.in                    pw.x nscf on the coarse k-grid (K_POINTS automatic:
                                    irreducible wedge, unfolded by symmetry; an explicit
                                    full list, as EPW needs it, works as well)

Run them in this order:

    cd phonon
    mpirun -np N pw.x -in scf.in > scf.out
    mpirun -np N ph.x -in ph.in  > ph.out
    cd ../nscf
    mkdir -p Si2.save          # the nscf starts from the scf charge density
    cp ../phonon/Si2.save/{charge-density.dat,data-file-schema.xml} Si2.save/
    mpirun -np N pw.x -in nscf.in > nscf.out
    cd ..
    mpirun -np N python main.py
    python plot.py

EPW route (COUPLING_SOURCE = 'epw'): python nscf/kmesh.py 6 6 6 --explicit, rerun the
nscf, python nscf/write_ukk.py, mpirun -np N epw.x -nk N -in nscf/epw.in (writes Si2.epb*).

Outputs (OUTPUTDIR), one set per temperature, in EPW's layout:
epsilon2_indabs_<T>K.dat and epsilon2_indabs_lorenz<T>K.dat (photon energy and
the phonon-assisted Im eps for EPW's nine intermediate-state broadenings eta),
epsilon2_dirabs_<T>K.dat (direct Im eps), alpha_<T>K.dat (absorption
coefficient, cm^-1) and absorption.npz.

alpha = omega Im eps / (n_r c) uses n(omega) of the DIRECT_DIELECTRIC_DIR run when
given (REFRACTIVE_INDEX = None), else the constant REFRACTIVE_INDEX (3.4 is EPW's
tutorial value for Si); the emissivity always uses the full dielectric function.
The Fermi level sits mid-gap (EPW: efermi_read + fermi_energy) unless
FERMI_ENERGY_EV is set.  The band energies are the PAO (DFT) ones unless a
scissor fixes the gap: TARGET_GAP_EV (--target-gap-ev, e.g. the experiment) or
GAP_REFERENCE_SAVE (--gap-reference-save, the QE save of an eACBN0 / DFT+U+V
run with ortho-atomic projectors, whose PAO gap on the dense grid becomes the
target), optionally temperature dependent with VARSHNI (--varshni ALPHA BETA).
Only the band energies change (rigid conduction shift) and, with
SCISSOR_VELOCITY = 'renormalized', the interband velocities (Levine-Allan);
the electron-phonon coupling stays that of the DFT (DFPT) run, since ph.x
supports Hubbard U only with atomic projectors.  POSITION_DIPOLES adds the
position dipoles of the PAO orbitals (the on-site s-p, p-d matrix elements
that dH/dk lacks, Löwdin-orthogonalised over the basis): with them the
velocity is the full commutator i[H, r] of the PAO Hamiltonian, which already
contains the non-local pseudopotential (checked against exact plane-wave
velocities: dipoles 0.99-1.01 of the exact Im eps; the former non-local
pseudopotential term, removed in October 2026, double counted it: 0.87).

ELECTRONS_SAVE (--electrons-save) replaces the scissor at 0 K: the band energies
and eigenvectors come from the PAO Hamiltonian of an eACBN0 (DFT+U+V,
ortho-atomic) nscf of the same crystal in the same PAO basis (same
pseudopotential, BASIS tables, configuration, PTHR), while the coupling in the
orbital basis stays that of the DFPT/EPW run: ph.x and epw.x cannot use U+V or
ortho-atomic projectors, so the mixing happens in PAOFLOW.  --varshni without a
target then applies the Varshni shift relative to the Hamiltonian's own gap.
Not included: the Hubbard contribution to the coupling (dV_Hub/du), the
Hubbard term of the velocity commutator and the change of screening (beyond
the SCREENING_EPS rescaling below).  The
direct dielectric function (DIRECT_DIELECTRIC_DIR) must then come from a
dielectric_tensor run on the matching eACBN0 save (a warning compares its
onset with the Hamiltonian's direct gap).  If its direct gap is already right
and a scissor (--target-gap-ev) only corrects the indirect gap, use
--direct-scissor thermal: the external spectrum then gets only the
temperature-dependent part of the scissor.

SCREENING_EPS (--screening-eps) corrects the DFT overscreening of the coupling
as Ponce, Margine and Giustino, PRB 97, 121201(R) (2018): every matrix element
is multiplied by eps_DFT(|q|)/eps(|q|), the ratio of two Resta model dielectric
functions with the DFPT static dielectric constant (stored in the .epb files,
or SCREENING_EPS_DFT) and the target one, e.g. the measured value (Si: 11.94).

With EMISSIVITY (or --emissivity) the run also gives the thermal emissivity of
a free-standing slab of THICKNESS_UM at every temperature (Kirchhoff's law:
emissivity = absorptance): Re eps from Kramers-Kronig of Im eps (direct term
over all bands up to KK_OMEGA_MAX_EV, phonon-assisted term below the direct
gap), the Fresnel slab absorptance, its hemispherical average and the
Planck-weighted total at the same temperature.  For the direct part, point
DIRECT_DIELECTRIC_DIR (--direct-dielectric) to the output of a separate
dielectric_tensor run on the extended basis with the position dipoles, as in
paoflow-gen's optical workflow: its eps1 is used throughout and its eps2 above
the direct gap.  The Fermi level is then the
charge-neutral one of each temperature, so the thermally excited carriers and
their (phonon-assisted) free-carrier absorption enter; OMEGA_EV should start
near 0.01 eV with a 0.01 eV step (and DEGAUSS_EV about 0.01 eV) to cover the
blackbody spectrum, e.g. --emissivity --omega-min 0.01 --omega-step 0.01
--degauss-ev 0.01 --temps 300 500 700 900 1100 1300 1500.  Extra outputs: eps_<T>K.dat,
emish_<T>K.dat, emis_th<deg>_<T>K.dat, emist.dat and emissivity.npz.  With the
DFT gap, the intrinsic carriers (and the rise of the emissivity with T) come at
too low a temperature.
"""

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

from PAOFLOW.elphon.coupling_screening import screening_report
from PAOFLOW.elphon.elph_bloch import atom_masses
from PAOFLOW.elphon.pao_electrons import (
    check_same_pao_basis,
    format_hubbard_info,
    pao_hamiltonian_from_save,
)
from PAOFLOW.elphon.long_range import LongRangeModel
from PAOFLOW.elphon.wavefunction_symmetry import stores_irreducible_wedge
from PAOFLOW.elphon.phonon_assisted_absorption import (
    pao_band_gaps,
    phonon_assisted_absorption_dense_q,
    external_refractive_index,
    read_direct_dielectric,
    thermal_emissivity,
    varshni_gap,
    varshni_shift,
    write_absorption_outputs,
    write_emissivity_outputs,
)

# ----------------------------------------------------------------------- #
# Configuration  (edit freely -- masses / NELEC are system-specific)      #
# ----------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = 'Si2'
COUPLING_SOURCE = (
    'dfpt'  # 'dfpt': native vertex from ph.x's dvscf + the nscf save; 'epw': EPW's .epb files
)
PHONON_DIR = os.path.join(HERE, 'phonon')  # ph.x run directory (_ph0/, Si2.dyn*) or its pp.py save/
NSCF_DIR = os.path.join(HERE, 'nscf')  # nscf run; with COUPLING_SOURCE='epw' also Si2.epb*, Si2.ukk
SAVEDIR = os.path.join(NSCF_DIR, 'Si2.save')  # the nscf save on the coarse k-grid (irreducible wedge or full list)
COUPLING_DIR = PHONON_DIR if COUPLING_SOURCE == 'dfpt' else NSCF_DIR
BASISDIR = os.path.join(HERE, 'BASIS_PS')  # paoflow-genbasis-ps --pseudo <upf> --out <BASISDIR>
OUTPUTDIR = os.path.join(HERE, 'output')

COARSE_GRID = (6, 6, 6)  # nscf k-grid (EPW nk1..3)
QGRID = (3, 3, 3)  # ph.x / EPW q-grid (nq1..3); must divide COARSE_GRID
MASSES_AMU = [28.085]  # mass of each species (amu), ATOMIC_SPECIES order (EPW amass)
NELEC = 8  # valence electrons (locates the gap)
PTHR = 0.95  # projectability threshold
PROJECTOR = (
    'internal'  # 'internal' (BASISDIR tables) or 'projwfc' (projwfc.x projections of the save)
)
SAVE_OVERLAPS = False  # projwfc overlaps (projwfc projector only)
NK_DENSE = 12  # dense electron grid (EPW nkf1..3)
NQ_DENSE = 6  # dense phonon grid (EPW nqf1..3); NK_DENSE % NQ_DENSE == 0
OMEGA_EV = (0.05, 3.0, 0.05)  # photon energies (min, max, step): EPW omegamin, omegamax, omegastep
TEMPS_K = [300.0]  # temperatures (EPW temps)
DEGAUSS_EV = 0.05  # energy-conserving delta smearing (EPW degaussw)
FSTHICK_EV = 4.0  # states within FSTHICK_EV of the Fermi level (EPW fsthick)
FERMI_ENERGY_EV = None  # Fermi level on the PAO energy scale; None = mid-gap
REFRACTIVE_INDEX = None  # n_r of alpha = omega Im eps / (n_r c); None = n(omega) of DIRECT_DIELECTRIC_DIR, else 3.4
POSITION_DIPOLES = True  # position dipoles of the PAO orbitals in the velocity (i[H, d])
POSITION_DIPOLE_LEVEL = (
    'full'  # 'full' (Löwdin-orthogonalised basis) or 'intra-atomic' (on-site only)
)
# Scissor: rigid conduction-band shift to a target indirect gap (the e-ph coupling stays DFT).
TARGET_GAP_EV = None  # e.g. 1.17 (Si, 0 K) or an eACBN0 gap; None = DFT gap
GAP_REFERENCE_SAVE = None  # QE save of an eACBN0 (DFT+U+V) run: target = its PAO gap
# eACBN0 electrons: QE save of a DFT+U+V (ortho-atomic) nscf of the same crystal with the SAME
# pseudopotential, BASIS tables, configuration and PTHR.  Its PAO Hamiltonian gives the band
# energies and eigenvectors (no 0 K scissor); A, the .epb coupling, the k list and the symmetries
# stay those of SAVEDIR, so the coupling in the orbital basis is the DFPT (PBE) one.
ELECTRONS_SAVE = None  # None = the electrons of SAVEDIR
VARSHNI = None  # (alpha eV/K, beta K): E_g(T) = E0 - alpha T^2 / (T + beta), E0 = target
SCISSOR_VELOCITY = 'renormalized'  # Levine-Allan velocities; 'ks' = EPW's scissor
# Coupling screening (Ponce et al. 2018): g -> g eps_DFT(|q|)/eps(|q|), Resta model dielectric functions.
SCREENING_EPS = None  # target static dielectric constant, e.g. 11.94 (Si, measured); None = DFPT
SCREENING_EPS_DFT = (
    None  # dielectric constant of the DFPT coupling; None = the tensor in the .epb files
)
# Long-range (polar) treatment: the analytic dipole (Frohlich) vertex and force constants are
# subtracted on the coarse grid and restored at every dense q (EPW lpolar).  Si is non-polar
# (Z* = 0), so the dipole model vanishes; a dynamical-quadrupole file (EPW quadrupole.fmt,
# Brunin et al. 2020) adds the next-order term at small q.
LONG_RANGE = True  # build the model when the crystal is polar or QUADRUPOLES is given
QUADRUPOLES = None  # path of an EPW quadrupole.fmt, or None
EMISSIVITY = False  # also the thermal emissivity of a slab (--emissivity)
THICKNESS_UM = 500.0  # slab thickness (um); float('inf') = opaque half-space
EMIS_ANGLES = (0.0, 30.0, 60.0)  # emission angles (deg) of the directional spectra
KK_OMEGA_MAX_EV = 10.0  # Kramers-Kronig range of the direct term (check the f-sum ratio)
# Output directory of a separate dielectric_tensor run on the EXTENDED basis with the
# position dipoles (recommended for the direct eps1 + i eps2; needs an nscf with more
# bands than extended-basis orbitals).  None = direct term from this run's PAO basis
# by Kramers-Kronig.
DIRECT_DIELECTRIC_DIR = None
# Shift of the direct dielectric function: 'same' = the scissor of the run (PBE optics with PBE
# electrons); 'thermal' = only its temperature-dependent part, when the optics run already has
# the right 0 K direct gap and the scissor corrects the indirect gap (eACBN0); 'none'.
DIRECT_SCISSOR = 'same'


def pao_electronic_structure(electrons_save=None):
    """PAO projections and Hamiltonian; the Hamiltonian optionally from another save.

    The coupling (the projections A, the .epb files, the k list and the
    symmetries) always comes from SAVEDIR, the coarse-grid nscf.  With
    ``electrons_save`` (an eACBN0 nscf of the same crystal in the same PAO basis)
    the band energies and eigenvectors are those of its PAO Hamiltonian, while the
    coupling in the orbital basis stays that of the DFPT/EPW run.  Returns the
    projections, HRs, the lattice data (with the orbital centres), the data
    controller of the coupling save and a provenance dict for the outputs.
    """
    options = dict(
        basispath=BASISDIR,
        configuration='standard',
        pthr=PTHR,
        projector=PROJECTOR,
        workpath=HERE,
        outputdir=OUTPUTDIR,
        save_overlaps=SAVE_OVERLAPS,
    )
    # The nscf may hold the irreducible wedge (K_POINTS automatic): its Bloch functions are then
    # unfolded to the full grid by symmetry; an explicit full list is used as is.
    coupling = pao_hamiltonian_from_save(
        SAVEDIR,
        expand_wedge=False,
        unfold_kpoints=stores_irreducible_wedge(SAVEDIR),
        verbose=False,
        **options,
    )
    kind = lambda run: run.hubbard['kind'] if run.hubbard else 'PBE'  # noqa: E731
    provenance = {
        'coupling_save': SAVEDIR,
        'electrons_save': '',
        'electrons_dft': kind(coupling),
        'electrons_hubbard': format_hubbard_info(coupling.hubbard),
    }
    if electrons_save is None:
        return (
            coupling.projections,
            coupling.HRs,
            coupling.nscf,
            coupling.data_controller,
            provenance,
        )
    # Same basis (pseudopotentials, BASIS tables, configuration, structure, atom order), or stop.
    electrons = pao_hamiltonian_from_save(electrons_save, keep_projections=False, **options)
    summary = check_same_pao_basis(coupling.data_controller, electrons.data_controller)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('  electrons from %s: %s' % (electrons_save, format_hubbard_info(electrons.hubbard)))
        print(
            '  same PAO basis as %s (bands with projectability > %g: %s vs %s)'
            % (SAVEDIR, PTHR, summary['bnd'][0], summary['bnd'][1])
        )
    nscf = dict(coupling.nscf, orbital_positions=electrons.nscf['orbital_positions'])
    provenance.update(
        electrons_save=os.path.abspath(electrons_save),
        electrons_dft=kind(electrons),
        electrons_hubbard=format_hubbard_info(electrons.hubbard),
    )
    return coupling.projections, electrons.HRs, nscf, coupling.data_controller, provenance


def reference_gap(savedir, nk):
    """Indirect gap (eV) of the PAO Hamiltonian of another QE run (e.g. eACBN0) on the dense grid."""
    electrons = pao_hamiltonian_from_save(
        savedir,
        BASISDIR,
        'standard',
        PTHR,
        projector=PROJECTOR,
        keep_projections=False,
        workpath=HERE,
        outputdir=OUTPUTDIR,
        save_overlaps=SAVE_OVERLAPS,
    )
    gaps = pao_band_gaps(
        electrons.HRs, electrons.nscf['at'], nk, NELEC, electrons.nscf['orbital_positions']
    )
    return gaps['indirect_gap_ev']


def scissor_settings(args):
    """Scissor of the band energies: ``(scissor_ev, target_gap_ev, mode)``.

    ``'target'``: the conduction bands are shifted to TARGET_GAP_EV (or to the gap
    of GAP_REFERENCE_SAVE), optionally temperature dependent (Varshni);
    ``'relative'``: no 0 K scissor, only the Varshni shift -alpha T^2/(T + beta)
    on top of the Hamiltonian's own gap (eACBN0 electrons, or the PBE gap at 0 K);
    ``'none'``: no scissor.
    """
    if args.electrons_save and args.gap_reference_save:
        sys.exit(
            '--gap-reference-save is a scissor target for PBE electrons; with --electrons-save '
            'the Hamiltonian already is the eACBN0 one (--target-gap-ev fine-tunes its gap).'
        )
    e0 = args.target_gap_ev
    if args.gap_reference_save:
        e0 = reference_gap(args.gap_reference_save, args.nk)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('  reference gap of %s: %.3f eV' % (args.gap_reference_save, e0))
    if e0 is not None:
        return None, (varshni_gap(e0, *args.varshni) if args.varshni else e0), 'target'
    if args.varshni:
        return varshni_shift(*args.varshni), None, 'relative'
    return None, None, 'none'



def long_range_model():
    """Dipole/quadrupole model of the coupling for polar crystals (or with a quadrupole file)."""
    if not LONG_RANGE:
        return None
    try:
        model = LongRangeModel.from_coupling(PHONON_DIR, SAVEDIR, quadrupoles=QUADRUPOLES)
    except ValueError:  # ph.x without epsil: no dielectric tensor, no long-range model
        return None
    return model if (model.is_polar() or model.has_quadrupoles) else None

def optical_absorption(
    projections, HRs, nscf, data_controller, args, scissor_ev=None, target_gap_ev=None, direct=None
):
    """Phonon-assisted + direct Im eps(omega) and alpha(omega) on the dense grids."""
    # n_r of alpha: the user constant, else n(omega) of the direct dielectric run, else 3.4 (EPW)
    if args.refractive_index is not None:
        refractive_index = args.refractive_index
    elif direct is not None:
        refractive_index = external_refractive_index(direct)
    else:
        refractive_index = 3.4
    return phonon_assisted_absorption_dense_q(
        projections,
        HRs,
        nscf['kpts_cryst'],
        nscf['bg'],
        nscf['at'],
        nscf['alat'],
        nscf['omega'],
        COUPLING_DIR,
        QGRID,
        COARSE_GRID,
        coupling_source=COUPLING_SOURCE,
        save_dir=SAVEDIR,
        long_range=long_range_model(),
        masses_amu=atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names']),
        nelec=NELEC,
        nk_dense=args.nk,
        nq_dense=args.nq,
        omega_ev=(args.omega_min, args.omega_max, args.omega_step),
        temps_k=args.temps,
        degauss_ev=args.degauss_ev,
        fsthick_ev=FSTHICK_EV,
        fermi_energy_ev='intrinsic' if args.emissivity else FERMI_ENERGY_EV,
        kk_omega_max_ev=KK_OMEGA_MAX_EV if args.emissivity and not args.direct_dielectric else None,
        refractive_index=refractive_index,
        scissor_ev=scissor_ev,
        target_gap_ev=target_gap_ev,
        scissor_velocity=args.scissor_velocity,
        screening_eps=args.screening_eps,
        screening_eps_dft=args.screening_eps_dft,
        position_dipoles=data_controller if args.position_dipoles else None,
        position_dipole_level=args.position_dipole_level,
        sym_rots=nscf['s_cryst'],
        tau_cryst=nscf['tau_cryst'],
        species=nscf['atom_names'],
        orbital_positions=nscf['orbital_positions'],
    )


def report(out, args, direct=None):
    """Print the gaps and write the spectra to OUTPUTDIR."""
    print(
        'PAOFLOW phonon-assisted absorption (k %d^3, q %d^3, degauss %.3f eV%s):'
        % (
            args.nk,
            args.nq,
            args.degauss_ev,
            ', position dipoles (%s)' % args.position_dipole_level if args.position_dipoles else '',
        )
    )
    n = out['refractive_index']
    print(
        '  alpha = omega Im eps / (n c) with n %s'
        % (
            '= %.3g' % n
            if np.ndim(n) == 0
            else 'of the direct dielectric run, %.3g-%.3g' % (n.min(), n.max())
        )
    )
    print(
        '  %s gaps: indirect %.3f eV, direct %.3f eV'
        % (out['electrons_dft'], out['indirect_gap_dft_ev'], out['direct_gap_dft_ev'])
    )
    if out['electrons_save']:
        print(
            '  band energies and eigenvectors from %s; coupling from %s'
            % (out['electrons_save'], out['coupling_save'])
        )
    print(screening_report(out))
    if abs(out['scissor_ev']).max() > 0.0:
        relative = out['scissor_mode'] == 'relative'
        print(
            '  scissor (%s velocities%s):'
            % (
                out['scissor_velocity'],
                '; Varshni shift relative to the Hamiltonian gap' if relative else '',
            )
        )
        for temp, delta, gap in zip(out['temps_k'], out['scissor_ev'], out['indirect_gap_ev_t']):
            print('    T = %7.1f K   scissor %+.3f eV   indirect gap %.3f eV' % (temp, delta, gap))
    for path in write_absorption_outputs(out, OUTPUTDIR):
        print('  wrote %s' % path)
    if not args.emissivity:
        return
    emissivity = thermal_emissivity(
        out,
        args.thickness_um * 1.0e-6,
        angles_deg=EMIS_ANGLES,
        direct_dielectric=direct,
        direct_scissor=args.direct_scissor,
    )
    print(
        'Thermal emissivity (slab, d = %g um; direct eps from %s, f-sum ratio %.2f):'
        % (
            args.thickness_um,
            args.direct_dielectric or 'Kramers-Kronig of this run',
            emissivity['f_sum_ratio'].mean(),
        )
    )
    if 'direct_onset_ev' in emissivity:
        print(
            '  external direct eps: onset %.2f eV, direct gap of the Hamiltonian %.2f eV; shifted by '
            '%+.3f ... %+.3f eV (%s)'
            % (
                emissivity['direct_onset_ev'],
                emissivity['direct_gap_h_ev'],
                emissivity['direct_scissor_ev'].min(),
                emissivity['direct_scissor_ev'].max(),
                emissivity['direct_scissor'],
            )
        )
    print('    T (K)   E_F (eV)   total slab   total opaque   Planck coverage')
    for it, temp in enumerate(emissivity['temps_k']):
        print(
            '  %7.1f  %9.4f  %11.4f  %13.4f  %16.4f'
            % (
                temp,
                out['fermi_levels_ev'][it],
                emissivity['total_slab'][it],
                emissivity['total_opaque'][it],
                emissivity['planck_coverage'][it],
            )
        )
    for path in write_emissivity_outputs(emissivity, OUTPUTDIR):
        print('  wrote %s' % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--nk', type=int, default=NK_DENSE, help='dense k-grid (default %(default)s)'
    )
    parser.add_argument(
        '--nq', type=int, default=NQ_DENSE, help='dense q-grid (default %(default)s)'
    )
    parser.add_argument('--temps', type=float, nargs='+', default=TEMPS_K, help='temperatures (K)')
    parser.add_argument('--degauss-ev', type=float, default=DEGAUSS_EV, help='delta smearing (eV)')
    parser.add_argument(
        '--omega-min', type=float, default=OMEGA_EV[0], help='min photon energy (eV)'
    )
    parser.add_argument(
        '--omega-max', type=float, default=OMEGA_EV[1], help='max photon energy (eV)'
    )
    parser.add_argument(
        '--omega-step', type=float, default=OMEGA_EV[2], help='photon-energy step (eV)'
    )
    parser.add_argument(
        '--position-dipoles',
        action=argparse.BooleanOptionalAction,
        default=POSITION_DIPOLES,
        help='position dipoles of the PAO orbitals in the velocity',
    )
    parser.add_argument(
        '--position-dipole-level',
        choices=('full', 'intra-atomic'),
        default=POSITION_DIPOLE_LEVEL,
        help='Löwdin-orthogonalised basis or on-site blocks only',
    )
    parser.add_argument(
        '--emissivity',
        action=argparse.BooleanOptionalAction,
        default=EMISSIVITY,
        help='also compute the thermal emissivity of a slab',
    )
    parser.add_argument(
        '--thickness-um',
        type=float,
        default=THICKNESS_UM,
        help='slab thickness in um (default %(default)s)',
    )
    parser.add_argument(
        '--target-gap-ev',
        type=float,
        default=TARGET_GAP_EV,
        help='target indirect gap (eV) of a scissor (0 K value with --varshni)',
    )
    parser.add_argument(
        '--gap-reference-save',
        default=GAP_REFERENCE_SAVE,
        help='QE save of an eACBN0 (DFT+U+V) run whose PAO gap is the target',
    )
    parser.add_argument(
        '--refractive-index',
        type=float,
        default=REFRACTIVE_INDEX,
        help='constant n_r of alpha (default: n(omega) of --direct-dielectric, else 3.4)',
    )
    parser.add_argument(
        '--electrons-save',
        default=ELECTRONS_SAVE,
        help='QE save of an eACBN0 (DFT+U+V) nscf in the same PAO basis: its PAO '
        'Hamiltonian replaces the PBE one (band energies and eigenvectors)',
    )
    parser.add_argument(
        '--varshni',
        type=float,
        nargs=2,
        default=VARSHNI,
        metavar=('ALPHA', 'BETA'),
        help='temperature-dependent gap E0 - alpha T^2/(T + beta): around the target, '
        'or relative to the Hamiltonian gap without one (Si: 4.73e-4 636)',
    )
    parser.add_argument(
        '--scissor-velocity',
        choices=('renormalized', 'ks'),
        default=SCISSOR_VELOCITY,
        help='velocities of the scissor-corrected Hamiltonian',
    )
    parser.add_argument(
        '--direct-scissor',
        choices=('same', 'thermal', 'none'),
        default=DIRECT_SCISSOR,
        help='shift of the direct dielectric function: the run scissor, its thermal part only, '
        'or none (default %(default)s)',
    )
    parser.add_argument(
        '--direct-dielectric',
        default=DIRECT_DIELECTRIC_DIR,
        help='output dir of an extended-basis dielectric_tensor run (epsr/epsi_*.dat)',
    )
    parser.add_argument(
        '--screening-eps',
        type=float,
        default=SCREENING_EPS,
        help='target dielectric constant of the Ponce screening correction of g (Si: 11.94)',
    )
    parser.add_argument(
        '--screening-eps-dft',
        type=float,
        default=SCREENING_EPS_DFT,
        help='DFT dielectric constant of the coupling (default: the .epb tensor)',
    )
    args = parser.parse_args()
    if args.nk % args.nq:
        sys.exit('--nk (%d) must be a multiple of --nq (%d).' % (args.nk, args.nq))
    if not os.path.isdir(SAVEDIR):
        sys.exit('%s not found. Run the pw.x nscf in nscf/ first.' % SAVEDIR)
    if COUPLING_SOURCE == 'epw' and not any(
        f.startswith(PREFIX + '.epb') for f in os.listdir(NSCF_DIR)
    ):
        sys.exit(
            'No %s.epb* files in %s. Run epw.x with epbwrite = .true. first.' % (PREFIX, NSCF_DIR)
        )
    if COUPLING_SOURCE == 'dfpt' and not (
        os.path.isdir(os.path.join(PHONON_DIR, '_ph0'))
        or any('dvscf' in f for f in os.listdir(PHONON_DIR))
    ):
        sys.exit(
            'No ph.x outputs (_ph0/ or *.dvscf_q*) in %s. Run ph.x with fildvscf first.'
            % PHONON_DIR
        )

    projections, HRs, nscf, data_controller, provenance = pao_electronic_structure(
        args.electrons_save
    )
    scissor_ev, target_gap_ev, provenance['scissor_mode'] = scissor_settings(args)
    direct = read_direct_dielectric(args.direct_dielectric) if args.direct_dielectric else None
    out = optical_absorption(
        projections, HRs, nscf, data_controller, args, scissor_ev, target_gap_ev, direct
    )
    out.update(provenance)

    # The result is identical on every rank; only rank 0 reports and writes files.
    if MPI.COMM_WORLD.Get_rank() == 0:
        report(out, args, direct)


if __name__ == '__main__':
    main()
