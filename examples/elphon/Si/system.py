#!/usr/bin/env python3
"""Si: configuration of the electron-phonon workflows (one folder per property).

Edit the values here; every driver of the workflow folders reads them through
``PAOFLOW.elphon.workflows.load_system``.  Paths are relative to this directory.
"""

import os

from PAOFLOW.elphon.workflows import ElphonSystem

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'Si'

SYSTEM = ElphonSystem(
    prefix='Si2',
    root=HERE,
    coupling_source='dfpt',  # 'dfpt': PAOFLOW's vertex from ph.x dvscf + the nscf save; 'epw': .epb files
    qe_dir='qe',  # qe/phonon (scf, ph.x), qe/nscf (<prefix>.save), qe/optics
    coarse_grid=(6, 6, 6),  # nscf k-grid
    qgrid=(3, 3, 3),  # ph.x q-grid (divides the k-grid)
    masses_amu=[28.085],  # one per species, ATOMIC_SPECIES order
    nelec=8,  # valence electrons
    basis_dir='BASIS_PS',  # paoflow-genbasis-ps tables
    configuration='standard',
    pthr=0.95,  # projectability threshold
    long_range=True,  # dipole (Fröhlich) + quadrupole model when polar or quadrupoles is set
    quadrupoles=None,  # 'coupling/quadrupole.fmt' from coupling/quadrupoles.py
    cache_dir='cache',  # coarse vertex g(R_e, R_p), shared by every workflow
    spin_degeneracy=None,  # 2 without spin-orbit coupling, 1 with it (None: from the save)
)

# Defaults of the workflow drivers (command-line options override them).
SUPERCONDUCTIVITY = {}
OPTICS = dict(
    nk=12,  # dense electron grid per axis (EPW nkf)
    nq=6,  # dense phonon grid per axis (EPW nqf); nk % nq == 0
    omega_ev=(0.05, 3.0, 0.05),  # photon energies (min, max, step)
    temps_k=[300.0],  # temperatures (K)
    degauss_ev=0.05,  # delta smearing (EPW degaussw); the floor with adaptive_smearing
    adaptive_smearing=None,  # Yates prefactor of the adaptive widths (None: fixed smearing)
    fsthick_ev=4.0,  # states within this window of the Fermi level (EPW fsthick)
    fermi_energy_ev=None,
    refractive_index=None,  # constant n_r of alpha (None: n(omega) of the direct run, else 3.4)
    position_dipoles=True,  # position dipoles of the PAO orbitals in the velocity
    position_dipole_level='full',
    target_gap_ev=None,  # scissor target (eV); None: the Hamiltonian gap
    gap_reference_save=None,  # QE save of an eACBN0 run whose PAO gap is the target
    varshni=None,  # (alpha eV/K, beta K) of a temperature-dependent gap
    electrons_save=None,  # QE save of an eACBN0 nscf in the same basis (eACBN0 electrons)
    screening_eps=None,  # target dielectric constant of the Poncé screening correction
    screening_eps_dft=None,
    scissor_velocity='renormalized',
    direct_scissor='same',
    emissivity=False,  # also the thermal emissivity of a slab
    thickness_um=500.0,  # slab thickness (um)
    angles=(0.0, 30.0, 60.0),  # emission angles (deg)
    direct_dielectric_dir=None,  # output of optics/dielectric.py (extended basis)
    kk_omega_max_ev=10.0,  # Kramers-Kronig range of the direct term without an external eps
    dielectric=dict(nfft=24, emin=0.01, emax=10.0, ne=1000, delta=0.1, intraband=True),  # optics/dielectric.py: interpolated grid, window and broadening
)
TRANSPORT = dict(
    nk=48,  # dense electron grid per axis (mobilities need >= 48)
    nq=48,  # dense phonon grid per axis; nk % nq == 0
    temps_k=[200.0, 250.0, 300.0, 350.0, 400.0, 500.0],  # temperatures (K)
    doping_cm3=0.0,  # net doping (cm^-3; > 0 n type, < 0 p type, 0 intrinsic)
    window_ev=0.3,  # states within this window of the band edges (EPW fsthick)
    smearing_ev=0.01,  # delta smearing (EPW degaussw); the floor with adaptive_smearing
    adaptive_smearing=None,  # Yates prefactor of the adaptive widths (None: fixed smearing)
    target_gap_ev=None,  # scissor of the carrier statistics (eV)
    varshni=None,  # (alpha eV/K, beta K) of a temperature-dependent gap
    electrons_save=None,  # QE save of an eACBN0 nscf in the same basis (eACBN0 electrons)
    screening_eps=None,  # target dielectric constant of the Poncé screening correction
    screening_eps_dft=None,
)
ARPES = dict(
    photon_ev=21.2,  # photon energy (He I: 21.2)
    work_function_ev=4.8,  # work function (eV)
    polarization=(1.0, 0.0, 1.0),  # light polarisation (Cartesian)
    direction=(0.5, 0.0, 0.5),  # end point of the line from Gamma (crystal coordinates)
    direction_label='X',  # its name
    window=(-13.0, 1.5, 581),  # (E_min, E_max, n) of the energy window (eV, relative to E_F)
    resolution=(0.05, 0.0),  # (energy, momentum) resolution (eV, 1/A)
    self_energy=['fan-migdal'],  # any of 'isotropic', 'anisotropic', 'fan-migdal'
    temps_k=[10.0, 300.0, 600.0],  # temperatures (K)
    segment_ev=None,  # segment of +- this energy across the Fermi crossing (None: the full line)
    npts=41,  # points of the path
    nq=12,  # dense q-grid of the Fan-Migdal run
    eta_ev=0.02,  # Lorentzian width / floor (eV) of the Fan-Migdal run
    adaptive_smearing=None,  # adaptive widths a |v| dq of the Fan-Migdal run (metals)
    intensity_power=0.3,  # power scale of the band maps (weak bands next to strong ones)
)
PHONONS = dict(
    units='cm-1',  # 'cm-1' or 'THz'
    mesh=(24, 24, 24),  # q-mesh of the DOS
    sigma=3.0,  # DOS broadening
)
COUPLING = dict(
    nk=24,  # dense grid of the path points of long_range_check.py
    direction=(1.0, 0.0, 0.0),  # Cartesian direction of q (2 pi / alat)
)
