#!/usr/bin/env python3
"""Pb: configuration of the electron-phonon workflows (one folder per property).

Edit the values here; every driver of the workflow folders reads them through
``PAOFLOW.elphon.workflows.load_system``.  Paths are relative to this directory.
"""

import os

from PAOFLOW.elphon.workflows import ElphonSystem

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'Pb'

SYSTEM = ElphonSystem(
    prefix='pb',
    root=HERE,
    coupling_source='dfpt',  # 'dfpt': PAOFLOW's vertex from ph.x dvscf + the nscf save; 'epw': .epb files
    qe_dir='qe',  # qe/phonon (scf, ph.x), qe/nscf (<prefix>.save), qe/optics
    coarse_grid=(6, 6, 6),  # nscf k-grid
    qgrid=(6, 6, 6),  # ph.x q-grid (divides the k-grid)
    masses_amu=[207.2],  # one per species, ATOMIC_SPECIES order
    nelec=14,  # valence electrons
    basis_dir='BASIS_PS',  # paoflow-genbasis-ps tables
    configuration='standard',
    pthr=0.95,  # projectability threshold
    long_range=True,  # dipole (Fröhlich) + quadrupole model when polar or quadrupoles is set
    quadrupoles=None,  # 'coupling/quadrupole.fmt' from coupling/quadrupoles.py
    cache_dir='cache',  # coarse vertex g(R_e, R_p), shared by every workflow
    spin_degeneracy=None,  # 2 without spin-orbit coupling, 1 with it (None: from the save)
)

# Defaults of the workflow drivers (command-line options override them).
SUPERCONDUCTIVITY = dict(
    nk=48,  # dense electron grid per axis (EPW nkf)
    nq=24,  # dense phonon grid per axis (EPW nqf); nk % nq == 0
    sigma_ev=0.05,  # Fermi-surface smearing (EPW degaussw)
    adaptive_smearing=None,  # Yates prefactor of the adaptive widths (None: fixed smearing)
    mu_star=0.1,  # Coulomb pseudopotential
    transport=False,  # also alpha^2F_tr (transport/resistivity.py)
    fs_coupling=False,  # Fermi-surface pair coupling for me_aniso.py (needs nk == nq)
    fsthick_ev=None,  # Fermi window of the anisotropic states (EPW fsthick)
    me_temps=[0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 4.0, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0],  # temperatures (K) of the isotropic Migdal-Eliashberg equations
    me_linear_temps=(0.25, 6.25, 25),  # (Tmin, Tmax, nstemp) of the linearised kernel
    me_aniso_temps=[0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 4.0, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0],  # temperatures (K) of the anisotropic equations
    wscut=0.1,  # Matsubara cutoff (eV, EPW wscut)
    wscut_aniso=0.5,  # Matsubara cutoff of the anisotropic equations (eV)
    degaussq_mev=0.15,  # phonon smearing of alpha^2F for the equations (meV, EPW degaussq)
    degaussq_linear_mev=0.5,  # ... for the linearised kernel (meV)
    npade=90.0,  # percentage of Matsubara points of the Padé approximant
)
OPTICS = dict(
    dielectric=dict(nfft=36, emin=0.002, emax=6.0, ne=3000, delta=0.1, intraband=False, smearing='gauss'),  # optics/dielectric.py: interpolated grid, window and broadening
)
TRANSPORT = dict(
    temps_k=[20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0],  # temperatures (K)
    rho0_uohm_cm=0.0,  # residual resistivity (micro-ohm cm)
    b_ee_per_ev=0.0,  # Fermi-liquid coefficient b (1/eV)
    omega_ev=(0.005, 2.0, 0.005),  # photon energies (min, max, step) of the metal optics
    thickness_um=float('inf'),  # film thickness (um; inf = opaque)
    angles=(0.0, 30.0, 60.0),  # emission angles (deg)
    interband_dielectric_dir=None,  # output of optics/dielectric.py --interband-only
    extended_drude=True,  # frequency-dependent rate + mass enhancement (False: DC rate)
)
ARPES = dict(
    photon_ev=21.2,  # photon energy (He I: 21.2)
    work_function_ev=4.25,  # work function (eV)
    polarization=(1.0, 0.0, 1.0),  # light polarisation (Cartesian)
    direction=(0.5, 0.0, 0.5),  # end point of the line from Gamma (crystal coordinates)
    direction_label='X',  # its name
    window=(-0.08, 0.02, 401),  # (E_min, E_max, n) of the energy window (eV, relative to E_F)
    resolution=(0.002, 0.0),  # (energy, momentum) resolution (eV, 1/A)
    self_energy=['isotropic'],  # any of 'isotropic', 'anisotropic', 'fan-migdal'
    temps_k=[10.0],  # temperatures (K)
    segment_ev=0.12,  # segment of +- this energy across the Fermi crossing (None: the full line)
    npts=81,  # points of the path
    nq=18,  # dense q-grid of the Fan-Migdal run
    eta_ev=0.005,  # Lorentzian width / floor (eV) of the Fan-Migdal run
    adaptive_smearing=0.7,  # adaptive widths a |v| dq of the Fan-Migdal run (metals)
    intensity_power=1.0,  # power scale of the band maps (weak bands next to strong ones)
)
PHONONS = dict(
    units='cm-1',  # 'cm-1' or 'THz'
    mesh=(24, 24, 24),  # q-mesh of the DOS
    sigma=3.0,  # DOS broadening
)
COUPLING = {}
