#!/usr/bin/env python3
"""3C-SiC: configuration of the electron-phonon workflows (one folder per property).

Edit the values here; every driver of the workflow folders reads them through
``PAOFLOW.elphon.workflows.load_system``.  Paths are relative to this directory.
"""

import os

from PAOFLOW.elphon.workflows import ElphonSystem

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = '3C-SiC'

SYSTEM = ElphonSystem(
    prefix='sic',
    root=HERE,
    coupling_source='dfpt',  # 'dfpt': PAOFLOW's vertex from ph.x dvscf + the nscf save; 'epw': .epb files
    qe_dir='qe',  # qe/phonon (scf, ph.x), qe/nscf (<prefix>.save), qe/optics
    coarse_grid=(6, 6, 6),  # nscf k-grid
    qgrid=(6, 6, 6),  # ph.x q-grid (divides the k-grid)
    masses_amu=[28.0855, 12.0107],  # one per species, ATOMIC_SPECIES order
    nelec=8,  # valence electrons
    basis_dir='BASIS_PS',  # paoflow-genbasis-ps tables
    configuration='standard',
    pthr=0.9,  # projectability threshold
    long_range=True,  # dipole (Fröhlich) + quadrupole model when polar or quadrupoles is set
    quadrupoles=None,  # 'coupling/quadrupole.fmt' from coupling/quadrupoles.py
    cache_dir='cache',  # coarse vertex g(R_e, R_p), shared by every workflow
    spin_degeneracy=None,  # 2 without spin-orbit coupling, 1 with it (None: from the save)
)

# Defaults of the workflow drivers (command-line options override them).
SUPERCONDUCTIVITY = {}
OPTICS = {}
TRANSPORT = dict(
    nk=36,  # dense electron grid per axis (mobilities need >= 48)
    nq=36,  # dense phonon grid per axis; nk % nq == 0
    temps_k=[200.0, 300.0, 400.0],  # temperatures (K)
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
ARPES = {}
PHONONS = dict(
    units='cm-1',  # 'cm-1' or 'THz'
    mesh=(24, 24, 24),  # q-mesh of the DOS
    sigma=3.0,  # DOS broadening
)
COUPLING = dict(
    nk=24,  # dense grid of the path points of long_range_check.py
    direction=(1.0, 0.0, 0.0),  # Cartesian direction of q (2 pi / alat)
)
