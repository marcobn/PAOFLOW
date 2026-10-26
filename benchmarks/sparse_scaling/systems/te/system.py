"""Trigonal Te (example11): a chiral semiconductor with spin-orbit coupling.

The base cell has three atoms and ``nawf = 54`` (5s, 5p, 4d with spin).  The
ladder runs to ``d = 4`` (``nawf = 864``) for now; ``d = 6`` would be
``nawf = 3456``, about Si's ``d = 7.6``.  Te has no
inversion centre, so its bands are spin split away from the time-reversal
invariant points and carry a spin texture.  Te tests:

- the Rashba-Edelstein response, band-diagonal through the spin and orbital
  textures (``rashba_edelstein``) and by the Kubo route
  (``linear_response:ree``).  Point group 32 allows the diagonal
  xx = yy and zz, and zeroes the off-diagonal xy;
- ``spin_texture`` on the mesh.  Its per-band files are folded by doubling,
  so only d = 0 is compared with the reference;
- transport and the conductivities of an anisotropic (trigonal)
  semiconductor.

The PAO shift is 0.47 eV, just above the conduction-band edge, so the
well-projected range ends at 0.27 eV: the properties stay at the valence-band
top, as in example11.  ``dielectric_tensor`` is left out, because the
conduction bands it needs are not projectable.  ``doping`` is left out too:
``solve_for_mu`` weights the DoS by 2 for spin, which double counts the
spinor bands of a spin-orbit run, so mu lands in the 4d shell (-36 eV), in
the dense pipeline as in the sparse one.  The DoS starts at -14 eV, below the
5s band and above the 4d semicore states (-38 eV), whose peak would dominate
its deviation norm; the bottom-of-the-spectrum window solves them anyway.

Energies are relative to E_F (eV). Unlike Si, the bond list keeps every H(R)
element (``hopping_threshold=0``): the largest shell cutoff the coarse QE
k-grid represents (47 shells, 22.0 Bohr) moves the bands near E_F by up to 110
meV on the band path, and the Hall-type curves with them, while without it
sparse d = 0 matches the dense reference (bands within 35 meV of it near E_F,
the remainder being the off-grid interpolation convention). H(k) is dense at
these sizes anyway. The 35 meV are the coarse 6x6x4 QE grid: off it, the dense
zero padding and the sparse Nyquist split interpolate differently."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW_PROPERTIES = [
    'conductivity',
    'transport',
    'spin_texture',
    'rashba_edelstein',
    'dos',
]
FULL_PROPERTIES = [
    'transport',
    'effective_mass',
    'linear_response:ree',
    'linear_response:cond',
]
SPARSE = {'hopping_threshold': 0.0, 'dense_n_max': 10**9}

SYSTEM = {
    'label': 'Te (example11, chiral, spin-orbit)',
    # extracted from the QE test assets (example11), not under version control
    'savedir': os.path.join(HERE, 'Te-L.save'),
    'projections': None,
    'pthr': 0.95,
    'smearing': 'gauss',
    # the band path of the reference and of d = 0 of 'sparse' (doubling folds it):
    # PAOFLOW's default path of the hexagonal lattice
    'bands': {'ibrav': 4, 'nk': 500},
    # c is 1.29 a; 36 halves twice along c, as d = 6 needs
    'base_nfft': (48, 48, 36),
    'min_nfft': 3,
    # point group 32 zeroes these, but at the 12 x 12 x 8 smoke mesh they are
    # 10-20% of xx (REE) and of the order of the allowed components (Hall,
    # Nernst), in the dense reference as in the sparse runs: check them at
    # the ladder's mesh before reading them as zeros.  R_xy,z (field along c)
    # is the smallest allowed Hall component, so the repeated-index RMS is
    # measured on R_xz,y instead
    'zeros': {
        'hall_zero': 'hall_xzy',
        'ree_spin_xy': 'ree_spin_xx',
        'ree_orbital_xy': 'ree_orbital_xx',
        'lr_ree_even_xy': 'lr_ree_even_xx',
    },
    'nonzero': [],
    'properties': {
        'conductivity': {
            'emin': -1.5,
            'emax': 0.2,
            'ne': 400,
            'cond_tensor': [[0, 0], [0, 1], [2, 2]],
        },
        'transport': {
            'emin': -1.5,
            'emax': 0.0,
            'ne': 400,
            'tmin': 300.0,
            'tmax': 300.0,
            'nt': 1,
            'do_hall': False,
        },
        'effective_mass': {'emin': -1.0, 'emax': 0.2},
        'spin_texture': {'fermi_dw': -0.5, 'fermi_up': 0.0},
        'rashba_edelstein': {'emin': -0.5, 'emax': 0.0, 'ne': 501, 'spin': True, 'orbital': True},
        # rows [i, j]: zz, xx and xy.  linear_response only reads a_tensor when
        # s_tensor is given too
        'linear_response:ree': {
            'response': 'ree',
            'gamma': 0.05,
            's_tensor': [[2, 0, 1]],
            'a_tensor': [[2, 2], [0, 0], [0, 1]],
            'eminH': -1.0,
            'emaxH': 0.2,
            'esize': 300,
        },
        'linear_response:cond': {
            'response': 'cond',
            'gamma': 0.05,
            's_tensor': [[2, 0, 1]],
            'a_tensor': [[0, 0], [0, 1], [2, 2]],
            'eminH': -1.5,
            'emaxH': 0.2,
            'esize': 300,
        },
        'dos': {'emin': -14.0, 'emax': 0.25, 'ne': 1500, 'do_pdos': True},
    },
    'pipelines': {
        'reference': {
            'sparse': False,
            'bands': True,
            'band_curvature': True,
            'properties': WINDOW_PROPERTIES + FULL_PROPERTIES[1:],
            'property_args': {'transport': {'do_hall': True}},
            'max_doublings': 0,
            'slurm': {'time': '12:00:00'},
        },
        'sparse': {
            'sparse': True,
            'bands': True,
            'sparse_config': SPARSE,
            'properties': WINDOW_PROPERTIES,
            'max_doublings': 4,
        },
        'sparse_curvature': {
            'sparse': True,
            'band_curvature': True,
            'sparse_config': SPARSE,
            'properties': FULL_PROPERTIES,
            'property_args': {'transport': {'do_hall': True}},
            'max_doublings': 4,
        },
    },
    'solver_sweep': {'enabled': False},
}
