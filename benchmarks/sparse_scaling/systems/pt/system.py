"""Bulk Pt (example05): a cubic metal with spin-orbit coupling from the DFT run.

The base cell has ``nawf = 18`` (6s, 6p, 5d with spin), the same as Si, so the
sizes match the Si ladder.  The ladder runs to ``d = 4`` (``nawf = 288``) for
now; ``d = 7`` would be ``nawf = 2304``.  Pt tests:

- the spin and orbital Hall conductivities, by the Berry-curvature route
  (``spin_Hall``, ``orbital_Hall``) and by the Kubo route
  (``linear_response:shc``).  They are the first properties of the
  benchmark that use the doubled spin and orbital operators ``Sj``/``Lj``;
- the spin circular dichroism (``spin_Hall(do_ac=True)``);
- every Si property that has a Fermi surface here: DoS, Boltzmann transport
  with the Hall and Nernst tensors, the band-diagonal and Kubo
  conductivities, the interband dielectric function and the effective
  masses, on the bottom-of-the-spectrum window and on an interior window
  through E_F.

Pt keeps inversion and time reversal, so every band is a Kramers pair: the
anomalous Hall conductivity, the Rashba-Edelstein response and the spin
texture vanish or have no fixed direction, and Fe and Te test those.
``doping`` is left out (a metal).

Energies are relative to E_F (eV). The PAO shift is 5.59 eV, so the well-
projected range ends at 5.39 eV. Unlike Si, the bond list keeps every H(R)
element (``hopping_threshold=0``): the largest shell cutoff the coarse QE
k-grid represents (23 shells, 25.7 Bohr) moves the bands near E_F by up to 47
meV on the band path, and the Hall-type curves with them, while without it
sparse d = 0 matches the dense reference (bands within 3 meV of it near E_F,
the remainder being the off-grid interpolation convention). H(k) is dense at
these sizes anyway."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW_PROPERTIES = ['conductivity', 'dielectric_tensor', 'transport', 'dos']
FULL_PROPERTIES = [
    'transport',
    'effective_mass',
    'spin_Hall',
    'orbital_Hall',
    'linear_response:shc',
    'linear_response:cond',
]
INTERIOR_PROPERTIES = ['conductivity', 'transport', 'dos']
SPARSE = {'hopping_threshold': 0.0, 'dense_n_max': 10**9}

SYSTEM = {
    'label': 'Pt (example05, spin-orbit)',
    # extracted from the QE test assets (example05), not under version control
    'savedir': os.path.join(HERE, 'pt.save'),
    'projections': None,
    'pthr': 0.95,
    'smearing': 'gauss',
    # the band path of the reference and of d = 0 of 'sparse' (doubling folds it):
    # PAOFLOW's default path of the fcc lattice
    'bands': {'ibrav': 2, 'nk': 500},
    'base_nfft': (48, 48, 48),
    'min_nfft': 3,
    # the components cubic symmetry zeroes, on the scale of an allowed one
    # (added to observables.DEFAULT_ZEROS)
    'zeros': {
        'berry_shc_z_xz': 'berry_shc_z_xy',
        'berry_ohc_z_xz': 'berry_ohc_z_xy',
        'lr_shc_even_z_xz': 'lr_shc_even_z_xy',
    },
    'nonzero': [],
    'properties': {
        'conductivity': {'emin': -3.0, 'emax': 3.0, 'ne': 600, 'cond_tensor': [[0, 0], [0, 1]]},
        # interband only: the Drude term of a metal is not in the Kubo-Greenwood sum
        'dielectric_tensor': {'emin': 0.0, 'emax': 5.0, 'ne': 251, 'd_tensor': 'diag'},
        'transport': {
            'emin': -3.0,
            'emax': 3.0,
            'ne': 600,
            'tmin': 300.0,
            'tmax': 300.0,
            'nt': 1,
            'do_hall': False,
        },
        'effective_mass': {},
        # rows [i, j, s]: sigma^z_xy and sigma^x_yz (equal in a cubic crystal),
        # and sigma^z_xz, which cubic symmetry zeroes
        'spin_Hall': {
            'emin': -3.0,
            'emax': 3.0,
            'ne': 601,
            'do_ac': True,
            's_tensor': [[0, 1, 2], [1, 2, 0], [0, 2, 2]],
        },
        'orbital_Hall': {
            'emin': -3.0,
            'emax': 3.0,
            'ne': 601,
            'o_tensor': [[0, 1, 2], [1, 2, 0], [0, 2, 2]],
        },
        # rows [s, j, i]: the same three components.  linear_response only
        # reads a_tensor when s_tensor is given too, so both are passed
        'linear_response:shc': {
            'response': 'shc',
            'gamma': 0.05,
            's_tensor': [[2, 0, 1], [0, 1, 2], [2, 0, 2]],
            'a_tensor': [[0, 0]],
            'eminH': -3.0,
            'emaxH': 3.0,
            'esize': 300,
        },
        'linear_response:cond': {
            'response': 'cond',
            'gamma': 0.05,
            's_tensor': [[2, 0, 1]],
            'a_tensor': [[0, 0], [0, 1]],
            'eminH': -3.0,
            'emaxH': 3.0,
            'esize': 300,
        },
        'dos': {'emin': -9.0, 'emax': 5.0, 'ne': 1400, 'do_pdos': True},
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
        # every property that sums over all states (spin/orbital Hall, Kubo)
        # or needs the band curvature (Hall, Nernst, effective masses)
        'sparse_curvature': {
            'sparse': True,
            'band_curvature': True,
            'sparse_config': SPARSE,
            'properties': FULL_PROPERTIES,
            'property_args': {'transport': {'do_hall': True}},
            'max_doublings': 4,
        },
        # a window through the Fermi surface: the use case of a metal
        'sparse_interior': {
            'sparse': True,
            'sparse_config': dict(SPARSE, interior_window={'elo': -2.0, 'ehi': 2.0}),
            'properties': INTERIOR_PROPERTIES,
            # 10 kT (0.26 eV at 300 K) and the smearing inside the window edges
            'property_args': {
                'transport': {'emin': -1.4, 'emax': 1.4},
                'conductivity': {'emin': -1.4, 'emax': 1.4},
            },
            'max_doublings': 4,
        },
    },
    'solver_sweep': {'enabled': False},
}
