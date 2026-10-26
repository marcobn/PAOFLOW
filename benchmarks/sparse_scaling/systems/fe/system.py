"""Bulk bcc Fe (example04): a ferromagnetic metal with spin-orbit coupling.

The base cell has ``nawf = 26`` (3s, 4s, 3p, 4p, 3d with spin).  The ladder runs
to ``d = 4`` (``nawf = 416``) for now; ``d = 7`` would be ``nawf = 3328``.  The
magnetization points along z (``starting_magnetization``,
default angles), which breaks time reversal.  Fe tests:

- the anomalous Hall conductivity sigma_xy, non-zero here, by the
  Berry-curvature route (``anomalous_Hall``) and by the Kubo route
  (``linear_response:ahc``), with sigma_yz, which the magnetization along z
  zeroes;
- the magnetic circular dichroism (``anomalous_Hall(do_ac=True)``);
- the Si properties on a magnetic metal: DoS, Boltzmann transport with the
  Hall and Nernst tensors, the band-diagonal and Kubo conductivities, the
  interband dielectric function and the effective masses, on the
  bottom-of-the-spectrum window and on an interior window through E_F.

The magnetization lowers the symmetry of the bands from cubic to tetragonal
(4/mmm for E(k)): xx = yy differs slightly from zz, but every off-diagonal
Boltzmann component and the Hall and Nernst components with a repeated index
still vanish.  ``doping`` is left out (a metal).

Energies are relative to E_F (eV). The basis holds the 3s and 3p semicore
states (down to -89 eV), which the bottom-of-the-spectrum window solves too.
The PAO shift is 6.94 eV, so the well-projected range ends at 6.74 eV. Unlike
Si, the bond list keeps every H(R) element (``hopping_threshold=0``): the
largest shell cutoff the coarse QE k-grid represents (25 shells, 23.0 Bohr)
moves the bands near E_F by up to 140 meV on the band path, and the Hall-type
curves with them, while without it sparse d = 0 matches the dense reference
(bands within 7 meV of it near E_F, the remainder being the off-grid
interpolation convention). H(k) is dense at these sizes anyway. The full bond
list peaks at about 63 GB per rank while doubling to d = 7 (8 GB at d = 4),
and every rank holds it."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW_PROPERTIES = ['conductivity', 'dielectric_tensor', 'transport', 'dos']
FULL_PROPERTIES = [
    'transport',
    'effective_mass',
    'anomalous_Hall',
    'linear_response:ahc',
    'linear_response:cond',
]
INTERIOR_PROPERTIES = ['conductivity', 'transport', 'dos']
SPARSE = {'hopping_threshold': 0.0, 'dense_n_max': 10**9}
# the bond list is replicated on every rank: d = 7 needs two ranks on a 200 GB
# node ('slurm': {'ntasks': 2, 'cpus_per_task': 6, 'npool': 2} on the sparse
# pipelines); up to d = 4 the defaults fit

SYSTEM = {
    'label': 'Fe (example04, ferromagnet, spin-orbit)',
    # extracted from the QE test assets (example04), not under version control
    'savedir': os.path.join(HERE, 'fe.save'),
    'projections': None,
    'pthr': 0.95,
    'smearing': 'gauss',
    # the band path of the reference and of d = 0 of 'sparse' (doubling folds it):
    # PAOFLOW's default path of the bcc lattice
    'bands': {'ibrav': 3, 'nk': 500},
    'base_nfft': (48, 48, 48),
    'min_nfft': 3,
    # sigma_yz vanishes with the magnetization along z; the anomalous Hall
    # sigma_xy is the signal here, not a zero
    'zeros': {
        'berry_ahc_yz': 'berry_ahc_xy',
        'lr_ahc_yz': 'lr_ahc_xy',
    },
    'nonzero': ['lr_ahc_xy'],
    'properties': {
        'conductivity': {'emin': -3.0, 'emax': 3.0, 'ne': 600, 'cond_tensor': [[0, 0], [0, 1]]},
        'dielectric_tensor': {'emin': 0.0, 'emax': 6.0, 'ne': 301, 'd_tensor': 'diag'},
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
        'anomalous_Hall': {
            'emin': -3.0,
            'emax': 3.0,
            'ne': 601,
            'do_ac': True,
            'a_tensor': [[0, 1], [1, 2]],
        },
        # linear_response only reads a_tensor when s_tensor is given too
        'linear_response:ahc': {
            'response': 'ahc',
            'gamma': 0.05,
            's_tensor': [[2, 0, 1]],
            'a_tensor': [[0, 1], [1, 2]],
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
        'dos': {'emin': -9.0, 'emax': 6.0, 'ne': 1500, 'do_pdos': True},
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
        'sparse_interior': {
            'sparse': True,
            'sparse_config': dict(SPARSE, interior_window={'elo': -2.0, 'ehi': 2.0}),
            'properties': INTERIOR_PROPERTIES,
            'property_args': {
                'transport': {'emin': -1.4, 'emax': 1.4},
                'conductivity': {'emin': -1.4, 'emax': 1.4},
            },
            'max_doublings': 4,
        },
    },
    'solver_sweep': {'enabled': False},
}
