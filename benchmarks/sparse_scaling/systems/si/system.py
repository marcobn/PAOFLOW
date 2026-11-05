"""Bulk Si (example01): a cubic semiconductor without spin-orbit coupling.

The base cell has ``nawf = 18``, so ``d = 8`` is ``nawf = 4608``.  Si tests
everything that needs neither spin-orbit coupling nor a Fermi surface:
the DoS, Boltzmann transport with the Hall and Nernst tensors, the
band-diagonal conductivity, doping, the dielectric function, effective masses
and the Kubo charge conductivity.  The anomalous Hall conductivity
(``linear_response:ahc``) is checked as a zero, because time reversal forces
it to vanish.  The interior-window ladders and the solver sweep run on Si only:
they measure the algorithm, and Si is the system with the measured history.

Energies are relative to E_F (eV).  ``bond_order=36`` is a cutoff of 32.1 Bohr;
on example01 the rcut of a bond_order (Bohr) and the H(k) nonzeros per row are
2: 7.5/153, 4: 10.6/315, 8: 15.0/891, 12: 18.4/1503, 24: 26.0/4401, 36: 32.1/8487.
"""

import os

from config import REPO

# the properties of the energy-window pass, and of the full-spectrum pass
WINDOW_PROPERTIES = ['doping', 'conductivity', 'dielectric_tensor', 'transport', 'dos']
CURVATURE_PROPERTIES = [
    'transport',
    'effective_mass',
    'linear_response:cond',
    'linear_response:ahc',
]

SYSTEM = {
    'label': 'Si (example01)',
    'savedir': os.path.join(REPO, 'examples', 'qe_examples', 'example01', 'silicon.save'),
    # None: read the QE atomic projections (read_atomic_proj_QE).  A dict
    # {'basispath': ..., 'configuration': ...} calls projections() instead.
    'projections': None,
    'pthr': 0.95,
    'smearing': 'gauss',
    # the band path of the reference and of d = 0 of 'sparse' (doubling folds it):
    # PAOFLOW's default path of the fcc lattice
    'bands': {'ibrav': 2, 'nk': 500},
    'base_nfft': (48, 48, 48),
    'min_nfft': 3,
    # the components cubic symmetry and time reversal zero are the defaults
    # of observables.DEFAULT_ZEROS
    'zeros': {},
    'nonzero': [],
    'properties': {
        # chemical potential against temperature at a p-type doping (1/cm^3);
        # integrates the DoS from emin, so emin must be below every valence band
        'doping': {
            'emin': -12.0,
            'emax': 2.2,
            'ne': 1000,
            'tmin': 200.0,
            'tmax': 800.0,
            'nt': 7,
            'doping_conc': 1.0e18,
        },
        # band-diagonal sigma_ij(E): xx, and xy, which cubic symmetry zeroes
        'conductivity': {'emin': -2.0, 'emax': 2.0, 'ne': 500, 'cond_tensor': [[0, 0], [0, 1]]},
        # Kubo-Greenwood epsilon(omega).  The sparse sum runs over the window
        # bands only; the window top follows the highest property emax (this
        # emax, unless capped at the well-projected top), so epsilon'' is
        # complete up to it and epsilon' is not
        'dielectric_tensor': {'emin': 0.0, 'emax': 6.0, 'ne': 301, 'd_tensor': 'diag'},
        'transport': {
            'emin': -2.0,
            'emax': 2.0,
            'ne': 500,
            'tmin': 300.0,
            'tmax': 300.0,
            'nt': 1,
            'do_hall': False,
        },
        'effective_mass': {},
        # Kubo charge conductivity (xx, and xy, which cubic symmetry zeroes) and
        # the anomalous Hall conductivity (zero by time reversal).
        # linear_response only reads a_tensor when s_tensor is given too
        # (otherwise it takes arrays['a_tensor']), so both are passed
        'linear_response:cond': {
            'response': 'cond',
            'gamma': 0.05,
            's_tensor': [[2, 0, 1]],
            'a_tensor': [[0, 0], [0, 1]],
            'eminH': -2.0,
            'emaxH': 2.0,
            'esize': 400,
        },
        'linear_response:ahc': {
            'response': 'ahc',
            'gamma': 0.05,
            's_tensor': [[2, 0, 1]],
            'a_tensor': [[0, 1]],
            'eminH': -2.0,
            'emaxH': 2.0,
            'esize': 400,
        },
        'dos': {'emin': -12.0, 'emax': 2.2, 'ne': 1000, 'do_pdos': True},
    },
    # each pipeline is a ladder d = 0 .. max_doublings; 'slurm' overrides
    # CONFIG['resources']
    'pipelines': {
        # the dense pipeline on the primitive cell, d = 0 only: the reference
        # every sparse case is compared with
        'reference': {
            'sparse': False,
            'bands': True,
            'band_curvature': True,
            'properties': WINDOW_PROPERTIES + CURVATURE_PROPERTIES[1:],
            'property_args': {'transport': {'do_hall': True}},
            'max_doublings': 0,
            'slurm': {'time': '12:00:00'},
        },
        'sparse': {
            'sparse': True,
            'bands': True,
            # the cap is lifted, so the per-k dense LAPACK kernel runs until it
            # really runs out of memory or time: that is the limit to measure.
            # The window of the fused pass follows the property ranges
            'sparse_config': {'bond_order': 36, 'dense_n_max': 10**9},
            'properties': WINDOW_PROPERTIES,
            'max_doublings': 8,
        },
        # the full-spectrum pass: every state at every k-point, for the band
        # curvature (effective masses, Hall coefficient) and the interband
        # sums of linear_response.  It costs a full (nawf, nawf) eigh and a
        # few nawf^3 products per k-point; expect the largest sizes to hit
        # the walltime
        'sparse_curvature': {
            'sparse': True,
            'band_curvature': True,
            'sparse_config': {'bond_order': 36, 'dense_n_max': 10**9},
            'properties': CURVATURE_PROPERTIES,
            'property_args': {'transport': {'do_hall': True}},
            'max_doublings': 8,
        },
        'sparse_interior': {
            'sparse': True,
            'sparse_config': {
                'bond_order': 36,
                'interior_window': {'elo': -1.5, 'ehi': 2.0},
                'dense_n_max': 10**9,
            },
            # doping and dielectric_tensor need the states below the window
            # and are skipped (with a warning) by the engine
            'properties': WINDOW_PROPERTIES,
            # explicit bounds: the engine clamps the DoS and conductivity ranges
            # to the window, and the transport scan to 10 kT inside it
            # (0.26 eV at 300 K)
            'property_args': {
                'transport': {'emin': -1.2, 'emax': 1.7},
                'conductivity': {'emin': -1.2, 'emax': 1.7},
            },
            'max_doublings': 8,
        },
        # Si PAO hoppings decay slowly: one row of H(k) holds ~8500 nonzeros
        # at bond_order=36 (rcut 32 Bohr) but ~890 at bond_order=8 (15 Bohr),
        # so with 36 shells H(k) is not sparse until nawf >~ 10^4.  This
        # ladder measures the truly sparse regime; its accuracy cost shows in
        # the per-cell comparison with the reference.
        'sparse_interior_bo8': {
            'sparse': True,
            'sparse_config': {
                'bond_order': 8,
                'interior_window': {'elo': -1.5, 'ehi': 2.0},
                'dense_n_max': 10**9,
            },
            'properties': WINDOW_PROPERTIES,
            'property_args': {
                'transport': {'emin': -1.2, 'emax': 1.7},
                'conductivity': {'emin': -1.2, 'emax': 1.7},
            },
            'max_doublings': 8,
        },
    },
    # per-k solver sweep: LAPACK against ARPACK at fixed n over nev/n, to
    # calibrate dense_ratio, and the per-k scratch memory of each kernel
    'solver_sweep': {
        'enabled': True,
        'doublings': [3, 5, 7, 8],
        'nev_fractions': [0.01, 0.02, 0.05, 0.10, 0.125, 0.20, 0.30, 0.50],
        'nk_probe': 3,
        # stop raising nev for ARPACK once it is this many times slower than LAPACK
        'stop_ratio': 20.0,
        'bond_order': 36,
        'slurm': {'time': '12:00:00'},
    },
}
