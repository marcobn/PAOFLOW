"""Settings of the sparse-scaling benchmark: the single file a user edits.

The benchmark grows bulk Si by ``doubling_Hamiltonian`` and runs every
property that makes sense for it (no spin-orbit coupling) through each sparse
pipeline, size by size.  Nothing is passed on the command line; :mod:`submit`
reads ``CONFIG`` from here, expands it into cases with
:func:`ladder.build_cases` and writes them to ``<output_root>/cases.json``,
which every Slurm array task reads back.

Sizes
-----
Doubling ``d`` times multiplies ``nawf`` by ``2**d``.  The doublings are
spread over the three axes in turn, (1,0,0), (1,1,0), (1,1,1), (2,1,1), ...,
so consecutive sizes differ by a factor of two.  The example01 base cell has
``nawf = 18``, so ``d = 8`` is ``nawf = 4608`` and ``d = 9`` is 9216.

k-mesh
------
``base_nfft`` is the property mesh of the undoubled cell.  Each doubling
along an axis halves the mesh along it (floored at ``min_nfft``), so the
physical k-point density, and therefore the per-cell result, stays the same
at every size.

Reference
---------
The ``reference`` pipeline is the dense PAOFLOW pipeline on the primitive
cell: no ``doubling_Hamiltonian`` call, the full ``base_nfft`` mesh, and every
property of every sparse pipeline.  Each sparse case is compared with it per
primitive cell.

Properties
----------
``properties`` holds each property's arguments, in the order they run.
``dos`` comes last because ``doping`` also writes ``dosdk_0.dat`` (over its
own energy range).  A pipeline lists the properties it runs, and may override
their arguments in ``property_args``.  ``band_curvature`` makes the pass solve
the full spectrum at every k-point (needed by ``effective_mass`` and the Hall
coefficient of ``transport``), so it has its own pipeline.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# the properties of the energy-window pass, and of the full-spectrum pass
WINDOW_PROPERTIES = ['doping', 'conductivity', 'dielectric_tensor', 'transport', 'dos']
CURVATURE_PROPERTIES = ['transport', 'effective_mass']

CONFIG = {
    # where cases.json, the job scripts and one directory per case go
    'output_root': os.path.join(HERE, 'runs'),
    # ------------------------------------------------------------------
    # system
    # ------------------------------------------------------------------
    'system': {
        'label': 'Si (example01)',
        'savedir': os.path.join(REPO, 'examples', 'qe_examples', 'example01', 'silicon.save'),
        # None: read the QE atomic projections (read_atomic_proj_QE).  A dict
        # {'basispath': ..., 'configuration': ...} calls projections() instead.
        'projections': None,
        'pthr': 0.95,
        'smearing': 'gauss',
    },
    'base_nfft': (48, 48, 48),
    'min_nfft': 3,
    # every pipeline adds its own 'sparse_config'; on example01 the rcut of a
    # bond_order (Bohr) and the H(k) nonzeros per row are 2: 7.5/153,
    # 4: 10.6/315, 8: 15.0/891, 12: 18.4/1503, 24: 26.0/4401, 36: 32.1/8487
    # random on-site energies (meV, Gaussian sigma) added to the sparse bond
    # list after doubling.  They break the exact folding degeneracies, the
    # worst case of ARPACK, at the price of the exact per-cell reference.
    # Sparse pipelines only; 0 disables.
    'disorder_meV': 0.0,
    'disorder_seed': 1234,
    # ------------------------------------------------------------------
    # properties (energies relative to E_F, eV), in the order they run
    # ------------------------------------------------------------------
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
        # bands only, so epsilon'' is complete up to about the window top
        # (energy_window emax + margin = 3.2 eV here), epsilon' is not
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
        'dos': {'emin': -12.0, 'emax': 2.2, 'ne': 1000, 'do_pdos': True},
    },
    # ------------------------------------------------------------------
    # pipelines: each one is a ladder d = 0 .. max_doublings
    # ------------------------------------------------------------------
    'pipelines': {
        # the dense pipeline on the primitive cell, d = 0 only: the reference
        # every sparse case is compared with
        'reference': {
            'enabled': True,
            'sparse': False,
            'band_curvature': True,
            'properties': WINDOW_PROPERTIES + ['effective_mass'],
            'property_args': {'transport': {'do_hall': True}},
            'max_doublings': 0,
            'slurm': {'ntasks': 4, 'cpus_per_task': 4, 'npool': 4, 'mem': '0', 'time': '12:00:00'},
        },
        'sparse': {
            'enabled': True,
            'sparse': True,
            # the cap is lifted, so the per-k dense LAPACK kernel runs until it
            # really runs out of memory or time: that is the limit to measure
            'sparse_config': {
                'bond_order': 36,
                'energy_window': {'emax': 2.2},
                'dense_n_max': 10**9,
            },
            'properties': WINDOW_PROPERTIES,
            'max_doublings': 9,
            'slurm': {'ntasks': 4, 'cpus_per_task': 4, 'npool': 4, 'mem': '0', 'time': '24:00:00'},
        },
        # the full-spectrum pass: every state at every k-point, for the band
        # curvature (effective masses, Hall coefficient).  It costs a full
        # (nawf, nawf) eigh and a few nawf^3 products per k-point; expect the
        # largest sizes to hit the walltime
        'sparse_curvature': {
            'enabled': True,
            'sparse': True,
            'band_curvature': True,
            'sparse_config': {
                'bond_order': 36,
                'energy_window': {'emax': 2.2},
                'dense_n_max': 10**9,
            },
            'properties': CURVATURE_PROPERTIES,
            'property_args': {'transport': {'do_hall': True}},
            'max_doublings': 9,
            'slurm': {'ntasks': 4, 'cpus_per_task': 4, 'npool': 4, 'mem': '0', 'time': '24:00:00'},
        },
        'sparse_interior': {
            'enabled': True,
            'sparse': True,
            'sparse_config': {
                'bond_order': 36,
                'interior_window': {'elo': -1.5, 'ehi': 2.0},
                'dense_n_max': 10**9,
            },
            # doping and dielectric_tensor need the states below the window
            # and are skipped (with a warning) by the engine
            'properties': WINDOW_PROPERTIES,
            # the DoS is clamped to the window by the engine; the transport and
            # conductivity scans have to sit inside [elo + 0.26, ehi - 0.26]
            'property_args': {
                'transport': {'emin': -1.2, 'emax': 1.7},
                'conductivity': {'emin': -1.2, 'emax': 1.7},
            },
            'max_doublings': 9,
            'slurm': {'ntasks': 4, 'cpus_per_task': 4, 'npool': 4, 'mem': '0', 'time': '24:00:00'},
        },
        # Si PAO hoppings decay slowly: on example01 one row of H(k) holds
        # ~8500 nonzeros at bond_order=36 (rcut 32 Bohr) but ~890 at
        # bond_order=8 (15 Bohr), so with 36 shells H(k) is not sparse until
        # nawf >~ 10^4.  This ladder measures the truly sparse regime; its
        # accuracy cost shows in the per-cell comparison with the reference.
        'sparse_interior_bo8': {
            'enabled': True,
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
            'max_doublings': 9,
            'slurm': {'ntasks': 4, 'cpus_per_task': 4, 'npool': 4, 'mem': '0', 'time': '24:00:00'},
        },
    },
    # ------------------------------------------------------------------
    # per-k solver sweep: LAPACK against ARPACK at fixed n over nev/n,
    # to calibrate dense_ratio, and the per-k scratch memory of each kernel
    # ------------------------------------------------------------------
    'solver_sweep': {
        'enabled': True,
        'doublings': [3, 5, 7, 8, 9],
        'nev_fractions': [0.01, 0.02, 0.05, 0.10, 0.125, 0.20, 0.30, 0.50],
        'nk_probe': 3,
        # stop raising nev for ARPACK once it is this many times slower than LAPACK
        'stop_ratio': 20.0,
        'bond_order': 36,
        'slurm': {'ntasks': 4, 'cpus_per_task': 4, 'npool': 4, 'mem': '0', 'time': '12:00:00'},
    },
    # ------------------------------------------------------------------
    # Slurm settings shared by every job
    # ------------------------------------------------------------------
    'slurm': {
        'account': None,
        'partition': None,
        'qos': None,
        'nodes': 1,
        'exclusive': False,  # set True only if your cluster allows whole-node jobs
        # extra '#SBATCH' lines
        'extra': [
            '--no-requeue',
            '--get-user-env',
        ],
        # shell lines run before Python
        'setup': [
            'ulimit -s unlimited',
            'MINICONDA_PATH=/home/jayn/miniconda3',
            'source $MINICONDA_PATH/etc/profile.d/conda.sh',
            'conda activate paoflow_new',
        ],
        'launcher': 'mpirun -n ${SLURM_NTASKS}',
        'python': 'python',
        'dry_run': False,
    },
}
