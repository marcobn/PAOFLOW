"""Settings of the sparse-scaling benchmark: the single file a user edits.

The benchmark grows bulk Si by ``doubling_Hamiltonian`` and runs ``dos`` and
Boltzmann ``transport`` through each pipeline until that pipeline fails.
Nothing is passed on the command line; :mod:`submit` reads ``CONFIG`` from
here, expands it into cases with :func:`ladder.build_cases` and writes them
to ``<output_root>/cases.json``, which every Slurm array task reads back.

Sizes
-----
Doubling ``d`` times multiplies ``nawf`` by ``2**d``.  The doublings are
spread over the three axes in turn, (1,0,0), (1,1,0), (1,1,1), (2,1,1), ...,
so consecutive sizes differ by a factor of two.  The example01 base cell has
``nawf = 18``, so ``d = 8`` is ``nawf = 4608`` and ``d = 11`` is 36864.

k-mesh
------
``base_nfft`` is the property mesh of the undoubled cell.  Each doubling
along an axis halves the mesh along it (floored at ``min_nfft``), so the
physical k-point density, and therefore the per-cell result, stays the same
at every size.  The dense pipeline can only interpolate *up* from the
H(R) grid of the QE run (12^3 here), so its mesh is clamped to that grid
once the halving would go below it; the record flags those cases
(``mesh.clamped``) and the analysis compares them per k-point.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

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
    # properties (energies relative to E_F, eV); a pipeline may override them
    # ------------------------------------------------------------------
    'dos': {'emin': -12.0, 'emax': 2.2, 'ne': 1000, 'do_pdos': False},
    'transport': {'emin': -2.0, 'emax': 2.0, 'ne': 500, 'tmin': 300.0, 'tmax': 300.0, 'nt': 1},
    # ------------------------------------------------------------------
    # pipelines: each one is a ladder d = 0 .. max_doublings
    # ------------------------------------------------------------------
    'pipelines': {
        'dense': {
            'enabled': True,
            'sparse': False,
            'max_doublings': 7,
            'slurm': {'ntasks': 8, 'cpus_per_task': 4, 'npool': 8, 'mem': '0', 'time': '12:00:00'},
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
            'max_doublings': 10,
            'slurm': {'ntasks': 4, 'cpus_per_task': 8, 'npool': 4, 'mem': '0', 'time': '24:00:00'},
        },
        'sparse_interior': {
            'enabled': True,
            'sparse': True,
            'sparse_config': {
                'bond_order': 36,
                'interior_window': {'elo': -1.5, 'ehi': 2.0},
                'dense_n_max': 10**9,
            },
            # the DoS is clamped to the window by the engine; the transport scan
            # has to sit inside [elo + 0.26, ehi - 0.26]
            'transport': {'emin': -1.2, 'emax': 1.7},
            'max_doublings': 11,
            'slurm': {'ntasks': 4, 'cpus_per_task': 8, 'npool': 4, 'mem': '0', 'time': '24:00:00'},
        },
        # Si PAO hoppings decay slowly: on example01 one row of H(k) holds
        # ~8500 nonzeros at bond_order=36 (rcut 32 Bohr) but ~890 at
        # bond_order=8 (15 Bohr), so with 36 shells H(k) is not sparse until
        # nawf >~ 10^4.  This ladder measures the truly sparse regime; its
        # accuracy cost shows in the per-cell comparison against dense d=0.
        'sparse_interior_bo8': {
            'enabled': True,
            'sparse': True,
            'sparse_config': {
                'bond_order': 8,
                'interior_window': {'elo': -1.5, 'ehi': 2.0},
                'dense_n_max': 10**9,
            },
            'transport': {'emin': -1.2, 'emax': 1.7},
            'max_doublings': 12,
            'slurm': {'ntasks': 4, 'cpus_per_task': 8, 'npool': 4, 'mem': '0', 'time': '24:00:00'},
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
        'slurm': {'ntasks': 1, 'cpus_per_task': 32, 'npool': 1, 'mem': '0', 'time': '12:00:00'},
    },
    # ------------------------------------------------------------------
    # Slurm settings shared by every job
    # ------------------------------------------------------------------
    'slurm': {
        'account': None,
        'partition': None,
        'qos': None,
        'nodes': 1,
        'exclusive': True,
        # extra '#SBATCH' lines, e.g. ['--constraint=bigmem']
        'extra': [],
        # shell lines run before Python, e.g. module loads and the venv
        'setup': [
            # 'module load python/3.14 openmpi',
            # 'source ~/venvs/paoflow/bin/activate',
        ],
        'launcher': 'srun',
        'python': 'python',
        # write the job scripts but do not call sbatch
        'dry_run': False,
    },
}
