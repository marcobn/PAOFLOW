"""Shared settings of the sparse-scaling benchmark, and the system loader.

The benchmark grows a crystal by ``doubling_Hamiltonian`` and runs its
properties through each sparse pipeline, size by size.  Each system has a
folder ``systems/<name>/``: ``system.py`` defines the crystal, its properties
and its pipelines (:data:`SYSTEM`), the notebook ``observables.ipynb`` plots
its curves, and ``runs/`` receives its cases.  This file holds what the
systems share: which systems :mod:`submit` submits, the Slurm settings and
the default resources of a job.  Nothing is passed on the command line.

Sizes
-----
Doubling ``d`` times multiplies ``nawf`` by ``2**d``.  The doublings are
spread over the three axes in turn, (1,0,0), (1,1,0), (1,1,1), (2,1,1), ...,
so consecutive sizes differ by a factor of two.

k-mesh
------
``base_nfft`` is the property mesh of the undoubled cell.  Each doubling
along an axis halves the mesh along it (floored at ``min_nfft``), so the
physical k-point density, and therefore the per-cell result, stays the same
at every size.

Reference
---------
The ``reference`` pipeline of a system is the dense PAOFLOW pipeline on the
primitive cell: no ``doubling_Hamiltonian`` call, the full ``base_nfft``
mesh, and every property of every sparse pipeline.  Each sparse case is
compared with it per primitive cell.

Properties
----------
``SYSTEM['properties']`` holds each property's arguments, in the order they
run.  A key ``'<method>:<tag>'`` calls ``<method>``, so one method can run
with several argument sets.  ``dos`` comes last because ``doping`` also
writes ``dosdk_0.dat`` (over its own energy range).  A pipeline lists the
properties it runs, and may override their arguments in ``property_args``.
``band_curvature`` makes the pass solve the full spectrum at every k-point
(needed by ``effective_mass`` and the Hall coefficient of ``transport``);
the properties that sum over all states (the Hall conductivities,
``linear_response``) need the full spectrum as well, so they share that
pipeline.
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SYSTEMS_DIR = os.path.join(HERE, 'systems')

CONFIG = {
    # the systems submit.py submits, each into systems/<name>/runs.  Si already
    # ran (2026-10): add it back to rerun it, which overwrites its runs
    'systems': ['pt', 'fe', 'te'],
    # random on-site energies (meV, Gaussian sigma) added to the sparse bond
    # list after doubling.  They break the exact folding degeneracies, the
    # worst case of ARPACK, at the price of the exact per-cell reference.
    # Sparse pipelines only; 0 disables.
    'disorder_meV': 0.0,
    'disorder_seed': 1234,
    # ------------------------------------------------------------------
    # the resources of every job, unless a pipeline's 'slurm' overrides them
    # ------------------------------------------------------------------
    'resources': {
        'ntasks': 4,
        'cpus_per_task': 3,
        'npool': 4,
        # MB; a request below the use (Si's sparse_curvature asked for 2500 and
        # used 35 GB) still runs where Slurm does not enforce it, and a larger
        # one waits in the queue
        'mem': '2500',
        'time': '24:00:00',
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


def system_dir(name):
    return os.path.join(SYSTEMS_DIR, name)


def output_root(name):
    """Where the cases of system ``name`` go: ``systems/<name>/runs``."""
    return os.path.join(system_dir(name), 'runs')


def load_system(name):
    """The :data:`SYSTEM` dict of ``systems/<name>/system.py``."""
    path = os.path.join(system_dir(name), 'system.py')
    spec = importlib.util.spec_from_file_location('benchmark_system_%s' % name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEM


def all_systems():
    """Every system that has a folder, in alphabetical order."""
    return sorted(
        name
        for name in os.listdir(SYSTEMS_DIR)
        if os.path.isfile(os.path.join(SYSTEMS_DIR, name, 'system.py'))
    )
