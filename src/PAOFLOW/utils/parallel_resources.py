"""Core budgeting for joblib/loky parallelism inside MPI launches.

PAOFLOW is normally run as ``mpirun -np N python main.py``, so every MPI rank
executes the same script.  A few modules (``models/sk_fitting.py``,
``models/band_unfold.py``, ``spectrum/sparse_bands.py``, ``pyskeaf/runner.py``)
additionally spawn joblib/loky worker processes.  Without coordination each
rank would size its pool for the whole node, oversubscribing the cores
``N``-fold.

This module assigns every rank a *core budget* — its share of the node — and
enforces it at two levels:

1. :func:`configure_joblib_for_mpi` (called when :mod:`PAOFLOW` is imported)
   exports ``LOKY_MAX_CPU_COUNT``.  loky's ``cpu_count()`` honours it, so
   ``n_jobs=-1`` resolves to the budget for *any* joblib call, including user
   scripts, and loky caps BLAS/OpenMP threads per worker at
   ``budget // n_jobs``.
2. :func:`resolve_n_jobs` clips explicit worker counts requested at PAOFLOW
   call sites, which joblib itself never caps.

Launch detection only reads launcher environment variables and never performs
MPI collectives, so it is cheap, does not import :mod:`mpi4py`, and is safe to
call from a single rank.
"""

from __future__ import annotations

import os
import re
import warnings

_RANK_ENV_VARS = (
    'OMPI_COMM_WORLD_RANK',
    'PMI_RANK',
    'PMIX_RANK',
    'SLURM_PROCID',
    'MV2_COMM_WORLD_RANK',
    'I_MPI_RANK',
)

# Number of ranks sharing this node, most specific launcher first.
_LOCAL_SIZE_ENV_VARS = (
    'OMPI_COMM_WORLD_LOCAL_SIZE',
    'PMIX_LOCAL_SIZE',
    'MPI_LOCALNRANKS',
    'MV2_COMM_WORLD_LOCAL_SIZE',
    'PALS_LOCAL_SIZE',
)

# Total number of ranks in the launch.
_WORLD_SIZE_ENV_VARS = (
    'OMPI_COMM_WORLD_SIZE',
    'PMI_SIZE',
    'MV2_COMM_WORLD_SIZE',
    'SLURM_STEP_NUM_TASKS',
)

_LOKY_CPU_ENV_VAR = 'LOKY_MAX_CPU_COUNT'

_SLURM_TASKS_ITEM_RE = re.compile(r'^(\d+)(?:\(x(\d+)\))?$')

_clip_warning_issued = False


def mpi_rank() -> int | None:
    """Return this process' MPI rank from common launcher env vars.

    Returns
    -------
    int or None
        The rank reported by the launcher, or ``None`` outside an MPI launch.
    """
    for name in _RANK_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _positive_env_int(name: str) -> int | None:
    """Return environment variable ``name`` as a positive int, else ``None``."""
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        return None
    return value if value > 0 else None


def _slurm_max_tasks_per_node() -> int | None:
    """Largest per-node task count of the current Slurm step.

    Returns
    -------
    int or None
        Maximum entry of ``SLURM_STEP_TASKS_PER_NODE`` (format such as
        ``'4(x2),3'``), or ``None`` when unset or unparsable.

    Notes
    -----
    The maximum is used rather than the entry for this node because the step
    node index is not exposed reliably; on heterogeneous steps this errs
    towards a smaller budget.
    """
    spec = os.environ.get('SLURM_STEP_TASKS_PER_NODE')
    if not spec:
        return None
    tasks_per_node = []
    for item in spec.split(','):
        match = _SLURM_TASKS_ITEM_RE.match(item.strip())
        if match is None:
            return None
        tasks_per_node.append(int(match.group(1)))
    return max(tasks_per_node) if tasks_per_node else None


def mpi_launch_sizes() -> tuple[int, int]:
    """Detect the MPI world size and the number of ranks on this node.

    Returns
    -------
    tuple of int
        ``(world_size, local_size)``.  Both are ``1`` outside an MPI launch.
        When the launcher reports a world size but no per-node size, the
        world size is used for both, which is conservative on multi-node runs.
    """
    world_size = next((size for size in map(_positive_env_int, _WORLD_SIZE_ENV_VARS) if size), None)
    local_size = next((size for size in map(_positive_env_int, _LOCAL_SIZE_ENV_VARS) if size), None)
    if local_size is None and os.environ.get('SLURM_STEP_NUM_TASKS'):
        local_size = _slurm_max_tasks_per_node()

    if world_size is None and local_size is None:
        return 1, 1
    if local_size is None:
        local_size = world_size
    if world_size is None:
        world_size = local_size
    return world_size, min(local_size, world_size)


def _available_cpus() -> int:
    """Number of CPUs this process may run on (affinity mask when available)."""
    if hasattr(os, 'sched_getaffinity'):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _mpi_core_share() -> int | None:
    """This rank's share of the node, or ``None`` outside a multi-rank launch.

    Returns
    -------
    int or None
        ``max(1, min(available_cpus, node_cpus // ranks_on_node))``.

    Notes
    -----
    The affinity mask captures launcher binding: a rank bound to one core
    gets a budget of 1, whereas ``--bind-to none`` or ``--map-by ...:pe=k``
    leaves room for hybrid MPI + joblib parallelism.
    """
    world_size, local_size = mpi_launch_sizes()
    if world_size <= 1:
        return None
    node_cpus = max(1, os.cpu_count() or 1)
    return max(1, min(_available_cpus(), node_cpus // local_size))


def core_budget() -> int:
    """Number of cores this process may use for joblib workers and threads.

    Returns
    -------
    int
        ``LOKY_MAX_CPU_COUNT`` if set (bounded by the available CPUs),
        otherwise this rank's share of the node under MPI, otherwise all
        available CPUs.
    """
    available = _available_cpus()
    user_limit = _positive_env_int(_LOKY_CPU_ENV_VAR)
    if user_limit is not None:
        return max(1, min(available, user_limit))
    share = _mpi_core_share()
    return share if share is not None else available


def configure_joblib_for_mpi() -> None:
    """Export ``LOKY_MAX_CPU_COUNT`` for multi-rank MPI launches.

    Returns
    -------
    None
        Sets ``os.environ['LOKY_MAX_CPU_COUNT']`` to this rank's core share
        when more than one MPI rank is detected and the variable is not
        already defined.  Serial and single-rank runs are left untouched.
    """
    share = _mpi_core_share()
    if share is not None:
        os.environ.setdefault(_LOKY_CPU_ENV_VAR, str(share))


def resolve_n_jobs(
    requested: int | None,
    n_tasks: int | None = None,
    threads_per_job: int = 1,
) -> int:
    """Convert a joblib ``n_jobs`` request into a worker count within budget.

    Parameters
    ----------
    requested : int or None
        joblib-style request: positive for an explicit count, ``-1`` for all
        cores, ``-k`` for all but ``k - 1`` cores, ``None`` for serial.
    n_tasks : int or None, optional
        Number of independent tasks; the result never exceeds it.
    threads_per_job : int, optional
        Threads each worker starts itself (for example a
        ``ThreadPoolExecutor``), so that ``workers * threads_per_job`` stays
        within :func:`core_budget`.

    Returns
    -------
    int
        Worker count ``>= 1``.  Emits a single ``RuntimeWarning`` (primary
        rank only) the first time an explicit request is reduced.

    Raises
    ------
    ValueError
        If ``requested == 0``, which has no meaning in joblib.
    """
    global _clip_warning_issued

    if requested is None:
        return 1
    if requested == 0:
        raise ValueError('n_jobs == 0 has no meaning; use 1 for serial or -1 for all cores')

    budget = core_budget()
    capacity = max(1, budget // max(1, threads_per_job))

    if requested < 0:
        n_workers = max(capacity + 1 + requested, 1)
    else:
        n_workers = requested
        if requested > capacity:
            n_workers = capacity
            if not _clip_warning_issued and mpi_rank() in (None, 0):
                _clip_warning_issued = True
                world_size, _ = mpi_launch_sizes()
                context = f' per MPI rank ({world_size} ranks)' if world_size > 1 else ''
                warnings.warn(
                    f'Requested {requested} parallel workers but only {budget} cores are '
                    f'available{context}'
                    + (f' at {threads_per_job} threads per worker' if threads_per_job > 1 else '')
                    + f'; using {n_workers}. Set {_LOKY_CPU_ENV_VAR} or change the MPI '
                    f'rank count/binding to change this budget.',
                    RuntimeWarning,
                    stacklevel=2,
                )

    if n_tasks is not None:
        n_workers = min(n_workers, max(1, n_tasks))
    return n_workers
