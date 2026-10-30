import os
import warnings

import pytest

import PAOFLOW.utils.parallel_resources as pr

_LAUNCHER_VARS = (
    pr._RANK_ENV_VARS
    + pr._LOCAL_SIZE_ENV_VARS
    + pr._WORLD_SIZE_ENV_VARS
    + ('SLURM_STEP_TASKS_PER_NODE', pr._LOKY_CPU_ENV_VAR)
)


@pytest.fixture
def node(monkeypatch):
    """Clean launcher environment on a fake 64-CPU node without binding."""
    for name in _LAUNCHER_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(os, 'cpu_count', lambda: 64)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(64)), raising=False)
    monkeypatch.setattr(pr, '_clip_warning_issued', False)
    return monkeypatch


@pytest.mark.unit
def test_serial_run_uses_all_cores_and_leaves_env_untouched(node):
    assert pr.mpi_launch_sizes() == (1, 1)
    assert pr.core_budget() == 64
    pr.configure_joblib_for_mpi()
    assert pr._LOKY_CPU_ENV_VAR not in os.environ


@pytest.mark.unit
@pytest.mark.parametrize(
    'env, expected_sizes, expected_budget',
    [
        ({'OMPI_COMM_WORLD_SIZE': '8', 'OMPI_COMM_WORLD_LOCAL_SIZE': '4'}, (8, 4), 16),
        ({'PMI_SIZE': '4', 'MPI_LOCALNRANKS': '4'}, (4, 4), 16),
        ({'PMI_SIZE': '3'}, (3, 3), 21),
        ({'SLURM_STEP_NUM_TASKS': '11', 'SLURM_STEP_TASKS_PER_NODE': '4(x2),3'}, (11, 4), 16),
        ({'SLURM_STEP_NUM_TASKS': '1', 'SLURM_STEP_TASKS_PER_NODE': '1'}, (1, 1), 64),
    ],
)
def test_launcher_detection_and_budget(node, env, expected_sizes, expected_budget):
    for name, value in env.items():
        node.setenv(name, value)
    assert pr.mpi_launch_sizes() == expected_sizes
    assert pr.core_budget() == expected_budget


@pytest.mark.unit
def test_bound_rank_budget_follows_affinity(node):
    node.setenv('OMPI_COMM_WORLD_SIZE', '4')
    node.setenv('OMPI_COMM_WORLD_LOCAL_SIZE', '4')
    node.setattr(os, 'sched_getaffinity', lambda pid: {3})
    assert pr.core_budget() == 1


@pytest.mark.unit
def test_configure_exports_share_but_respects_user_value(node):
    node.setenv('OMPI_COMM_WORLD_SIZE', '4')
    node.setenv('OMPI_COMM_WORLD_LOCAL_SIZE', '4')
    pr.configure_joblib_for_mpi()
    assert os.environ[pr._LOKY_CPU_ENV_VAR] == '16'

    node.setenv(pr._LOKY_CPU_ENV_VAR, '32')
    pr.configure_joblib_for_mpi()
    assert os.environ[pr._LOKY_CPU_ENV_VAR] == '32'
    assert pr.core_budget() == 32


@pytest.mark.unit
def test_resolve_n_jobs_follows_joblib_semantics_within_budget(node):
    node.setenv('OMPI_COMM_WORLD_SIZE', '4')
    node.setenv('OMPI_COMM_WORLD_LOCAL_SIZE', '4')
    assert pr.resolve_n_jobs(None) == 1
    assert pr.resolve_n_jobs(1) == 1
    assert pr.resolve_n_jobs(-1) == 16
    assert pr.resolve_n_jobs(-2) == 15
    assert pr.resolve_n_jobs(-100) == 1
    assert pr.resolve_n_jobs(-1, n_tasks=5) == 5
    assert pr.resolve_n_jobs(-1, n_tasks=0) == 1
    assert pr.resolve_n_jobs(-1, threads_per_job=5) == 3
    with pytest.raises(ValueError):
        pr.resolve_n_jobs(0)


@pytest.mark.unit
def test_resolve_n_jobs_clips_explicit_request_and_warns_once(node):
    node.setenv('OMPI_COMM_WORLD_SIZE', '4')
    node.setenv('OMPI_COMM_WORLD_LOCAL_SIZE', '4')
    with pytest.warns(RuntimeWarning, match='using 16'):
        assert pr.resolve_n_jobs(64) == 16
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert pr.resolve_n_jobs(64) == 16


@pytest.mark.unit
def test_resolve_n_jobs_warns_only_on_rank_zero(node):
    node.setenv('OMPI_COMM_WORLD_SIZE', '4')
    node.setenv('OMPI_COMM_WORLD_LOCAL_SIZE', '4')
    node.setenv('OMPI_COMM_WORLD_RANK', '2')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert pr.resolve_n_jobs(64) == 16
