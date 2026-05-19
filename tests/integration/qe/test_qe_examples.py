from __future__ import annotations

from pathlib import Path

import pytest

from .compare import CompareFailure, compare_dat_dirs
from .jobs import JobSpec, discover_jobs
from .runner import run_example_in_sandbox

HERE = Path(__file__).resolve().parent
EXAMPLES_ROOT = HERE


def _discover_jobs() -> list[JobSpec]:
    return discover_jobs(EXAMPLES_ROOT)


@pytest.mark.integration
@pytest.mark.parametrize('job', _discover_jobs(), ids=lambda j: j.id)
def test_qe_example(job: JobSpec, tmp_path: Path, qe_assets_root, qe_assets_link_mode) -> None:
    sandbox_root = tmp_path / 'sandbox'
    sandbox_root.mkdir(parents=True, exist_ok=True)

    result = run_example_in_sandbox(
        job.example_root,
        sandbox_root,
        job_relpath=job.job_relpath,
        assets_root=qe_assets_root,
        assets_link_mode=qe_assets_link_mode,
    )
    plots_dir = result.workdir / '_compare_plots'

    try:
        compare_dat_dirs(result.outdir, result.refdir, tolerance=0.01, plot_dir=plots_dir)
    except CompareFailure as e:
        raise AssertionError(
            f'{result.job_id} failed.\nSandbox: {result.workdir}\nPlots: {plots_dir}\n{e}'
        ) from e
