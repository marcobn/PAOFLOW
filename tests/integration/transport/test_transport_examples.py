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
def test_transport_example(
    job: JobSpec,
    tmp_path: Path,
    transport_assets_root,
    transport_assets_link_mode,
) -> None:
    sandbox_root = tmp_path / 'sandbox'
    sandbox_root.mkdir(parents=True, exist_ok=True)

    repo_job_dir = job.job_dir
    repo_has_ref = (repo_job_dir / 'Reference').is_dir()
    repo_has_qe_save = any(repo_job_dir.glob('*.save'))

    if transport_assets_root is None and not repo_has_ref:
        pytest.skip(
            'Transport integration assets are not configured and Reference/ is missing. '
            'Set PAOFLOW_TRANSPORT_ASSET_ARCHIVE (or --transport-assets-archive) '
            'to a local tar.gz to enable these tests.'
        )

    if transport_assets_root is None and not repo_has_qe_save:
        pytest.skip(
            'Transport integration assets are not configured and no *.save directory exists. '
            'Set PAOFLOW_TRANSPORT_ASSET_ARCHIVE (or --transport-assets-archive) '
            'to a local tar.gz to enable these tests.'
        )

    if transport_assets_root is not None and not repo_has_ref and not repo_has_qe_save:
        asset_job_root = (transport_assets_root / job.example_root.name / job.job_relpath).resolve()
        if not asset_job_root.exists():
            pytest.skip(
                f'Assets are missing for job {job.id} in configured archive and no local '
                'Reference/*.save data exists yet.'
            )

    result = run_example_in_sandbox(
        job.example_root,
        sandbox_root,
        job_relpath=job.job_relpath,
        assets_root=transport_assets_root,
        assets_link_mode=transport_assets_link_mode,
    )

    try:
        compare_dat_dirs(result.outdir, result.refdir, tolerance=0.01)
    except CompareFailure as e:
        raise AssertionError(f'{result.job_id} failed.\nSandbox: {result.workdir}\n{e}') from e
