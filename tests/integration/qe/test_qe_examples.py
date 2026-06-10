from __future__ import annotations

from pathlib import Path

import pytest

from .compare import CompareFailure, compare_dat_dirs
from .jobs import JobSpec, discover_jobs
from .runner import _overlay_internal_basis_assets, run_example_in_sandbox

HERE = Path(__file__).resolve().parent
EXAMPLES_ROOT = HERE


def _discover_jobs() -> list[JobSpec]:
    return discover_jobs(EXAMPLES_ROOT)


@pytest.mark.integration
@pytest.mark.parametrize('job', _discover_jobs(), ids=lambda j: j.id)
def test_qe_example(
    job: JobSpec,
    tmp_path: Path,
    qe_test_assets_root,
    qe_test_assets_link_mode,
) -> None:
    sandbox_root = tmp_path / 'sandbox'
    sandbox_root.mkdir(parents=True, exist_ok=True)

    result = run_example_in_sandbox(
        job.example_root,
        sandbox_root,
        job_relpath=job.job_relpath,
        qe_test_assets_root=qe_test_assets_root,
        assets_link_mode=qe_test_assets_link_mode,
    )
    plots_dir = result.workdir / '_compare_plots'

    try:
        compare_dat_dirs(result.outdir, result.refdir, tolerance=0.01, plot_dir=plots_dir)
    except CompareFailure as e:
        raise AssertionError(
            f'{result.job_id} failed.\nSandbox: {result.workdir}\nPlots: {plots_dir}\n{e}'
        ) from e


def test_internal_basis_assets_overlay(tmp_path: Path) -> None:
    sandbox_root = tmp_path / 'sandbox'
    job_dir = sandbox_root / 'tests' / 'integration' / 'qe' / 'example15'
    job_dir.mkdir(parents=True)

    qe_test_assets_root = tmp_path / 'qe_test_assets'
    basis_src = qe_test_assets_root / 'BASIS' / 'Si'
    basis_src.mkdir(parents=True)
    (basis_src / 'basis.py').write_text('basis_data = {}\n', encoding='utf-8')

    _overlay_internal_basis_assets(
        qe_test_assets_root=qe_test_assets_root,
        sandbox_root=sandbox_root,
        link_mode='symlink',
    )

    basis_dst = (job_dir / '../../../../BASIS').resolve()
    assert basis_dst.is_dir()
    assert (basis_dst / 'Si' / 'basis.py').is_file()
