from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ExampleRunResult:
    example_name: str
    job_id: str
    workdir: Path
    outdir: Path
    refdir: Path


def _run_paoflow(example_dir: Path) -> None:
    main_py = example_dir / 'main.py'
    if not main_py.exists():
        raise RuntimeError(f'No main.py found in {example_dir}')

    subprocess.run([sys.executable, str(main_py)], cwd=example_dir, check=True)


def _infer_outputdir(job_dir: Path) -> Path:
    """Infer PAOFLOW output directory from a job's main.py.

    Defaults to "output" if not found.
    """

    main_py = job_dir / 'main.py'
    try:
        text = main_py.read_text(encoding='utf-8')
    except Exception:
        return job_dir / 'output'

    m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        return job_dir / 'output'

    raw = m.group(1).strip()
    outp = Path(raw)
    if not outp.is_absolute():
        outp = job_dir / outp
    return outp


def _overlay_assets(
    *,
    assets_root: Path,
    example_name: str,
    job_relpath: Path,
    job_dir: Path,
    link_mode: str,
) -> None:
    assets_job_root = (assets_root / example_name / job_relpath).resolve()
    if not assets_job_root.exists():
        raise RuntimeError(f'Assets missing for job: {example_name}/{job_relpath}')

    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink(missing_ok=True)

        if link_mode == 'symlink':
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src, target_is_directory=True)
            return

        shutil.copytree(src, dst)

    # Reference
    ref_src = assets_job_root / 'Reference'
    if ref_src.is_dir():
        _link_or_copy(ref_src, job_dir / 'Reference')

    for savedir in sorted(assets_job_root.glob('*.save')):
        if savedir.is_dir():
            _link_or_copy(savedir, job_dir / savedir.name)


def run_example_in_sandbox(
    example_dir: Path,
    sandbox_root: Path,
    *,
    job_relpath: Path | None = None,
    assets_root: Optional[Path] = None,
    assets_link_mode: str = 'symlink',
) -> ExampleRunResult:
    """Run a QE integration job in an isolated sandbox.

    - Copies the full example directory into the sandbox.
    - Overlays `Reference/` and `*.save/` from assets into the job directory (if provided).
    """

    example_name = example_dir.name
    job_relpath = job_relpath or Path('.')

    # Keep sandbox directory stable per example, since tmp_path is unique per test.
    sandbox_example_root = sandbox_root / example_name

    if sandbox_example_root.exists():
        shutil.rmtree(sandbox_example_root)

    shutil.copytree(example_dir, sandbox_example_root)
    sandbox_job_dir = sandbox_example_root / job_relpath
    if not sandbox_job_dir.exists():
        raise RuntimeError(f'Sandbox job directory missing: {sandbox_job_dir}')

    if assets_root is not None:
        _overlay_assets(
            assets_root=assets_root,
            example_name=example_name,
            job_relpath=job_relpath,
            job_dir=sandbox_job_dir,
            link_mode=assets_link_mode,
        )

    _run_paoflow(sandbox_job_dir)

    outdir = _infer_outputdir(sandbox_job_dir)
    refdir = sandbox_job_dir / 'Reference'

    job_id = (
        example_name
        if str(job_relpath) in ('.', '')
        else f'{example_name}/{job_relpath.as_posix()}'
    )
    return ExampleRunResult(
        example_name=example_name,
        job_id=job_id,
        workdir=sandbox_job_dir,
        outdir=outdir,
        refdir=refdir,
    )
