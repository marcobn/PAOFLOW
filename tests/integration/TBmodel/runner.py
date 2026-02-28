from __future__ import annotations

import os
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


def _run_paoflow(script_path: Path, workdir: Path) -> None:
    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')

    subprocess.run([sys.executable, str(script_path)], cwd=workdir, check=True, env=env)


def _infer_outputdir(script_path: Path, workdir: Path) -> Path:
    """Infer PAOFLOW output directory from a TBmodel script.

    Defaults to a folder with the script stem if not found.
    """

    try:
        text = script_path.read_text(encoding='utf-8')
    except Exception:
        return workdir / script_path.stem

    m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        return workdir / script_path.stem

    raw = m.group(1).strip()
    outp = Path(raw)
    if not outp.is_absolute():
        outp = workdir / outp
    return outp


def _overlay_assets(
    *,
    assets_root: Path,
    example_name: str,
    job_dir: Path,
    link_mode: str,
) -> None:
    assets_job_root = (assets_root / example_name).resolve()
    if not assets_job_root.exists():
        raise RuntimeError(f'Assets missing for job: {example_name}')

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

    ref_src = assets_job_root / 'Reference'
    if ref_src.is_dir():
        _link_or_copy(ref_src, job_dir / 'Reference')

    for savedir in sorted(assets_job_root.glob('*.save')):
        if savedir.is_dir():
            _link_or_copy(savedir, job_dir / savedir.name)


def run_example_in_sandbox(
    script_path: Path,
    sandbox_root: Path,
    *,
    assets_root: Optional[Path] = None,
    assets_link_mode: str = 'symlink',
) -> ExampleRunResult:
    """Run a TBmodel integration job in an isolated sandbox."""

    example_name = script_path.stem

    sandbox_example_root = sandbox_root / example_name
    if sandbox_example_root.exists():
        shutil.rmtree(sandbox_example_root)
    sandbox_example_root.mkdir(parents=True, exist_ok=True)

    sandbox_script = sandbox_example_root / script_path.name
    shutil.copy2(script_path, sandbox_script)

    if assets_root is not None:
        _overlay_assets(
            assets_root=assets_root,
            example_name=example_name,
            job_dir=sandbox_example_root,
            link_mode=assets_link_mode,
        )

    _run_paoflow(sandbox_script, sandbox_example_root)

    outdir = _infer_outputdir(sandbox_script, sandbox_example_root)
    refdir = sandbox_example_root / 'Reference'

    return ExampleRunResult(
        example_name=example_name,
        job_id=example_name,
        workdir=sandbox_example_root,
        outdir=outdir,
        refdir=refdir,
    )
