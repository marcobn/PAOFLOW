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


def _infer_outputdir_raw(script_path: Path) -> str:
    try:
        text = script_path.read_text(encoding='utf-8')
    except Exception:
        return script_path.stem

    m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        return script_path.stem
    return m.group(1).strip()


def _resolve_outputdir(script_path: Path, workdir: Path) -> tuple[Path, Path]:
    """Return output directory (absolute) and its relative path under workdir."""

    raw = _infer_outputdir_raw(script_path)
    outp = Path(raw)
    if not outp.is_absolute():
        outp = (workdir / outp).resolve()

    try:
        rel = outp.relative_to(workdir)
    except ValueError:
        rel = Path(outp.name)
    return outp, rel


def run_example_in_sandbox(
    script_path: Path,
    sandbox_root: Path,
    *,
    assets_root: Optional[Path] = None,
) -> ExampleRunResult:
    """Run a TBmodel integration job in an isolated sandbox."""

    example_name = script_path.stem

    sandbox_example_root = sandbox_root / example_name
    if sandbox_example_root.exists():
        shutil.rmtree(sandbox_example_root)
    sandbox_example_root.mkdir(parents=True, exist_ok=True)

    sandbox_script = sandbox_example_root / script_path.name
    shutil.copy2(script_path, sandbox_script)

    _run_paoflow(sandbox_script, sandbox_example_root)

    outdir, rel_outdir = _resolve_outputdir(sandbox_script, sandbox_example_root)
    if assets_root is not None:
        refdir = (assets_root / example_name / rel_outdir).resolve()
    else:
        refdir = outdir

    return ExampleRunResult(
        example_name=example_name,
        job_id=example_name,
        workdir=sandbox_example_root,
        outdir=outdir,
        refdir=refdir,
    )
