from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ExampleRunResult:
    example_name: str
    job_id: str
    workdir: Path
    outdir: Path
    refdir: Path


def build_commands(job_dir: Path) -> List[List[str]]:
    """Discover runnable commands in a transport job directory."""

    commands: List[List[str]] = []

    main_py = job_dir / 'main.py'
    if main_py.exists():
        commands.append([sys.executable, str(main_py)])

    mc = job_dir / 'main_conductor.py'
    if mc.exists():
        if (job_dir / 'conductor.yaml').exists():
            commands.append([sys.executable, str(mc)])
        else:
            for y in sorted(job_dir.glob('conductor*.yaml')):
                commands.append([sys.executable, str(mc), str(y.name)])

    mcu = job_dir / 'main_current.py'
    if mcu.exists():
        if (job_dir / 'current.yaml').exists():
            commands.append([sys.executable, str(mcu)])
        else:
            for y in sorted(job_dir.glob('current*.yaml')):
                commands.append([sys.executable, str(mcu), str(y.name)])

    return commands


def _infer_outputdir(job_dir: Path) -> Path:
    """Infer PAOFLOW output directory from main scripts.

    Defaults to ``output/paoflow`` if not found.
    """

    for main_name in ('main.py', 'main_conductor.py', 'main_current.py'):
        main_py = job_dir / main_name
        if not main_py.exists():
            continue

        try:
            text = main_py.read_text(encoding='utf-8')
        except Exception:
            continue

        m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
        if not m:
            continue

        raw = m.group(1).strip()
        outp = Path(raw)
        if not outp.is_absolute():
            outp = job_dir / outp
        return outp

    return job_dir / 'output' / 'paoflow'


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

    ref_src = assets_job_root / 'Reference'
    if ref_src.is_dir():
        _link_or_copy(ref_src, job_dir / 'Reference')

    output_qe_src = assets_job_root / 'output' / 'qe'
    if output_qe_src.is_dir():
        dst_output = job_dir / 'output'
        dst_output.mkdir(parents=True, exist_ok=True)
        _link_or_copy(output_qe_src, dst_output / 'qe')


def run_example_in_sandbox(
    example_dir: Path,
    sandbox_root: Path,
    *,
    job_relpath: Path | None = None,
    assets_root: Optional[Path] = None,
    assets_link_mode: str = 'symlink',
) -> ExampleRunResult:
    """Run a transport integration job in an isolated sandbox."""

    example_name = example_dir.name
    job_relpath = job_relpath or Path('.')

    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')

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

    commands = build_commands(sandbox_job_dir)
    if not commands:
        raise RuntimeError(f'No runnable scripts found in {sandbox_job_dir}')

    for cmd in commands:
        subprocess.run(cmd, cwd=sandbox_job_dir, check=True, env=env)

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
