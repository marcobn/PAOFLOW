from __future__ import annotations

import sys
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ExampleRunResult:
    example_name: str
    workdir: Path
    outdir: Path
    refdir: Path


def build_commands(example_dir: Path) -> List[List[str]]:
    """Reproduce your current command discovery logic, but without chdir()."""
    commands: List[List[str]] = []

    mc = example_dir / 'main_conductor.py'
    if mc.exists():
        if (example_dir / 'conductor.yaml').exists():
            commands.append([sys.executable, str(mc)])
        else:
            for y in sorted(example_dir.glob('conductor*.yaml')):
                commands.append([sys.executable, str(mc), str(y.name)])

    mcu = example_dir / 'main_current.py'
    if mcu.exists():
        if (example_dir / 'current.yaml').exists():
            commands.append([sys.executable, str(mcu)])
        else:
            for y in sorted(example_dir.glob('current*.yaml')):
                commands.append([sys.executable, str(mcu), str(y.name)])

    return commands


def run_example_in_sandbox(example_dir: Path, sandbox_root: Path) -> ExampleRunResult:
    """
    Copy the example into a sandbox directory, run there, and return paths for output+reference.
    This prevents overwriting committed outputs or developer local outputs.
    """
    example_name = example_dir.name
    sandbox_dir = sandbox_root / example_name

    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)

    # Copy everything needed to run (includes qe inputs etc). Exclude Reference in the sandbox if you want,
    # but keeping it is fine; we’ll point refdir at the *original* reference.
    shutil.copytree(example_dir, sandbox_dir)

    commands = build_commands(sandbox_dir)
    if not commands:
        raise RuntimeError(f'No runnable scripts found in {example_dir}')

    # Run commands in the sandbox
    for cmd in commands:
        subprocess.run(cmd, cwd=sandbox_dir, check=True)

    outdir = sandbox_dir / 'output' / 'paoflow'
    refdir = example_dir / 'Reference'  # reference remains the committed gold data
    return ExampleRunResult(
        example_name=example_name, workdir=sandbox_dir, outdir=outdir, refdir=refdir
    )
