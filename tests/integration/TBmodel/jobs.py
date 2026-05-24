from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobSpec:
    script_path: Path

    @property
    def example_root(self) -> Path:
        return self.script_path.parent

    @property
    def example_name(self) -> str:
        return self.script_path.stem

    @property
    def job_dir(self) -> Path:
        return self.example_root / self.example_name

    @property
    def id(self) -> str:
        return self.example_name


def discover_jobs(examples_root: Path) -> list[JobSpec]:
    """Discover runnable TBmodel jobs under tests/integration/TBmodel.

    A job is any top-level python script that is not part of the test harness.
    """

    excluded = {
        '__init__.py',
        'assets.py',
        'build_assets.py',
        'compare.py',
        'conftest.py',
        'jobs.py',
        'runner.py',
    }

    jobs: list[JobSpec] = []
    for script in sorted(examples_root.glob('*.py')):
        if script.name in excluded:
            continue
        if script.name.startswith('test_') or script.name.startswith('_'):
            continue
        jobs.append(JobSpec(script_path=script))

    uniq: dict[str, JobSpec] = {}
    for job in jobs:
        uniq[job.id] = job
    return [uniq[k] for k in sorted(uniq.keys())]
