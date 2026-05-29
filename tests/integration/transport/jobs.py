from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobSpec:
    example_root: Path
    job_relpath: Path

    @property
    def job_dir(self) -> Path:
        return self.example_root / self.job_relpath

    @property
    def id(self) -> str:
        rel = self.job_relpath.as_posix()
        if rel in ('.', ''):
            return self.example_root.name
        return f'{self.example_root.name}/{rel}'


def discover_jobs(examples_root: Path) -> list[JobSpec]:
    """Discover runnable PAOFLOW transport jobs under tests/integration/transport.

    A job is any directory (at any depth) under an example folder that contains
    at least one transport entrypoint: ``main.py``, ``main_conductor.py`` or
    ``main_current.py``.
    """

    jobs: list[JobSpec] = []
    entrypoints = ('main.py', 'main_conductor.py', 'main_current.py')

    for example_root in sorted([p for p in examples_root.glob('example*') if p.is_dir()]):
        if any((example_root / name).is_file() for name in entrypoints):
            jobs.append(JobSpec(example_root=example_root, job_relpath=Path('.')))

        for path in sorted(example_root.rglob('*')):
            if not path.is_file() or path.name not in entrypoints:
                continue

            job_dir = path.parent
            if job_dir == example_root:
                continue

            jobs.append(
                JobSpec(
                    example_root=example_root,
                    job_relpath=job_dir.relative_to(example_root),
                )
            )

    uniq: dict[str, JobSpec] = {}
    for job in jobs:
        uniq[job.id] = job
    return [uniq[k] for k in sorted(uniq.keys())]
