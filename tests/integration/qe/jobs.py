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
    """Discover runnable PAOFLOW jobs under tests/integration/qe.

    A job is any directory (at any depth) under an example folder that contains
    a `main.py`. This matches the behavior of the bash-based `job.sh` runner and
    allows nested examples (e.g. qe-soc/ad_hoc_soc) to be tested.
    """

    jobs: list[JobSpec] = []
    for example_root in sorted([p for p in examples_root.glob('example*') if p.is_dir()]):
        # Root job.
        if (example_root / 'main.py').is_file():
            jobs.append(JobSpec(example_root=example_root, job_relpath=Path('.')))

        # Nested jobs.
        for main_py in sorted(example_root.rglob('main.py')):
            job_dir = main_py.parent
            if job_dir == example_root:
                continue
            jobs.append(
                JobSpec(
                    example_root=example_root,
                    job_relpath=job_dir.relative_to(example_root),
                )
            )

    # Stable ordering, de-dupe just in case.
    uniq: dict[str, JobSpec] = {}
    for j in jobs:
        uniq[j.id] = j
    return [uniq[k] for k in sorted(uniq.keys())]
