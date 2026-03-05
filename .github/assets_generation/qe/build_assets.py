from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tests.integration.qe.jobs import discover_jobs


def _add_dir(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    # Add directory recursively, preserving permissions.
    tf.add(str(path), arcname=arcname, recursive=True)


def build_assets(*, qe_root: Path, out_tar_gz: Path) -> None:
    jobs = discover_jobs(qe_root)
    if not jobs:
        raise SystemExit(f'No jobs found under {qe_root}')

    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, 'w:gz') as tf:
        for job in jobs:
            job_dir = job.job_dir

            # Reference
            refdir = job_dir / 'Reference'
            if refdir.is_dir():
                arc = (Path(job.example_root.name) / job.job_relpath / 'Reference').as_posix()
                _add_dir(tf, refdir, arc)

            # *.save dirs
            for savedir in sorted(job_dir.glob('*.save')):
                if not savedir.is_dir():
                    continue
                arc = (Path(job.example_root.name) / job.job_relpath / savedir.name).as_posix()
                _add_dir(tf, savedir, arc)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            'Build a local qe_test_assets tar.gz from existing *.save/ and Reference/ folders. '
            'Run .github/assets_generation/qe/job.sh first to generate Reference outputs.'
        )
    )
    p.add_argument(
        '--qe-root',
        type=Path,
        default=repo_root / 'tests' / 'integration' / 'qe',
        help='Path to tests/integration/qe (default: tests/integration/qe)',
    )
    p.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output tar.gz path',
    )
    args = p.parse_args()

    build_assets(qe_root=args.qe_root.resolve(), out_tar_gz=args.out.resolve())


if __name__ == '__main__':
    main()
