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
    tf.add(str(path), arcname=arcname, recursive=True)


def build_assets(*, qe_root: Path, out_tar_gz: Path) -> None:
    jobs = discover_jobs(qe_root)
    if not jobs:
        raise SystemExit(f'No jobs found under {qe_root}')

    added_dirs = 0
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_tar_gz, 'w:gz') as tf:
        for job in jobs:
            job_dir = job.job_dir

            for savedir in sorted(job_dir.glob('*.save')):
                if not savedir.is_dir():
                    continue

                arc = (Path(job.example_root.name) / job.job_relpath / savedir.name).as_posix()
                _add_dir(tf, savedir, arc)
                added_dirs += 1

    if added_dirs == 0:
        out_tar_gz.unlink(missing_ok=True)
        raise SystemExit(f'No *.save directories found under {qe_root}')


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            'Build a local QE assets tar.gz from existing *.save folders. '
            'Run create_assets.sh first if the *.save folders do not exist.'
        )
    )
    p.add_argument(
        '--qe-root',
        type=Path,
        default=repo_root / 'examples' / 'qe_examples',
        help='Path to examples/qe_examples (default: examples/qe_examples)',
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
