from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from .jobs import discover_jobs


def _add_dir(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    tf.add(str(path), arcname=arcname, recursive=True)


def build_assets(*, tbmodel_root: Path, out_tar_gz: Path) -> None:
    jobs = discover_jobs(tbmodel_root)
    if not jobs:
        raise SystemExit(f'No jobs found under {tbmodel_root}')

    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, 'w:gz') as tf:
        for job in jobs:
            job_dir = job.job_dir
            if not job_dir.exists():
                continue

            refdir = job_dir / 'Reference'
            if refdir.is_dir():
                _add_dir(tf, refdir, (Path(job.example_name) / 'Reference').as_posix())

            for savedir in sorted(job_dir.glob('*.save')):
                if savedir.is_dir():
                    _add_dir(tf, savedir, (Path(job.example_name) / savedir.name).as_posix())


def main() -> None:
    p = argparse.ArgumentParser(
        description=('Build a local tbmodel_test_assets tar.gz from existing Reference folders.')
    )
    p.add_argument(
        '--tbmodel-root',
        type=Path,
        default=Path(__file__).resolve().parent,
        help='Path to tests/integration/TBmodel (default: this directory)',
    )
    p.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output tar.gz path',
    )
    args = p.parse_args()

    build_assets(tbmodel_root=args.tbmodel_root.resolve(), out_tar_gz=args.out.resolve())


if __name__ == '__main__':
    main()
