from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from tests.integration.transport.jobs import discover_jobs


def _add_dir(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    tf.add(str(path), arcname=arcname, recursive=True)


def build_assets(*, transport_root: Path, out_tar_gz: Path) -> None:
    jobs = discover_jobs(transport_root)
    if not jobs:
        raise SystemExit(f'No jobs found under {transport_root}')

    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, 'w:gz') as tf:
        for job in jobs:
            job_dir = job.job_dir
            base_arc = Path(job.example_root.name) / job.job_relpath

            refdir = job_dir / 'Reference'
            if refdir.is_dir():
                _add_dir(tf, refdir, (base_arc / 'Reference').as_posix())

            for savedir in sorted(job_dir.glob('*.save')):
                if savedir.is_dir():
                    _add_dir(tf, savedir, (base_arc / savedir.name).as_posix())


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            'Build a local transport_test_assets tar.gz from existing *.save and Reference folders.'
        )
    )
    p.add_argument(
        '--transport-root',
        type=Path,
        default=Path(__file__).resolve().parents[2] / 'integration' / 'transport',
        help='Path to tests/integration/transport (default: tests/integration/transport)',
    )
    p.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output tar.gz path',
    )
    args = p.parse_args()

    build_assets(transport_root=args.transport_root.resolve(), out_tar_gz=args.out.resolve())


if __name__ == '__main__':
    main()
