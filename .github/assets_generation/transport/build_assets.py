from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Iterable

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tests.integration.transport.jobs import discover_jobs


def _add_dir(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    tf.add(str(path), arcname=arcname, recursive=True)


def _iter_savedirs(job_dir: Path) -> Iterable[Path]:
    savedirs = sorted(path for path in job_dir.rglob('*.save') if path.is_dir())
    seen_names: set[str] = set()

    for savedir in savedirs:
        name = savedir.name
        if name in seen_names:
            raise SystemExit(f'Ambiguous savedir name under {job_dir}: {name}')
        seen_names.add(name)
        yield savedir


def _reference_dir(job_id: str, job_dir: Path, reference_root: Path | None) -> Path | None:
    if reference_root is not None:
        staged_root = reference_root / job_id
        staged_ref = staged_root / 'Reference'
        if staged_ref.is_dir():
            return staged_ref
        if staged_root.is_dir():
            return staged_root

    refdir = job_dir / 'Reference'
    if refdir.is_dir():
        return refdir
    return None


def build_assets(
    *,
    transport_root: Path,
    out_tar_gz: Path,
    reference_root: Path | None = None,
    selected_examples: set[str] | None = None,
) -> None:
    jobs = discover_jobs(transport_root)
    if not jobs:
        raise SystemExit(f'No jobs found under {transport_root}')

    added_dirs = 0
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, 'w:gz') as tf:
        for job in jobs:
            if selected_examples is not None and job.example_root.name not in selected_examples:
                continue

            job_dir = job.job_dir
            base_arc = Path(job.example_root.name) / job.job_relpath

            refdir = _reference_dir(job.id, job_dir, reference_root)
            if refdir is not None:
                _add_dir(tf, refdir, (base_arc / 'Reference').as_posix())
                added_dirs += 1

            for savedir in _iter_savedirs(job_dir):
                _add_dir(tf, savedir, (base_arc / savedir.name).as_posix())
                added_dirs += 1

    if added_dirs == 0:
        out_tar_gz.unlink(missing_ok=True)
        raise SystemExit(
            f'No staged Reference folders or *.save directories found under {transport_root}'
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            'Build a local transport_test_assets tar.gz from existing *.save and Reference folders.'
        )
    )
    p.add_argument(
        '--transport-root',
        type=Path,
        default=repo_root / 'examples' / 'transport_examples',
        help='Path to examples/transport_examples (default: examples/transport_examples)',
    )
    p.add_argument(
        '--reference-root',
        type=Path,
        default=None,
        help=(
            'Optional root containing staged Reference folders keyed by job id '
            '(for example tests/integration/transport/_assets/staging).'
        ),
    )
    p.add_argument(
        '--examples',
        type=str,
        default=None,
        help='Optional comma-separated list of top-level example names to include.',
    )
    p.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output tar.gz path',
    )
    args = p.parse_args()

    selected_examples = None
    if args.examples:
        selected_examples = {name.strip() for name in args.examples.split(',') if name.strip()}

    build_assets(
        transport_root=args.transport_root.resolve(),
        out_tar_gz=args.out.resolve(),
        reference_root=args.reference_root.resolve() if args.reference_root else None,
        selected_examples=selected_examples,
    )


if __name__ == '__main__':
    main()
