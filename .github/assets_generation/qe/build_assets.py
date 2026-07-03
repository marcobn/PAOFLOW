from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Iterable

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tests.integration.qe.jobs import JobSpec, discover_jobs


def _add_dir(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    tf.add(str(path), arcname=arcname, recursive=True)


def _render_progress(current: int, total: int, arcname: str) -> None:
    if total <= 0:
        return

    percent = current / total
    if sys.stderr.isatty():
        width = 24
        filled = min(width, int(percent * width))
        bar = '#' * filled + '-' * (width - filled)
        message = f'[{bar}] {current}/{total} {percent:>6.1%} {arcname}'
        end = '\n' if current == total else '\r'
        print(message, file=sys.stderr, end=end, flush=True)
        return

    checkpoints = {1, total}
    for fraction in (0.25, 0.5, 0.75):
        checkpoint = max(1, int(total * fraction))
        checkpoints.add(checkpoint)

    if current in checkpoints:
        print(f'[{current}/{total}] packing {arcname}', file=sys.stderr, flush=True)


def _iter_savedirs(job_dir: Path) -> Iterable[Path]:
    for savedir in sorted(job_dir.glob('*.save')):
        if savedir.is_dir():
            yield savedir


def _reference_dir(job_dir: Path, reference_root: Path | None, job_id: str) -> Path | None:
    if reference_root is None:
        return None

    staged_job_root = reference_root / job_id
    staged_ref = staged_job_root / 'Reference'
    if staged_ref.is_dir():
        return staged_ref
    if staged_job_root.is_dir():
        return staged_job_root
    return None


def _internal_basis_species_for_example(example_name: str) -> tuple[str, ...]:
    if example_name == 'example10':
        return ('Ga', 'As')
    if example_name == 'example15':
        return ('Si',)
    return ()


def _has_repack_payload(job_dir: Path) -> bool:
    if (job_dir / 'Reference').is_dir():
        return True
    return any(savedir.is_dir() for savedir in job_dir.glob('*.save'))


def _discover_repack_jobs(assets_root: Path) -> list[JobSpec]:
    jobs: list[JobSpec] = []

    for example_root in sorted([p for p in assets_root.glob('example*') if p.is_dir()]):
        if _has_repack_payload(example_root):
            jobs.append(JobSpec(example_root=example_root, job_relpath=Path('.')))

        candidate_dirs: set[Path] = set()
        for ref_dir in example_root.rglob('Reference'):
            if ref_dir.is_dir() and ref_dir.parent != example_root:
                candidate_dirs.add(ref_dir.parent)

        for savedir in example_root.rglob('*.save'):
            if savedir.is_dir() and savedir.parent != example_root:
                candidate_dirs.add(savedir.parent)

        for job_dir in sorted(candidate_dirs):
            if not _has_repack_payload(job_dir):
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
    return [uniq[key] for key in sorted(uniq.keys())]


def _collect_entries(
    *,
    jobs: list[JobSpec],
    reference_root: Path | None,
    basis_root: Path | None,
    selected_examples: set[str] | None,
) -> tuple[list[tuple[Path, str]], int]:
    entries: list[tuple[Path, str]] = []
    payload_count = 0

    for job in jobs:
        if selected_examples is not None and job.example_root.name not in selected_examples:
            continue

        job_dir = job.job_dir
        base_arc = Path(job.example_root.name) / job.job_relpath

        for savedir in _iter_savedirs(job_dir):
            entries.append((savedir, (base_arc / savedir.name).as_posix()))
            payload_count += 1

        refdir = _reference_dir(job_dir, reference_root, job.id)
        if refdir is not None:
            entries.append((refdir, (base_arc / 'Reference').as_posix()))
            payload_count += 1

    if selected_examples is None:
        examples_for_basis = {job.example_root.name for job in jobs}
    else:
        examples_for_basis = selected_examples

    added_basis: set[str] = set()
    for example_name in sorted(examples_for_basis):
        for species in _internal_basis_species_for_example(example_name):
            if species in added_basis:
                continue

            basis_parent = basis_root if basis_root is not None else repo_root / 'BASIS'
            basis_dir = basis_parent / species
            if not basis_dir.is_dir():
                raise SystemExit(f'Missing BASIS directory for internal-basis species: {basis_dir}')

            entries.append((basis_dir, (Path('BASIS') / species).as_posix()))
            added_basis.add(species)

    return entries, payload_count


def build_assets(
    *,
    qe_root: Path,
    out_tar_gz: Path,
    reference_root: Path | None = None,
    basis_root: Path | None = None,
    repack_layout: bool = False,
    selected_examples: set[str] | None = None,
) -> None:
    jobs = _discover_repack_jobs(qe_root) if repack_layout else discover_jobs(qe_root)
    if not jobs:
        raise SystemExit(f'No jobs found under {qe_root}')

    entries, added_payload_dirs = _collect_entries(
        jobs=jobs,
        reference_root=reference_root,
        basis_root=basis_root,
        selected_examples=selected_examples,
    )

    if added_payload_dirs == 0:
        out_tar_gz.unlink(missing_ok=True)
        raise SystemExit(
            f'No *.save directories, staged Reference folders, or BASIS directories found under {qe_root}'
        )

    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    print(
        f'Packing {len(entries)} archive entries into {out_tar_gz.name}',
        file=sys.stderr,
        flush=True,
    )

    with tarfile.open(out_tar_gz, 'w:gz') as tf:
        total_entries = len(entries)
        for index, (path, arcname) in enumerate(entries, start=1):
            _add_dir(tf, path, arcname)
            _render_progress(index, total_entries, arcname)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            'Build a local qe_test_assets tar.gz from existing *.save folders, '
            'staged Reference folders, and BASIS directories.'
        )
    )
    p.add_argument(
        '--qe-root',
        type=Path,
        default=repo_root / 'examples' / 'qe_examples',
        help='Path to examples/qe_examples (default: examples/qe_examples)',
    )
    p.add_argument(
        '--reference-root',
        type=Path,
        default=repo_root / 'tests' / 'integration' / 'qe' / '_assets' / 'staging',
        help='Path to staged Reference folders (default: tests/integration/qe/_assets/staging)',
    )
    p.add_argument(
        '--basis-root',
        type=Path,
        default=repo_root / 'BASIS',
        help='Path to BASIS directories (default: BASIS)',
    )
    p.add_argument(
        '--repack-layout',
        action='store_true',
        help='Discover jobs from an unpacked asset tree containing Reference/ and *.save entries.',
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
        qe_root=args.qe_root.resolve(),
        out_tar_gz=args.out.resolve(),
        reference_root=args.reference_root.resolve() if args.reference_root else None,
        basis_root=args.basis_root.resolve() if args.basis_root else None,
        repack_layout=bool(args.repack_layout),
        selected_examples=selected_examples,
    )


if __name__ == '__main__':
    main()
