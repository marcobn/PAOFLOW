from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Iterable

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tests.integration.qe.jobs import discover_jobs


def _add_dir(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    tf.add(str(path), arcname=arcname, recursive=True)


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


def build_assets(
    *,
    qe_root: Path,
    out_tar_gz: Path,
    reference_root: Path | None = None,
    selected_examples: set[str] | None = None,
) -> None:
    jobs = discover_jobs(qe_root)
    if not jobs:
        raise SystemExit(f'No jobs found under {qe_root}')

    added_payload_dirs = 0
    added_basis: set[str] = set()
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_tar_gz, 'w:gz') as tf:
        for job in jobs:
            if selected_examples is not None and job.example_root.name not in selected_examples:
                continue

            job_dir = job.job_dir
            base_arc = Path(job.example_root.name) / job.job_relpath

            for savedir in _iter_savedirs(job_dir):
                _add_dir(tf, savedir, (base_arc / savedir.name).as_posix())
                added_payload_dirs += 1

            refdir = _reference_dir(job_dir, reference_root, job.id)
            if refdir is not None:
                _add_dir(tf, refdir, (base_arc / 'Reference').as_posix())
                added_payload_dirs += 1

        if selected_examples is None:
            examples_for_basis = {job.example_root.name for job in jobs}
        else:
            examples_for_basis = selected_examples

        for example_name in sorted(examples_for_basis):
            for species in _internal_basis_species_for_example(example_name):
                if species in added_basis:
                    continue

                basis_dir = repo_root / 'BASIS' / species
                if not basis_dir.is_dir():
                    raise SystemExit(
                        f'Missing BASIS directory for internal-basis species: {basis_dir}'
                    )

                _add_dir(tf, basis_dir, (Path('BASIS') / species).as_posix())
                added_basis.add(species)

    if added_payload_dirs == 0:
        out_tar_gz.unlink(missing_ok=True)
        raise SystemExit(
            f'No *.save directories, staged Reference folders, or BASIS directories found under {qe_root}'
        )


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
        selected_examples=selected_examples,
    )


if __name__ == '__main__':
    main()
