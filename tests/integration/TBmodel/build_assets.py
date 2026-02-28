from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path

try:
    from .jobs import discover_jobs
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tests.integration.TBmodel.jobs import discover_jobs


def _infer_outputdir(script_path: Path) -> str:
    try:
        text = script_path.read_text(encoding='utf-8')
    except Exception:
        return script_path.stem

    m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        return script_path.stem
    return m.group(1).strip()


def _resolve_outputdir(script_path: Path) -> tuple[Path, Path]:
    raw = _infer_outputdir(script_path)
    base_dir = script_path.parent
    outp = Path(raw)
    if not outp.is_absolute():
        outp = (base_dir / outp).resolve()

    try:
        rel = outp.relative_to(base_dir)
    except ValueError:
        rel = Path(outp.name)
    return outp, rel


def build_assets(*, tbmodel_root: Path, out_tar_gz: Path) -> None:
    jobs = discover_jobs(tbmodel_root)
    if not jobs:
        raise SystemExit(f'No jobs found under {tbmodel_root}')

    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, 'w:gz') as tf:
        for job in jobs:
            script_path = job.script_path
            if not script_path.exists():
                continue

            outdir, rel_outdir = _resolve_outputdir(script_path)
            if not outdir.is_dir():
                continue

            base_arc = Path(job.example_name) / rel_outdir
            for datap in sorted(outdir.glob('*.dat')):
                tf.add(str(datap), arcname=(base_arc / datap.name).as_posix())


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
