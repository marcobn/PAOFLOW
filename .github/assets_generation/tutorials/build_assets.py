from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def build_assets(source_root: Path, output_path: Path, selected: set[str] | None = None) -> None:
    tutorial_dirs = sorted(path for path in source_root.glob('tutorial[0-9][0-9]') if path.is_dir())
    if selected is not None:
        tutorial_dirs = [path for path in tutorial_dirs if path.name in selected]
    if not tutorial_dirs:
        raise SystemExit(f'No tutorial asset directories found under {source_root}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, 'w:gz') as archive:
        for tutorial_dir in tutorial_dirs:
            savedirs = sorted(path for path in tutorial_dir.glob('*.save') if path.is_dir())
            if not savedirs:
                raise SystemExit(
                    f'{tutorial_dir.name}: at least one QE .save directory is required'
                )

            for savedir in savedirs:
                files = sorted(
                    path
                    for path in savedir.rglob('*')
                    if path.is_file() and (path.suffix == '.xml' or path.suffix == '.UPF')
                )
                if not files:
                    raise SystemExit(f'{savedir}: no XML or UPF files found')
                for path in files:
                    archive.add(path, arcname=path.relative_to(source_root))

            output_dir = tutorial_dir / 'output'
            if output_dir.is_dir():
                archive.add(output_dir, arcname=output_dir.relative_to(source_root))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description='Build the release archive for website tutorial assets.'
    )
    parser.add_argument('--source-root', type=Path, default=script_dir)
    parser.add_argument(
        '--out',
        type=Path,
        default=script_dir / 'tutorial_assets.tar.gz',
    )
    parser.add_argument(
        '--tutorial',
        action='append',
        help='Optional tutorial ID to include; repeat to include several',
    )
    args = parser.parse_args()

    build_assets(args.source_root.resolve(), args.out.resolve(), set(args.tutorial or []) or None)


if __name__ == '__main__':
    main()
