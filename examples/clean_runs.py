#!/usr/bin/env python3
"""Remove files produced by example runs, keeping only inputs and references.

Mirrors the ``examples/**`` rules in the repository ``.gitignore``: every
file that is *not* an input, pseudopotential, or curated reference is
deleted.  Empty directories left behind are removed too, except for the
top-level example directories themselves.

Usage
-----
    # Show what would be deleted (default, no changes on disk):
    python examples/clean_runs.py

    # Actually delete:
    python examples/clean_runs.py --apply

    # Restrict to one or more example subtrees:
    python examples/clean_runs.py --apply qe_examples/example01 vasp_examples
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Files kept by exact basename (case-sensitive on POSIX).
KEEP_NAMES = {
    'inputfile.xml',
    'POTCAR',
    'INCAR',
    'KPOINTS',
    'POSCAR',
}

# Files kept by lowercase suffix.
KEEP_SUFFIXES = {
    '.in',
    '.py',
    '.ipynb',
    '.md',
    '.upf',  # match .UPF case-insensitively
    '.pptx',  # tutorial slide decks
    '.yaml',  # transport_examples configs
    '.yml',
    '.sh',  # job scripts shipped with examples
    '.bxsf',  # pyskeaf_examples Fermi-surface inputs
}

# Basename prefixes that are always kept (e.g. README, README.md, README.rst).
KEEP_PREFIXES = ('README',)

# Directory names whose entire subtree is preserved verbatim.
# - Reference/Reference2: curated expected outputs.
# - BASIS: per-tutorial atomic basis files (inputs to PAOFLOW).
# - data_files: shipped reference data consumed by examples/plot_examples/.
KEEP_DIRS = {'Reference', 'Reference2', 'BASIS', 'data_files'}

# Directory name suffixes (lowercase) whose entire subtree is removed.
# e.g. 'pt.save', 'silicon.save', ... produced by pw.x.
REMOVE_DIR_SUFFIXES = {'.save'}


def _should_keep_file(path: Path) -> bool:
    name = path.name
    if name in KEEP_NAMES:
        return True
    if any(name.startswith(p) for p in KEEP_PREFIXES):
        return True
    if path.suffix.lower() in KEEP_SUFFIXES:
        return True
    return False


def _iter_targets(roots: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (files_to_delete, dirs_to_consider_for_removal, save_dirs_to_delete)."""
    files: list[Path] = []
    dirs: list[Path] = []
    save_dirs: list[Path] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            d = Path(dirpath)
            # Collect *.save dirs for whole-tree removal; prune from walk.
            to_remove = [x for x in dirnames if x.lower().endswith(tuple(REMOVE_DIR_SUFFIXES))]
            for x in to_remove:
                save_dirs.append(d / x)
            # Skip Reference/Reference2 subtrees and *.save dirs entirely.
            dirnames[:] = [x for x in dirnames if x not in KEEP_DIRS and x not in to_remove]
            for fname in filenames:
                fpath = d / fname
                if not _should_keep_file(fpath):
                    files.append(fpath)
            dirs.append(d)
    return files, dirs, save_dirs


def _prune_empty_dirs(dirs: list[Path], protected: set[Path]) -> list[Path]:
    """Remove empty directories bottom-up, skipping any in ``protected``."""
    removed: list[Path] = []
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        if d in protected or not d.exists():
            continue
        try:
            d.rmdir()
            removed.append(d)
        except OSError:
            # Not empty (or permission issue) -- leave it alone.
            pass
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'paths',
        nargs='*',
        help='Paths to clean (absolute or relative to CWD). Defaults to the whole examples/ tree.',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually delete files. Without this flag the script only prints '
        'what would be deleted.',
    )
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress per-file output.')
    args = parser.parse_args(argv)

    examples_root = Path(__file__).resolve().parent
    if args.paths:
        roots = [Path(p).resolve() for p in args.paths]
    else:
        roots = [examples_root]

    missing = [r for r in roots if not r.exists()]
    if missing:
        for r in missing:
            print(f'error: path does not exist: {r}', file=sys.stderr)
        return 2

    files, dirs, save_dirs = _iter_targets(roots)

    if not args.quiet:
        for f in files:
            print(
                ('delete ' if args.apply else 'would delete ')
                + str(f.relative_to(examples_root.parent))
            )
        for sd in save_dirs:
            print(
                ('rmtree ' if args.apply else 'would rmtree ')
                + str(sd.relative_to(examples_root.parent))
            )

    if args.apply:
        import shutil

        for f in files:
            try:
                f.unlink()
            except OSError as exc:
                print(f'warning: could not delete {f}: {exc}', file=sys.stderr)
        for sd in save_dirs:
            try:
                shutil.rmtree(sd)
            except OSError as exc:
                print(f'warning: could not remove {sd}: {exc}', file=sys.stderr)
        # Protect the user-supplied roots so we never rmdir them.
        protected = {r.resolve() for r in roots}
        removed_dirs = _prune_empty_dirs(dirs, protected)
        if not args.quiet:
            for d in removed_dirs:
                print(f'rmdir {d.relative_to(examples_root.parent)}')

    verb = 'deleted' if args.apply else 'would delete'
    n_save = len(save_dirs)
    print(f'\n{verb} {len(files)} file(s) and {n_save} *.save dir(s) under {len(roots)} root(s).')
    if not args.apply:
        print('Re-run with --apply to perform the deletions.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
