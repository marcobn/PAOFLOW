"""Command-line entry point for the pyskeaf port.

Usage::

    python -m pyskeaf [config.in] [--bxsf FILE] [--out DIR] [--no-write]

Mirrors the Fortran ``skeaf.x`` driver: by default it reads ``config.in``
from the current directory, loads the BXSF file named within it, runs the
analysis, and writes the five Fortran-compatible output files to ``DIR``
(default: cwd).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PAOFLOW.pyskeaf.config import read_config_in
from PAOFLOW.pyskeaf.io_bxsf import read_bxsf
from PAOFLOW.pyskeaf.runner import run_skeaf


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyskeaf",
        description="Python port of SKEAF — Supercell K-space Extremal Area Finder.",
    )
    p.add_argument("config", nargs="?", default="config.in",
                   help="Path to a SKEAF config.in file (default: ./config.in).")
    p.add_argument("--bxsf", default=None,
                   help="Override the BXSF filename listed in config.in.")
    p.add_argument("--out", default=None,
                   help="Directory to write output files into (default: cwd).")
    p.add_argument("--no-write", dest="write", action="store_false",
                   help="Run the analysis but skip writing output files.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"pyskeaf: config file not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = read_config_in(cfg_path)
    bxsf_path = Path(args.bxsf) if args.bxsf else Path(cfg.filename)
    if not bxsf_path.exists():
        alt = cfg_path.parent / bxsf_path
        if alt.exists():
            bxsf_path = alt
        else:
            print(f"pyskeaf: BXSF file not found: {bxsf_path}", file=sys.stderr)
            return 3

    bxsf = read_bxsf(bxsf_path)
    out_dir = Path(args.out) if args.out else Path.cwd()
    result = run_skeaf(cfg, bxsf, write_files=args.write, output_dir=out_dir)

    n_orbits = len(result.orbits)
    n_angles = result.angles.shape[0] if result.angles is not None else 1
    print(
        f"pyskeaf: completed {n_angles} angle(s); "
        f"found {n_orbits} extremal orbit(s) in total.",
        file=sys.stderr,
    )
    if args.write:
        print(f"pyskeaf: outputs written to {out_dir.resolve()}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
