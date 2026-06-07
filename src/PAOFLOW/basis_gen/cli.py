"""CLI entry point: ``paoflow-genbasis-ps``.

Generate per-element ``BASIS_PS/<elem>/<shell>.dat`` files by solving
the pseudo-atom radial Schrödinger equation in a confining box for a
given UPF (norm-conserving, ultrasoft, or PAW) or every UPF in a
directory.  For ultrasoft/PAW pseudos the generalized eigenproblem
``H u = eps S u`` is solved with the augmentation overlap operator
``S = I + sum_ij q_ij |beta_i><beta_j|`` built from ``PP_AUGMENTATION``.
Output files are consumed by ``PAOFLOW.projections(configuration='standard'
or 'extended', basispath=<out_dir>/)`` and respect the j-resolved naming
convention picked up by ``build_aewfc_basis``.
"""

from __future__ import annotations

import argparse
import os
import sys

from .driver import generate_basis_for_directory, generate_basis_for_pseudo


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog='paoflow-genbasis-ps',
        description='Generate pseudized AE-style basis files from UPF '
        'pseudopotentials (norm-conserving, ultrasoft, or PAW).',
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--pseudo', help='Path to a single UPF file.')
    src.add_argument('--pseudo-dir', help='Directory containing one or more *.UPF / *.upf files.')
    p.add_argument(
        '--out',
        required=True,
        help='Output directory (per-element subdirectories are created underneath).',
    )
    p.add_argument(
        '--preset',
        default='extended',
        choices=('minimal', 'standard', 'extended'),
        help='Augmentation preset (default: extended).',
    )
    p.add_argument(
        '--shells',
        default=None,
        help=(
            'Comma-separated explicit shell labels (e.g. "3S,3P,3D"). '
            'Overrides --preset.  Only valid with --pseudo.'
        ),
    )
    p.add_argument(
        '--r-box',
        type=float,
        default=None,
        help='Confining box radius in Bohr (default: min(upf.r[-1], 10)).',
    )
    p.add_argument(
        '--n-points',
        type=int,
        default=2000,
        help='Uniform-mesh resolution (default: 2000).',
    )
    p.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Skip files that already exist.',
    )
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.shells is not None and args.pseudo_dir is not None:
        print('error: --shells cannot be combined with --pseudo-dir', file=sys.stderr)
        return 2
    shells = [s.strip() for s in args.shells.split(',') if s.strip()] if args.shells else None
    os.makedirs(args.out, exist_ok=True)

    if args.pseudo is not None:
        generate_basis_for_pseudo(
            args.pseudo,
            args.out,
            shells=shells,
            preset=args.preset,
            r_box=args.r_box,
            n_points=args.n_points,
            overwrite=not args.no_overwrite,
            verbose=args.verbose,
        )
    else:
        generate_basis_for_directory(
            args.pseudo_dir,
            args.out,
            preset=args.preset,
            r_box=args.r_box,
            n_points=args.n_points,
            overwrite=not args.no_overwrite,
            verbose=args.verbose,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
