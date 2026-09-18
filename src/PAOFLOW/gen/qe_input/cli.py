"""Command-line interface for the multi-database QE-input generator.

Generates a Quantum ESPRESSO scf input from a database entry.  The database is
selected with ``--source`` (default ``auto`` detects it from the identifier):

* AFLOW  -- AFLOWDATA URL, material page ``?id=aflow:<auid>``, or ``aflow:<auid>``
* C2DB   -- material UID (e.g. ``2C-1``), material-page URL, or ``c2db:<uid>``
"""

import argparse
import os
import sys

from .sources import available_sources, detect_source, get_source
from .writer import DEFAULT_DEGAUSS, build_qe_input


def _build_parser():
    parser = argparse.ArgumentParser(
        prog='paoflow-gen-qe',
        description='Generate a Quantum ESPRESSO scf input from a materials-database entry.',
    )
    parser.add_argument(
        'identifier',
        help='Database entry. AFLOW: an AFLOWDATA directory URL, a material '
        'page URL (?id=aflow:<auid>), or a bare aflow:<auid> token. '
        'C2DB: a material UID (e.g. 2C-1), a material-page URL, or a '
        'c2db:<uid> token.',
    )
    parser.add_argument(
        '--source',
        choices=['auto', *available_sources()],
        default='auto',
        help="Database to read from (default: 'auto' detects it from the identifier)",
    )
    parser.add_argument(
        '-p',
        '--pseudo-dir',
        required=True,
        help='Folder with <element>.upf, PeriodicTableJSON.json and reference.json',
    )
    parser.add_argument(
        '--soc',
        action='store_true',
        help='Force noncolin/lspinorb cards. Auto-enabled when a pseudopotential '
        'is fully relativistic (relativistic="full")',
    )
    parser.add_argument(
        '-o',
        '--out',
        default=None,
        help='Output path (default: <compound>.scf.in in the current directory)',
    )
    parser.add_argument(
        '--degauss',
        type=float,
        default=DEFAULT_DEGAUSS,
        help='Smearing width in Ry for metallic systems (default: %(default)s)',
    )
    parser.add_argument(
        '--nbnd',
        type=int,
        default=None,
        help='Override nbnd (default: estimated from the extended PAO basis)',
    )
    parser.add_argument(
        '--ibrav',
        choices=('auto', '0'),
        default='auto',
        help="Lattice mode: 'auto' detects the QE ibrav + celldm from the cell "
        '(2D entries keep the in-plane symmetry with vacuum on c; falls back to '
        "ibrav=0 when undetectable), '0' always writes explicit CELL_PARAMETERS "
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--symprec',
        type=float,
        default=1.0e-4,
        help='Symmetry tolerance for ibrav detection (default: %(default)s)',
    )
    parser.add_argument(
        '--metallic',
        dest='metallic',
        action='store_true',
        default=None,
        help='Force metallic treatment (smearing). Overrides database metadata',
    )
    parser.add_argument(
        '--insulating',
        dest='metallic',
        action='store_false',
        default=None,
        help='Force insulating treatment (fixed occupations). Overrides metadata',
    )
    parser.add_argument(
        '--magnetic',
        dest='magnetic',
        action='store_true',
        default=None,
        help='Force a spin-polarized input. Overrides database metadata',
    )
    parser.add_argument(
        '--nonmagnetic',
        dest='magnetic',
        action='store_false',
        default=None,
        help='Force a non-magnetic input. Overrides database metadata',
    )
    parser.add_argument(
        '--kpoints',
        type=int,
        nargs=3,
        metavar=('K1', 'K2', 'K3'),
        default=None,
        help='Monkhorst-Pack grid override (k3 forced to 1 for 2D entries)',
    )
    parser.add_argument(
        '--no-assume-isolated-2d',
        dest='assume_isolated_2d',
        action='store_false',
        default=True,
        help="Do not emit assume_isolated='2D' for 2D entries",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    pseudo_dir = os.path.abspath(os.path.expanduser(args.pseudo_dir))
    if not os.path.isdir(pseudo_dir):
        parser.error('pseudo-dir does not exist: {}'.format(pseudo_dir))

    try:
        if args.source == 'auto':
            source = detect_source(args.identifier)
        else:
            source = get_source(args.source)

        record = source.fetch(
            args.identifier,
            metallic=args.metallic,
            magnetic=args.magnetic,
            kpoints=args.kpoints,
        )
        if args.kpoints is not None:
            record.kpoints = tuple(args.kpoints)

        content = build_qe_input(
            record,
            pseudo_dir,
            soc=args.soc,
            degauss=args.degauss,
            nbnd_override=args.nbnd,
            ibrav_mode=args.ibrav,
            symprec=args.symprec,
            assume_isolated_2d=args.assume_isolated_2d,
        )
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write('Error: {}\n'.format(exc))
        return 1

    out_path = args.out or '{}.scf.in'.format(record.compound or 'system')
    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write(content)

    sys.stderr.write('Wrote {}\n'.format(os.path.abspath(out_path)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
