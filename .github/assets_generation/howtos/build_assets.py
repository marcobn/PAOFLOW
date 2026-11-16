"""Build ``howto_assets.tar.gz``, the data of the website how-tos.

The archive holds ``plotting/<system>/``: PAOFLOW output files of the QE examples, read by
``docs/tutorials/howtos/plotting.md``. Si and Bi come from fresh runs of
``examples/qe_examples/example01/main.py`` and ``examples/pyskeaf_examples/Bi/main.py`` (pass
their ``output/`` directories with ``--si-output`` and ``--bi-output``), because the committed
Reference files predate the orbital-named PDOS files and the BXSF header with E_F = 0. The other
systems come from the committed ``Reference/`` directories of their examples.

It also holds ``tight-binding/<example>/``: the ``Reference/`` files of the tight-binding model
examples that ``docs/tutorials/howtos/tight-binding-model.md`` replaces (``Cubium``, ``Graphene``,
``Graphene2``, ``Kane_mele``; ``examples/TBmodel_examples`` up to commit e439e4e7). Pass a
directory with one ``<example>/Reference/`` folder per example with ``--tb-reference``. In that
directory the Graphene2 Berry-phase files (``circle_*``, ``square_berry_phase_{0,1,01}.dat``,
``berry_phase_k.pdf``) and the Kane_mele ``shcEf_z_xy.dat``, ``*.bxsf`` and ``shc_z_xy.png`` come
from reruns of the unchanged example scripts (2026-11-16): the committed ones predate commit
9fe821d7, which swaps K and K' of the ``graphene`` model, and a change of the k-resolved
Berry-curvature files. The other files are the committed ones.

Upload next to the tutorial assets, without replacing them::

    sha256sum howto_assets.tar.gz > howto_SHA256SUMS
    gh release upload tutorial-assets-v1 howto_assets.tar.gz howto_SHA256SUMS --repo marcobn/PAOFLOW
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
QE = REPO_ROOT / 'examples' / 'qe_examples'

# system -> (source directory or name of a --<name>-output option, file patterns)
PLOTTING = {
    'si': ('si', ['bands_0.dat', 'kpath_points.txt', 'dosdk_0.dat', 'Si*_pdosdk_0.dat']),
    'al': (QE / 'example02' / 'Reference', ['sigmagauss_0.dat']),
    'fe': (QE / 'example04' / 'Reference', ['bands_0.dat', 'kpath_points.txt', 'Omega_z_yz.dat']),
    'pt': (QE / 'example05' / 'Reference', ['shcEf_z_xy.dat']),
    'alp': (QE / 'example06' / 'Reference', ['Seebeckgauss_0.dat']),
    'snte': (QE / 'example08' / 'Reference', ['shcEf_z_*.dat']),
    'bi': ('bi', ['FermiSurf_0.bxsf']),
}


def build_assets(runs: dict[str, Path], output_path: Path) -> None:
    with tarfile.open(output_path, 'w:gz') as archive:
        for system, (source, patterns) in PLOTTING.items():
            source = runs[source] if isinstance(source, str) else source
            for pattern in patterns:
                files = sorted(source.glob(pattern))
                if not files:
                    raise SystemExit(f'{source}: no file matches {pattern}')
                for path in files:
                    archive.add(path, arcname=f'plotting/{system}/{path.name}')
        for reference in sorted(runs['tb'].glob('*/Reference')):
            for path in sorted(reference.iterdir()):
                archive.add(path, arcname=f'tight-binding/{reference.parent.name}/{path.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the release archive for the how-tos.')
    parser.add_argument('--si-output', type=Path, required=True, help='output/ of example01')
    parser.add_argument('--bi-output', type=Path, required=True, help='output/ of the Bi example')
    parser.add_argument(
        '--tb-reference',
        type=Path,
        required=True,
        help='directory with <example>/Reference/ folders of the tight-binding examples',
    )
    parser.add_argument(
        '--out', type=Path, default=Path(__file__).resolve().parent / 'howto_assets.tar.gz'
    )
    args = parser.parse_args()
    runs = {
        'si': args.si_output.resolve(),
        'bi': args.bi_output.resolve(),
        'tb': args.tb_reference.resolve(),
    }
    build_assets(runs, args.out.resolve())


if __name__ == '__main__':
    main()
