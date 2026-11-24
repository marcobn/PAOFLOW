"""Build ``howto_assets.tar.gz``, the data of the website how-tos.

The archive holds ``plotting/<system>/``: PAOFLOW output files of the QE examples, read by
``docs/learn/howtos/plotting.md``. Si and Bi come from fresh runs of
``examples/qe_examples/example01/main.py`` and ``examples/pyskeaf_examples/Bi/main.py`` (pass
their ``output/`` directories with ``--si-output`` and ``--bi-output``), because the committed
Reference files predate the orbital-named PDOS files and the BXSF header with E_F = 0. The other
systems come from the committed ``Reference/`` directories of their examples.

It also holds ``quantum-oscillations/``, the inputs of
``docs/learn/howtos/quantum-oscillations.md``, copied unchanged from
``examples/qo_analysis_examples/NbSn2/``: ``pyskeaf/`` holds the pyskeaf rotations of the SOC
bands (``calc/soc/qo_EF_0_freqvsangle_1[1-8].out.gz``, 49^3 BXSF grid, azimuth 29.95 deg, polar
0-90 deg in 90 angles) and ``experiment/`` the measured tables (``data/exp_sweep_peaks.csv``,
``data/exp_LK_table.csv``).

It also holds ``twisted-bilayer/``, what ``examples/cases/tbg_unfold/twisted_bilayer.py``
writes, because running it takes longer than the page budget (6 min on 4 cores): the page test
runs ``docs/learn/howtos/twisted-bilayer/figures.py`` inside this folder. Make it by running, in an empty directory,
``OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 python main.py > main.log``, and pass that directory,
with the output files and ``main.log``, as ``--twisted-bilayer``.

To rebuild an archive whose Si and Bi files are unchanged, extract ``plotting/si`` and
``plotting/bi`` of the previous archive and pass those directories as ``--si-output`` and
``--bi-output``.

Upload to a new release tag next to an unchanged copy of the tutorial assets, never replacing
a published archive (``upload_release_assets.sh`` of the tutorials creates the tag)::

    sha256sum howto_assets.tar.gz > howto_SHA256SUMS
    gh release upload tutorial-assets-vN howto_assets.tar.gz howto_SHA256SUMS --repo marcobn/PAOFLOW
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
QE = REPO_ROOT / 'examples' / 'qe_examples'
NBSN2 = REPO_ROOT / 'examples' / 'qo_analysis_examples' / 'NbSn2'

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

# archive folder -> (source directory, file patterns)
QUANTUM_OSCILLATIONS = {
    'pyskeaf': (NBSN2 / 'calc' / 'soc', ['qo_EF_0_freqvsangle_1[1-8].out.gz']),
    'experiment': (NBSN2 / 'data', ['exp_sweep_peaks.csv', 'exp_LK_table.csv']),
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
        for path in sorted(p for p in runs['tbg'].rglob('*') if p.is_file()):
            archive.add(path, arcname=f'twisted-bilayer/{path.relative_to(runs["tbg"])}')
        for folder, (source, patterns) in QUANTUM_OSCILLATIONS.items():
            for pattern in patterns:
                files = sorted(source.glob(pattern))
                if not files:
                    raise SystemExit(f'{source}: no file matches {pattern}')
                for path in files:
                    archive.add(path, arcname=f'quantum-oscillations/{folder}/{path.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the release archive for the how-tos.')
    parser.add_argument('--si-output', type=Path, required=True, help='output/ of example01')
    parser.add_argument('--bi-output', type=Path, required=True, help='output/ of the Bi example')
    parser.add_argument(
        '--twisted-bilayer',
        type=Path,
        required=True,
        help='twisted-bilayer/ written by examples/cases/tbg_unfold/twisted_bilayer.py',
    )
    parser.add_argument(
        '--out', type=Path, default=Path(__file__).resolve().parent / 'howto_assets.tar.gz'
    )
    args = parser.parse_args()
    runs = {
        'si': args.si_output.resolve(),
        'bi': args.bi_output.resolve(),
        'tbg': args.twisted_bilayer.resolve(),
    }
    build_assets(runs, args.out.resolve())


if __name__ == '__main__':
    main()
