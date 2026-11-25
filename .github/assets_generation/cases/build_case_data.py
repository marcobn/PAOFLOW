"""Build ``<case>-v<N>.tar.gz``, the data of a case, for the release ``case-data``.

The data of a case are what its pages need and the repository does not hold: the trimmed QE
``.save`` its driver reads (the DFT output, which a change in PAOFLOW does not invalidate) or,
for a page above the run-time budget of the learn skill, the outputs of its slow step. The
archive holds paths relative to the case folder; ``PAOFLOW.gen.cases.fetch_case_data`` extracts
it into a copy of the case, the tutorials' setup cells into the folder of the notebook.

Run the case's steps in a copy (``paoflow-gen --example <case> -d <run>``, then its QE steps, or
the slow step of the case) and pass that folder. The patterns below pick what goes in::

    python build_case_data.py si_bands_transport /tmp/case-data/si_bands_transport --version 1

The script prints the ``data`` entry of ``case.json``. A published archive is never replaced:
new data get a new version, uploaded with ``upload_case_data.sh``.

How the published data were made (QE 7.4.1, serial, in a copy of the case):

- ``si_bands_transport``: v1 ``pw.x -in scf.in``, ``pw.x -in nscf.in`` (wavefunctions for the
  internal projections); v2 ``pw.x -in scf.in``, ``pw.x -in bands.in > bands.out`` (the DFT
  bands along the path, before the nscf run, which overwrites them), ``pw.x -in nscf.in`` and
  ``projwfc.x -in proj.in`` (``atomic_proj.xml``), for the projections how-to, with ``bands.in``
  and ``proj.in`` copied in from ``docs/learn/howtos/projections/``.
- ``si_relaxation_times``, ``si_sparse_backend``: the same two runs and ``projwfc.x -in proj.in``
  (``tutorial01/silicon.save`` of the unpublished ``tutorial-assets-v2`` archive).
- ``tbg_unfold``: with ``twisted_bilayer.py`` copied in from ``docs/learn/howtos/twisted-bilayer/``,
  ``OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 python twisted_bilayer.py >
  twisted_bilayer.log`` (6 min on 4 cores).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

SAVE = ['silicon.save/data-file-schema.xml', 'silicon.save/*.UPF']
PROJWFC = [*SAVE, 'silicon.save/atomic_proj.xml']

# case -> glob patterns relative to the run folder; each must match at least one file
CONTENT = {
    'si_bands_transport': [*PROJWFC, 'silicon.save/wfc*.dat', 'bands.out'],
    'si_relaxation_times': PROJWFC,
    'si_sparse_backend': PROJWFC,
    'tbg_unfold': ['twisted_bilayer.log', 'tbg_4_3/*', 'tbg_6_5/*', 'near_K_4_3/*', 'near_K_6_5/*'],
}


def build(case: str, run: Path, version: int, out: Path) -> Path:
    """Write the archive of ``case`` from the files of ``run``; byte-identical on a rebuild."""
    files = []
    for pattern in CONTENT[case]:
        found = sorted(path for path in run.glob(pattern) if path.is_file())
        if not found:
            raise SystemExit(f'{run}: no file matches {pattern}')
        files += found
    archive = out / f'{case}-v{version}.tar.gz'
    if archive.exists():
        raise SystemExit(f'{archive} exists; a published version is never rebuilt')
    with gzip.GzipFile(archive, 'wb', mtime=0) as raw, tarfile.open(fileobj=raw, mode='w') as tar:
        for path in files:
            info = tar.gettarinfo(path, arcname=str(path.relative_to(run)))
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, '', ''
            with path.open('rb') as handle:
                tar.addfile(info, handle)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the data archive of a case.')
    parser.add_argument('case', choices=sorted(CONTENT))
    parser.add_argument('run', type=Path, help='folder where the steps of the case ran')
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    archive = build(args.case, args.run.resolve(), args.version, args.out.resolve())
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f'{archive} ({archive.stat().st_size / 1e6:.1f} MB)')
    print(json.dumps({'data': {'file': archive.name, 'sha256': sha256}}, indent=2))


if __name__ == '__main__':
    main()
