"""``paoflow_outputs.json``: what a PAOFLOW run wrote, with the facts the file names do not carry.

PAOFLOW shifts every energy to the Fermi level and writes files without the prefix, the units or
the DFT Fermi energy.  At ``finish_execution`` rank 0 records, for the run, the PAOFLOW version
(with the commit and the state of the working tree when PAOFLOW runs from a git checkout), the
save, the DFT Fermi energy, the spin treatment, the basis size and the list of the files of the
output folder written since the run started, each with the output kind of the results catalogue.
Several PAOFLOW instances of one script (a standard and an optical run sharing ``output/``) append
their entries.  The manifest is an output for readers (the GUI, reports); nothing reads it back
as configuration.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import time
from typing import Any

MANIFEST = 'paoflow_outputs.json'
MAX_RUNS = 20


def _scalar(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover
        pass
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


@functools.lru_cache(maxsize=1)
def source_revision() -> dict[str, Any] | None:
    """Commit of the PAOFLOW source tree this process imported, if it is a git checkout.

    Returns
    -------
    dict or None
        ``{'commit': <40-hex sha>, 'modified': bool}``; ``modified`` is True when tracked files
        differ from the commit (an editable install with local edits). None for an installed
        package (no git checkout around it), without ``git``, or on any git error.

    Notes
    -----
    The checkout must be the one that contains this package (``<top>/src/PAOFLOW``): a package
    installed in a virtual environment that happens to sit inside another repository reports
    None, not that repository's commit.
    """
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*args: str) -> str:
        return subprocess.run(
            ['git', '-C', package, *args], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        top = git('rev-parse', '--show-toplevel')
        if os.path.realpath(os.path.join(top, 'src', 'PAOFLOW')) != os.path.realpath(package):
            return None
        commit = git('rev-parse', 'HEAD')
        modified = bool(git('status', '--porcelain', '--untracked-files=no'))
    except (OSError, subprocess.SubprocessError):
        return None
    return {'commit': commit, 'modified': modified}


def run_record(attributes: dict, start_time: float, output_path: str) -> dict:
    """The manifest entry of one PAOFLOW instance.

    Parameters
    ----------
    attributes : dict
        ``DataController.data_attributes``.
    start_time : float
        Start of the run (epoch seconds); files modified since then are listed.
    output_path : str
        The output folder.

    Returns
    -------
    dict
    """
    from .. import __version__

    try:
        from ..results.discovery import match_file
    except ImportError:  # pragma: no cover - the results layer ships with PAOFLOW
        match_file = None
    files = []
    for name in sorted(os.listdir(output_path)):
        path = os.path.join(output_path, name)
        if (
            name == MANIFEST
            or not os.path.isfile(path)
            or os.path.getmtime(path) < start_time - 1.0
        ):
            continue
        entry: dict[str, Any] = {'file': name, 'bytes': os.path.getsize(path)}
        found = match_file(name, '.') if match_file is not None else None
        if found is not None:
            kind, groups = found
            entry.update(kind=kind.name, category=kind.category, labels=groups)
        files.append(entry)
    keys = (
        'savedir',
        'Efermi',
        'nspin',
        'dftSO',
        'dftMAG',
        'nawf',
        'nbnds',
        'nelec',
        'natoms',
        'nk1',
        'nk2',
        'nk3',
        'smearing',
        'insulator',
    )
    facts = {key: _scalar(attributes.get(key)) for key in keys if key in attributes}
    if 'Efermi' in facts:
        facts['fermi_energy_dft_ev'] = facts.pop('Efermi')
    revision = source_revision()
    if revision is not None:
        facts = {
            'paoflow_commit': revision['commit'],
            'paoflow_tree_modified': revision['modified'],
            **facts,
        }
    return {
        'paoflow': __version__,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time)),
        'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'energy_reference': 'electronic energies are relative to the DFT Fermi energy (E_F = 0)',
        **facts,
        'files': files,
    }


def write_output_manifest(attributes: dict, start_time: float) -> str | None:
    """Append the entry of this run to ``<output>/paoflow_outputs.json`` (rank 0 only).

    Parameters
    ----------
    attributes : dict
        ``DataController.data_attributes`` (``opath`` is the output folder).
    start_time : float
        Start of the run (epoch seconds).

    Returns
    -------
    str or None
        The manifest path, or ``None`` without an output folder.
    """
    output_path = attributes.get('opath')
    if not output_path or not os.path.isdir(output_path):
        return None
    path = os.path.join(output_path, MANIFEST)
    runs: list = []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            runs = json.load(handle).get('runs', [])
    except (OSError, ValueError, AttributeError):
        runs = []
    runs.append(run_record(attributes, start_time, output_path))
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump({'manifest': 1, 'runs': runs[-MAX_RUNS:]}, handle, indent=1)
    os.replace(temporary, path)
    return path
