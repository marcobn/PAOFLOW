"""``paoflow_outputs.json``: what a PAOFLOW run wrote, with the facts the file names do not carry.

PAOFLOW shifts every energy to the Fermi level and writes files without the prefix, the units or
the DFT Fermi energy.  At ``finish_execution`` rank 0 records, for the run, the PAOFLOW version
(with the commit and the state of the working tree when PAOFLOW runs from a git checkout), the
save, the DFT Fermi energy, the spin treatment, the basis size and the list of the files of the
output folder written since the run started, each with the output kind of the results catalogue.
Several PAOFLOW instances of one script (a standard and an optical run sharing ``output/``) append
their entries.  The manifest is an output for readers (the GUI, reports); nothing reads it back
as configuration.

Version 2 of the record (``"manifest": 2``) only adds keys.  :func:`recorded` wraps the public
methods of the orchestrators: rank 0 rewrites the record after each top-level call, so a killed
script leaves a record with ``"status": "unfinished"``; ``finish_execution`` marks it finished and
hashes the output files.  The added keys are the run identifier and UTC times, the DFT inputs
(engine, version, functional, structure, pseudopotentials, file hashes), the ordered method calls
with their arguments, the software environment and the infrastructure.  No hostname, user name,
scheduler job or absolute path is written.  Design: ``docs/internals/fair-records.md``.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Callable

MANIFEST = 'paoflow_outputs.json'
VERSION = 2
# Larger files are recorded by size and modification time; paoflow-pack hashes them.
# shortcut: 50 MB is a guess, set it from real sizes of .save XMLs and volume outputs.
HASH_LIMIT = 50 * 1024**2
_STATE = '_run_record'
_PROCESS_START = time.time()
# Runs of the plain-function paths (record_function_call), one per output folder of this process.
_FUNCTION_RUNS: dict[str, dict] = {}


def _utc(seconds: float | None = None) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(seconds))


def sha256_file(path: str) -> str:
    """Hex sha256 of a file, read in 1 MB blocks.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    str
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_stamp(path: str, root: str) -> dict[str, Any]:
    """Identity of one file: path relative to ``root``, size, and its hash or modification time.

    Parameters
    ----------
    path : str
        The file.
    root : str
        Folder the recorded path is relative to.

    Returns
    -------
    dict
        ``file``, ``bytes`` and ``sha256``; a file above :data:`HASH_LIMIT` carries
        ``modified_utc`` in place of the hash.
    """
    size = os.path.getsize(path)
    stamp: dict[str, Any] = {
        'file': os.path.relpath(path, root).replace(os.sep, '/'),
        'bytes': size,
    }
    if size <= HASH_LIMIT:
        stamp['sha256'] = sha256_file(path)
    else:
        stamp['modified_utc'] = _utc(os.path.getmtime(path))
    return stamp


def encode_argument(value: Any) -> tuple[Any, bool]:
    """A JSON form of one method argument.

    Parameters
    ----------
    value : Any
        The argument.

    Returns
    -------
    tuple of (Any, bool)
        The encoded value, and whether the value can be rebuilt from it.  Scalars, and lists and
        dicts of them, are kept; an array becomes its shape, dtype and sha256, a callable its
        qualified name, any other object its class name and ``repr``: these three cannot be
        rebuilt.  An absolute path is made relative to the current folder, or cut to its last
        name (not rebuildable) when it lies outside.
    """
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str) and os.path.isabs(value):
        # no absolute path in the record: inside the current folder it becomes relative,
        # outside only the last name is kept
        relative = os.path.relpath(value, os.getcwd())
        if relative.startswith('..'):
            return {'path_outside_folder': os.path.basename(value.rstrip(os.sep))}, False
        return relative.replace(os.sep, '/'), True
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, True
    if isinstance(value, complex):
        return {'complex': [value.real, value.imag]}, True
    if isinstance(value, np.ndarray):
        data = np.ascontiguousarray(value)
        return {
            'array': list(value.shape),
            'dtype': str(value.dtype),
            'sha256': hashlib.sha256(data.tobytes()).hexdigest(),
        }, False
    if isinstance(value, (list, tuple)):
        items = [encode_argument(item) for item in value]
        return [item for item, _ in items], all(ok for _, ok in items)
    if isinstance(value, dict):
        items = {str(key): encode_argument(item) for key, item in value.items()}
        return {key: item for key, (item, _) in items.items()}, all(ok for _, ok in items.values())
    if callable(value):
        name = getattr(value, '__qualname__', type(value).__name__)
        return {'callable': '%s.%s' % (getattr(value, '__module__', '?'), name)}, False
    return {'object': type(value).__name__, 'repr': repr(value)[:500]}, False


@functools.lru_cache(maxsize=1)
def environment() -> dict[str, Any]:
    """Versions of Python and of the numerical libraries, and whether the Rust kernels load.

    Returns
    -------
    dict
    """
    from importlib.metadata import PackageNotFoundError, version

    def installed(package: str) -> str | None:
        try:
            return version(package)
        except PackageNotFoundError:
            return None

    try:
        from ..acbn0_native import available

        rust = bool(available())
    except Exception:  # noqa: BLE001
        rust = False
    return {
        'python': platform.python_version(),
        **{package: installed(package) for package in ('numpy', 'scipy', 'mpi4py')},
        'rust_kernels': rust,
    }


@functools.lru_cache(maxsize=1)
def infrastructure() -> dict[str, Any]:
    """The machine as far as it affects the run, without anything that names it.

    Returns
    -------
    dict
        MPI ranks, threads per rank, nodes and cores (scheduler variables, when set), CPU model,
        total memory and the BLAS library of numpy.  No hostname, user, partition or job ID.
    """
    import numpy as np
    from mpi4py import MPI

    from ..utils.parallel_resources import core_budget

    env = os.environ

    def integer(*names: str) -> int | None:
        for name in names:
            if env.get(name, '').isdigit():
                return int(env[name])
        return None

    cpu = platform.processor() or None
    try:
        with open('/proc/cpuinfo', 'r', encoding='utf-8') as handle:
            cpu = re.search(r'model name\s*:\s*(.+)', handle.read()).group(1).strip()
    except (OSError, AttributeError):
        pass
    try:
        memory = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        memory = None
    try:
        blas = np.show_config(mode='dicts')['Build Dependencies']['blas']
        blas = ('%s %s' % (blas.get('name', ''), blas.get('version', ''))).strip() or None
    except Exception:  # noqa: BLE001
        blas = None
    cores = integer('SLURM_NTASKS', 'PBS_NP')
    if cores is not None and 'SLURM_NTASKS' in env:
        cores *= integer('SLURM_CPUS_PER_TASK') or 1
    return {
        'mpi_ranks': MPI.COMM_WORLD.Get_size(),
        'threads_per_rank': core_budget(),
        'nodes': integer('SLURM_JOB_NUM_NODES', 'SLURM_NNODES', 'PBS_NUM_NODES'),
        'cores': cores,
        'cpu_model': cpu,
        'memory_total_gb': memory,
        'blas': blas,
    }


def dft_inputs(attributes: dict, arrays: dict) -> dict[str, Any] | None:
    """The DFT calculation a run started from: engine, version, functional, structure, files.

    Parameters
    ----------
    attributes : dict
        ``DataController.data_attributes``.
    arrays : dict
        ``DataController.data_arrays``.

    Returns
    -------
    dict or None
        None before a reader has run and for model Hamiltonians (no ``a_vectors``).  ``files``
        lists the XML files and pseudopotentials of the save folder with :func:`file_stamp`,
        relative to the working folder.

    Notes
    -----
    The version and the functional are read from the text of the XML, since the readers keep the
    version as a ``major.minor`` float and do not read the functional.
    """
    import numpy as np

    if 'a_vectors' not in arrays or 'dft' not in attributes:
        return None
    record: dict[str, Any] = {
        'engine': {'QE': 'Quantum ESPRESSO'}.get(attributes['dft'], attributes['dft']),
        'cell_bohr': (np.asarray(arrays['a_vectors']) * attributes.get('alat', 1.0)).tolist(),
    }
    if 'species' in arrays:
        record['species'] = [
            {'element': element, 'pseudopotential': pseudo} for element, pseudo in arrays['species']
        ]
    if 'atoms' in arrays and 'tau' in arrays:
        record['atoms'] = [str(atom) for atom in arrays['atoms']]
        record['positions_bohr'] = np.asarray(arrays['tau']).tolist()
    folder = attributes.get('fpath')
    if not folder or not os.path.isdir(folder):
        return record
    root = attributes.get('workpath') or '.'
    names = sorted(
        name
        for pattern in ('*.xml', '*.UPF', '*.upf', 'POTCAR')
        for name in glob.glob(os.path.join(folder, pattern))
    )
    record['files'] = [file_stamp(name, root) for name in names]
    searches = {
        'data-file-schema.xml': (r'<creator[^>]*VERSION="([^"]+)"', r'<functional>([^<]+)<'),
        'data-file.xml': (r'<CREATOR[^>]*VERSION="([^"]+)"', None),
        # shortcut: only the GGA tag; METAGGA and hybrids are not read, extend when VASP
        # cases with them are shared.
        'vasprun.xml': (r'<i name="version"[^>]*>\s*([^<\s]+)', r'name="GGA"[^>]*>\s*([^<\s]+)'),
    }
    for name, (version, functional) in searches.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or os.path.getsize(path) > HASH_LIMIT:
            continue
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            text = handle.read()
        for key, pattern in (('version', version), ('functional', functional)):
            found = re.search(pattern, text) if pattern else None
            if found is not None:
                record[key] = ('GGA=' if name == 'vasprun.xml' and key == 'functional' else '') + (
                    found.group(1).strip()
                )
        break
    return record


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


def _output_files(
    output_path: str, start_time: float, final: bool, extra: tuple, shallow: bool = False
) -> list:
    """Files written since ``start_time`` under the output folder (recursive) and, top level
    only, in the ``extra`` folders (the output folder of a ``Transport`` set elsewhere)."""
    try:
        from ..results.discovery import match_file
    except ImportError:  # pragma: no cover - the results layer ships with PAOFLOW
        match_file = None
    found = []
    if shallow:  # a folder PAOFLOW did not create (the current folder of pyskeaf): no walk
        found += [os.path.join(output_path, name) for name in os.listdir(output_path)]
    else:
        for folder, _, names in os.walk(output_path):
            found += [os.path.join(folder, name) for name in names]
    for folder in extra:
        if os.path.isdir(folder) and os.path.relpath(folder, output_path).startswith('..'):
            found += [os.path.join(folder, name) for name in os.listdir(folder)]
    files = []
    for path in sorted(found):
        name = os.path.basename(path)
        if (
            name in (MANIFEST, MANIFEST + '.tmp')
            or not os.path.isfile(path)
            or os.path.getmtime(path) < start_time - 1.0
        ):
            continue
        relative = os.path.relpath(path, output_path).replace(os.sep, '/')
        entry: dict[str, Any] = {'file': relative, 'bytes': os.path.getsize(path)}
        folder = os.path.dirname(relative) or '.'
        match = match_file(name, folder) if match_file is not None else None
        if match is not None:
            kind, groups = match
            entry.update(kind=kind.name, category=kind.category, labels=groups)
            x, y = kind.x, kind.y
            if kind.describe is not None:
                try:
                    y = kind.describe(groups)[1:3]
                except Exception:  # noqa: BLE001
                    pass
            axes = [{'quantity': label, 'unit': unit} for label, unit in (x, y) if label]
            if axes:
                entry['axes'] = axes
        if final:
            entry.update({k: v for k, v in file_stamp(path, output_path).items() if k != 'file'})
        files.append(entry)
    return files


def run_record(
    attributes: dict,
    start_time: float,
    output_path: str,
    arrays: dict | None = None,
    state: dict | None = None,
    final: bool = True,
) -> dict:
    """The manifest entry of one PAOFLOW instance.

    Parameters
    ----------
    attributes : dict
        ``DataController.data_attributes``.
    start_time : float
        Start of the run (epoch seconds); files modified since then are listed.
    output_path : str
        The output folder.
    arrays : dict, optional
        ``DataController.data_arrays``, for the structure of :func:`dft_inputs`.
    state : dict, optional
        State of :func:`recorded` (run identifier, method calls, driver script).
    final : bool, optional
        True when the run finished: the output files are hashed and ``finished`` is set.

    Returns
    -------
    dict
    """
    from .. import __version__

    state = state if state is not None else {}
    files = _output_files(
        output_path, start_time, final, tuple(state.get('folders', ())), state.get('shallow', False)
    )
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
    if isinstance(facts.get('savedir'), str) and os.path.isabs(facts['savedir']):
        # no absolute path in the record, as for the arguments of the calls
        cleaned, _ = encode_argument(facts['savedir'])
        facts['savedir'] = cleaned if isinstance(cleaned, str) else cleaned['path_outside_folder']
    revision = source_revision()
    if revision is not None:
        facts = {
            'paoflow_commit': revision['commit'],
            'paoflow_tree_modified': revision['modified'],
            **facts,
        }
    record = {
        'paoflow': __version__,
        'run_id': state.get('run_id') or str(uuid.uuid4()),
        'status': 'finished' if final else 'unfinished',
        'started': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time)),
        'started_utc': _utc(start_time),
    }
    if final:
        record.update(finished=time.strftime('%Y-%m-%dT%H:%M:%S'), finished_utc=_utc())
    record.update(
        energy_reference='electronic energies are relative to the DFT Fermi energy (E_F = 0)',
        **facts,
    )
    if 'inputs' not in state:  # hashed once, when the reader has run
        inputs = dft_inputs(attributes, arrays or {})
        if inputs is None:
            record_inputs = None
        else:
            record_inputs = state['inputs'] = inputs
    else:
        record_inputs = state['inputs']
    if record_inputs is not None:
        record['inputs'] = record_inputs
    for key in ('input_files', 'input_versions'):
        if state.get(key):
            record[key] = state[key]
    if state.get('script'):
        record['script'] = state['script']
    if 'calls' in state:
        record['calls'] = state['calls']
        record['replayable'] = state['replayable']
    if state.get('calls'):
        # the call after which each file was last written: the record is rewritten after every
        # call, so a file that is new or newer than at the previous write belongs to this one
        written = state.setdefault('written', {})
        for entry in files:
            modified = os.path.getmtime(os.path.join(output_path, entry['file']))
            if entry['file'] not in written or modified > written[entry['file']][0]:
                written[entry['file']] = (modified, len(state['calls']) - 1)
            entry['call'] = written[entry['file']][1]
            # The unit of a frequency axis is the ``units`` argument of the call (phonon, IR
            # and Raman writers), not a property of the file name: the catalogue leaves it empty.
            units = state['calls'][entry['call']]['arguments'].get('units')
            for axis in entry.get('axes', []):
                if axis['quantity'] == 'Frequency' and not axis['unit'] and isinstance(units, str):
                    axis['unit'] = units
    record.update(environment=environment(), infrastructure=infrastructure(), files=files)
    return record


def call_text(call: dict, limit: int = 200) -> str:
    """One recorded call as a line of text.

    Parameters
    ----------
    call : dict
        An entry of ``calls`` of a run.
    limit : int, optional
        Longest argument list kept; the rest is cut.

    Returns
    -------
    str
        ``PAOFLOW.bands(ibrav=2, nk=20)``: the arguments that are not None, as JSON values.
    """
    arguments = ', '.join(
        '%s=%s' % (key, json.dumps(value, ensure_ascii=False))
        for key, value in call.get('arguments', {}).items()
        if value is not None
    )
    if len(arguments) > limit:
        arguments = arguments[:limit] + ' ...'
    return '%s(%s)' % (call['method'], arguments)


def axes_text(entry: dict) -> str:
    """Quantities and units of an output file of the record: ``Energy (eV); DOS (states/eV)``.

    Parameters
    ----------
    entry : dict
        An entry of ``files`` of a run.

    Returns
    -------
    str
        Empty for a file without recorded axes.
    """
    return '; '.join(
        '%s (%s)' % (axis['quantity'], axis['unit']) if axis.get('unit') else axis['quantity']
        for axis in entry.get('axes', [])
    )


def formula(atoms: list) -> str:
    """Formula of a cell from its list of chemical symbols: ``['Si', 'Si']`` gives ``Si2``.

    Parameters
    ----------
    atoms : list of str

    Returns
    -------
    str
    """
    counts = {atom: atoms.count(atom) for atom in sorted(set(atoms))}
    return ''.join('%s%s' % (atom, n if n > 1 else '') for atom, n in counts.items())


def run_of_file(root: str, file: str) -> tuple[dict, dict] | None:
    """The run that wrote an output file, from the run record next to it or above it.

    Parameters
    ----------
    root : str
        Folder the search stops at.
    file : str
        The file, relative to ``root``.

    Returns
    -------
    tuple of (dict, dict) or None
        The run entry and the entry of the file in it: the last run that lists the file.  None
        without a record or when no run lists it.
    """
    root = os.path.abspath(root)
    path = os.path.join(root, file)
    folder = os.path.dirname(path)
    while True:
        try:
            with open(os.path.join(folder, MANIFEST), 'r', encoding='utf-8') as handle:
                runs = json.load(handle).get('runs', [])
        except (OSError, ValueError, AttributeError):
            runs = []
        relative = os.path.relpath(path, folder).replace(os.sep, '/')
        for run in reversed(runs):
            for entry in run.get('files', []):
                if entry.get('file') == relative:
                    return run, entry
        if folder == root or os.path.dirname(folder) == folder:
            return None
        folder = os.path.dirname(folder)


def write_output_manifest(
    attributes: dict,
    start_time: float,
    arrays: dict | None = None,
    state: dict | None = None,
    final: bool = True,
) -> str | None:
    """Write the entry of this run to ``<output>/paoflow_outputs.json`` (rank 0 only).

    Parameters
    ----------
    attributes : dict
        ``DataController.data_attributes`` (``opath`` is the output folder).
    start_time : float
        Start of the run (epoch seconds).
    arrays : dict, optional
        ``DataController.data_arrays``.
    state : dict, optional
        State of :func:`recorded`; its ``run_id`` selects the entry to replace.  Without it a new
        entry is appended.
    final : bool, optional
        True when the run finished.

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
    record = run_record(attributes, start_time, output_path, arrays, state, final)
    runs = [run for run in runs if run.get('run_id') != record['run_id']] + [record]
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump({'manifest': VERSION, 'runs': runs}, handle, indent=1)
    os.replace(temporary, path)
    return path


def _script() -> dict[str, Any] | None:
    """Path (relative to the current folder) and hash of the driver script, if there is one."""
    path = sys.argv[0] if sys.argv else ''
    if not path.endswith('.py') or not os.path.isfile(path):
        return None
    stamp = file_stamp(path, os.getcwd())
    if stamp['file'].startswith('..'):  # no path outside the folder in the record
        stamp['file'] = os.path.basename(path)
    return stamp


def _new_state(start: float) -> dict[str, Any]:
    return {
        'run_id': str(uuid.uuid4()),
        'start': start,
        'calls': [],
        'replayable': True,
        'script': _script(),
        'folders': [],
    }


# Files of an input folder worth a checksum: the DFT save, the ph.x and phonopy phonons, EPW.
_INPUT_PATTERNS = (
    '*.xml',
    '*.UPF',
    '*.upf',
    'POTCAR',
    '*.dyn*',
    '*.fc',
    'phonopy*.yaml',
    '*dvscf*',
    '_ph0/*dvscf*',
    '_ph0/*/*dvscf*',
    '*.epb*',
)
_VERSIONS = {
    'Quantum ESPRESSO': r'<creator[^>]*NAME="PWSCF"[^>]*VERSION="([^"]+)"',
    'phonopy': r'^phonopy:\s*\n\s+version:\s*"?([^"\s]+)',
}


def stamp_inputs(paths: list, root: str) -> tuple[list, dict]:
    """Checksums of the input files of a step, and the code versions those files state.

    Parameters
    ----------
    paths : list of str
        Files, or folders (a DFT save, a ``ph.x`` or phonopy run folder): of a folder the files
        of :data:`_INPUT_PATTERNS` are taken.  Missing and empty entries are skipped.
    root : str
        Folder the recorded paths are relative to.

    Returns
    -------
    tuple of (list, dict)
        The :func:`file_stamp` of each file, and ``{code: version}`` for the codes whose version
        is written in one of the files (``data-file-schema.xml``, ``phonopy.yaml``).  No log file
        is parsed.
    """
    files: list[str] = []
    for path in paths:
        if path and os.path.isdir(path):
            for pattern in _INPUT_PATTERNS:
                files += glob.glob(os.path.join(path, pattern))
        elif path and os.path.isfile(path):
            files.append(path)
    files = sorted({os.path.abspath(name) for name in files if os.path.isfile(name)})
    versions: dict[str, str] = {}
    for name in files:
        if name.endswith(('.xml', '.yaml')) and os.path.getsize(name) <= HASH_LIMIT:
            with open(name, 'r', encoding='utf-8', errors='replace') as handle:
                head = handle.read(4096)
            for code, pattern in _VERSIONS.items():
                found = re.search(pattern, head, re.MULTILINE)
                if found is not None:
                    versions.setdefault(code, found.group(1))
    stamps = []
    for name in files:
        stamp = file_stamp(name, root)
        # a driver of a system folder runs one level down (transport/ reads ../qe/phonon);
        # a path that climbs further would spell out folders above the calculation
        if stamp['file'].startswith('../../../'):
            stamp['file'] = os.path.basename(name)
        stamps.append(stamp)
    return stamps, versions


def record_function_call(
    output_path: str, name: str, arguments: dict | None = None, inputs: list | None = None
) -> None:
    """Record a step of a run path that is a plain function, not an orchestrator method.

    For the electron-phonon writers and the ``pyskeaf`` command line.  Call it on the way out,
    after the files are written.

    Parameters
    ----------
    output_path : str
        Folder the step wrote into; its record gets the entry.
    name : str
        Name of the step (``'elphon.write_mobility'``).
    arguments : dict, optional
        The arguments worth keeping, encoded with :func:`encode_argument`.
    inputs : list of str, optional
        Input files or folders of the step (:func:`stamp_inputs`): the record gets their
        checksums under ``input_files`` and the versions they state under ``input_versions``.

    Notes
    -----
    Calls of one process into one folder share one run entry, which lists the files written
    since the process started (top level of the folder only) and is marked finished after
    every call.  Only MPI rank 0 writes.
    A failure prints one line and never raises.
    """
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() != 0 or not os.path.isdir(output_path):
            return
        state = _FUNCTION_RUNS.setdefault(os.path.abspath(output_path), _new_state(_PROCESS_START))
        state['shallow'] = True
        encoded = {}
        for key, value in (arguments or {}).items():
            encoded[key], rebuildable = encode_argument(value)
            state['replayable'] = state['replayable'] and rebuildable
        state['calls'].append({'method': name, 'arguments': encoded, 'status': 'ok'})
        new = [path for path in inputs or () if path and path not in state.setdefault('seen', [])]
        if new:  # each input is hashed once per process
            state['seen'] += new
            stamps, versions = stamp_inputs(new, os.getcwd())
            known = {stamp['file'] for stamp in state.setdefault('input_files', [])}
            state['input_files'] += [stamp for stamp in stamps if stamp['file'] not in known]
            state.setdefault('input_versions', {}).update(versions)
        write_output_manifest({'opath': output_path}, state['start'], None, state, final=True)
    except Exception as exc:  # noqa: BLE001
        print('(run record not written: %s)' % exc)


def _record_call(
    host: Any, method: Callable, args: tuple, kwargs: dict, started: float, ok: bool
) -> None:
    """Append one finished call to the state on the data controller and rewrite the record."""
    import resource

    controller = host.data_controller
    state = getattr(controller, _STATE, None)
    if state is None:
        state = _new_state(getattr(host, 'start_time', None) or started)
        setattr(controller, _STATE, state)
    bound = inspect.signature(method).bind(host, *args, **kwargs)
    bound.apply_defaults()
    arguments = {}
    for name, value in list(bound.arguments.items())[1:]:
        arguments[name], rebuildable = encode_argument(value)
        state['replayable'] = state['replayable'] and rebuildable
    owner = type(host).__name__
    name = owner if method.__name__ == '__init__' else '%s.%s' % (owner, method.__name__)
    to_mb = 1024.0**2 if sys.platform == 'darwin' else 1024.0
    state['calls'].append(
        {
            'method': name,
            'arguments': arguments,
            'seconds': round(time.time() - started, 3),
            'peak_memory_rank0_mb': round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / to_mb, 1
            ),
            'status': 'ok' if ok else 'failed',
        }
    )
    folder = getattr(host, '_output_config', {}).get('output_dir')
    if folder and folder not in state['folders']:
        state['folders'].append(folder)
    attributes, arrays = controller.data_attributes, controller.data_arrays
    final = ok and method.__name__ == 'finish_execution'
    write_output_manifest(attributes, state['start'], arrays, state, final)


def _recording(method: Callable) -> Callable:
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Nested public calls (a method calling another, the constructor's memory check, the
        # sparse engine calling the host) belong to the call the script made.
        if getattr(self, '_recording_call', False):
            return method(self, *args, **kwargs)
        started, ok = time.time(), False
        self._recording_call = True
        try:
            result = method(self, *args, **kwargs)
            ok = True
            return result
        finally:
            self._recording_call = False
            controller = getattr(self, 'data_controller', None)
            if controller is not None and getattr(controller, 'rank', 0) == 0:
                try:  # the record is informational: a failure never stops the run
                    _record_call(self, method, args, kwargs, started, ok)
                except Exception as exc:  # noqa: BLE001
                    print('(run record not written: %s)' % exc)

    return wrapper


def recorded(cls: type) -> type:
    """Class decorator: record every public method call (and the constructor) in the run record.

    Parameters
    ----------
    cls : type
        An orchestrator whose instances hold ``data_controller``.

    Returns
    -------
    type
        The same class.  Each wrapper keeps the name, docstring, signature and attributes of the
        method (``functools.wraps``) and only observes: arguments and results pass unchanged.

    Notes
    -----
    Apply it above ``@sparse_aware`` so that a call is recorded once, before it is routed to the
    dense or the sparse engine.  Only rank 0 records, and only the calls the script makes.
    """
    for name, attribute in list(vars(cls).items()):
        if (name.startswith('_') and name != '__init__') or not inspect.isfunction(attribute):
            continue
        setattr(cls, name, _recording(attribute))
    return cls
