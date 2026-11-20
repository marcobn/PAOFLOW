"""``paoflow-pack`` and ``paoflow-unpack``: a calculation as one file to share, and back.

``paoflow-pack`` runs in the folder of a calculation (the folder the script was run in).  It reads
the run records (``paoflow_outputs.json``) below it, asks the few things only a person knows
(title, authors, licence), and writes ``<folder>-package.zip``: the outputs, the driver scripts,
the DFT inputs and pseudopotentials, ``case.json``, and at the top a ``README.md`` (the summary a
person reads) and ``ro-crate-metadata.json`` (the same for programs: a Workflow Run Crate when a
driver script is there, a Process Run Crate otherwise, with the checksum of every file).
Wavefunctions are not included; their checksums are listed.  ``paoflow-unpack`` checks every
checksum, lays the folder out again and prints the steps to reproduce.

Neither command touches the network.  Design: ``docs/internals/fair-records.md``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote

from ..writers.output_manifest import MANIFEST, axes_text, call_text, formula, sha256_file
from .cases import CASE_FILE, SCRIPTS

CRATE = 'ro-crate-metadata.json'
README = 'README.md'
CASE_README = 'README.case.md'  # the README of the folder itself, next to the generated one
DEFAULT_LICENSE = 'CC-BY-4.0'
# A zip from elsewhere can expand to any size: paoflow-unpack stops above this (--max-size).
MAX_UNPACKED_GB = 20.0
#: Files taken wherever they are in the folder, besides those the run records name.
EXTRA_FILES = (
    CASE_FILE,
    README,
    '*.py',
    '*.in',
    'INCAR',
    'KPOINTS',
    'POSCAR',
    '*.UPF',
    '*.upf',
    'figures.json',
    'g_ReRp_*.json',
    'JOURNAL.md',
    'STATUS.md',
)
PROFILES = {
    'process': ('https://w3id.org/ro/wfrun/process/0.6', 'Process Run Crate', '0.6'),
    'workflow': ('https://w3id.org/ro/wfrun/workflow/0.6', 'Workflow Run Crate', '0.6'),
    'workflow-ro-crate': (
        'https://w3id.org/workflowhub/workflow-ro-crate/1.1',
        'Workflow RO-Crate',
        '1.1',
    ),
}
PYTHON = 'https://www.python.org/'
SOFTWARE = 'https://github.com/marcobn/PAOFLOW#'  # + version: an absolute id (run-crate SHOULD)
STATUS = 'http://schema.org/%sActionStatus'
NO_RECORD = (
    'No run record (%s) in %s or below. Run paoflow-pack in the folder where you ran your '
    'PAOFLOW script.'
)
ORCID = re.compile(r'(?:https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])')


def parse_author(text: str) -> dict[str, str]:
    """An author from one line of text.

    Parameters
    ----------
    text : str
        A name, optionally followed by an ORCID and, after a comma, the affiliation
        (``'Ada Lovelace 0000-0002-1825-0097, University of London'``).

    Returns
    -------
    dict
        ``name`` and, when the text holds them, ``orcid`` and ``affiliation``.
    """
    text, _, affiliation = text.partition(',')
    found = ORCID.search(text)
    author = {'name': ORCID.sub('', text).strip(' ,()<>')}
    if found:
        author['orcid'] = found.group(1)
    if affiliation.strip():
        author['affiliation'] = affiliation.strip()
    return author


def author_text(author: dict[str, str]) -> str:
    """The line :func:`parse_author` reads back: name, ORCID and affiliation of one author."""
    text = ('%s %s' % (author['name'], author.get('orcid', ''))).strip()
    return text + (', ' + author['affiliation'] if author.get('affiliation') else '')


def _walk(root: Path):
    """The files below ``root``, hidden folders and ``__pycache__`` left out."""
    for folder, folders, names in os.walk(root):
        folders[:] = sorted(d for d in folders if not d.startswith('.') and d != '__pycache__')
        for name in sorted(names):
            yield Path(folder) / name


def _find(root: Path, record_folder: Path, name: str) -> Path | None:
    """The file a record names, looked up from the record's folder up to ``root``.

    A record does not know where the packed folder starts: its paths are relative to the folder
    the script ran in, which is the output folder or one of the folders above it.
    """
    folder = record_folder
    while True:
        candidate = Path(os.path.normpath(folder / name))
        if candidate.is_file() and root in candidate.parents:
            return candidate
        if folder == root:
            return None
        folder = folder.parent


def collect(root: Path) -> dict[str, Any]:
    """What the package of a folder holds, from its run records.

    Parameters
    ----------
    root : Path
        The folder of the calculation.

    Returns
    -------
    dict
        ``files``: ``{relative path: entry}`` with ``sha256``, ``bytes``, ``role`` (``output``,
        ``input``, ``script``, ``record`` or ``other``) and, for an output, the ``kind`` and
        ``axes`` of the record and the ``run`` that describes it (the last run that wrote it);
        ``runs``: the run entries in the order they started, each with its ``folder``;
        ``left_out``: files named but not packed (wavefunctions, ``POTCAR``), with checksum;
        ``missing``: input files and scripts the records name that are not in the folder;
        ``changed``: files whose content differs from the checksum of their record.

    Raises
    ------
    FileNotFoundError
        If the folder holds no run record.
    """
    root = root.resolve()
    records = [path for path in _walk(root) if path.name == MANIFEST]
    if not records:
        raise FileNotFoundError(NO_RECORD % (MANIFEST, root))
    files: dict[str, dict[str, Any]] = {}
    left_out: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    changed: set[str] = set()
    runs: list[dict[str, Any]] = []

    def add(path: Path, role: str, recorded: dict | None = None, **facts: Any) -> str:
        relative = path.relative_to(root).as_posix()
        entry = files.setdefault(relative, {'bytes': path.stat().st_size, 'role': role})
        if 'sha256' not in entry:
            entry['sha256'] = sha256_file(str(path))
        if recorded is not None:  # the last run that names a file decides
            if recorded.get('sha256') not in (None, entry['sha256']):
                changed.add(relative)
            else:
                changed.discard(relative)
        entry.update(facts)
        return relative

    for record in records:
        add(record, 'record')
        try:
            entries = json.loads(record.read_text(encoding='utf-8')).get('runs', [])
        except (ValueError, AttributeError):
            entries = []
        runs += [{**run, 'folder': record.parent} for run in entries if isinstance(run, dict)]
    runs.sort(key=lambda run: run.get('started_utc', ''))
    save_folders: set[Path] = set()
    for run in runs:
        folder = run['folder']
        run['results'], run['objects'], run['scripts'] = [], [], []
        for entry in run.get('files', []):
            path = Path(os.path.normpath(folder / entry['file']))
            if not path.is_file() or root not in path.parents:
                continue
            facts = {key: entry[key] for key in ('kind', 'axes', 'labels') if key in entry}
            run['results'].append(add(path, 'output', entry, run=run['run_id'], **facts))
        named = list((run.get('inputs') or {}).get('files', [])) + list(run.get('input_files', []))
        for role, entry in [('input', e) for e in named] + [('script', run.get('script'))]:
            if not entry:
                continue
            path = _find(root, folder, entry['file'])
            if path is None:
                if entry['file'] not in missing:
                    missing.append(entry['file'])
            elif path.name == 'POTCAR':  # the VASP licence does not allow sharing it
                relative = path.relative_to(root).as_posix()
                left_out[relative] = {'bytes': path.stat().st_size, 'why': 'VASP licence'}
            else:
                run['objects' if role == 'input' else 'scripts'].append(add(path, role, entry))
                if role == 'input' and path.suffix == '.xml':
                    save_folders.add(path.parent)
    # the last run that lists a file describes it: earlier runs lose it from their results
    for run in runs:
        run['results'] = [
            name for name in run['results'] if files[name].get('run') == run['run_id']
        ]
    for path in _walk(root):
        relative = path.relative_to(root).as_posix()
        if relative not in files and any(fnmatch.fnmatch(path.name, p) for p in EXTRA_FILES):
            add(path, 'script' if path.suffix == '.py' else 'other')
    for folder in sorted(save_folders):  # wavefunctions and charge density of the DFT saves
        for path in sorted(folder.iterdir()):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and relative not in files:
                left_out[relative] = {'bytes': path.stat().st_size, 'why': 'large DFT data'}
    for relative, entry in left_out.items():
        entry['sha256'] = sha256_file(str(root / relative))
    return {
        'files': files,
        'runs': runs,
        'left_out': left_out,
        'missing': missing,
        'changed': sorted(changed),
    }


def _material(runs: list[dict]) -> dict | None:
    """The ``inputs`` of the last run that has a DFT calculation behind it."""
    for run in reversed(runs):
        if run.get('inputs'):
            return run['inputs']
    return None


def update_case(root: Path, content: dict, answers: dict[str, Any]) -> dict[str, Any]:
    """Store the answers in ``case.json``, creating it for a folder that has none.

    Parameters
    ----------
    root : Path
        The folder of the calculation.
    content : dict
        Result of :func:`collect`.
    answers : dict
        ``title``, ``authors`` (list of dict), ``license``, ``machine``; empty values are not
        written.

    Returns
    -------
    dict
        The content of ``case.json``.  A new one is a ``scripts`` case (the layout of
        the example cases) whose steps are the driver scripts of the runs, in the order they ran.
    """
    path = root / CASE_FILE
    case: dict[str, Any] = {}
    if path.is_file():
        case = json.loads(path.read_text(encoding='utf-8'))
    else:
        inputs = _material(content['runs'])
        kinds = sorted({e['kind'] for e in content['files'].values() if e.get('kind')})
        material = formula(inputs['atoms']) if inputs and inputs.get('atoms') else 'Model'
        steps, seen = [], set()
        for run in content['runs']:
            for script in run.get('scripts', []):
                if script not in seen:
                    seen.add(script)
                    posix = PurePosixPath(script)
                    steps.append(
                        {
                            'label': 'Run %s' % posix.name,
                            'command': ['python', posix.name],
                            'cwd': posix.parent.as_posix(),
                            'mpi': True,
                        }
                    )
        case = {
            'title': root.name,
            'summary': '%s: %s.' % (material, ', '.join(kinds) or 'PAOFLOW calculation'),
            'workflow': SCRIPTS,
            'steps': steps,
        }
    case.update({key: value for key, value in answers.items() if value})
    path.write_text(json.dumps(case, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return case


def _size(size: float) -> str:
    for unit in ('bytes', 'kB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return '%d %s' % (size, unit) if unit == 'bytes' else '%.1f %s' % (size, unit)
        size /= 1024
    return ''


def readme_text(case: dict, content: dict) -> str:
    """The ``README.md`` of the package: the calculation for a person to read.

    Parameters
    ----------
    case : dict
        Content of ``case.json``.
    content : dict
        Result of :func:`collect`.

    Returns
    -------
    str
        Markdown: material and structure, DFT settings, PAOFLOW version, the steps in order, a
        table of every file with its quantity and units, and what is not included.
    """
    from ..results.vocabulary import TERMS

    runs = [run for run in content['runs'] if run['results']] or content['runs'][-1:]
    last = runs[-1]
    lines = ['# %s' % case.get('title', ''), '', case.get('summary', ''), '']
    authors = ', '.join(
        a['name']
        + (' ([%s](https://orcid.org/%s))' % (a['orcid'], a['orcid']) if 'orcid' in a else '')
        + (', %s' % a['affiliation'] if a.get('affiliation') else '')
        for a in case.get('authors', [])
    )
    lines += ['- Authors: %s' % (authors or 'not given')]
    lines += ['- Licence: %s' % case.get('license', 'not given')]
    lines += ['- Packaged on %s with PAOFLOW %s' % (time.strftime('%Y-%m-%d'), last['paoflow'])]
    inputs = _material(content['runs'])
    lines += ['', '## Material', '']
    if inputs is None:
        lines += ['A model Hamiltonian: no DFT calculation behind it.']
    else:
        if inputs.get('atoms'):
            lines += [
                '- Formula of the cell: %s (%d atoms)'
                % (formula(inputs['atoms']), len(inputs['atoms']))
            ]
        lines += ['- Lattice vectors (bohr):', '']
        lines += ['      % .6f  % .6f  % .6f' % tuple(vector) for vector in inputs['cell_bohr']]
        if inputs.get('positions_bohr'):
            lines += ['', '- Atomic positions (bohr):', '']
            lines += [
                '      %-3s % .6f  % .6f  % .6f' % (atom, *position)
                for atom, position in zip(inputs['atoms'], inputs['positions_bohr'])
            ]
        lines += ['', '## DFT calculation', '']
        lines += [
            '- Code: %s %s' % (inputs['engine'], inputs.get('version', '(version not recorded)'))
        ]
        lines += [
            '- Exchange-correlation functional: %s' % inputs.get('functional', 'not recorded')
        ]
        for species in inputs.get('species', []):
            lines += [
                '- Pseudopotential of %s: %s' % (species['element'], species['pseudopotential'])
            ]
        if 'nk1' in last:
            lines += ['- k grid: %s x %s x %s' % (last['nk1'], last['nk2'], last['nk3'])]
        if 'fermi_energy_dft_ev' in last:
            lines += [
                '- Fermi energy: %.6f eV (the output energies are relative to it)'
                % last['fermi_energy_dft_ev']
            ]
    versions = {k: v for run in content['runs'] for k, v in run.get('input_versions', {}).items()}
    lines += ['- %s version: %s' % item for item in sorted(versions.items())]
    environment, machine = last.get('environment', {}), last.get('infrastructure', {})
    commit = last.get('paoflow_commit')
    lines += ['', '## PAOFLOW', '']
    lines += [
        '- Version %s%s%s'
        % (
            last['paoflow'],
            ', commit %s' % commit[:12] if commit else '',
            ' with local changes' if last.get('paoflow_tree_modified') else '',
        )
    ]
    lines += [
        '- Python %s, numpy %s, scipy %s, mpi4py %s'
        % tuple(environment.get(key) for key in ('python', 'numpy', 'scipy', 'mpi4py'))
    ]
    lines += [
        '- Run on %s MPI rank(s), %s thread(s) each; %s; BLAS: %s'
        % (
            machine.get('mpi_ranks'),
            machine.get('threads_per_rank'),
            machine.get('cpu_model') or 'CPU not recorded',
            machine.get('blas') or 'not recorded',
        )
    ]
    if case.get('machine'):
        lines += ['- Machine: %s' % case['machine']]
    lines += ['', '## Steps', '']
    for run in runs:
        script = ', script `%s`' % run['scripts'][0] if run.get('scripts') else ''
        state = '' if run['status'] == 'finished' else ' (**this run did not finish**)'
        lines += ['Run started %s%s%s:' % (run['started_utc'], script, state), '']
        lines += [
            '%d. `%s`%s'
            % (n, call_text(call), '' if call.get('status') == 'ok' else '  **(failed)**')
            for n, call in enumerate(run.get('calls', []), 1)
        ]
        lines += ['']
    lines += [
        '## Files',
        '',
        '| File | What it is | Quantities and units | Size |',
        '|---|---|---|---|',
    ]
    what = {'record': 'run record', 'script': 'script', 'input': 'input of the calculation'}
    for name, entry in sorted(content['files'].items()):
        kind = '`%s`' % entry['kind'] if entry.get('kind') else what.get(entry['role'], '')
        shown = CASE_README if name == README else name
        lines += [
            '| `%s` | %s | %s | %s |' % (shown, kind, axes_text(entry), _size(entry['bytes']))
        ]
    kinds = sorted({e['kind'] for e in content['files'].values() if e.get('kind') in TERMS})
    if kinds:
        lines += ['', 'What the names in the second column mean:', '']
        lines += ['- `%s`: %s' % (kind, TERMS[kind][0]) for kind in kinds]
    if content['left_out']:
        lines += ['', '## Not included', '']
        lines += [
            'These files of the calculation are not in the package. The checksums (sha256) identify them.',
            '',
        ]
        lines += ['| File | Why | Size | sha256 |', '|---|---|---|---|']
        lines += [
            '| `%s` | %s | %s | `%s` |' % (name, e['why'], _size(e['bytes']), e['sha256'])
            for name, e in sorted(content['left_out'].items())
        ]
    lines += [
        '',
        '## Reproducing',
        '',
        '`paoflow-unpack <this file>` checks every file and prints the steps in order. The '
        'wavefunctions are not included, so the DFT steps run first.',
        '',
        'The run records (`%s`) hold the complete arguments of every step; '
        '`%s` describes the package for programs.' % (MANIFEST, CRATE),
        '',
    ]
    return '\n'.join(lines)


def crate(case: dict, content: dict, readme_sha256: str, readme_bytes: int) -> dict[str, Any]:
    """``ro-crate-metadata.json`` of the package.

    Parameters
    ----------
    case : dict
        Content of ``case.json``.
    content : dict
        Result of :func:`collect`.
    readme_sha256 : str
        Checksum of the generated ``README.md``.
    readme_bytes : int
        Its size.

    Returns
    -------
    dict
        An RO-Crate 1.3 graph: a Workflow Run Crate (profile 0.6) when a run has its driver
        script in the package, a Process Run Crate otherwise.  One ``CreateAction`` per run with
        results, every file with ``sha256`` and ``contentSize``, each PAOFLOW term as a
        ``DefinedTerm`` with ``sameAs`` links to the shared vocabularies.  The method calls stay
        in the run records.
    """
    from ..results.vocabulary import STRUCTURE, TERMS

    def ref(identifier: str) -> dict[str, str]:
        return {'@id': identifier}

    def utc(stamp: str) -> str:  # the run-crate profiles want an offset, not 'Z'
        return stamp[:-1] + '+00:00' if stamp.endswith('Z') else stamp

    def file_id(name: str) -> str:
        return quote(CASE_README if name == README else name)

    files, runs = content['files'], [run for run in content['runs'] if run['results']]
    workflows = [s for run in runs for s in run.get('scripts', [])]
    graph: list[dict[str, Any]] = []
    people = []
    for index, author in enumerate(case.get('authors', [])):
        identifier = (
            'https://orcid.org/' + author['orcid']
            if 'orcid' in author
            else '#author-%d' % (index + 1)
        )
        people.append(ref(identifier))
        person = {'@id': identifier, '@type': 'Person', 'name': author['name']}
        if author.get('affiliation'):
            names = [a['affiliation'] for a in case['authors'] if a.get('affiliation')]
            organisation = '#organisation-%d' % (names.index(author['affiliation']) + 1)
            person['affiliation'] = ref(organisation)
            if not any(entity['@id'] == organisation for entity in graph):
                graph.append(
                    {'@id': organisation, '@type': 'Organization', 'name': author['affiliation']}
                )
        graph.append(person)
    profiles = ['process'] + (['workflow', 'workflow-ro-crate'] if workflows else [])
    licence = case.get('license') or ''
    root: dict[str, Any] = {
        '@id': './',
        '@type': 'Dataset',
        'name': case.get('title', ''),
        'description': case.get('summary', ''),
        'datePublished': time.strftime('%Y-%m-%d', time.gmtime()),
        'conformsTo': [ref(PROFILES[key][0]) for key in profiles],
        'hasPart': [ref(README)] + [ref(file_id(name)) for name in sorted(files)],
        'mentions': [ref('#run-' + run['run_id']) for run in runs],
    }
    if licence:
        spdx = re.fullmatch(r'[A-Za-z0-9.+-]+', licence)
        root['license'] = ref('https://spdx.org/licenses/' + licence) if spdx else licence
        if spdx:
            graph.append(
                {
                    '@id': 'https://spdx.org/licenses/' + licence,
                    '@type': 'CreativeWork',
                    'name': licence,
                    'description': 'Licence of the files of this package (SPDX identifier %s).'
                    % licence,
                }
            )
    if people:
        root['author'] = people
    if case.get('tags'):
        root['keywords'] = ', '.join(case['tags'])
    if workflows:
        root['mainEntity'] = ref(file_id(workflows[-1]))
    if _material(content['runs']):
        root['about'] = [ref('#structure-' + key) for key in STRUCTURE]
    graph = [
        {
            '@id': CRATE,
            '@type': 'CreativeWork',
            'conformsTo': ref('https://w3id.org/ro/crate/1.3'),
            'about': ref('./'),
        },
        root,
        *graph,
    ]
    for key in profiles:
        identifier, name, version = PROFILES[key]
        graph.append(
            {
                '@id': identifier,
                '@type': ['CreativeWork', 'Profile'],  # RO-Crate 1.3: conformsTo names Profiles
                'name': name,
                'version': version,
            }
        )
    software = {}
    for run in runs:
        version = run['paoflow']
        software[version] = {
            '@id': SOFTWARE + version,
            '@type': 'SoftwareApplication',
            'name': 'PAOFLOW',
            'url': 'https://github.com/marcobn/PAOFLOW',
            'version': version,
        }
        if run.get('paoflow_commit'):
            software[version]['identifier'] = run['paoflow_commit']
    graph += software.values()
    if workflows:
        graph.append(
            {
                '@id': PYTHON,
                '@type': 'ComputerLanguage',
                'name': 'Python',
                'url': PYTHON,
                # RO-Crate 1.3 requires a version; a version-1 record states none
                'version': runs[-1].get('environment', {}).get('python') or '3',
            }
        )
    for run in runs:
        scripts = run.get('scripts', [])
        action: dict[str, Any] = {
            '@id': '#run-' + run['run_id'],
            '@type': 'CreateAction',
            'name': 'PAOFLOW run started %s' % run['started_utc'],
            'description': 'PAOFLOW %s run with %d calls that wrote %d files.'
            % (run['paoflow'], len(run.get('calls', [])), len(run['results'])),
            'startTime': utc(run['started_utc']),
            'instrument': ref(file_id(scripts[0]) if scripts else SOFTWARE + run['paoflow']),
            'result': [ref(file_id(name)) for name in run['results']],
        }
        if run.get('objects'):
            action['object'] = [ref(file_id(name)) for name in run['objects']]
        if run['status'] == 'finished':
            action['endTime'] = utc(run['finished_utc'])
        status = STATUS % ('Completed' if run['status'] == 'finished' else 'Failed')
        action['actionStatus'] = ref(status)
        if not any(entity['@id'] == status for entity in graph):
            name = status.rsplit('/', 1)[1]
            graph.append({'@id': status, '@type': 'ActionStatusType', 'name': name})
        if run['status'] != 'finished':
            action['error'] = 'The run did not reach finish_execution().'
        if people:
            action['agent'] = people[0]
        graph.append(action)
        for script in scripts:
            files[script]['workflow'] = SOFTWARE + run['paoflow']
    graph.append(
        {
            '@id': README,
            '@type': 'File',
            'name': README,
            'about': ref('./'),
            'description': 'Summary of the package for people: material, steps and files.',
            'encodingFormat': 'text/markdown',
            'contentSize': str(readme_bytes),
            'sha256': readme_sha256,
        }
    )
    formats = {'.md': 'text/markdown', '.json': 'application/json', '.py': 'text/x-python'}
    formats.update({'.xml': 'application/xml', '.npz': 'application/octet-stream'})
    formats.update(dict.fromkeys(('.dat', '.in', '.txt', '.upf', '.bxsf', '.xsf'), 'text/plain'))
    roles = {'output': 'Output', 'input': 'Input', 'script': 'Driver script'}
    kinds = set()
    for name, entry in sorted(files.items()):
        entity: dict[str, Any] = {
            '@id': file_id(name),
            '@type': 'File',
            'name': PurePosixPath(name).name,
            'contentSize': str(entry['bytes']),
            'sha256': entry['sha256'],
        }
        # shortcut: an unknown extension is declared binary, extend `formats` when one matters
        suffix = PurePosixPath(name).suffix.lower()
        entity['encodingFormat'] = formats.get(suffix, 'application/octet-stream')
        entity['description'] = '%s file of the calculation.' % roles.get(entry['role'], 'A')
        if 'workflow' in entry:
            entity['@type'] = ['File', 'SoftwareSourceCode', 'ComputationalWorkflow']
            entity['programmingLanguage'] = ref(PYTHON)
            entity['softwareRequirements'] = ref(entry['workflow'])
        if entry.get('kind') in TERMS:
            kinds.add(entry['kind'])
            entity['about'] = ref('#term-' + entry['kind'])
            entity['description'] = TERMS[entry['kind']][0]
        if entry.get('axes'):
            entity['description'] = axes_text(entry)
        graph.append(entity)
    terms = [(('#term-' + kind, kind) + TERMS[kind]) for kind in sorted(kinds)]
    if 'about' in root:
        terms += [('#structure-' + key, key) + STRUCTURE[key] for key in STRUCTURE]
    if terms:
        graph.append(
            {
                '@id': '#paoflow-terms',
                '@type': 'DefinedTermSet',
                'name': 'PAOFLOW terms',
                'description': 'Names of the output types of PAOFLOW and of the structure in the run record.',
            }
        )
    for identifier, code, definition, links in terms:
        term = {
            '@id': identifier,
            '@type': 'DefinedTerm',
            'name': code,
            'termCode': code,
            'description': definition,
            'inDefinedTermSet': ref('#paoflow-terms'),
        }
        if links:
            term['sameAs'] = list(links)
        graph.append(term)
    for entity in graph:  # RO-Crate 1.3: a single value is written bare, not as a one-item list
        for key, value in entity.items():
            if key != '@type' and isinstance(value, list) and len(value) == 1:
                entity[key] = value[0]
    return {
        '@context': [
            'https://w3id.org/ro/crate/1.3/context',
            'https://w3id.org/ro/terms/workflow-run/context',
        ],
        '@graph': graph,
    }


def checklist(root: Path, case: dict, content: dict) -> list[tuple[bool, str]]:
    """What a reader of the package will look for, in ordinary words.

    Parameters
    ----------
    root : Path
        The folder of the calculation.
    case : dict
        Content of ``case.json``.
    content : dict
        Result of :func:`collect`.

    Returns
    -------
    list of (bool, str)
        One line per item, True when it is in order.  Nothing here stops the package.
    """
    files, runs = content['files'], content['runs']
    outputs = [name for name, entry in files.items() if entry['role'] == 'output']
    items = [
        (
            bool(outputs),
            'Run record found: %d run(s), %d output file(s)' % (len(runs), len(outputs)),
        )
    ]
    unfinished = [run for run in runs if run['status'] != 'finished' and run['results']]
    if unfinished:
        items.append(
            (
                False,
                '%d run(s) did not finish (killed, crashed, or no finish_execution())'
                % len(unfinished),
            )
        )
    else:
        items.append((True, 'All runs finished'))
    inputs = _material(runs)
    named = [name for name, entry in files.items() if entry['role'] == 'input']
    if inputs is None and not named:
        items.append((True, 'No DFT calculation behind these runs (model Hamiltonian)'))
    elif content['missing']:
        items.append(
            (False, 'Input files missing from this folder: %s' % ', '.join(content['missing']))
        )
    else:
        items.append((True, 'DFT inputs found (%d files)' % len(named)))
    if inputs is not None and inputs.get('engine') == 'Quantum ESPRESSO':
        qe = sorted(name for name in files if name.endswith('.in'))
        items.append(
            (bool(qe), 'Quantum ESPRESSO input files found: %s' % ', '.join(qe))
            if qe
            else (
                False,
                'No Quantum ESPRESSO input files (*.in) here: add them so others can redo the DFT steps',
            )
        )
    scripts = sorted({s for run in runs for s in run.get('scripts', [])})
    if scripts:
        items.append((True, 'Driver script found: %s' % ', '.join(scripts)))
    elif any(run.get('script') for run in runs):
        items.append(
            (False, 'Driver script missing: run paoflow-pack in the folder where you ran it')
        )
    if content['changed']:
        items.append(
            (False, 'Changed since the run that recorded them: %s' % ', '.join(content['changed']))
        )
    else:
        items.append((True, 'Files unchanged since their runs'))
    items.append(
        (
            bool(case.get('title')),
            'Title: %s' % case['title'] if case.get('title') else 'Title missing',
        )
    )
    authors = ', '.join(author['name'] for author in case.get('authors', []))
    items.append(
        (True, 'Authors: %s' % authors)
        if authors
        else (False, 'Authors missing: answer the question, or pass --author "Name ORCID"')
    )
    items.append((bool(case.get('license')), 'Licence: %s' % (case.get('license') or 'missing')))
    if content['left_out']:
        size = sum(entry['bytes'] for entry in content['left_out'].values())
        items.append(
            (
                True,
                'Not included, listed by checksum in README.md: %d file(s), %s (wavefunctions, POTCAR)'
                % (len(content['left_out']), _size(size)),
            )
        )
    return items


def pack(
    folder: str | os.PathLike = '.',
    output: str | os.PathLike | None = None,
    title: str | None = None,
    authors: list[dict[str, str]] | None = None,
    license: str | None = None,
    machine: str | None = None,
) -> dict[str, Any]:
    """Write the package of a calculation.

    Parameters
    ----------
    folder : path-like
        The folder of the calculation: the one the script was run in.
    output : path-like, optional
        The zip to write (default ``<folder>/<folder name>-package.zip``).
    title, license, machine : str, optional
        Stored in ``case.json`` when given; the stored values are used otherwise.
    authors : list of dict, optional
        ``{'name': ..., 'orcid': ...}`` per author (:func:`parse_author`).

    Returns
    -------
    dict
        ``path`` of the zip, ``checklist`` (:func:`checklist`), ``files`` (number packed).

    Raises
    ------
    FileNotFoundError
        If the folder holds no run record.

    Notes
    -----
    The folder gains or updates ``case.json``; nothing else in it changes.  ``README.md`` and
    ``ro-crate-metadata.json`` exist only inside the zip; a ``README.md`` of the folder is packed
    as ``README.case.md``.
    """
    root = Path(folder).resolve()
    content = collect(root)
    answers = {'title': title, 'authors': authors, 'license': license, 'machine': machine}
    case = update_case(root, content, answers)
    content['files'][CASE_FILE] = {  # written just now: part of the package
        'bytes': (root / CASE_FILE).stat().st_size,
        'role': 'other',
        'sha256': sha256_file(str(root / CASE_FILE)),
    }
    text = readme_text(case, content).encode('utf-8')
    metadata = crate(case, content, hashlib.sha256(text).hexdigest(), len(text))
    path = Path(output) if output else root / ('%s-package.zip' % root.name)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(README, text)
        archive.writestr(CRATE, json.dumps(metadata, indent=1, ensure_ascii=False))
        for name in sorted(content['files']):
            archive.write(root / name, CASE_README if name == README else name)
    return {
        'path': path,
        'checklist': checklist(root, case, content),
        'files': len(content['files']) + 2,
    }


def main_pack(argv: list[str] | None = None) -> int:
    """Command line of ``paoflow-pack``."""
    from .paoflow_driver import ask

    parser = argparse.ArgumentParser(
        prog='paoflow-pack',
        description='Package a PAOFLOW calculation for sharing: one zip with the results, the '
        'scripts, the inputs and a description. Run it in the folder where you ran your script.',
    )
    parser.add_argument(
        'folder', nargs='?', default='.', help='folder of the calculation (default: here)'
    )
    parser.add_argument('-o', '--output', help='zip file to write (default: <folder>-package.zip)')
    parser.add_argument('--title', help='title of the calculation')
    parser.add_argument(
        '--author',
        action='append',
        metavar='"NAME [ORCID][, AFFILIATION]"',
        help='an author; repeat for more',
    )
    parser.add_argument('--license', help='licence of the data (default %s)' % DEFAULT_LICENSE)
    parser.add_argument('--machine', help='name of the machine the runs were made on (optional)')
    parser.add_argument(
        '--no-questions',
        action='store_true',
        help='ask nothing: use the options and what case.json holds',
    )
    args = parser.parse_args(argv)
    root = Path(args.folder).resolve()
    if not any(path.name == MANIFEST for path in _walk(root)):
        print(NO_RECORD % (MANIFEST, root), file=sys.stderr)
        return 1
    stored: dict[str, Any] = {}
    if (root / CASE_FILE).is_file():
        stored = json.loads((root / CASE_FILE).read_text(encoding='utf-8'))
    title, licence, machine = args.title, args.license, args.machine
    authors = [parse_author(text) for text in args.author or []]
    if not args.no_questions:
        print(
            'A few questions for the description of the package (Enter keeps the value in brackets).'
        )
        if title is None:
            title = ask('Title', stored.get('title') or root.name)
        if not authors:
            known = '; '.join(author_text(a) for a in stored.get('authors', []))
            answer = ask(
                'Authors, separated by ";" (after a name: an ORCID, then "," and the affiliation)',
                known or None,
            )
            authors = [parse_author(text) for text in answer.split(';') if text.strip()]
        if licence is None:
            licence = ask('Licence', stored.get('license') or DEFAULT_LICENSE)
        if machine is None:
            machine = ask('Name of the machine (leave empty to omit)', stored.get('machine'))
    if licence is None and not stored.get('license'):
        licence = DEFAULT_LICENSE
    print('Computing checksums and writing the package ...')
    result = pack(root, args.output, title, authors or None, licence, machine)
    print()
    for fine, line in result['checklist']:
        print('  [%s] %s' % ('x' if fine else ' ', line))
    open_items = sum(1 for fine, _ in result['checklist'] if not fine)
    print()
    print(
        'Package: %s (%s, %d files)'
        % (result['path'], _size(result['path'].stat().st_size), result['files'])
    )
    print(
        'Checklist: %s'
        % ('complete' if not open_items else '%d item(s) to look at, marked [ ] above' % open_items)
    )
    print('Upload this file to a repository such as Zenodo to get a DOI.')
    return 0


def unpack(
    archive: str | os.PathLike,
    folder: str | os.PathLike | None = None,
    max_gb: float | None = MAX_UNPACKED_GB,
) -> dict[str, Any]:
    """Extract a package and check every file against its checksum.

    Parameters
    ----------
    archive : path-like
        A zip written by :func:`pack`.
    folder : path-like, optional
        Where to lay the calculation out (default: the name of the zip without
        ``-package.zip``, in the current folder).  It must not exist or be empty.
    max_gb : float, optional
        Largest unpacked size accepted, in GB; None for no limit.  Checked on the sizes the zip
        declares, before anything is written.

    Returns
    -------
    dict
        ``folder``, ``files`` (number checked), ``bad`` (files whose checksum differs or that
        are missing), ``steps`` (lines of text: the steps to reproduce, in order).

    Raises
    ------
    FileExistsError
        If the target folder holds files.
    ValueError
        If the zip is not a package (no ``ro-crate-metadata.json``), holds a path that leaves
        the target folder, or unpacks to more than ``max_gb``.
    """
    archive = Path(archive)
    name = archive.name
    for ending in ('-package.zip', '.zip'):
        if name.endswith(ending):
            name = name[: -len(ending)]
            break
    target = Path(folder) if folder else Path.cwd() / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('%s exists and is not empty: choose another folder with -o' % target)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if CRATE not in names:
            raise ValueError('%s is not a PAOFLOW package (no %s inside)' % (archive, CRATE))
        for member in names:  # the zip comes from elsewhere: no path may leave the folder
            parts = PurePosixPath(member).parts
            if member.startswith(('/', '\\')) or '..' in parts or ':' in parts[0]:
                raise ValueError('%s holds a path outside the package: %s' % (archive, member))
        size = sum(info.file_size for info in bundle.infolist())
        if max_gb is not None and size > max_gb * 1024**3:
            raise ValueError(
                '%s unpacks to %s, more than the limit of %g GB. If you expect a package this '
                'large, run again with --max-size %d (in GB), or --max-size 0 for no limit.'
                % (archive, _size(size), max_gb, size // 1024**3 + 1)
            )
        target.mkdir(parents=True, exist_ok=True)
        bundle.extractall(target)
    metadata = json.loads((target / CRATE).read_text(encoding='utf-8'))
    bad, checked = [], 0
    for entity in metadata.get('@graph', []):
        types = entity.get('@type', [])
        if 'File' not in ([types] if isinstance(types, str) else types) or 'sha256' not in entity:
            continue
        path = target / unquote(entity['@id'])
        checked += 1
        if not path.is_file() or sha256_file(str(path)) != entity['sha256']:
            bad.append(unquote(entity['@id']))
    return {'folder': target, 'files': checked, 'bad': bad, 'steps': _steps(target)}


def _steps(folder: Path) -> list[str]:
    """The steps to reproduce an unpacked calculation, from its ``case.json``."""
    from .cases import case_plan, load_case

    lines = []
    try:
        case = load_case(folder)
        for step in case_plan(case):
            where = '' if step.cwd in ('.', '') else '  (in %s/)' % step.cwd
            command = ' '.join(step.argv) if step.argv else step.note
            lines.append('%s: %s%s' % (step.label, command, where))
    except (OSError, ValueError, KeyError):
        pass
    if not lines:
        lines = ['Run the scripts listed under "Steps" in README.md']
    return lines


def main_unpack(argv: list[str] | None = None) -> int:
    """Command line of ``paoflow-unpack``."""
    parser = argparse.ArgumentParser(
        prog='paoflow-unpack',
        description='Open a package made by paoflow-pack: check every file, lay the calculation '
        'out as a folder and print the steps to reproduce it.',
    )
    parser.add_argument('package', help='the zip file')
    parser.add_argument('-o', '--output', help='folder to create (default: the name of the zip)')
    parser.add_argument(
        '--max-size',
        type=float,
        default=MAX_UNPACKED_GB,
        metavar='GB',
        help='largest unpacked size accepted (default %g GB; 0: no limit)' % MAX_UNPACKED_GB,
    )
    args = parser.parse_args(argv)
    try:
        result = unpack(args.package, args.output, args.max_size or None)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(exc, file=sys.stderr)
        return 1
    folder = result['folder']
    if result['bad']:
        print(
            '%d file(s) differ from the checksum of the package or are missing:'
            % len(result['bad'])
        )
        for name in result['bad']:
            print('  ' + name)
        print('The package was changed or damaged after it was made. Unpacked in %s.' % folder)
        return 1
    print('%d files checked: all match the checksums of the package.' % result['files'])
    print('Unpacked in %s. Start with README.md.' % folder)
    print('The scripts in this package were written by its author. Read them before you run them.')
    print()
    print('To reproduce, in this order:')
    has_dft = any(folder.rglob('*.in')) or any(folder.rglob('data-file-schema.xml'))
    if has_dft:
        print(
            '  0. Run the DFT steps first: the wavefunctions are not in the package (README.md lists them).'
        )
    for number, line in enumerate(result['steps'], 1):
        print('  %d. %s' % (number, line))
    print()
    print('To look at the results: paoflow-studio --root %s' % folder)
    return 0
