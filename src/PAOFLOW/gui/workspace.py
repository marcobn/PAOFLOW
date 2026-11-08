"""Workspace endpoints: session facts, the sandboxed file browser and the QE artefact detection."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .server import HttpError, Request, Response, cpu_budget, environment_features, json_response

if TYPE_CHECKING:
    from .server import GuiApp

#: Largest file the text preview returns.
MAX_PREVIEW_BYTES = 2 * 1024 * 1024
HIDDEN = {'__pycache__', '.git', '.paoflow-studio', '.DS_Store', '.ipynb_checkpoints'}


def _is_pw_input(path: Path) -> bool:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            head = handle.read(4096).lower()
    except OSError:
        return False
    return '&control' in head or ('&system' in head and 'atomic_species' in head)


def entry_tags(path: Path) -> list[str]:
    """Tags of a folder entry shown by the file browser.

    Parameters
    ----------
    path : Path
        File or folder.

    Returns
    -------
    list of str
        Any of ``qe-save``, ``upf``, ``pw-input``, ``driver``, ``elphon-system``,
        ``paoflow-output``, ``dynamical-matrices``.
    """
    name = path.name
    tags = []
    if path.is_dir():
        if name.endswith('.save') and (path / 'data-file-schema.xml').is_file():
            tags.append('qe-save')
        if (path / 'system.py').is_file():
            tags.append('elphon-system')
        if name == 'output' or (path / 'output').is_dir():
            tags.append('paoflow-output')
        return tags
    lower = name.lower()
    if lower.endswith('.upf'):
        tags.append('upf')
    elif lower.endswith('.in') and _is_pw_input(path):
        tags.append('pw-input')
    elif (
        name.startswith('main')
        and name.endswith('.py')
        or name.startswith('plot')
        and name.endswith('.py')
    ):
        tags.append('driver')
    elif lower.endswith('.dyn0'):
        tags.append('dynamical-matrices')
    return tags


def list_directory(app: 'GuiApp', relative: str | None) -> dict:
    """Entries of a folder inside the root or a read-only folder (hidden and cache folders skipped).

    Parameters
    ----------
    app : GuiApp
        The application (sandbox).
    relative : str or None
        Folder relative to the root, or an absolute path inside a read-only folder.

    Returns
    -------
    dict
        ``path`` (relative to the root; absolute in a read-only folder), ``absolute``, ``parent``
        (``None`` at the top of the root or of the read-only folder), ``place`` (``path``, ``label``
        and ``writable`` of that top folder) and ``entries`` (name, kind, size, modification time,
        tags).
    """
    folder = app.resolve_readable(relative)
    if not folder.is_dir():
        raise HttpError(404, '%s is not a folder' % relative)
    entries = []
    try:
        children = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as exc:
        raise HttpError(403, str(exc)) from None
    for child in children:
        if child.name in HIDDEN or child.name.startswith('.'):
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        if not app.sandbox.readable(child):
            continue
        entries.append(
            {
                'name': child.name,
                'kind': 'dir' if child.is_dir() else 'file',
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'tags': entry_tags(child),
            }
        )
    place = app.sandbox.place(folder)
    path = app.sandbox.relative(folder)
    parent = None if folder == place else app.sandbox.relative(folder.parent)
    return {
        'path': path,
        'absolute': str(folder),
        'parent': parent,
        'place': {
            'path': app.sandbox.relative(place),
            'label': 'root' if place == app.sandbox.root else str(place),
            'writable': place == app.sandbox.root,
        },
        'entries': entries,
    }


def detect(app: 'GuiApp', relative: str | None) -> dict:
    """What a working folder holds for the generators, with summaries and suggestions.

    Parameters
    ----------
    app : GuiApp
        The application.
    relative : str or None
        Folder relative to the root.

    Returns
    -------
    dict
        ``saves``, ``upfs``, ``pw_inputs``, ``drivers``, ``dyn0``, ``system_py``,
        ``outputs``, ``save_summary`` (first save), ``template_summary`` (first pw.x input),
        ``suggestions`` (workflow, reason) and ``warnings``.
    """
    from ..gen.paoflow_driver import detect_pw_input, detect_upfs
    from ..gen.system_summary import find_saves, read_pw_input_summary, read_save_summary

    folder = app.resolve(relative)
    if not folder.is_dir():
        raise HttpError(404, '%s is not a folder' % relative)
    workdir = str(folder)
    saves = find_saves(workdir)
    pw_inputs = sorted(p.name for p in folder.glob('*.in') if p.is_file() and _is_pw_input(p))
    preferred_input = detect_pw_input(workdir)
    if preferred_input in pw_inputs:
        pw_inputs.remove(preferred_input)
        pw_inputs.insert(0, preferred_input)
    drivers = sorted(p.name for p in folder.glob('main*.py')) + sorted(
        p.name for p in folder.glob('plot*.py')
    )
    dyn0 = sorted(
        os.path.relpath(p, workdir)
        for p in glob.glob(os.path.join(workdir, '**', '*.dyn0'), recursive=True)
    )[:20]
    outputs = sorted(
        os.path.relpath(p, workdir)
        for p in glob.glob(os.path.join(workdir, '**', 'output'), recursive=True)
        if os.path.isdir(p)
    )[:50]
    save_summary = read_save_summary(os.path.join(workdir, saves[0])) if saves else None
    template_summary = (
        read_pw_input_summary(os.path.join(workdir, pw_inputs[0])) if pw_inputs else None
    )
    suggestions, warnings = [], []
    if saves:
        suggestions.append(('run', 'a QE save is present: PAOFLOW properties'))
        suggestions.append(('acbn0', 'Hubbard U (and V) from the same save'))
        suggestions.append(('phonon', 'finite-displacement phonons from the scf'))
        suggestions.append(('sscha', 'anharmonic phonons at temperature'))
    if pw_inputs:
        suggestions.append(('elphon', 'a pw.x input is present: electron-phonon system folder'))
    if not saves and not pw_inputs:
        suggestions.append(('qe_input', 'no QE run here: start from a database entry'))
        warnings.append('No <prefix>.save and no pw.x input in this folder.')
    if save_summary is not None:
        if save_summary['augmented']:
            warnings.append(
                'Ultrasoft/PAW pseudopotentials: internal projections use the augmented metric; '
                'the native DFPT e-ph vertex needs norm-conserving ones (use the EPW source).'
            )
        missing = [s['name'] for s in save_summary['species'] if s['pseudo'] is None]
        if missing:
            warnings.append('No UPF found in the save for: %s.' % ', '.join(missing))
    if (folder / 'system.py').is_file():
        suggestions.insert(0, ('results', 'an electron-phonon system folder: browse its outputs'))
    return {
        'path': app.sandbox.relative(folder),
        'absolute': workdir,
        'saves': saves,
        'upfs': detect_upfs(workdir, saves[0] if saves else None),
        'pw_inputs': pw_inputs,
        'drivers': drivers,
        'dyn0': dyn0,
        'system_py': (folder / 'system.py').is_file(),
        'outputs': outputs,
        'save_summary': save_summary,
        'template_summary': template_summary,
        'suggestions': [{'workflow': w, 'reason': r} for w, r in suggestions],
        'warnings': warnings,
    }


def read_text(app: 'GuiApp', relative: str) -> dict:
    """Content of a text file inside the root or a read-only folder (for previews).

    Parameters
    ----------
    app : GuiApp
        The application.
    relative : str
        File relative to the root.

    Returns
    -------
    dict
        ``path``, ``size``, ``text`` (``None`` for binary files), ``truncated``.
    """
    path = app.resolve_readable(relative)
    if not path.is_file():
        raise HttpError(404, '%s is not a file' % relative)
    size = path.stat().st_size
    with open(path, 'rb') as handle:
        data = handle.read(MAX_PREVIEW_BYTES)
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        text = None
    return {
        'path': app.sandbox.relative(path),
        'size': size,
        'text': text,
        'truncated': size > MAX_PREVIEW_BYTES,
    }


def register(app: 'GuiApp') -> None:
    """Register the workspace endpoints.

    Parameters
    ----------
    app : GuiApp
        The application.
    """

    @app.route('GET', '/api/session')
    def session(request: Request) -> Response:
        from .. import __version__ as version

        return json_response(
            {
                'root': str(app.sandbox.root),
                'read_roots': [str(folder) for folder in app.sandbox.read_roots],
                'start': app.start,
                'read_only': app.read_only,
                'version': version,
                'max_ranks': app.max_ranks or cpu_budget(),
                'features': environment_features(),
            }
        )

    @app.route('GET', '/api/fs/list')
    def fs_list(request: Request) -> Response:
        return json_response(list_directory(app, request.query.get('path')))

    @app.route('GET', '/api/fs/read')
    def fs_read(request: Request) -> Response:
        return json_response(read_text(app, request.query.get('path', '')))

    @app.route('GET', '/api/workspace/detect')
    def workspace_detect(request: Request) -> Response:
        return json_response(detect(app, request.query.get('path')))
