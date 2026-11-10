"""Endpoints of the Plots tab: the plot scripts of a folder and the figures they made.

The scripts are the generated ``plot*.py`` of the working folder and the ``plot*.py`` one level
down (the workflows of an electron–phonon system folder: ``superconductivity/plot_me.py``, ...).
The tab runs them through the job manager with ``capture`` (see :func:`.jobs.build_command`):
:mod:`PAOFLOW.graphics.figure_capture` runs them without a display and writes their figures to
``figures/<script>/`` as PNG and PDF files listed in ``figures.json``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .server import HttpError, Request, Response, json_response

if TYPE_CHECKING:
    from .server import GuiApp

PLOT_SCRIPT = re.compile(r'^plot[\w.-]*\.py$')
OPTION = re.compile(r"""add_argument\(\s*['"](--[\w-]+)['"]""")
SKIPPED_FOLDERS = {'output', 'figures', '__pycache__'}
FIGURE_TYPES = {'.png': 'image/png', '.pdf': 'application/pdf'}


def plot_scripts(folder: Path) -> list[str]:
    """The plot scripts of a folder, relative to it.

    Parameters
    ----------
    folder : Path
        The working folder.

    Returns
    -------
    list of str
        ``plot*.py`` of the folder, then those of its subfolders (one level), sorted.
    """
    top = sorted(p.name for p in folder.glob('plot*.py') if PLOT_SCRIPT.match(p.name))
    nested = sorted(
        '%s/%s' % (p.parent.name, p.name)
        for p in folder.glob('*/plot*.py')
        if PLOT_SCRIPT.match(p.name)
        and not p.parent.name.startswith('.')
        and p.parent.name not in SKIPPED_FOLDERS
    )
    return [name for name in top + nested if (folder / name).is_file()]


def script_facts(path: Path) -> dict:
    """What the Plots tab shows of a script: its title and command-line options.

    Parameters
    ----------
    path : Path
        The script.

    Returns
    -------
    dict
        ``title`` (first line of the docstring), ``options`` (``--name`` flags) and
        ``arguments`` (the default arguments: ``--all`` when the script has it, which plots every
        quantity without asking).
    """
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        text = ''
    try:
        docstring = ast.get_docstring(ast.parse(text)) or ''
    except SyntaxError:
        docstring = ''
    options = list(dict.fromkeys(OPTION.findall(text)))
    return {
        'title': docstring.strip().splitlines()[0] if docstring.strip() else '',
        'options': options,
        'arguments': ['--all'] if '--all' in options else [],
    }


def captured_figures(app: 'GuiApp', folder: Path) -> dict | None:
    """The manifest of a capture folder, with the paths of the files relative to the root.

    Parameters
    ----------
    app : GuiApp
        The application.
    folder : Path
        ``figures/<script>`` of a script.

    Returns
    -------
    dict or None
        ``figures`` (``title``, ``files`` by format, ``saved_as``, ``mtime``), ``status``,
        ``started``, ``ended`` and ``arguments``; None before the first run.
    """
    try:
        manifest = json.loads((folder / 'figures.json').read_text())
    except (OSError, ValueError):
        return None
    figures = []
    for entry in manifest.get('figures', []):
        files = {}
        mtime = 0.0
        for fmt, name in (entry.get('files') or {}).items():
            path = folder / str(name)
            if path.parent == folder and path.is_file():
                files[fmt] = app.sandbox.relative(path)
                mtime = max(mtime, path.stat().st_mtime)
        if files:
            figures.append(
                {
                    'title': entry.get('title', ''),
                    'files': files,
                    'saved_as': entry.get('saved_as'),
                    'mtime': mtime,
                }
            )
    return {
        'figures': figures,
        'status': manifest.get('status'),
        'started': manifest.get('started'),
        'ended': manifest.get('ended'),
        'arguments': manifest.get('arguments', []),
        'folder': app.sandbox.relative(folder),
    }


def _last_job(app: 'GuiApp', cwd: str, script: str) -> dict | None:
    """The latest capture job of a script in a folder (running or ended), for its log."""
    manager = app.services.get('jobs')
    if manager is None:
        return None
    found = None
    for job in manager.jobs.values():
        argv = job.argv
        if job.cwd != cwd or 'PAOFLOW.graphics.figure_capture' not in argv:
            continue
        index = argv.index('--out') + 2 if '--out' in argv else -1
        if 0 <= index < len(argv) and argv[index] == script:
            if found is None or job.started > found.started:
                found = job
    return found.to_json() if found is not None else None


def list_plots(app: 'GuiApp', relative: str | None) -> dict:
    """The plot scripts of a folder and their figures.

    Parameters
    ----------
    app : GuiApp
        The application.
    relative : str or None
        Folder relative to the root, or absolute.

    Returns
    -------
    dict
        ``folder``, ``absolute``, ``writable`` and ``scripts`` (``script``, ``title``,
        ``options``, ``arguments``, ``capture`` (see :func:`captured_figures`) and ``job``).
    """
    from ..graphics.figure_capture import capture_folder

    folder = app.resolve_readable(relative)
    if not folder.is_dir():
        raise HttpError(404, '%s is not a folder' % relative)
    cwd = app.sandbox.relative(folder)
    scripts = []
    for script in plot_scripts(folder):
        scripts.append(
            {
                'script': script,
                **script_facts(folder / script),
                'capture': captured_figures(app, folder / capture_folder(script)),
                'job': _last_job(app, cwd, script),
            }
        )
    return {
        'folder': cwd,
        'absolute': str(folder),
        'writable': app.sandbox.writable(folder) and not app.read_only,
        'scripts': scripts,
    }


def figure_file(app: 'GuiApp', relative: str, download: bool = False) -> Response:
    """A captured figure (PNG shown in the page, PDF downloaded).

    Parameters
    ----------
    app : GuiApp
        The application.
    relative : str
        File relative to the root, or absolute.
    download : bool, optional
        Send it as an attachment.

    Returns
    -------
    Response
    """
    path = app.resolve_readable(relative)
    content_type = FIGURE_TYPES.get(path.suffix.lower())
    if content_type is None:
        raise HttpError(400, 'only PNG and PDF figures are served')
    if not path.is_file():
        raise HttpError(404, '%s not found' % relative)
    # A PDF is downloaded: the page's content policy (object-src 'none') keeps it from opening.
    disposition = 'attachment' if download or content_type == 'application/pdf' else 'inline'
    return Response(
        body=path.read_bytes(),
        content_type=content_type,
        headers={
            'Cache-Control': 'no-cache',
            'Content-Disposition': '%s; filename="%s"'
            % (disposition, re.sub(r'[^\w.-]', '_', path.name)),
        },
    )


def register(app: 'GuiApp') -> None:
    """Register the Plots endpoints.

    Parameters
    ----------
    app : GuiApp
        The application.
    """

    @app.route('GET', '/api/plots')
    def plots(request: Request) -> Response:
        return json_response(list_plots(app, request.query.get('path')))

    @app.route('GET', '/api/plots/file')
    def plot_file(request: Request) -> Response:
        return figure_file(app, request.query.get('path', ''), bool(request.query.get('download')))
