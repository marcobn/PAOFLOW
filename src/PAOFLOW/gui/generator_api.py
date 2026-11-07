"""Generator endpoints: the ``paoflow-gen`` workflows as forms, previews and written files.

The browser sends the raw field values; the server resolves them with the same questions the CLI
asks (:func:`PAOFLOW.gen.questions.resolve.resolve_form`), assembles the configuration and builds
the files with the same builders, so a form and a terminal session with the same answers write
the same bytes.  Writing re-resolves the answers on the server (the browser never sends a
configuration), refuses to overwrite a different existing file unless the browser confirms it,
and can keep a timestamped backup.
"""

from __future__ import annotations

import difflib
import os
import pprint
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .server import HttpError, Request, Response, json_response

if TYPE_CHECKING:
    from ..gen.outputs import GeneratedFile
    from .server import GuiApp

#: Tools that are not ``paoflow-gen`` workflows but share the Setup tab.
TOOLS = (
    {
        'name': 'qe_input',
        'title': 'QE input from a database',
        'description': 'A pw.x scf input from an AFLOW or C2DB entry (paoflow-gen-qe).',
    },
    {
        'name': 'basis',
        'title': 'Pseudo-atom basis tables',
        'description': 'Radial tables of the PAO basis from the UPF pseudopotentials '
        '(paoflow-genbasis-ps).',
    },
)
MAX_DIFF_LINES = 400


def _workflow(name: str):
    from ..gen.questions import WORKFLOWS

    if name not in WORKFLOWS:
        raise HttpError(404, 'unknown workflow %r' % name)
    return WORKFLOWS[name]


def _context(app: 'GuiApp', payload: dict):
    from ..gen.questions.model import Context

    workdir = app.resolve(payload.get('workdir'))
    if not workdir.is_dir():
        raise HttpError(404, 'the working folder %s does not exist' % payload.get('workdir'))
    options = payload.get('options') or {}
    overrides = {}
    if payload.get('workflow') == 'elphon':
        overrides['out'] = str(app.resolve(options.get('folder') or payload.get('workdir')))
    return Context(workdir=str(workdir), overrides=overrides), options


def _target_folder(app: 'GuiApp', payload: dict, context) -> Path:
    if payload.get('workflow') == 'elphon':
        return Path(context.overrides['out'])
    return Path(context.workdir)


def _diff(old: str, new: str, path: str) -> str:
    lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile='existing/' + path,
            tofile='generated/' + path,
        )
    )
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + ['... (%d more lines)\n' % (len(lines) - MAX_DIFF_LINES)]
    return ''.join(lines)


def describe_files(
    app: 'GuiApp', folder: Path, files: list['GeneratedFile'], with_text: bool = True
) -> list[dict]:
    """Preview of the generated files against what is on disk.

    Parameters
    ----------
    app : GuiApp
        The application (sandbox).
    folder : Path
        Folder the files are written to.
    files : list of GeneratedFile
        The files.
    with_text : bool, optional
        Include the contents.

    Returns
    -------
    list of dict
        ``path``, ``location`` (relative to the root), ``status`` (``new``, ``identical`` or
        ``changed``), ``executable``, ``template`` (copied driver), ``text`` and ``diff``.
    """
    described = []
    for generated in files:
        destination = app.resolve(str(folder / generated.path))
        text = generated.content()
        status, diff = 'new', ''
        if destination.is_file():
            try:
                existing = destination.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                existing = None
            if existing == text:
                status = 'identical'
            else:
                status = 'changed'
                diff = _diff(existing or '', text, generated.path) if existing is not None else ''
        entry = {
            'path': generated.path,
            'location': app.sandbox.relative(destination),
            'status': status,
            'executable': generated.executable,
            'template': generated.template is not None,
            'diff': diff,
        }
        if with_text:
            entry['text'] = text
        described.append(entry)
    return described


def python_equivalent(workflow: str, config: dict, options: dict) -> str:
    """A script that writes the same files without the GUI (reproducible generation).

    Parameters
    ----------
    workflow : str
        Workflow name.
    config : dict
        The assembled configuration.
    options : dict
        GUI options (driver and plot file names).

    Returns
    -------
    str
    """
    shown = dict(config)
    template = None
    if shown.get('pw_template_text') and shown.get('pw_template'):
        template = shown['pw_template']
        shown['pw_template_text'] = '__TEMPLATE__'
    text = pprint.pformat(shown, sort_dicts=False, width=96)
    if template is not None:
        text = text.replace("'__TEMPLATE__'", 'open(%r).read()' % template)
    arguments = ''
    if workflow != 'elphon':
        extras = []
        if options.get('out'):
            extras.append('out=%r' % options['out'])
        if options.get('plot_out'):
            extras.append('plot_out=%r' % options['plot_out'])
        arguments = ''.join(', ' + extra for extra in extras)
    folder = 'cfg[%r]' % 'out' if workflow == 'elphon' else "'.'"
    return (
        '"""Write the %s files of this configuration (generated by the PAOFLOW Studio)."""\n\n'
        'from PAOFLOW.gen.outputs import workflow_outputs\n\n'
        'cfg = %s\n\n'
        'for generated in workflow_outputs(%r, cfg%s):\n'
        '    print(generated.write(%s))\n' % (workflow, text, workflow, arguments, folder)
    )


def resolve_payload(app: 'GuiApp', payload: dict, with_text: bool = True) -> dict:
    """Resolve a form and, when complete, preview its files and its run plan.

    Parameters
    ----------
    app : GuiApp
        The application.
    payload : dict
        ``workflow``, ``workdir``, ``answers`` and ``options`` (``out``, ``plot_out``, ``plot``,
        ``folder`` of the electron-phonon system).
    with_text : bool, optional
        Include the file contents.

    Returns
    -------
    dict
        The resolved form (:meth:`ResolvedForm.to_json`), ``target`` (relative folder),
        ``files``, ``plan`` and ``python`` when complete.
    """
    from ..gen.outputs import run_plan, workflow_outputs
    from ..gen.questions.resolve import resolve_form

    workflow = _workflow(payload.get('workflow', ''))
    context, options = _context(app, payload)
    form = resolve_form(workflow, payload.get('answers') or {}, context)
    result: dict[str, Any] = form.to_json()
    folder = _target_folder(app, payload, context)
    result['target'] = app.sandbox.relative(folder) if folder.exists() else str(folder)
    result['workflow'] = workflow.name
    if not form.complete:
        return result
    try:
        files = workflow_outputs(
            workflow.name,
            form.config,
            plot=bool(options.get('plot')),
            out=options.get('out') or None,
            plot_out=options.get('plot_out') or None,
        )
    except Exception as exc:  # a builder rejected the configuration
        result['complete'] = False
        result['errors']['_build'] = 'the script builder failed: %s: %s' % (type(exc).__name__, exc)
        return result
    result['files'] = describe_files(app, folder, files, with_text=with_text)
    result['plan'] = [
        {
            'label': step.label,
            'argv': list(step.argv),
            'command': step.command(),
            'cwd': step.cwd,
            'mpi': step.mpi,
            'ranks_env': step.ranks_env,
            'interactive': step.interactive,
            'manual': step.manual,
            'note': step.note,
        }
        for step in run_plan(workflow.name, form.config, files)
    ]
    result['python'] = python_equivalent(workflow.name, form.config, options)
    result['_files'] = files
    result['_folder'] = folder
    result['_config'] = form.config
    return result


def write_files(app: 'GuiApp', payload: dict) -> dict:
    """Write the files of a complete form (conflicts need the browser's confirmation).

    Parameters
    ----------
    app : GuiApp
        The application.
    payload : dict
        As :func:`resolve_payload`, plus ``overwrite`` (paths the user confirmed), ``keep``
        (changed paths left as they are) and ``backup`` (keep ``<file>.bak-<time>`` copies of
        overwritten files).

    Returns
    -------
    dict
        ``written``, ``unchanged``, ``kept``, ``backups`` (relative paths), ``folder`` and
        ``plan``.

    Raises
    ------
    HttpError
        400 for an incomplete form, 403 in read-only mode, 409 listing the files that exist
        with another content and were not confirmed.
    """
    app.require_writable()
    resolved = resolve_payload(app, payload, with_text=False)
    if not resolved.get('complete'):
        raise HttpError(400, 'the form is not complete', {'errors': resolved.get('errors', {})})
    folder: Path = resolved['_folder']
    files = resolved['_files']
    confirmed = set(payload.get('overwrite') or [])
    kept_paths = set(payload.get('keep') or []) - confirmed
    conflicts = [
        f['path']
        for f in resolved['files']
        if f['status'] == 'changed' and f['path'] not in confirmed and f['path'] not in kept_paths
    ]
    if conflicts:
        raise HttpError(409, 'existing files differ', {'conflicts': conflicts})
    stamp = time.strftime('%Y%m%d-%H%M%S')
    written, unchanged, kept, backups = [], [], [], []
    status = {f['path']: f['status'] for f in resolved['files']}
    with app.lock:
        for generated in files:
            destination = app.resolve(str(folder / generated.path))
            if status[generated.path] == 'identical':
                unchanged.append(app.sandbox.relative(destination))
                continue
            if status[generated.path] == 'changed' and generated.path in kept_paths:
                kept.append(app.sandbox.relative(destination))
                continue
            if status[generated.path] == 'changed' and payload.get('backup', True):
                backup = destination.with_name(destination.name + '.bak-' + stamp)
                shutil.copy2(destination, backup)
                backups.append(app.sandbox.relative(backup))
            written.append(app.sandbox.relative(generated.write(str(folder))))
        if resolved['workflow'] == 'elphon':
            from ..gen.elphon_system import SYSTEM_FOLDERS

            for name in SYSTEM_FOLDERS:
                os.makedirs(folder / name, exist_ok=True)
    return {
        'written': written,
        'unchanged': unchanged,
        'kept': kept,
        'backups': backups,
        'folder': app.sandbox.relative(folder),
        'plan': resolved['plan'],
    }


# ------------------------------------------------------------------------------ QE input
def _qe_input_arguments(app: 'GuiApp', payload: dict) -> dict:
    identifier = (payload.get('identifier') or '').strip()
    if not identifier:
        raise HttpError(400, 'give an AFLOW or C2DB identifier')
    workdir = app.resolve(payload.get('workdir'))
    pseudo_dir = app.resolve(str(workdir / (payload.get('pseudo_dir') or '.')))
    if not pseudo_dir.is_dir():
        raise HttpError(400, 'the pseudopotential folder does not exist')
    kpoints = payload.get('kpoints')
    if kpoints:
        try:
            kpoints = tuple(int(k) for k in kpoints)
        except (TypeError, ValueError):
            raise HttpError(400, 'the k-point grid needs three integers') from None
        if len(kpoints) != 3:
            raise HttpError(400, 'the k-point grid needs three integers')
    nbnd = payload.get('nbnd')

    def optional_bool(key: str):
        value = payload.get(key)
        return None if value in (None, '', 'auto') else bool(value)

    return dict(
        identifier=identifier,
        pseudo_dir=str(pseudo_dir),
        source=payload.get('source') or 'auto',
        soc=bool(payload.get('soc')),
        degauss=float(payload.get('degauss') or 0.05),
        nbnd=int(nbnd) if nbnd not in (None, '') else None,
        ibrav=payload.get('ibrav') or 'auto',
        symprec=float(payload.get('symprec') or 1.0e-4),
        metallic=optional_bool('metallic'),
        magnetic=optional_bool('magnetic'),
        kpoints=kpoints or None,
        assume_isolated_2d=payload.get('assume_isolated_2d', True) is not False,
    )


def qe_input_preview(app: 'GuiApp', payload: dict) -> dict:
    """Fetch a database entry and preview its scf input (needs the network).

    Parameters
    ----------
    app : GuiApp
        The application.
    payload : dict
        ``identifier``, ``pseudo_dir``, ``workdir`` and the options of ``paoflow-gen-qe``.

    Returns
    -------
    dict
        ``filename`` (default name), ``text``, ``record`` (compound, atoms, species, metallic,
        magnetic, dimensionality, k-points, source) and ``status`` of the file on disk.
    """
    from ..gen.qe_input.cli import default_output_name, generate_qe_input

    arguments = _qe_input_arguments(app, payload)
    try:
        record, text = generate_qe_input(**arguments)
    except (RuntimeError, ValueError, OSError) as exc:
        raise HttpError(502, 'could not build the input: %s' % exc) from None
    filename = payload.get('filename') or default_output_name(record)
    workdir = app.resolve(payload.get('workdir'))
    destination = app.resolve(str(workdir / filename))
    status = 'new'
    if destination.is_file():
        status = 'identical' if destination.read_text(errors='replace') == text else 'changed'
    return {
        'filename': filename,
        'text': text,
        'status': status,
        'record': {
            'compound': record.compound,
            'natoms': record.natoms,
            'species': list(record.species),
            'metallic': record.metallic,
            'magnetic': record.magnetic,
            'dimensionality': record.dimensionality,
            'kpoints': list(record.kpoints) if record.kpoints else None,
            'source': record.source,
            'spacegroup': str(record.spacegroup) if record.spacegroup is not None else None,
        },
    }


def qe_input_write(app: 'GuiApp', payload: dict) -> dict:
    """Write the scf input of a database entry (rebuilt on the server).

    Parameters
    ----------
    app : GuiApp
        The application.
    payload : dict
        As :func:`qe_input_preview`, plus ``filename`` and ``overwrite`` (bool).

    Returns
    -------
    dict
        ``written`` (relative path).
    """
    app.require_writable()
    preview = qe_input_preview(app, payload)
    workdir = app.resolve(payload.get('workdir'))
    destination = app.resolve(str(workdir / preview['filename']))
    if preview['status'] == 'changed' and not payload.get('overwrite'):
        raise HttpError(409, 'the file exists', {'conflicts': [preview['filename']]})
    destination.write_text(preview['text'], encoding='utf-8')
    return {'written': app.sandbox.relative(destination)}


def basis_command(app: 'GuiApp', payload: dict) -> dict:
    """``paoflow-genbasis-ps`` command line of the basis form.

    Parameters
    ----------
    app : GuiApp
        The application.
    payload : dict
        ``workdir``, ``pseudo`` or ``pseudo_dir``, ``out``, ``preset``, ``shells``, ``r_box``,
        ``n_points``, ``no_overwrite``.

    Returns
    -------
    dict
        ``argv`` (relative to the working folder), ``command`` and ``cwd``.
    """
    import shlex

    workdir = app.resolve(payload.get('workdir'))
    argv = ['paoflow-genbasis-ps']
    if payload.get('pseudo'):
        app.resolve(str(workdir / payload['pseudo']))
        argv += ['--pseudo', payload['pseudo']]
    else:
        pseudo_dir = payload.get('pseudo_dir') or '.'
        app.resolve(str(workdir / pseudo_dir))
        argv += ['--pseudo-dir', pseudo_dir]
    out = payload.get('out') or 'BASIS_PS'
    app.resolve(str(workdir / out))
    argv += ['--out', out]
    if payload.get('shells') and payload.get('pseudo'):
        argv += ['--shells', str(payload['shells'])]
    else:
        preset = payload.get('preset') or 'extended'
        if preset not in ('minimal', 'standard', 'extended'):
            raise HttpError(400, 'unknown preset %r' % preset)
        argv += ['--preset', preset]
    if payload.get('r_box') not in (None, ''):
        argv += ['--r-box', '%g' % float(payload['r_box'])]
    if payload.get('n_points') not in (None, ''):
        argv += ['--n-points', '%d' % int(payload['n_points'])]
    if payload.get('no_overwrite'):
        argv.append('--no-overwrite')
    return {
        'argv': argv,
        'command': ' '.join(shlex.quote(a) for a in argv),
        'cwd': app.sandbox.relative(workdir),
    }


def register(app: 'GuiApp') -> None:
    """Register the generator endpoints.

    Parameters
    ----------
    app : GuiApp
        The application.
    """

    @app.route('GET', '/api/gen/workflows')
    def workflows(request: Request) -> Response:
        from ..gen.questions import WORKFLOWS

        items = [
            {'name': w.name, 'title': w.title, 'description': w.description, 'kind': 'workflow'}
            for w in WORKFLOWS.values()
        ]
        items += [dict(tool, kind='tool') for tool in TOOLS]
        return json_response({'workflows': items})

    @app.route('POST', '/api/gen/resolve')
    def resolve(request: Request) -> Response:
        result = resolve_payload(app, request.json())
        return json_response({k: v for k, v in result.items() if not k.startswith('_')})

    @app.route('POST', '/api/gen/write')
    def write(request: Request) -> Response:
        return json_response(write_files(app, request.json()))

    @app.route('POST', '/api/gen/qe-input/preview')
    def qe_preview(request: Request) -> Response:
        return json_response(qe_input_preview(app, request.json()))

    @app.route('POST', '/api/gen/qe-input/write')
    def qe_write(request: Request) -> Response:
        return json_response(qe_input_write(app, request.json()))

    @app.route('POST', '/api/gen/basis/command')
    def basis(request: Request) -> Response:
        return json_response(basis_command(app, request.json()))
