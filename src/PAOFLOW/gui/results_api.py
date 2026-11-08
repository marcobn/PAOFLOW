"""Results endpoints: scan a folder, draw a dataset, compare runs, overlay experiment, export.

The browser sends a folder (inside the root), a dataset identifier and the selected options; the
server loads the files with :mod:`PAOFLOW.results` and answers with the Plotly specification
(colour roles resolved in the browser), the options of the dataset, its scalar results and notes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from .server import HttpError, Request, Response, json_response

if TYPE_CHECKING:
    from .server import GuiApp


def _folder(app: 'GuiApp', payload: dict) -> str:
    folder = app.resolve_readable(payload.get('path'))
    if not folder.is_dir():
        raise HttpError(404, '%s is not a folder' % payload.get('path'))
    return str(folder)


def _reference(folder: str, identifier: str):
    from ..results.discovery import discover

    for ref in discover(folder):
        if ref.id == identifier:
            return ref
    raise HttpError(404, 'dataset %s not found in the folder (rescan?)' % identifier)


def _dataset(app: 'GuiApp', payload: dict):
    from ..results.loaders import load

    folder = _folder(app, payload)
    ref = _reference(folder, payload.get('id', ''))
    options = payload.get('options') or {}
    try:
        dataset = load(ref, folder, options)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise HttpError(422, 'cannot read %s: %s' % (', '.join(ref.files[:3]), exc)) from None
    return folder, ref, dataset


def _experiment(app: 'GuiApp', dataset, spec: dict | None) -> dict | None:
    """Overlay an experimental curve and compare the first computed series with it."""
    from ..results.columns import read_columns
    from ..results.model import Series

    request = getattr(dataset, '_experiment', None)
    if not request:
        return None
    path = app.resolve_readable(request.get('file'))
    if not path.is_file():
        raise HttpError(404, 'experiment file not found')
    data, header = read_columns(str(path))
    x_column = int(request.get('x_column', 0))
    y_column = int(request.get('y_column', 1))
    if data.shape[0] == 0 or data.shape[1] <= max(x_column, y_column):
        raise HttpError(422, 'the experiment file has no columns %d and %d' % (x_column, y_column))
    scale = float(request.get('scale', 1.0) or 1.0)
    label = request.get('label') or os.path.basename(str(path))
    series = Series(
        label,
        data[:, x_column],
        data[:, y_column] * scale,
        role='neutral',
        mode='markers',
        meta={'file': str(path)},
    )
    dataset.series.append(series)
    metrics = None
    model = next((s for s in dataset.series if s is not series and np.ndim(s.y) == 1), None)
    if model is not None:
        from ..agentic.thermo_optics.metrics import curve_deviation

        order = np.argsort(model.x)
        try:
            linear = curve_deviation(
                series.x,
                series.y,
                np.asarray(model.x)[order],
                np.asarray(model.y)[order],
                space='linear',
            )
            relative = curve_deviation(
                series.x,
                series.y,
                np.asarray(model.x)[order],
                np.asarray(model.y)[order],
                space='relative',
            )
            metrics = {
                'compared_with': model.name,
                'points': linear.get('n'),
                'rms': linear.get('rms'),
                'bias': linear.get('bias'),
                'relative_rms': relative.get('rms'),
            }
        except (ValueError, ZeroDivisionError):
            metrics = None
    return metrics


def figure_payload(app: 'GuiApp', payload: dict) -> dict:
    """Figure, options, results and notes of a dataset.

    Parameters
    ----------
    app : GuiApp
        The application.
    payload : dict
        ``path`` (folder), ``id``, ``options`` and optionally ``experiment`` (``file``,
        ``x_column``, ``y_column``, ``scale``, ``label``).

    Returns
    -------
    dict
    """
    from ..results.figures import figure_spec

    folder, ref, dataset = _dataset(app, payload)
    dataset._experiment = payload.get('experiment')  # type: ignore[attr-defined]
    metrics = _experiment(app, dataset, None)
    spec = figure_spec(dataset)
    return {
        'ref': ref.to_json(),
        'title': dataset.title,
        'family': dataset.family,
        'figure': spec,
        'options': dataset.options,
        'table': dataset.table,
        'text': dataset.text,
        'notes': dataset.notes,
        'x_label': dataset.x_label,
        'y_label': dataset.y_label,
        'metrics': metrics,
        'folder': app.sandbox.relative(folder),
    }


def compare_payload(app: 'GuiApp', payload: dict) -> dict:
    """One figure with the same kind of dataset from several folders.

    Parameters
    ----------
    app : GuiApp
        The application.
    payload : dict
        ``items``: list of ``{path, id, options, label}``.

    Returns
    -------
    dict
        ``figure`` and ``notes``.
    """
    from ..results.figures import figure_spec
    from ..results.model import CATEGORICAL, Dataset

    items = payload.get('items') or []
    if len(items) < 2:
        raise HttpError(400, 'pick at least two datasets to compare')
    if len(items) > len(CATEGORICAL):
        raise HttpError(400, 'at most %d datasets at once' % len(CATEGORICAL))
    loaded = [_dataset(app, item) for item in items]
    kinds = {ref.kind for _, ref, _ in loaded}
    if len(kinds) > 1:
        raise HttpError(
            400, 'compare datasets of the same kind (got %s)' % ', '.join(sorted(kinds))
        )
    first = loaded[0][2]
    merged = Dataset(
        ref=first.ref,
        title='%s — comparison' % first.title,
        family=first.family,
        x_label=first.x_label,
        y_label=first.y_label,
        ticks=first.ticks,
        zero=first.zero,
        options={k: v for k, v in first.options.items() if k == 'window'},
    )
    dashes = ('solid', 'dash', 'dot')
    for index, ((folder, ref, dataset), item) in enumerate(zip(loaded, items)):
        label = item.get('label') or app.sandbox.relative(folder)
        for position, series in enumerate(dataset.series[:3]):
            series.name = label if len(dataset.series) == 1 else '%s · %s' % (label, series.name)
            series.role = CATEGORICAL[index]
            series.dash = dashes[position % len(dashes)]
            merged.series.append(series)
        if len(dataset.series) > 3:
            merged.notes.append('%s: first 3 of %d series shown' % (label, len(dataset.series)))
        if dataset.ticks and dataset.ticks != first.ticks:
            merged.notes.append('%s: different k path; ticks of the first dataset shown' % label)
    if first.surface is not None or first.mesh is not None:
        raise HttpError(400, 'maps and 3D surfaces are compared side by side, not overlaid')
    return {'figure': figure_spec(merged), 'notes': merged.notes, 'title': merged.title}


def register(app: 'GuiApp') -> None:
    """Register the results endpoints.

    Parameters
    ----------
    app : GuiApp
        The application.
    """

    @app.route('GET', '/api/results/palette')
    def palette(request: Request) -> Response:
        from ..results.palette import PALETTE

        return json_response(PALETTE)

    @app.route('POST', '/api/results/scan')
    def scan(request: Request) -> Response:
        from ..results.catalogue import CATEGORY_ORDER
        from ..results.discovery import discover

        payload = request.json()
        folder = _folder(app, payload)
        references = discover(folder)
        return json_response(
            {
                'folder': app.sandbox.relative(folder),
                'absolute': folder,
                'categories': list(CATEGORY_ORDER),
                'datasets': [ref.to_json() for ref in references],
            }
        )

    @app.route('POST', '/api/results/figure')
    def figure(request: Request) -> Response:
        return json_response(figure_payload(app, request.json()))

    @app.route('POST', '/api/results/compare')
    def compare(request: Request) -> Response:
        return json_response(compare_payload(app, request.json()))

    @app.route('POST', '/api/results/export')
    def export(request: Request) -> Response:
        from ..results.export import matplotlib_figure, python_snippet, to_csv

        payload = request.json()
        folder, ref, dataset = _dataset(app, payload)
        fmt = payload.get('format', 'csv')
        name = (
            ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in dataset.title)[:60]
            or ref.kind
        )
        if fmt == 'csv':
            return json_response({'text': to_csv(dataset), 'filename': name + '.csv'})
        if fmt == 'python':
            return json_response(
                {
                    'text': python_snippet(dataset, folder, payload.get('options') or {}),
                    'filename': name + '.py',
                }
            )
        if fmt in ('pdf', 'png', 'svg'):
            try:
                body = matplotlib_figure(dataset, fmt=fmt)
            except (ValueError, ImportError) as exc:
                raise HttpError(422, str(exc)) from None
            types: dict[str, Any] = {
                'pdf': 'application/pdf',
                'png': 'image/png',
                'svg': 'image/svg+xml',
            }
            return Response(
                body=body,
                content_type=types[fmt],
                headers={
                    'Content-Disposition': 'attachment; filename="%s.%s"' % (name, fmt),
                    'Cache-Control': 'no-store',
                },
            )
        raise HttpError(400, 'unknown export format %r' % fmt)

    @app.route('POST', '/api/results/report')
    def report(request: Request) -> Response:
        from ..results.report import build_report

        payload = request.json()
        folder = _folder(app, payload)
        try:
            text = build_report(folder, title=payload.get('title'))
        except ImportError as exc:
            raise HttpError(422, str(exc)) from None
        name = (os.path.basename(folder) or 'results') + '_report.html'
        return Response(
            body=text.encode('utf-8'),
            content_type='text/html; charset=utf-8',
            headers={
                'Content-Disposition': 'attachment; filename="%s"' % name,
                'Cache-Control': 'no-store',
            },
        )
