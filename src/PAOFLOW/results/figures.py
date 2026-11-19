"""Plotly figure specifications of datasets (plain dictionaries, no plotting library needed).

Colours are *roles* (``series-1`` ... ``series-8``, ``seq-450``, ``neutral``) and colour scales are
``role:sequential`` / ``role:diverging``: the browser (and the static report) resolves them for
the light or dark theme, so a specification does not depend on the theme.  Marks follow the
GUI's chart rules: 2 px lines, hairline axes and grids, a legend for two or more series, labelled
high-symmetry ticks, a reference line at the Fermi level of energy axes, and no second y axis
(quantities of different units are separate figures through the dataset options).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .model import CATEGORICAL, Dataset, Series

DIGITS = 6


def _list(values: Any) -> list:
    array = np.asarray(values, dtype=float)
    rounded = np.where(np.isfinite(array), np.round(array, DIGITS), np.nan)
    return [None if not np.isfinite(v) else float(v) for v in rounded.ravel()]


def _packed(x: np.ndarray, y: np.ndarray) -> tuple[list, list]:
    """One trace for many bands: columns of ``y`` joined with gaps (``None``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return _list(x), _list(y)
    count = y.shape[1]
    xs = np.tile(np.append(x, np.nan), count)
    ys = np.concatenate([np.append(y[:, band], np.nan) for band in range(count)])
    return _list(xs), _list(ys)


def _line_trace(series: Series, show_legend: bool) -> dict:
    x, y = _packed(series.x, series.y)
    trace: dict[str, Any] = {
        'type': 'scatter',
        'mode': series.mode,
        'name': series.name,
        'x': x,
        'y': y,
        'showlegend': show_legend,
        'hovertemplate': '%{x:.4g}, %{y:.4g}<extra>' + series.name + '</extra>',
    }
    if 'lines' in series.mode:
        trace['line'] = {
            'color': series.role,
            'width': 2 if np.ndim(series.y) == 1 else 1.4,
            'dash': series.dash,
        }
    if 'stack' in series.meta:
        # Stacked areas (part of a whole): solid fills split by a thin line of the surface colour.
        trace['stackgroup'] = series.meta['stack']
        trace['fillcolor'] = '%s@0.8' % series.role
        trace['line'] = {'color': 'surface', 'width': 1}
    if 'markers' in series.mode:
        marker: dict[str, Any] = {'size': 6, 'color': series.role}
        if 'color' in series.meta:
            marker = {
                'size': 5,
                'color': _list(series.meta['color']),
                'colorscale': 'role:%s'
                % ('diverging' if series.role == 'diverging' else 'sequential'),
                'showscale': True,
                'colorbar': {
                    'title': {'text': series.meta.get('color_label', '')},
                    'thickness': 12,
                },
            }
            if series.role == 'diverging':
                limit = float(np.nanmax(np.abs(series.meta['color']))) or 1.0
                marker.update(cmin=-limit, cmax=limit)
        trace['marker'] = marker
    if 'error' in series.meta:
        trace['error_y'] = {
            'type': 'data',
            'array': _list(series.meta['error']),
            'color': series.role,
            'thickness': 1,
        }
    return trace


def _axis(title: str, **extra) -> dict:
    axis = {
        'title': {'text': title},
        'showline': True,
        'linewidth': 1,
        'mirror': False,
        'ticks': 'outside',
        'ticklen': 4,
        'zeroline': False,
        'showgrid': True,
        'gridwidth': 1,
        'automargin': True,
        'exponentformat': 'power',
    }
    axis.update(extra)
    return axis


def _tick_axis(dataset: Dataset, title: str) -> dict:
    if not dataset.ticks:
        return _axis(title)
    return _axis(
        '',
        tickmode='array',
        tickvals=[t[0] for t in dataset.ticks],
        ticktext=[t[1] for t in dataset.ticks],
        showgrid=False,
    )


def _reference_shapes(dataset: Dataset) -> list[dict]:
    shapes = []
    if dataset.zero == 'y':
        shapes.append(
            {
                'type': 'line',
                'xref': 'paper',
                'x0': 0,
                'x1': 1,
                'yref': 'y',
                'y0': 0,
                'y1': 0,
                'line': {'color': 'neutral', 'width': 1, 'dash': 'dot'},
            }
        )
    elif dataset.zero == 'x':
        shapes.append(
            {
                'type': 'line',
                'yref': 'paper',
                'y0': 0,
                'y1': 1,
                'xref': 'x',
                'x0': 0,
                'x1': 0,
                'line': {'color': 'neutral', 'width': 1, 'dash': 'dot'},
            }
        )
    for position, _label in dataset.ticks:
        shapes.append(
            {
                'type': 'line',
                'yref': 'paper',
                'y0': 0,
                'y1': 1,
                'xref': 'x',
                'x0': position,
                'x1': position,
                'line': {'color': 'axis', 'width': 1},
            }
        )
    return shapes


def _layout(dataset: Dataset, x_axis: dict, y_axis: dict, legend: bool) -> dict:
    return {
        'title': {'text': dataset.title, 'x': 0.0, 'xanchor': 'left', 'font': {'size': 15}},
        'xaxis': x_axis,
        'yaxis': y_axis,
        'showlegend': legend,
        'legend': {'orientation': 'h', 'y': -0.18, 'x': 0, 'yanchor': 'top'},
        'shapes': _reference_shapes(dataset),
        'margin': {'l': 70, 'r': 20, 't': 48, 'b': 60},
        'hovermode': 'closest',
    }


def lines_figure(dataset: Dataset) -> dict:
    """Line and band plots (every family with series).

    Parameters
    ----------
    dataset : Dataset
        The dataset.

    Returns
    -------
    dict
        ``{'data': [...], 'layout': {...}}``.
    """
    legend = len({s.name for s in dataset.series}) > 1
    data = [_line_trace(series, legend) for series in dataset.series]
    x_axis = _tick_axis(dataset, dataset.x_label)
    y_axis = _axis(dataset.y_label)
    window = dataset.options.get('window', {}).get('value')
    if window:
        y_axis['range'] = list(window)
    log = dataset.options.get('log_y', {}).get('value')
    if log:
        y_axis['type'] = 'log'
    if dataset.ticks and dataset.series:
        x_axis['range'] = [
            float(np.nanmin(dataset.series[0].x)),
            float(np.nanmax(dataset.series[0].x)),
        ]
    layout = _layout(dataset, x_axis, y_axis, legend)
    if dataset.ticks:
        layout['hovermode'] = 'closest'
    elif len(dataset.series) > 1 and all(s.mode == 'lines' for s in dataset.series):
        layout['hovermode'] = 'x unified'
    return {'data': data, 'layout': layout}


#: Line styles of the series (components) of an overlay; the colour is the property.
OVERLAY_DASHES = ('solid', 'dash', 'dot', 'dashdot')
#: Families that draw as lines on one x axis, the only ones an overlay can combine.
OVERLAY_FAMILIES = ('curve', 'columns', 'pdos', 'temperature_columns', 'transport', 'hall', 'bands')


def axis_unit(label: str) -> str:
    """The unit of an axis title (``'Absorption coefficient (1/m)'`` -> ``'1/m'``; none: ``''``).

    Parameters
    ----------
    label : str
        Axis title as written by :func:`~PAOFLOW.results.model.axis_title`.

    Returns
    -------
    str
    """
    match = re.search(r'\(([^()]*)\)\s*$', label or '')
    return match.group(1).strip() if match else ''


def axis_name(label: str) -> str:
    """An axis title without its unit (``'Absorption coefficient (1/m)'`` -> its name)."""
    return re.sub(r'\s*\([^()]*\)\s*$', '', label or '').strip()


def overlay_figure(
    datasets: list[Dataset], series: list[str] | None = None, slots: list[int] | None = None
) -> tuple[dict, list[str], list[str]]:
    """Several properties on their common x axis: one panel per y unit, stacked, sharing x.

    Quantities of one unit (Re and Im of ε, n and k) share a panel; another unit gets its own
    panel under it, never a second y axis on the same panel. The colour is the property, in the
    fixed categorical order of its slot (it stays when other properties come and go), and the
    line style is the series (the tensor component).

    Parameters
    ----------
    datasets : list of Dataset
        The properties, the first one being the open dataset; same x axis.
    series : list of str, optional
        Series names to draw (default: the first name all the datasets share, e.g. ``xx``). A
        dataset with none of them draws its first series, so every property appears.
    slots : list of int, optional
        Colour slot of each dataset (1-based; default 1, 2, ...).

    Returns
    -------
    figure : dict
        Plotly specification with colour roles.
    names : list of str
        Every series name of the datasets (the choices of ``series``).
    shown : list of str
        The series names drawn.
    """
    slots = list(slots or range(1, len(datasets) + 1))
    names: list[str] = []
    for dataset in datasets:
        for item in dataset.series:
            if item.name not in names:
                names.append(item.name)
    # The series every property has (the tensor components of the optics); by default the first.
    common = [n for n in names if all(any(s.name == n for s in d.series) for d in datasets)]
    wanted = [name for name in (series or common[:1]) if name in names]
    units: list[str] = []
    for dataset in datasets:
        if axis_unit(dataset.y_label) not in units:
            units.append(axis_unit(dataset.y_label))
    labels = [axis_name(d.y_label) or d.title for d in datasets]
    if len(set(labels)) < len(labels):
        labels = [d.title for d in datasets]
    data = []
    drawn: set[str] = set()
    for dataset, label, slot in zip(datasets, labels, slots):
        panel = units.index(axis_unit(dataset.y_label))
        # A property without the chosen series still appears, with its first series.
        chosen = [item for item in dataset.series if item.name in wanted] or dataset.series[:1]
        for item in chosen:
            drawn.add(item.name)
            plain = len(dataset.series) == 1 or (len(wanted) == 1 and item.name in wanted)
            name = label if plain else '%s · %s' % (label, item.name)
            trace = _line_trace(
                Series(
                    name,
                    item.x,
                    item.y,
                    role=CATEGORICAL[(slot - 1) % len(CATEGORICAL)],
                    dash=OVERLAY_DASHES[names.index(item.name) % len(OVERLAY_DASHES)],
                    mode=item.mode,
                    meta={k: v for k, v in item.meta.items() if k != 'stack'},
                ),
                True,
            )
            trace['yaxis'] = 'y' if panel == 0 else 'y%d' % (panel + 1)
            data.append(trace)
    shown = [name for name in names if name in drawn]
    first = datasets[0]
    count = len(units)
    gap = 0.07 if count > 1 else 0.0
    height = (1.0 - gap * (count - 1)) / count
    layout = _layout(first, _tick_axis(first, first.x_label), {}, True)
    layout['title']['text'] = ', '.join(labels)
    layout['shapes'] = [s for s in layout['shapes'] if s.get('yref') == 'paper']
    for panel, unit in enumerate(units):
        members = [
            label for dataset, label in zip(datasets, labels) if axis_unit(dataset.y_label) == unit
        ]
        title = ', '.join(members) + (' (%s)' % unit if unit else '')
        top = 1.0 - panel * (height + gap)
        axis = _axis(title, domain=[round(max(0.0, top - height), 6), round(top, 6)], anchor='x')
        layout['yaxis' if panel == 0 else 'yaxis%d' % (panel + 1)] = axis
        if any(d.zero == 'y' for d in datasets if axis_unit(d.y_label) == unit):
            layout['shapes'].append(
                {
                    'type': 'line',
                    'xref': 'paper',
                    'x0': 0,
                    'x1': 1,
                    'yref': 'y' if panel == 0 else 'y%d' % (panel + 1),
                    'y0': 0,
                    'y1': 0,
                    'line': {'color': 'neutral', 'width': 1, 'dash': 'dot'},
                }
            )
    layout['xaxis']['anchor'] = 'y' if count == 1 else 'y%d' % count
    if first.ticks and first.series:
        layout['xaxis']['range'] = [
            float(np.nanmin(first.series[0].x)),
            float(np.nanmax(first.series[0].x)),
        ]
    layout['hovermode'] = 'closest' if first.ticks else 'x unified'
    return {'data': data, 'layout': layout}, names, shown


def map_figure(dataset: Dataset) -> dict:
    """Heatmaps (ARPES intensity, spectral functions, field slices) with overlaid curves.

    Parameters
    ----------
    dataset : Dataset
        A dataset with ``surface``.

    Returns
    -------
    dict
    """
    surface = dataset.surface
    z = np.asarray(surface['z'], dtype=float)
    heatmap: dict[str, Any] = {
        'type': 'heatmap',
        'x': _list(surface['x']),
        'y': _list(surface['y']),
        'z': [_list(row) for row in z],
        'colorscale': 'role:%s' % ('diverging' if surface.get('diverging') else 'sequential'),
        'colorbar': {'title': {'text': surface.get('z_label', '')}, 'thickness': 12},
        'hovertemplate': 'x %{x:.4g}<br>y %{y:.4g}<br>'
        + surface.get('z_label', 'z')
        + ' %{z:.4g}<extra></extra>',
    }
    if surface.get('diverging'):
        limit = float(np.nanmax(np.abs(z))) or 1.0
        heatmap.update(zmin=-limit, zmax=limit, zmid=0.0)
    data = [heatmap] + [_line_trace(series, False) for series in dataset.series]
    x_axis = _tick_axis(dataset, surface.get('x_label', ''))
    y_axis = _axis(surface.get('y_label', ''))
    layout = _layout(dataset, x_axis, y_axis, False)
    layout['xaxis']['showgrid'] = False
    layout['yaxis']['showgrid'] = False
    return {'data': data, 'layout': layout}


def volume_figure(dataset: Dataset) -> dict:
    """Fermi sheets as ``mesh3d`` coloured by the Fermi velocity, with the zone edges.

    Parameters
    ----------
    dataset : Dataset
        A dataset with ``mesh``.

    Returns
    -------
    dict
    """
    mesh = dataset.mesh
    low, high = mesh['speed_range']
    data = []
    for index, sheet in enumerate(mesh['sheets']):
        data.append(
            {
                'type': 'mesh3d',
                'name': 'band %d' % sheet['band'],
                'x': _list(sheet['x']),
                'y': _list(sheet['y']),
                'z': _list(sheet['z']),
                'i': [int(v) for v in sheet['i']],
                'j': [int(v) for v in sheet['j']],
                'k': [int(v) for v in sheet['k']],
                'intensity': _list(sheet['speed']),
                'colorscale': 'role:sequential-mesh',
                'cmin': low,
                'cmax': high,
                'showscale': index == 0,
                'colorbar': {'title': {'text': '|v_F| (a.u.)'}, 'thickness': 12},
                'flatshading': False,
                'lighting': {'ambient': 0.55, 'diffuse': 0.7, 'specular': 0.15, 'roughness': 0.6},
                'hovertemplate': 'band %d<br>|v_F| %%{intensity:.3g}<extra></extra>'
                % sheet['band'],
                'showlegend': True,
            }
        )
    if mesh['edges']:
        xs, ys, zs = [], [], []
        for edge in mesh['edges']:
            xs += [edge[0], edge[3], None]
            ys += [edge[1], edge[4], None]
            zs += [edge[2], edge[5], None]
        data.append(
            {
                'type': 'scatter3d',
                'mode': 'lines',
                'name': 'Brillouin zone',
                'x': xs,
                'y': ys,
                'z': zs,
                'line': {'color': 'axis', 'width': 3},
                'hoverinfo': 'skip',
                'showlegend': False,
            }
        )
    axis = {
        'showbackground': False,
        'showgrid': False,
        'zeroline': False,
        'showticklabels': False,
        'title': {'text': ''},
    }
    layout = {
        'title': {'text': dataset.title, 'x': 0.0, 'xanchor': 'left', 'font': {'size': 15}},
        'scene': {
            'xaxis': dict(axis, title={'text': 'kx'}),
            'yaxis': dict(axis, title={'text': 'ky'}),
            'zaxis': dict(axis, title={'text': 'kz'}),
            'aspectmode': 'data',
        },
        'margin': {'l': 0, 'r': 0, 't': 40, 'b': 0},
        'showlegend': len(mesh['sheets']) > 1,
        'legend': {'orientation': 'h', 'y': 0, 'x': 0},
    }
    return {'data': data, 'layout': layout}


def figure_spec(dataset: Dataset) -> dict | None:
    """The Plotly specification of a dataset (``None`` for text and array listings).

    Parameters
    ----------
    dataset : Dataset
        The loaded dataset.

    Returns
    -------
    dict or None
    """
    if dataset.family == 'volume' and dataset.mesh is not None:
        return volume_figure(dataset)
    if dataset.surface is not None:
        return map_figure(dataset)
    if dataset.series:
        return lines_figure(dataset)
    return None
