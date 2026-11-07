"""Plotly figure specifications of datasets (plain dictionaries, no plotting library needed).

Colours are *roles* (``series-1`` ... ``series-8``, ``seq-450``, ``neutral``) and colour scales are
``role:sequential`` / ``role:diverging``: the browser (and the static report) resolves them for
the light or dark theme, so a specification does not depend on the theme.  Marks follow the
GUI's chart rules: 2 px lines, hairline axes and grids, a legend for two or more series, labelled
high-symmetry ticks, a reference line at the Fermi level of energy axes, and no second y axis
(quantities of different units are separate figures through the dataset options).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import Dataset, Series

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
