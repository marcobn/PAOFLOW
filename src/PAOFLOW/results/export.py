"""Exports of a dataset: CSV of the plotted data, a reproducing Python snippet, matplotlib files."""

from __future__ import annotations

import io
import json
import math
import os

import numpy as np

from .model import Dataset


def to_csv(dataset: Dataset) -> str:
    """The plotted data as CSV (one x column per series unless the x values are shared).

    Parameters
    ----------
    dataset : Dataset
        The loaded dataset.

    Returns
    -------
    str
    """
    buffer = io.StringIO()
    if dataset.surface is not None:
        surface = dataset.surface
        buffer.write('# %s\n' % dataset.title)
        buffer.write(
            '%s,%s,%s\n'
            % (
                surface.get('x_label', 'x'),
                surface.get('y_label', 'y'),
                surface.get('z_label', 'z'),
            )
        )
        z = np.asarray(surface['z'])
        for iy, y in enumerate(surface['y']):
            for ix, x in enumerate(surface['x']):
                buffer.write('%.8g,%.8g,%.8g\n' % (x, y, z[iy, ix]))
        return buffer.getvalue()
    if not dataset.series:
        return dataset.text or ''
    buffer.write('# %s\n' % dataset.title)
    columns: list[tuple[str, np.ndarray]] = []
    shared = all(
        len(s.x) == len(dataset.series[0].x)
        and np.allclose(s.x, dataset.series[0].x, equal_nan=True)
        for s in dataset.series
    )
    if shared:
        columns.append((dataset.x_label or 'x', np.asarray(dataset.series[0].x)))
    for series in dataset.series:
        if not shared:
            columns.append(('%s: %s' % (series.name, dataset.x_label or 'x'), np.asarray(series.x)))
        y = np.asarray(series.y)
        if y.ndim == 1:
            columns.append(('%s: %s' % (series.name, dataset.y_label or 'y'), y))
        else:
            for band in range(y.shape[1]):
                columns.append(('%s band %d' % (series.name, band + 1), y[:, band]))
    length = max(len(values) for _, values in columns)
    buffer.write(','.join('"%s"' % name.replace('"', "'") for name, _ in columns) + '\n')
    for row in range(length):
        buffer.write(
            ','.join(
                '%.8g' % values[row] if row < len(values) and np.isfinite(values[row]) else ''
                for _, values in columns
            )
            + '\n'
        )
    return buffer.getvalue()


def python_snippet(dataset: Dataset, root: str, options: dict, limits: dict | None = None) -> str:
    """A script that rebuilds the figure from the files (results API, plus GPAO when it applies).

    Parameters
    ----------
    dataset : Dataset
        The loaded dataset.
    root : str
        Absolute folder the reference paths are relative to.
    options : dict
        The selected options.
    limits : dict, optional
        Axis limits set in the Results tab (:func:`clean_limits`).

    Returns
    -------
    str
    """
    from .catalogue import KINDS

    ref = dataset.ref
    kind = next(k for k in KINDS if k.name == ref.kind)
    shown = {k: v for k, v in options.items() if v not in (None, '', [])}
    lines = [
        '"""%s (rebuilt from the PAOFLOW output files)."""' % dataset.title,
        '',
        'from PAOFLOW.results import ResultsFolder',
        '',
        'results = ResultsFolder(%r)' % root,
        'dataset = results.load(%r, options=%s)' % (ref.id, json.dumps(shown, ensure_ascii=False)),
        "results.save(dataset, 'figure.pdf'%s)        # matplotlib; or results.figure(dataset) -> Plotly spec"
        % (', limits=%r' % clean_limits(limits) if clean_limits(limits) else ''),
    ]
    files = [os.path.join(root, f) for f in ref.files]
    if kind.gpao == 'plot_bands' and dataset.family == 'bands':
        kpath = os.path.join(root, ref.folder, 'kpath_points.txt')
        window = dataset.options.get('window', {}).get('value')
        lines += [
            '',
            '# The same with GPAO (matplotlib):',
            'from PAOFLOW import GPAO',
            '',
            'GPAO.GPAO().plot_bands(%r, sym_points=%r%s)'
            % (
                files if len(files) > 1 else files[0],
                kpath,
                ', y_lim=%r' % (tuple(window),) if window else '',
            ),
        ]
    elif kind.gpao == 'plot_dos' and len(files) == 1:
        lines += [
            '',
            '# The same with GPAO (matplotlib):',
            'from PAOFLOW import GPAO',
            '',
            'GPAO.GPAO().plot_dos(%r)' % files[0],
        ]
    return '\n'.join(lines) + '\n'


def clean_limits(limits: dict | None) -> dict[str, list]:
    """Axis limits of a request: ``{'x': [low, high], 'y': [low, high]}``, ``None`` for automatic.

    Parameters
    ----------
    limits : dict or None
        As sent by the Results tab (numbers, or ``None`` / missing for an automatic end).

    Returns
    -------
    dict
        Only the axes with at least one finite end; ends as floats or ``None``.

    Raises
    ------
    ValueError
        For an end that is not a number, or a low end not below the high one.
    """
    cleaned: dict[str, list] = {}
    for axis in ('x', 'y'):
        pair = (limits or {}).get(axis)
        if not pair:
            continue
        ends = []
        for value in list(pair)[:2]:
            if value is None or value == '':
                ends.append(None)
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError('the %s limits must be finite numbers' % axis)
            ends.append(number)
        ends += [None] * (2 - len(ends))
        if ends[0] is not None and ends[1] is not None and ends[0] >= ends[1]:
            raise ValueError('the %s minimum must be below its maximum' % axis)
        if ends != [None, None]:
            cleaned[axis] = ends
    return cleaned


def matplotlib_figure(
    dataset: Dataset, fmt: str = 'pdf', theme: str = 'light', limits: dict | None = None
) -> bytes:
    """A static figure of the dataset (lines, bands, maps) rendered with matplotlib.

    Parameters
    ----------
    dataset : Dataset
        The loaded dataset.
    fmt : str, optional
        ``pdf``, ``png`` or ``svg``.
    theme : str, optional
        Palette column used for the colours.
    limits : dict, optional
        Axis limits set in the Results tab (:func:`clean_limits`); an end left ``None`` stays
        automatic.

    Returns
    -------
    bytes

    Raises
    ------
    ValueError
        For datasets without a 2D figure (text, arrays, 3D surfaces).
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from .palette import PALETTE, resolve

    if dataset.mesh is not None or (not dataset.series and dataset.surface is None):
        raise ValueError('no 2D figure for this dataset (use the browser view or the CSV)')
    table = PALETTE[theme]
    figure, axes = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    if dataset.surface is not None:
        surface = dataset.surface
        key = 'diverging' if surface.get('diverging') else 'sequential'
        cmap = LinearSegmentedColormap.from_list(key, [c for _, c in table['scales'][key]])
        z = np.asarray(surface['z'], dtype=float)
        kwargs = {}
        if surface.get('diverging'):
            limit = float(np.nanmax(np.abs(z))) or 1.0
            kwargs = {'vmin': -limit, 'vmax': limit}
        mesh = axes.pcolormesh(surface['x'], surface['y'], z, cmap=cmap, shading='auto', **kwargs)
        figure.colorbar(mesh, ax=axes, label=surface.get('z_label', ''))
        axes.set_xlabel(surface.get('x_label', ''))
        axes.set_ylabel(surface.get('y_label', ''))
    stacked = [s for s in dataset.series if 'stack' in s.meta]
    if stacked:
        axes.stackplot(
            stacked[0].x,
            [np.asarray(s.y) for s in stacked],
            labels=[s.name for s in stacked],
            colors=[resolve(s.role, theme) for s in stacked],
            alpha=0.8,
            edgecolor=table['surface'],
            linewidth=0.6,
        )
    for series in dataset.series:
        if 'stack' in series.meta:
            continue
        color = (
            resolve(series.role, theme)
            if not series.role.startswith(('diverging', 'sequential'))
            else table['sequential']
        )
        style = {'dash': '--', 'dot': ':'}.get(series.dash, '-')
        y = np.asarray(series.y)
        if 'markers' in series.mode and 'color' in series.meta:
            cmap = LinearSegmentedColormap.from_list(
                'c',
                [
                    c
                    for _, c in table['scales'][
                        'diverging' if series.role == 'diverging' else 'sequential'
                    ]
                ],
            )
            points = axes.scatter(series.x, series.y, c=series.meta['color'], s=4, cmap=cmap)
            figure.colorbar(points, ax=axes, label=series.meta.get('color_label', ''))
        elif y.ndim == 2:
            for band in range(y.shape[1]):
                axes.plot(
                    series.x,
                    y[:, band],
                    style,
                    color=color,
                    lw=1.0,
                    label=series.name if band == 0 else None,
                )
        else:
            marker = 'o' if 'markers' in series.mode else None
            line = style if 'lines' in series.mode else 'none'
            axes.plot(
                series.x,
                y,
                linestyle=line,
                marker=marker,
                ms=3,
                color=color,
                lw=1.6,
                label=series.name,
            )
    if dataset.ticks:
        axes.set_xticks([t[0] for t in dataset.ticks], [t[1] for t in dataset.ticks])
        for position, _ in dataset.ticks:
            axes.axvline(position, color=table['axis'], lw=0.8)
        axes.set_xlim(min(t[0] for t in dataset.ticks), max(t[0] for t in dataset.ticks))
    elif dataset.surface is None:
        axes.set_xlabel(dataset.x_label)
    if dataset.surface is None:
        axes.set_ylabel(dataset.y_label)
    window = dataset.options.get('window', {}).get('value')
    if window:
        axes.set_ylim(*window)
    if dataset.options.get('log_y', {}).get('value'):
        axes.set_yscale('log')
    for axis, (low, high) in clean_limits(limits).items():
        (axes.set_xlim if axis == 'x' else axes.set_ylim)(low, high)
    if dataset.zero == 'y':
        axes.axhline(0.0, color=table['neutral'], lw=0.8, ls=':')
    elif dataset.zero == 'x':
        axes.axvline(0.0, color=table['neutral'], lw=0.8, ls=':')
    if len({s.name for s in dataset.series}) > 1:
        axes.legend(frameon=False, fontsize=8)
    axes.set_title(dataset.title, loc='left', fontsize=11)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format=fmt)
    plt.close(figure)
    return buffer.getvalue()
