"""Band paths for the generator: the default path of a lattice and the geometry of a custom one.

The default path is the one ``PAOFLOW.bands`` follows without an explicit path
(:func:`PAOFLOW.spectrum.kpnts_interpolation_mesh._getHighSymPoints`, from the ``ibrav`` and the
cell of the QE save) or, for slabs, the in-plane path of :mod:`PAOFLOW.gen.paoflow_driver`.  The
GUI starts its band-path editor from it and previews the path in the first Brillouin zone.
"""

from __future__ import annotations

import os

import numpy as np

Point = tuple[str, float, float, float]


def default_band_path(
    workdir: str, savedir: str | None, ibrav: int, is_2d: bool
) -> tuple[str, list[Point]]:
    """The path ``PAOFLOW.bands`` follows by default, as a path string and labelled points.

    Parameters
    ----------
    workdir : str
        Working directory.
    savedir : str or None
        The ``<prefix>.save`` folder (relative to ``workdir``); its cell gives the points.
    ibrav : int
        QE Bravais-lattice index (0: no default path).
    is_2d : bool
        Slab: the in-plane path (``kz = 0``).

    Returns
    -------
    band_path : str
        ``'gG-X-W-K-gG-L-U-W-L-K|U-X'``-style string (empty when there is no default).
    points : list of (str, float, float, float)
        Labels and crystal coordinates (in the order of first appearance on the path).
    """
    if not ibrav:
        return '', []
    if is_2d:
        from .paoflow_driver import _band_path_2d

        band_path, points = _band_path_2d(ibrav)
        if band_path is None:
            return '', []
        return band_path, [(label, *map(float, k)) for label, k in points.items()]
    cell = _cell_in_alat(workdir, savedir)
    if cell is None:
        return '', []
    from ..spectrum.kpnts_interpolation_mesh import _getHighSymPoints

    alat, vectors = cell
    try:
        points, band_path = _getHighSymPoints(ibrav, alat, vectors)
    except Exception:  # unsupported lattice or degenerate cell
        return '', []
    order = []
    for label in band_path.replace('|', '-').split('-'):
        if label in points and label not in order:
            order.append(label)
    order += [label for label in points if label not in order]
    return band_path, [(label, *(round(float(v), 10) for v in points[label])) for label in order]


def _cell_in_alat(workdir: str, savedir: str | None) -> tuple[float, np.ndarray] | None:
    from .system_summary import read_save_summary

    if not savedir:
        return None
    summary = read_save_summary(os.path.join(workdir, savedir))
    if summary is None or not summary.get('alat_bohr'):
        return None
    alat = float(summary['alat_bohr'])
    return alat, np.asarray(summary['cell_bohr'], dtype=float) / alat


def path_labels(band_path: str) -> list[str]:
    """Labels of a path string (``'G-X|M-R'`` -> ``['G', 'X', 'M', 'R']``).

    Parameters
    ----------
    band_path : str
        The path.

    Returns
    -------
    list of str
    """
    return [label for segment in band_path.split('|') for label in segment.split('-') if label]


def check_band_path(band_path: str, points: list[Point]) -> str | None:
    """Why a custom path cannot be used, or ``None``.

    Parameters
    ----------
    band_path : str
        The path string.
    points : list of (str, float, float, float)
        Labelled crystal coordinates.

    Returns
    -------
    str or None
    """
    if not band_path.strip():
        return 'give the path, e.g. gG-X-W-K-gG-L'
    known = {p[0] for p in points}
    for segment in band_path.split('|'):
        labels = [label for label in segment.split('-')]
        if len(labels) < 2 or any(not label for label in labels):
            return 'each segment needs two or more labels joined by "-" (segments split by "|")'
    missing = [label for label in path_labels(band_path) if label not in known]
    if missing:
        return 'no coordinates for %s' % ', '.join(sorted(set(missing)))
    return None


def camera_eye(points: np.ndarray, distance: float = 1.6) -> dict:
    """Viewing direction of the zone that keeps the labelled points apart.

    A fixed camera hides a point behind another on some lattice (fcc seen along about [111] puts
    Γ under L). Among elevations of 15-75 degrees and azimuths in steps of 15 degrees, the direction
    with the largest smallest distance between the points projected on the screen plane is chosen.

    Parameters
    ----------
    points : ndarray, shape (n, 3)
        Cartesian coordinates of the labelled points.
    distance : float, optional
        Distance of the eye in Plotly scene units.

    Returns
    -------
    dict
        Plotly ``scene.camera.eye`` (``x``, ``y``, ``z``).
    """
    points = np.asarray(points, dtype=float)
    best_score, best = -1.0, np.array([1.05, 1.05, 0.7]) / np.linalg.norm([1.05, 1.05, 0.7])
    if len(points) > 1:
        pairs = np.triu_indices(len(points), 1)
        for elevation in np.radians([15, 30, 45, 60, 75]):
            for azimuth in np.radians(np.arange(0, 360, 15)):
                direction = np.array(
                    [
                        np.cos(elevation) * np.cos(azimuth),
                        np.cos(elevation) * np.sin(azimuth),
                        np.sin(elevation),
                    ]
                )
                projected = points - np.outer(points @ direction, direction)
                gaps = np.linalg.norm(projected[pairs[0]] - projected[pairs[1]], axis=1)
                if gaps.min() > best_score + 1e-9:
                    best_score, best = float(gaps.min()), direction
    eye = distance * best
    return {
        'x': round(float(eye[0]), 3),
        'y': round(float(eye[1]), 3),
        'z': round(float(eye[2]), 3),
    }


def path_figure(workdir: str, savedir: str | None, band_path: str, points: list[Point]) -> dict:
    """Plotly specification of the first Brillouin zone with the path (crystal points to Cartesian).

    Parameters
    ----------
    workdir : str
        Working directory.
    savedir : str or None
        The save (its cell gives the zone).
    band_path : str
        The path string.
    points : list of (str, float, float, float)
        Labelled crystal coordinates.

    Returns
    -------
    dict
        ``figure`` (Plotly specification with colour roles) and ``length`` (path length in units of
        ``2π/alat``), or ``error``.
    """
    from .fermi_plotter import brillouin_zone, bz_edges

    cell = _cell_in_alat(workdir, savedir)
    if cell is None:
        return {
            'error': 'the zone needs the cell of a QE save (<prefix>.save) in the working folder: '
            'run pw.x first, or choose the save above'
        }
    problem = check_band_path(band_path, points)
    if problem:
        return {'error': problem}
    _alat, vectors = cell
    reciprocal = np.linalg.inv(vectors).T  # rows b_i, units of 2π/alat (a_i · b_j = δ_ij)
    coordinates = {p[0]: np.asarray(p[1:], dtype=float) @ reciprocal for p in points}
    zone_vertices, zone_faces = brillouin_zone(reciprocal)
    xs, ys, zs = [], [], []
    for start, end in bz_edges(zone_vertices, zone_faces):
        xs += [float(start[0]), float(end[0]), None]
        ys += [float(start[1]), float(end[1]), None]
        zs += [float(start[2]), float(end[2]), None]
    px, py, pz = [], [], []
    length = 0.0
    for segment in band_path.split('|'):
        labels = segment.split('-')
        for index, label in enumerate(labels):
            k = coordinates[label]
            px.append(float(k[0]))
            py.append(float(k[1]))
            pz.append(float(k[2]))
            if index:
                length += float(np.linalg.norm(k - coordinates[labels[index - 1]]))
        px.append(None)
        py.append(None)
        pz.append(None)
    used = list(dict.fromkeys(path_labels(band_path)))
    names = ['Γ' if label in ('G', 'gG', 'Gamma') else label for label in used]
    axis = {
        'showbackground': False,
        'showgrid': False,
        'zeroline': False,
        'showticklabels': False,
        'title': {'text': ''},
    }
    figure = {
        'data': [
            {
                'type': 'scatter3d',
                'mode': 'lines',
                'name': 'Brillouin zone',
                'x': xs,
                'y': ys,
                'z': zs,
                'line': {'color': 'axis', 'width': 3},
                'hoverinfo': 'skip',
            },
            {
                'type': 'scatter3d',
                'mode': 'lines',
                'name': 'path',
                'x': px,
                'y': py,
                'z': pz,
                'line': {'color': 'series-1', 'width': 6},
                'hoverinfo': 'skip',
            },
            {
                'type': 'scatter3d',
                'mode': 'markers+text',
                'name': 'points',
                'x': [float(coordinates[label][0]) for label in used],
                'y': [float(coordinates[label][1]) for label in used],
                'z': [float(coordinates[label][2]) for label in used],
                'text': names,
                'textposition': 'top center',
                'marker': {'size': 4, 'color': 'series-2'},
                'hovertemplate': '%{text}<extra></extra>',
            },
        ],
        'layout': {
            'scene': {
                'xaxis': dict(axis),
                'yaxis': dict(axis),
                'zaxis': dict(axis),
                'aspectmode': 'data',
                'camera': {'eye': camera_eye(np.array([coordinates[label] for label in used]))},
            },
            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
            'showlegend': False,
        },
    }
    return {'figure': figure, 'length': length}
