"""BXSF volumes for the browser: Fermi-surface meshes and slices of scalar fields.

The geometry is the one of ``fermi-plotter`` (:mod:`PAOFLOW.gen.fermi_plotter`): marching cubes at
the Fermi level of each band, Cartesian vertices from the reciprocal vectors, the Fermi velocity
:math:`|\\nabla_{\\mathbf k} E|` interpolated on the vertices, and the sheets folded into the
Wigner-Seitz zone whose edges are drawn.  The result is plain arrays (vertices, triangles,
values, zone edges) that Plotly's ``mesh3d`` renders, so no Mayavi or VTK is needed.  Scalar
fields that are not band energies (the Berry curvature written by the Hall runs) are shown as
plane slices.
"""

from __future__ import annotations

import os

import numpy as np

from .model import Dataset, DatasetRef

#: Triangles sent to the browser at most.  A larger mesh is computed again by marching cubes on
#: every second (third, ...) grid point: larger triangles, the sheets stay closed.
MAX_FACES = 400000


def _require_skimage() -> None:
    try:
        import skimage  # noqa: F401
    except ImportError:
        raise ImportError(
            'the Fermi-surface view needs scikit-image: pip install "PAOFLOW[fermisurface]"'
        ) from None


def weld_vertices(
    xyz: np.ndarray, faces: np.ndarray, scalars: np.ndarray, decimals: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge the vertices at the same position (the fold into the zone gives three per face).

    Shared vertices let the browser shade the sheet smoothly and cut the vertex data about
    five-fold.  Faces left with a repeated vertex (slivers of the clipping) are dropped.

    Parameters
    ----------
    xyz : ndarray, shape (nv, 3)
        Vertices.
    faces : ndarray, shape (nf, 3)
        Vertex indices of the triangles.
    scalars : ndarray, shape (nv,)
        Per-vertex values (equal at equal positions).
    decimals : int, optional
        Positions equal to this many decimals are one vertex.

    Returns
    -------
    tuple of ndarray
        ``(xyz, faces, scalars)`` with distinct vertices.
    """
    if not len(xyz):
        return xyz, faces, scalars
    _unique, first, inverse = np.unique(
        np.round(xyz, decimals), axis=0, return_index=True, return_inverse=True
    )
    faces = inverse.reshape(-1)[faces]
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    return xyz[first], faces[keep], scalars[first]


def fermi_mesh(
    paths: list[str],
    bands: list[int] | None = None,
    interp: int = 1,
    shift: float = 0.0,
    zone: str = 'ws',
    skeaf: bool = False,
) -> dict:
    """Fermi sheets of BXSF files as triangle meshes.

    Parameters
    ----------
    paths : list of str
        One multi-band ``FermiSurf_<s>.bxsf`` or several single-band SKEAF sheets.
    bands : list of int, optional
        Band labels to draw (default: every band crossing the level; an empty list draws none).
    interp : int, optional
        Fourier upsampling factor of the grids (smoother sheets).
    shift : float, optional
        Offset (eV, or the file's energy unit) of the iso-level from the Fermi energy.
    zone : str, optional
        ``'ws'`` folds the sheets into the first Brillouin zone and draws it; ``'cell'`` keeps
        the reciprocal parallelepiped.
    skeaf : bool, optional
        The files are the SKEAF inputs ``Fermi_surf_band_<n>.bxsf``
        (:func:`PAOFLOW.writers.write2bxsf4skeaf.write2bxsf4skeaf`): energies in Rydberg and
        reciprocal vectors without the :math:`2\\pi`.  They are brought to eV and to the
        vectors of ``FermiSurf_<s>.bxsf``, and the bands take the label of their ``BAND`` line
        (``Fermi_surf_band_1.bxsf`` holds band 2 when the first band is below the window), so
        both kinds of file give the same picture.

    Returns
    -------
    dict
        ``sheets`` (``band``, ``x``, ``y``, ``z``, ``i``, ``j``, ``k``, ``speed``), ``edges``
        (segments), ``available`` (crossing band labels), ``speed_range``, ``fermi_energy`` and
        ``step`` (the marching-cubes step: 1, or more when the mesh would exceed
        :data:`MAX_FACES`).
    """
    _require_skimage()
    from scipy.ndimage import map_coordinates
    from skimage import measure

    from ..gen.fermi_plotter import (
        brillouin_zone,
        bz_edges,
        bz_planes,
        fermi_velocity_field,
        fft_upsample,
        fold_into_bz,
        read_fermi_bxsf,
    )

    grids: dict[int, np.ndarray] = {}
    recip = None
    fermi = 0.0
    for path in paths:
        data = read_fermi_bxsf(path, skeaf=skeaf)  # eV and bohr⁻¹ with 2π, whatever the file
        recip = np.asarray(data.recip, dtype=float) if recip is None else recip
        fermi = data.fermi_energy
        label_offset = 0
        if len(paths) > 1:
            digits = ''.join(ch for ch in os.path.basename(path) if ch.isdigit())
            label_offset = int(digits) if digits else len(grids) + 1
        for label, grid in data.bands.items():
            # SKEAF files: the band of the BAND line, unless two files claim the same one.
            key = label if skeaf and label not in grids else label_offset or label
            grids[key] = grid
    level = fermi + shift
    available = [label for label, grid in grids.items() if grid.min() <= level <= grid.max()]
    header_ignored = False
    if not available and fermi != 0.0:
        # Older writers put the absolute Fermi energy in the header of grids already shifted to it.
        crossing = [label for label, grid in grids.items() if grid.min() <= shift <= grid.max()]
        if crossing:
            fermi, level, available, header_ignored = 0.0, shift, crossing, True
    chosen = [b for b in (available if bands is None else bands) if b in available]
    planes = None
    edges: list[list[float]] = []
    if zone == 'ws':
        vertices, faces = brillouin_zone(recip)
        planes = bz_planes(vertices, faces)
        edges = [list(a) + list(b) for a, b in bz_edges(vertices, faces)]
    else:
        corners = (
            np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
            @ recip
        )
        for a in range(8):
            for b in range(a + 1, 8):
                if bin(a ^ b).count('1') == 1:
                    edges.append(list(corners[a]) + list(corners[b]))
    energies = {
        label: fft_upsample(grids[label], interp) if interp > 1 else grids[label]
        for label in chosen
    }

    def surfaces(step: int) -> dict:
        found = {}
        for label, energy in energies.items():
            try:
                found[label] = measure.marching_cubes(energy, level=level, step_size=step)[:2]
            except (ValueError, RuntimeError):
                continue
        return found

    # A mesh too large for the browser is computed again on a coarser step: marching cubes on
    # every step-th grid point gives larger triangles and closed sheets.  (Dropping faces at
    # random, as before 11 November 2026, punched holes through which the backs showed.)
    step = 1
    meshes = surfaces(step)
    total_faces = sum(len(faces) for _verts, faces in meshes.values())
    while total_faces > MAX_FACES:
        step = max(step + 1, int(np.ceil(np.sqrt(total_faces / MAX_FACES) * step)))
        meshes = surfaces(step)
        total_faces = sum(len(faces) for _verts, faces in meshes.values())
    sheets = []
    for label, (verts, faces) in meshes.items():
        energy = energies[label]
        speed = fermi_velocity_field(energy, recip)
        span = np.where(np.array(energy.shape) > 1, np.array(energy.shape) - 1, 1)
        cart = (verts / span) @ recip
        vspeed = map_coordinates(speed, verts.T, order=1, mode='nearest')
        if planes is not None:
            cart, faces, vspeed = fold_into_bz(cart, faces, vspeed, recip, *planes)
            cart, faces, vspeed = weld_vertices(cart, faces, vspeed)
        sheets.append({'band': int(label), 'xyz': cart, 'faces': faces, 'speed': vspeed})
    speeds = [s['speed'] for s in sheets if len(s['speed'])]
    speed_range = (
        [float(min(s.min() for s in speeds)), float(max(s.max() for s in speeds))]
        if speeds
        else [0.0, 1.0]
    )
    return {
        'sheets': [
            {
                'band': sheet['band'],
                'x': np.round(sheet['xyz'][:, 0], 5),
                'y': np.round(sheet['xyz'][:, 1], 5),
                'z': np.round(sheet['xyz'][:, 2], 5),
                'i': sheet['faces'][:, 0],
                'j': sheet['faces'][:, 1],
                'k': sheet['faces'][:, 2],
                'speed': np.round(sheet['speed'], 5),
            }
            for sheet in sheets
        ],
        'edges': edges,
        'available': sorted(int(b) for b in available),
        'speed_range': speed_range,
        'fermi_energy': fermi,
        'step': step,
        'header_ignored': header_ignored,
    }


def scalar_slice(
    path: str, axis: int = 2, index: int | None = None, band: int | None = None
) -> dict:
    """A plane of a BXSF scalar field (the Berry curvature of the Hall runs).

    Parameters
    ----------
    path : str
        The BXSF file.
    axis : int, optional
        Grid axis normal to the plane (0, 1, 2).
    index : int, optional
        Plane index (default: the middle plane).
    band : int, optional
        Band label (default: the first).

    Returns
    -------
    dict
        ``x``, ``y`` (fractional coordinates), ``z`` (2D values), ``index``, ``count``.
    """
    import contextlib
    import io

    from ..gen.fermi_plotter import read_fermi_bxsf

    with contextlib.redirect_stderr(io.StringIO()):  # the 2-band Berry files trigger a warning
        data = read_fermi_bxsf(path)
    label = band if band in data.bands else sorted(data.bands)[0]
    grid = data.bands[label]
    count = grid.shape[axis]
    index = count // 2 if index is None else max(0, min(count - 1, index))
    plane = np.take(grid, index, axis=axis)
    rows, cols = plane.shape
    return {
        'x': np.linspace(0.0, 1.0, cols),
        'y': np.linspace(0.0, 1.0, rows),
        'z': plane,
        'index': index,
        'count': count,
    }


def load_volume(ref: DatasetRef, root: str, options: dict, dataset: Dataset) -> Dataset:
    """Dataset of a BXSF reference: a Fermi-surface mesh or a field slice.

    Parameters
    ----------
    ref : DatasetRef
        The reference (``fermi_surface``, ``fermi_sheet`` or ``berry_bxsf``).
    root : str
        Root of the reference paths.
    options : dict
        ``bands``, ``interp``, ``zone``, ``shift`` (meshes); ``axis``, ``index`` (slices).
    dataset : Dataset
        Empty dataset to fill.

    Returns
    -------
    Dataset
    """
    paths = [os.path.join(root, f) for f in ref.files]
    if ref.kind == 'berry_bxsf':
        axes = ['k1', 'k2', 'k3']
        axis = axes.index(options.get('axis', 'k3')) if options.get('axis', 'k3') in axes else 2
        result = scalar_slice(paths[0], axis=axis, index=options.get('index'))
        dataset.options['axis'] = {'label': 'Plane normal', 'choices': axes, 'value': axes[axis]}
        dataset.options['index'] = {
            'label': 'Plane index',
            'range': [0, result['count'] - 1],
            'value': result['index'],
            'integer': True,
        }
        others = [a for a in axes if a != axes[axis]]
        dataset.surface = {
            'x': result['x'],
            'y': result['y'],
            'z': result['z'],
            'x_label': '%s (crystal)' % others[1],
            'y_label': '%s (crystal)' % others[0],
            'z_label': 'Ω',
            'diverging': True,
        }
        dataset.family = 'map'
        return dataset
    try:
        interp = int(options.get('interp', 1))
    except (TypeError, ValueError):
        interp = 1
    interp = max(1, min(3, interp))
    zone = options.get('zone', 'ws') if options.get('zone', 'ws') in ('ws', 'cell') else 'ws'
    try:
        shift = float(options.get('shift', 0.0) or 0.0)
    except (TypeError, ValueError):
        shift = 0.0
    bands = options.get('bands')
    try:
        mesh = fermi_mesh(
            paths,
            bands=bands if isinstance(bands, list) else None,
            interp=interp,
            shift=shift,
            zone=zone,
            skeaf=ref.kind == 'fermi_sheet',
        )
    except ImportError as exc:
        dataset.notes.append(str(exc))
        dataset.family = 'text'
        dataset.text = str(exc)
        return dataset
    dataset.mesh = mesh
    dataset.family = 'volume'
    shown = [s['band'] for s in mesh['sheets']]
    dataset.options['bands'] = {
        'label': 'Bands',
        'choices': mesh['available'],
        'value': shown,
        'multiple': True,
    }
    dataset.options['interp'] = {
        'label': 'Smoothing (Fourier ×)',
        'choices': [1, 2, 3],
        'value': interp,
    }
    dataset.options['zone'] = {'label': 'Zone', 'choices': ['ws', 'cell'], 'value': zone}
    dataset.options['shift'] = {'label': 'Level − E_F', 'value': shift, 'number': True}
    if ref.kind == 'fermi_sheet':
        dataset.notes.append(
            'pyskeaf input files: energies converted from Rydberg to eV and reciprocal vectors '
            'multiplied by 2π, as in FermiSurf_<s>.bxsf; the bands are those of the BAND lines '
            '(the number of a file name is its position, not its band).'
        )
    if not mesh['available']:
        dataset.notes.append('No band crosses the chosen level (move Level − E_F into a band).')
    elif not mesh['sheets']:
        dataset.notes.append('No band selected.')
    if mesh.get('header_ignored'):
        dataset.notes.append(
            'The BXSF header gives an absolute Fermi energy but the grid is already relative to '
            'it (older writer): the sheets are drawn at 0.'
        )
    if mesh['step'] > 1:
        effective = interp / mesh['step']
        dataset.notes.append(
            'Large mesh: the sheets are drawn on every %s grid point of the smoothed grid, so the '
            'browser gets at most %d triangles%s.'
            % (
                {2: 'second', 3: 'third'}.get(mesh['step'], '%dth' % mesh['step']),
                MAX_FACES,
                ''
                if effective > 1
                else (
                    ': this is the original grid, so the smoothing shows nothing here'
                    if effective == 1
                    else ': coarser than the original grid'
                ),
            )
        )
    dataset.table.append(
        {'name': 'bands crossing', 'value': ', '.join(map(str, mesh['available'])), 'unit': ''}
    )
    dataset.table.append(
        {'name': '|v_F| range', 'value': '%.3g – %.3g' % tuple(mesh['speed_range']), 'unit': 'a.u.'}
    )
    return dataset
