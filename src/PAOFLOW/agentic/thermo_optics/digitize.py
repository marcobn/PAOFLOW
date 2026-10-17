"""Digitising figures: axis calibration, frame/tick detection, masks, markers, curves.

Procedure for a raster figure (scan, embedded bitmap, slide image):

1. Load it (:func:`load_image`) and find the plot frame (:func:`find_frame`).
2. Locate the ticks along the frame (:func:`find_ticks`) and pair two or more of
   them with their printed values (read from the figure) to build an
   :class:`AxisCalibration` per axis (``linear`` or ``log``).
3. Isolate the data: a colour mask (:func:`color_mask`) or a dark-ink mask
   (:func:`dark_mask`), minus the frame and text regions (:func:`erase_box`).
4. Extract points: marker centroids (:func:`marker_centroids`, :func:`ring_markers`
   for open circles, :func:`filled_markers` for dots on lines) for symbols,
   :func:`trace_curve` (single-valued lines) or :func:`trace_path` (any slope)
   for solid lines, :func:`link_dashes` for closely packed dashed lines, or
   :func:`column_profile` for an isolated single curve.
5. Validate: map known anchors (tick positions, values quoted in the text) back
   and quote the deviation (:func:`calibration_residuals`).

The same :class:`AxisCalibration` maps PDF vector coordinates
(:func:`~PAOFLOW.agentic.thermo_optics.ingest.pdf_vector_polylines`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class Axis:
    """One axis: two or more ``pixel <-> value`` anchors, linear or logarithmic.

    Parameters
    ----------
    pixels : sequence of float
        Pixel (or PDF-point) coordinates of the anchors along this axis.
    values : sequence of float
        Data values of the anchors.
    scale : {'linear', 'log'}
        Axis scale.
    piecewise : bool, optional
        Interpolate linearly (in ``value`` or ``log10(value)``) between
        consecutive anchors instead of fitting one line, for hand-drawn or
        broken axes whose tick spacing changes along the axis.  Extrapolates
        with the end segments.

    Notes
    -----
    With more than two anchors the map is a least-squares line in
    ``value`` (linear) or ``log10(value)`` (log); :meth:`residuals` reports how
    well the anchors fit it (a large residual means a misread tick).  With
    ``piecewise=True`` the anchors are exact; :meth:`residuals` then reports
    the deviation from the global line, to show where the drawn axis departs
    from its nominal scale.
    """

    pixels: Sequence[float]
    values: Sequence[float]
    scale: str = 'linear'
    piecewise: bool = False

    def __post_init__(self) -> None:
        if self.scale not in ('linear', 'log'):
            raise ValueError("scale must be 'linear' or 'log'")
        pixels = np.asarray(self.pixels, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if pixels.size < 2 or pixels.size != values.size:
            raise ValueError('need at least two pixel/value anchors of equal count')
        target = np.log10(values) if self.scale == 'log' else values
        self._slope, self._intercept = np.polyfit(pixels, target, 1)
        order = np.argsort(pixels)
        self._pixels, self._target = pixels[order], target[order]
        if self.piecewise and np.any(np.diff(self._target) * np.sign(self._slope) <= 0):
            raise ValueError('piecewise anchors must be strictly monotonic')

    @staticmethod
    def _interp(x: NDArray, xp: NDArray, fp: NDArray) -> NDArray[np.float64]:
        """Piecewise-linear map with linear extrapolation by the end segments."""
        out = np.interp(x, xp, fp)
        low, high = x < xp[0], x > xp[-1]
        out[low] = fp[0] + (x[low] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        out[high] = fp[-1] + (x[high] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return out

    def to_value(self, pixel: ArrayLike) -> NDArray[np.float64]:
        """Data value at pixel coordinates."""
        pixel = np.asarray(pixel, dtype=float)
        if self.piecewise:
            linear = self._interp(np.atleast_1d(pixel), self._pixels, self._target).reshape(
                pixel.shape
            )
        else:
            linear = self._slope * pixel + self._intercept
        return 10.0**linear if self.scale == 'log' else linear

    def to_pixel(self, value: ArrayLike) -> NDArray[np.float64]:
        """Pixel coordinate of data values."""
        value = np.asarray(value, dtype=float)
        target = np.log10(value) if self.scale == 'log' else value
        if self.piecewise:
            order = np.argsort(self._target)
            flat = self._interp(np.atleast_1d(target), self._target[order], self._pixels[order])
            return flat.reshape(target.shape)
        return (target - self._intercept) / self._slope

    def residuals(self) -> NDArray[np.float64]:
        """Anchor residuals in pixels: global line minus given anchors."""
        target = (
            np.log10(np.asarray(self.values, dtype=float))
            if self.scale == 'log'
            else np.asarray(self.values, dtype=float)
        )
        return (target - self._intercept) / self._slope - np.asarray(self.pixels, dtype=float)


@dataclass
class AxisCalibration:
    """Pixel <-> data map of a 2-D plot (x along columns, y along rows)."""

    x: Axis
    y: Axis

    def to_data(self, points: ArrayLike) -> NDArray[np.float64]:
        """``(n, 2)`` pixel ``(col, row)`` -> data ``(x, y)``."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return np.column_stack([self.x.to_value(points[:, 0]), self.y.to_value(points[:, 1])])

    def to_pixels(self, data: ArrayLike) -> NDArray[np.float64]:
        """``(n, 2)`` data ``(x, y)`` -> pixel ``(col, row)``."""
        data = np.atleast_2d(np.asarray(data, dtype=float))
        return np.column_stack([self.x.to_pixel(data[:, 0]), self.y.to_pixel(data[:, 1])])


def calibration_residuals(calibration: AxisCalibration) -> dict[str, NDArray[np.float64]]:
    """Anchor residuals (pixels) of both axes."""
    return {'x': calibration.x.residuals(), 'y': calibration.y.residuals()}


def load_image(path: str | Path) -> NDArray[np.float64]:
    """RGB image as floats in [0, 1], shape ``(rows, cols, 3)`` (alpha dropped)."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=float) / 255.0


def dark_mask(image: NDArray, threshold: float = 0.45) -> NDArray[np.bool_]:
    """Pixels darker than ``threshold`` (mean of RGB): black ink of scans and line art."""
    return image.mean(axis=2) < threshold


def color_mask(
    image: NDArray, rgb: Sequence[float], tolerance: float = 0.18, min_saturation: float = 0.0
) -> NDArray[np.bool_]:
    """Pixels within ``tolerance`` (Euclidean in RGB [0, 1]) of a colour.

    ``min_saturation`` (max - min channel) drops gray antialiasing of black lines.
    """
    distance = np.linalg.norm(image - np.asarray(rgb, dtype=float)[None, None, :], axis=2)
    saturation = image.max(axis=2) - image.min(axis=2)
    return (distance < tolerance) & (saturation >= min_saturation)


def erase_box(
    mask: NDArray[np.bool_], rows: tuple[int, int], cols: tuple[int, int]
) -> NDArray[np.bool_]:
    """Copy of a mask with the box ``rows=(r0, r1), cols=(c0, c1)`` cleared (labels, legends)."""
    out = mask.copy()
    out[rows[0] : rows[1], cols[0] : cols[1]] = False
    return out


def find_frame(
    mask: NDArray[np.bool_], min_fraction: float = 0.5, skew_tolerance: int = 0
) -> dict[str, int]:
    """Bounding frame of a plot: the outermost long dark rows and columns.

    Parameters
    ----------
    mask : ndarray of bool
        Ink mask (:func:`dark_mask`).
    min_fraction : float, optional
        A frame line covers at least this fraction of the image width/height.
    skew_tolerance : int, optional
        Half-width (pixels) of the band a frame line may wander over (skewed
        scans): the mask is dilated by it across each line before projecting,
        and the returned position is the centre of the band.

    Returns
    -------
    dict
        ``left``, ``right``, ``top``, ``bottom`` (pixel indices of the frame lines).

    Raises
    ------
    ValueError
        If fewer than two long lines are found in either direction.
    """
    horizontal, vertical = mask, mask
    if skew_tolerance > 0:
        from scipy import ndimage

        size = 2 * skew_tolerance + 1
        horizontal = ndimage.binary_dilation(mask, structure=np.ones((size, 1), dtype=bool))
        vertical = ndimage.binary_dilation(mask, structure=np.ones((1, size), dtype=bool))
    rows = np.nonzero(horizontal.sum(axis=1) > min_fraction * mask.shape[1])[0]
    cols = np.nonzero(vertical.sum(axis=0) > min_fraction * mask.shape[0])[0]
    if rows.size < 2 or cols.size < 2:
        raise ValueError('frame not found: lower min_fraction or crop the image to the plot')

    def ends(lines):  # first and last runs of qualifying lines: centre of each band
        runs = np.split(lines, np.nonzero(np.diff(lines) > 1)[0] + 1)
        return int(round(runs[0].mean())), int(round(runs[-1].mean()))

    if skew_tolerance > 0:
        (top, bottom), (left, right) = ends(rows), ends(cols)
        if top == bottom or left == right:
            raise ValueError('only one frame line found: lower min_fraction or crop the image')
    else:
        top, bottom, left, right = (
            int(rows.min()),
            int(rows.max()),
            int(cols.min()),
            int(cols.max()),
        )
    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}


def find_ticks(
    mask: NDArray[np.bool_], frame: dict[str, int], axis: str, length: int = 4, inside: bool = True
) -> NDArray[np.float64]:
    """Pixel positions of tick marks along the bottom (``'x'``) or left (``'y'``) frame line.

    A tick is a run of ink perpendicular to the frame, ``length`` pixels deep,
    just inside (or outside) it; adjacent tick columns are merged into their
    centre.
    """
    if axis == 'x':
        line = frame['bottom']
        band = (
            mask[line - length - 1 : line - 1, :]
            if inside
            else mask[line + 2 : line + length + 2, :]
        )
        hits = band.all(axis=0)
        hits[: frame['left'] + 2] = False
        hits[frame['right'] - 1 :] = False
    elif axis == 'y':
        line = frame['left']
        band = (
            mask[:, line + 2 : line + length + 2]
            if inside
            else mask[:, line - length - 1 : line - 1]
        )
        hits = band.all(axis=1)
        hits[: frame['top'] + 2] = False
        hits[frame['bottom'] - 1 :] = False
    else:
        raise ValueError("axis must be 'x' or 'y'")
    positions = np.nonzero(hits)[0]
    if positions.size == 0:
        return positions.astype(float)
    groups = np.split(positions, np.nonzero(np.diff(positions) > 1)[0] + 1)
    return np.array([group.mean() for group in groups])


def marker_centroids(
    mask: NDArray[np.bool_], min_area: int = 6, max_area: int = 400, fill_holes: bool = True
) -> NDArray[np.float64]:
    """Centroids ``(col, row)`` of connected blobs of a mask (markers: dots, circles, squares).

    Open symbols (circles) are filled first so their centre is the centroid.
    Blobs outside ``[min_area, max_area]`` pixels (text, lines) are dropped.
    """
    from scipy import ndimage

    work = ndimage.binary_fill_holes(mask) if fill_holes else mask
    labels, count = ndimage.label(work)
    if count == 0:
        return np.empty((0, 2))
    areas = ndimage.sum(work, labels, index=np.arange(1, count + 1))
    keep = np.nonzero((areas >= min_area) & (areas <= max_area))[0] + 1
    centres = ndimage.center_of_mass(work, labels, keep)
    return np.array([(c[1], c[0]) for c in centres]) if centres else np.empty((0, 2))


def filled_markers(
    mask: NDArray[np.bool_], radius: float, min_area: int = 20, max_area: int = 2000
) -> NDArray[np.float64]:
    """Centroids ``(col, row)`` of filled symbols sitting on lines (dots on fitted curves).

    Parameters
    ----------
    mask : ndarray of bool
        Ink mask.
    radius : float
        Radius of the disk used for a morphological opening: larger than half
        the line width, smaller than the marker radius.
    min_area, max_area : int, optional
        Area range (pixels, after the opening) of a marker.

    Notes
    -----
    Holes are filled first (open or scanned-hollow symbols), then the opening
    erases every stroke thinner than ``2 * radius`` -- lines, axes, most
    text -- and leaves the symbols.
    """
    from scipy import ndimage

    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = xx**2 + yy**2 <= radius**2
    work = ndimage.binary_opening(ndimage.binary_fill_holes(mask), structure=disk)
    labels, count = ndimage.label(work)
    if count == 0:
        return np.empty((0, 2))
    areas = ndimage.sum(work, labels, index=np.arange(1, count + 1))
    keep = np.nonzero((areas >= min_area) & (areas <= max_area))[0] + 1
    centres = ndimage.center_of_mass(work, labels, keep)
    return np.array([(c[1], c[0]) for c in centres]) if centres else np.empty((0, 2))


def ring_markers(
    mask: NDArray[np.bool_],
    radius: float,
    width: float = 2.5,
    min_score: float = 0.45,
    min_distance: float | None = None,
) -> NDArray[np.float64]:
    """Centres ``(col, row)`` of open circular markers, also broken or overlapping ones.

    Parameters
    ----------
    mask : ndarray of bool
        Mask of the marker colour.
    radius : float
        Ring radius (pixels, centre of the stroke).
    width : float, optional
        Stroke width (pixels).
    min_score : float, optional
        Minimum fraction of the annulus covered by ink at a centre.
    min_distance : float, optional
        Minimum distance between centres (default ``radius``).

    Returns
    -------
    ndarray, shape ``(n, 2)``

    Notes
    -----
    The mask is correlated with an annulus template; local maxima of the
    coverage above ``min_score`` are the centres.  Unlike blob centroids this is
    insensitive to rings broken by compression and separates overlapping
    circles whose centres are at least ``min_distance`` apart.
    """
    from scipy import ndimage

    size = int(np.ceil(radius + width)) + 1
    yy, xx = np.mgrid[-size : size + 1, -size : size + 1]
    distance = np.hypot(xx, yy)
    kernel = (np.abs(distance - radius) <= 0.5 * width).astype(float)
    kernel /= kernel.sum()
    score = ndimage.correlate(mask.astype(float), kernel, mode='constant')
    min_distance = radius if min_distance is None else min_distance
    footprint = np.hypot(*np.mgrid[-int(min_distance) : int(min_distance) + 1,
                                    -int(min_distance) : int(min_distance) + 1]) <= min_distance  # fmt: skip
    peaks = (score == ndimage.maximum_filter(score, footprint=footprint)) & (score >= min_score)
    rows, cols = np.nonzero(peaks)
    candidates = np.column_stack([cols, rows]).astype(float)
    order = np.argsort(-score[rows, cols])
    centres: list[NDArray] = []
    groups: list[list[NDArray]] = []
    for point in candidates[order]:  # merge plateau duplicates closer than min_distance
        for k, centre in enumerate(centres):
            if np.hypot(*(point - centre)) < min_distance:
                groups[k].append(point)
                break
        else:
            centres.append(point)
            groups.append([point])
    return np.array([np.mean(group, axis=0) for group in groups]) if groups else np.empty((0, 2))


def trace_curve(
    mask: NDArray[np.bool_],
    seed: tuple[float, float],
    stop_col: int,
    max_jump: float = 6.0,
    max_gap: int = 12,
    step: int = 1,
) -> NDArray[np.float64]:
    """Follow one curve column by column from a seed pixel ``(col, row)``.

    Parameters
    ----------
    mask : ndarray of bool
        Ink mask of the curves (frame and text erased).
    seed : tuple of float
        A pixel on the wanted curve.
    stop_col : int
        Last column to reach (left or right of the seed).
    max_jump : float, optional
        Largest row change from the extrapolated position accepted per column.
    max_gap : int, optional
        Columns without a matching ink run tolerated (dashes, crossings, labels).
    step : int, optional
        Column step.

    Returns
    -------
    ndarray, shape ``(n, 2)``
        Pixel ``(col, row)`` of the curve (one row per column with ink).

    Notes
    -----
    In each column the ink runs are found; the run whose centre is closest to the
    position extrapolated from the last two accepted points (slope continuity) is
    taken, if within ``max_jump``.  Continuity carries the trace through crossings
    with other curves and through the gaps of dashed lines.
    """
    col, row = float(seed[0]), float(seed[1])
    direction = 1 if stop_col >= col else -1
    points = [(col, row)]
    slope, gap = 0.0, 0
    for current in range(
        int(round(col)) + direction * step, stop_col + direction, direction * step
    ):
        if current < 0 or current >= mask.shape[1]:
            break
        predicted = points[-1][1] + slope * (current - points[-1][0])
        column = mask[:, current]
        runs = np.nonzero(column)[0]
        best = None
        if runs.size:
            groups = np.split(runs, np.nonzero(np.diff(runs) > 1)[0] + 1)
            centres = np.array([g.mean() for g in groups])
            k = int(np.argmin(np.abs(centres - predicted)))
            if abs(centres[k] - predicted) <= max_jump + gap * 0.5:
                best = centres[k]
        if best is None:
            gap += 1
            if gap > max_gap:
                break
            continue
        if len(points) >= 2:
            new_slope = (best - points[-1][1]) / (current - points[-1][0])
            slope = 0.7 * slope + 0.3 * new_slope
        else:
            slope = (best - points[-1][1]) / (current - points[-1][0])
        points.append((float(current), float(best)))
        gap = 0
    return np.array(points)


def trace_path(
    mask: NDArray[np.bool_],
    seed: tuple[float, float],
    direction: tuple[float, float],
    step: float = 3.0,
    max_turn_deg: float = 40.0,
    probe_radius: float = 1.5,
    max_gap_steps: int = 4,
    bounds: tuple[int, int, int, int] | None = None,
    max_steps: int = 5000,
) -> NDArray[np.float64]:
    """Follow a curve of any slope by arc-length steps from a seed ``(col, row)``.

    Parameters
    ----------
    mask : ndarray of bool
        Ink mask of the curves.
    seed : tuple of float
        A pixel on the wanted curve.
    direction : tuple of float
        Initial direction ``(dcol, drow)`` (e.g. ``(1, -1)``: right and up).
    step : float, optional
        Step length (pixels).
    max_turn_deg : float, optional
        Largest change of direction per step.
    probe_radius : float, optional
        Radius of the ink probe at each candidate point.
    max_gap_steps : int, optional
        Steps without ink allowed (dashes, open markers), continued straight.
    bounds : tuple of int, optional
        ``(col_min, col_max, row_min, row_max)`` to stay inside (the plot frame).
    max_steps : int, optional
        Safety limit.

    Returns
    -------
    ndarray, shape ``(n, 2)``
        Pixel ``(col, row)`` points along the curve.

    Notes
    -----
    At each step the candidates ``p + step (cos a, sin a)`` for angles within
    ``max_turn_deg`` of the current direction are probed; among those with ink
    the one with the smallest turn wins, which keeps the trace on its curve
    through crossings and across open markers.  Unlike :func:`trace_curve` it
    handles vertical segments.
    """
    rows, cols = mask.shape
    if bounds is None:
        bounds = (0, cols - 1, 0, rows - 1)
    point = np.array(seed, dtype=float)
    heading = np.array(direction, dtype=float)
    heading /= np.linalg.norm(heading)
    angles = np.deg2rad(np.linspace(-max_turn_deg, max_turn_deg, 2 * int(max_turn_deg // 5) + 1))
    order = np.argsort(np.abs(angles))
    radius = int(np.ceil(probe_radius))
    offsets = [(dc, dr) for dc in range(-radius, radius + 1) for dr in range(-radius, radius + 1)
               if dc * dc + dr * dr <= probe_radius**2]  # fmt: skip

    def ink(position):
        c, r = int(round(position[0])), int(round(position[1]))
        return sum(
            mask[r + dr, c + dc] for dc, dr in offsets if 0 <= r + dr < rows and 0 <= c + dc < cols
        )

    path, gap = [point.copy()], 0
    for _ in range(max_steps):
        chosen = None
        for k in order:
            a = angles[k]
            rotated = np.array([heading[0] * np.cos(a) - heading[1] * np.sin(a),
                                heading[0] * np.sin(a) + heading[1] * np.cos(a)])  # fmt: skip
            candidate = point + step * rotated
            if ink(candidate) > 0:
                chosen = (candidate, rotated)
                break
        if chosen is None:
            gap += 1
            if gap > max_gap_steps:
                break
            candidate, rotated = point + step * heading, heading
        else:
            gap = 0
            candidate, rotated = chosen
        if not (bounds[0] <= candidate[0] <= bounds[1] and bounds[2] <= candidate[1] <= bounds[3]):
            break
        heading = 0.6 * heading + 0.4 * rotated
        heading /= np.linalg.norm(heading)
        point = candidate
        if gap == 0:
            path.append(point.copy())
    return np.array(path)


def dash_segments(
    mask: NDArray[np.bool_], min_area: int = 10, bounds: tuple[int, int, int, int] | None = None
) -> list[dict]:
    """Connected ink blobs (dashes) with centroid, principal axis and end points.

    Parameters
    ----------
    mask : ndarray of bool
        Ink mask.
    min_area : int, optional
        Smallest blob (pixels) kept; smaller ones are specks.
    bounds : tuple of int, optional
        ``(col_min, col_max, row_min, row_max)`` region to label.

    Returns
    -------
    list of dict
        Per blob: ``centroid`` ``(col, row)``, unit ``axis``, ``length`` (extent
        along the axis), ``ends`` ``(2, 2)``, ``area`` and ``pixels`` ``(n, 2)``.
    """
    from scipy import ndimage

    c0, r0 = 0, 0
    if bounds is not None:
        c0, c1, r0, r1 = bounds
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    segments = []
    for k, region in enumerate(ndimage.find_objects(labels), start=1):
        rows, cols = np.nonzero(labels[region] == k)
        if rows.size < min_area:
            continue
        pixels = np.column_stack([cols + region[1].start + c0, rows + region[0].start + r0]).astype(
            float
        )
        centroid = pixels.mean(0)
        _, _, vt = np.linalg.svd(pixels - centroid, full_matrices=False)
        axis = vt[0]
        projection = (pixels - centroid) @ axis
        ends = centroid + np.outer([projection.min(), projection.max()], axis)
        segments.append({'centroid': centroid, 'axis': axis, 'length': float(np.ptp(projection)),
                         'ends': ends, 'area': int(rows.size), 'pixels': pixels})  # fmt: skip
    return segments


def link_dashes(
    segments: list[dict],
    seed: tuple[float, float],
    direction: tuple[float, float],
    max_link: float = 30.0,
    max_turn_deg: float = 30.0,
    max_length: float = 80.0,
    lateral_weight: float = 4.0,
    sample: float = 3.0,
) -> NDArray[np.float64]:
    """Follow one dashed line by chaining dashes from a seed.

    Parameters
    ----------
    segments : list of dict
        Dashes from :func:`dash_segments`.
    seed : tuple of float
        A point on (or next to) the first dash of the wanted line.
    direction : tuple of float
        Direction ``(dcol, drow)`` to follow from the seed.
    max_link : float, optional
        Largest gap (pixels) between consecutive dashes.
    max_turn_deg : float, optional
        Largest angle between the current heading and the link to the next dash.
    max_length : float, optional
        Blobs longer than this are not dashes (solid lines, merged bundles); the
        chain stops there.
    lateral_weight : float, optional
        Penalty on the sideways offset of a candidate relative to its distance,
        so that a neighbouring parallel dashed line loses against the next dash
        of the same line.
    sample : float, optional
        Spacing (pixels) of the returned centre-line points.

    Returns
    -------
    ndarray, shape ``(n, 2)``
        Centre-line points ``(col, row)`` of the chained dashes, in order.

    Notes
    -----
    Each dash is a straight piece (its principal axis).  From the current end
    point ``p`` with heading ``h`` the next dash is the unused one whose nearer
    end ``e`` minimises ``|e - p| (1 + w |sin θ|)`` with ``θ`` the angle between
    ``e - p`` and ``h``, subject to ``|e - p| <= max_link`` and ``θ <= max_turn``.
    The heading then becomes that dash's axis.  The arc-length tracer
    :func:`trace_path` cannot do this when parallel dashed lines are closer than
    their gaps: in a gap it continues straight and lands on the neighbour.
    """
    seed = np.asarray(seed, dtype=float)
    heading = np.asarray(direction, dtype=float)
    heading /= np.linalg.norm(heading)
    usable = [k for k, seg in enumerate(segments) if seg['length'] <= max_length]
    if not usable:
        return np.empty((0, 2))
    distance = [np.min(np.linalg.norm(segments[k]['pixels'] - seed, axis=1)) for k in usable]
    current = usable[int(np.argmin(distance))]
    used, chain = {current}, []
    cos_turn = np.cos(np.deg2rad(max_turn_deg))
    while True:
        seg = segments[current]
        axis = seg['axis'] if seg['axis'] @ heading >= 0 else -seg['axis']
        heading, chain_end = axis, seg['centroid'] + axis * seg['length'] / 2
        projection = (seg['pixels'] - seg['centroid']) @ axis
        edges = np.arange(projection.min(), projection.max() + sample, sample)
        for lo in edges[:-1]:
            sel = (projection >= lo) & (projection < lo + sample)
            if sel.any():
                chain.append(seg['pixels'][sel].mean(0))
        best, best_cost = None, np.inf
        for k in usable:
            if k in used:
                continue
            ends = segments[k]['ends']
            gaps = ends - chain_end
            j = int(np.argmin(np.linalg.norm(gaps, axis=1)))
            length = float(np.linalg.norm(gaps[j]))
            if length > max_link:
                continue
            if length > 1e-9:
                cosine = float(gaps[j] @ heading) / length
                if cosine < cos_turn:
                    continue
                cost = length * (1 + lateral_weight * np.sqrt(max(0.0, 1 - cosine**2)))
            else:
                cost = 0.0
            if cost < best_cost:
                best, best_cost = k, cost
        if best is None:
            break
        used.add(best)
        current = best
    return np.array(chain)


def assign_to_curves(
    points: ArrayLike, curves: Sequence[ArrayLike], max_distance: float
) -> NDArray[np.int64]:
    """Index of the nearest traced curve for each point (``-1`` beyond ``max_distance``).

    Distances are pixel distances to the curve taken as a polyline through its
    points (in the given order), so the markers of a figure are owned by the
    fitted lines drawn through them; symbols far from every line (label glyphs,
    legend entries) get ``-1``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))

    def polyline_distance(curve):
        curve = np.atleast_2d(np.asarray(curve, dtype=float))
        if len(curve) == 1:
            return np.linalg.norm(points - curve[0], axis=1)
        start, segment = curve[:-1], np.diff(curve, axis=0)  # (nseg, 2)
        length2 = np.maximum((segment**2).sum(axis=1), 1e-12)
        t = np.clip(
            ((points[:, None, :] - start[None]) * segment[None]).sum(axis=2) / length2, 0.0, 1.0
        )
        nearest = start[None] + t[..., None] * segment[None]
        return np.min(np.linalg.norm(points[:, None, :] - nearest, axis=2), axis=1)

    distance = np.array([polyline_distance(curve) for curve in curves])
    owner = np.argmin(distance, axis=0)
    owner[np.min(distance, axis=0) > max_distance] = -1
    return owner


def column_profile(
    mask: NDArray[np.bool_], cols: Sequence[int] | None = None
) -> NDArray[np.float64]:
    """``(col, row)`` of the mean ink row in each column (a single isolated curve)."""
    cols = np.arange(mask.shape[1]) if cols is None else np.asarray(cols)
    points = []
    for col in cols:
        rows = np.nonzero(mask[:, col])[0]
        if rows.size:
            points.append((float(col), float(rows.mean())))
    return np.array(points)


def resample(points: ArrayLike, x_values: ArrayLike, log_y: bool = False) -> NDArray[np.float64]:
    """Interpolate a digitised data curve ``(x, y)`` (sorted by x) at given x (NaN outside)."""
    points = np.asarray(points, dtype=float)
    order = np.argsort(points[:, 0])
    x, y = points[order, 0], points[order, 1]
    target = np.log10(y) if log_y else y
    values = np.interp(np.asarray(x_values, dtype=float), x, target, left=np.nan, right=np.nan)
    return 10.0**values if log_y else values


def overlay_check(
    image: NDArray, points_px: ArrayLike, path: str | Path, color: Sequence[float] = (1.0, 0.0, 0.0)
) -> Path:
    """Save the image with digitised pixels painted in ``color`` (visual validation)."""
    from PIL import Image

    out = (np.clip(image, 0, 1) * 255).astype(np.uint8).copy()
    points_px = np.atleast_2d(np.asarray(points_px, dtype=float))
    for col, row in np.round(points_px).astype(int):
        out[max(row - 1, 0) : row + 2, max(col - 1, 0) : col + 2] = (
            np.asarray(color) * 255
        ).astype(np.uint8)
    Image.fromarray(out).save(str(path))
    return Path(path)
