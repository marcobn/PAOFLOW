"""Raster digitisation on synthetic figures: calibration, frame, markers, tracing, dashes."""

import numpy as np
import pytest

from PAOFLOW.agentic.thermo_optics import digitize as dg


def disk(mask, centre, radius):
    rows, cols = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    mask[(rows - centre[1]) ** 2 + (cols - centre[0]) ** 2 <= radius**2] = True


def thick_polyline(mask, points, width=2):
    for (c0, r0), (c1, r1) in zip(points[:-1], points[1:]):
        n = int(max(abs(c1 - c0), abs(r1 - r0))) + 1
        for c, r in zip(np.linspace(c0, c1, n), np.linspace(r0, r1, n)):
            mask[
                int(round(r)) - width : int(round(r)) + width + 1,
                int(round(c)) - width : int(round(c)) + width + 1,
            ] = True


def test_axis_log_and_piecewise():
    log_axis = dg.Axis([100, 200, 300], [1, 10, 100], 'log')
    assert log_axis.to_value(250) == pytest.approx(10**1.5)
    assert np.allclose(log_axis.residuals(), 0, atol=1e-9)
    pixels, values = [0, 100, 150, 300], [1, 2, 3, 4]  # uneven drawn spacing
    axis = dg.Axis(pixels, values, 'linear', piecewise=True)
    assert np.allclose(axis.to_value(pixels), values)
    assert axis.to_value(125) == pytest.approx(2.5)
    assert axis.to_value(350) == pytest.approx(4 + 50 / 150)  # extrapolated with the end segment
    assert np.allclose(axis.to_pixel(axis.to_value([10.0, 220.0])), [10.0, 220.0])
    assert np.abs(axis.residuals()).max() > 1  # departure from the global line is reported


def test_find_frame_with_skew_and_ticks():
    mask = np.zeros((220, 320), dtype=bool)
    for c in range(20, 301):  # bottom line drifting by 3 rows (skewed scan)
        mask[200 + (c - 20) * 3 // 280, c] = True
    mask[20, 20:301] = True
    mask[20:201, 20] = True
    mask[20:201, 300] = True
    for c in (80, 140, 200, 260):
        mask[188:201, c] = True
    with pytest.raises(ValueError):
        dg.find_frame(mask, 0.8)
    frame = dg.find_frame(mask, 0.8, skew_tolerance=3)
    assert (frame['left'], frame['right'], frame['top']) == (20, 300, 20) and abs(
        frame['bottom'] - 201
    ) <= 2
    frame['bottom'] = 200
    assert np.allclose(dg.find_ticks(mask, frame, 'x', length=8), [80, 140, 200, 260])


def test_filled_markers_on_a_line_and_assignment():
    mask = np.zeros((200, 300), dtype=bool)
    line = [(10, 150), (290, 50)]
    thick_polyline(mask, line, width=1)
    centres = [(50, 135.7), (150, 100.0), (250, 64.3)]
    for c in centres:
        disk(mask, c, 7)
    mask[20:40, 20:23] = True  # a thin glyph stroke: removed by the opening
    disk(mask, (60, 30), 6)  # a dot far from the line: kept as a marker, not assigned
    found = dg.filled_markers(mask, 3.0)
    assert len(found) == 4
    owner = dg.assign_to_curves(found, [np.array(line, dtype=float)], max_distance=10)
    on_line = found[owner == 0]
    assert len(on_line) == 3
    assert np.allclose(on_line[np.argsort(on_line[:, 0])], centres, atol=1.0)


def test_ring_markers_find_broken_circles():
    mask = np.zeros((120, 200), dtype=bool)
    rows, cols = np.mgrid[0:120, 0:200]
    for centre in ((50, 60), (140, 60)):
        r = np.hypot(cols - centre[0], rows - centre[1])
        ring = (r > 7.5) & (r < 10.5)
        ring &= ~((cols - centre[0] > 0) & (np.abs(rows - centre[1]) < 2))  # JPEG-like break
        mask |= ring
    found = dg.ring_markers(mask, radius=9.0, width=3.0, min_score=0.5, min_distance=6)
    assert len(found) == 2
    assert np.allclose(found[np.argsort(found[:, 0])], [(50, 60), (140, 60)], atol=1.5)


def test_trace_path_follows_steep_curve():
    mask = np.zeros((300, 300), dtype=bool)
    x = np.linspace(20, 280, 400)
    y = 150 - 120 * np.tanh((x - 150) / 20)  # nearly vertical in the middle
    thick_polyline(mask, list(zip(x, y)), width=1)
    path = dg.trace_path(mask, (20, float(y[0])), (1, 0), step=2, max_turn_deg=45)
    assert path[:, 0].max() > 270
    curve = np.column_stack([x, y])
    distance = np.min(np.linalg.norm(path[:, None, :] - curve[None], axis=2), axis=1)
    assert distance.max() < 4  # half the stroke width + the probe radius


def test_link_dashes_separates_close_parallel_dashed_lines():
    mask = np.zeros((300, 120), dtype=bool)
    for col in (40, 52):  # two vertical dashed lines 12 px apart, dashes 20 px, gaps 16 px
        for top in range(10, 290, 36):
            mask[top : top + 20, col - 2 : col + 3] = True
    segments = dg.dash_segments(mask, min_area=10)
    chain = dg.link_dashes(segments, (40, 15), (0, 1), max_link=25, max_turn_deg=30)
    assert np.all(np.abs(chain[:, 0] - 40) < 3) and chain[:, 1].max() > 270
