#!/usr/bin/env python3
"""Sources -> canonical datasets with provenance (template: copy to <study>/digitize_sources.py).

    python digitize_sources.py        # -> data/*.csv, checks/*_overlay.png

One function per source figure/table; every anchor, seed and threshold lives in this
file, and every dataset is written with a provenance header.  Re-running overwrites
data/ and checks/ only.  Playbook: PAOFLOW/agentic/thermo_optics/AGENT_WORKFLOW.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PAOFLOW.agentic.thermo_optics import digitize as dg
from PAOFLOW.agentic.thermo_optics import ingest
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, write_dataset

HERE = Path(__file__).resolve().parent
PAPERS = Path('<folder with the source PDFs>')  # read-only
DATA, CHECKS, SOURCES = HERE / 'data', HERE / 'checks', HERE / 'sources'
REF_A = '<Author et al., Journal Vol, page (year)>'


def source_table():
    """A table (CSV/XLSX or transcribed from a paper)."""
    table = ingest.read_table(PAPERS / '<table.csv>')
    frame = make_dataset(
        '<Author year_quantity_T>', '<quantity>', table['<x column>'], table['<y column>'], '<x unit>', '<y unit>',
        'transcribed', REF_A, 'Table <N>', temperature_k=float('nan'),
        conditions={'<key>': '<value>'}, y_rel_uncertainty=0.02,
    )  # fmt: skip
    return write_dataset(
        DATA / '<source>_table.csv', frame, [REF_A + ', Table <N>: <what, conditions>.']
    )


def source_vector_figure():
    """A vector-drawn PDF figure: exact coordinates from the paths."""
    pdf, page, region = (
        PAPERS / '<paper.pdf>',
        0,
        (0, 0, 600, 800),
    )  # region in PDF points (x0, y0, x1, y1)
    print(ingest.pdf_drawing_summary(pdf, page))  # colours and segment counts
    ticks = ingest.pdf_frame_ticks(pdf, page, rect=region)
    calibration = dg.AxisCalibration(
        dg.Axis(ticks['x_major'], ['<printed x values>']),
        dg.Axis(ticks['y_major'], ['<printed y values, top to bottom>']),  # page y grows downwards
    )
    lines = ingest.pdf_vector_polylines(pdf, page, (0.0, 0.0, 1.0), min_segments=40, rect=region)
    frames = []
    for label, line in zip(['<curve labels in the order of `lines`>'], lines):
        data = calibration.to_data(line)
        frames.append(make_dataset('<name>_%s' % label, '<quantity>', data[:, 0], data[:, 1], '<x unit>', '<y unit>',
                                   'vector-extracted', REF_A, 'Fig. <N>'))  # fmt: skip
    residuals = dg.calibration_residuals(calibration)
    header = [REF_A + ', Fig. <N>: <what>. Vector paths; anchor residuals x %.1e, y %.1e pt.'
              % (np.abs(residuals['x']).max(), np.abs(residuals['y']).max())]  # fmt: skip
    return write_dataset(DATA / '<source>_fig<N>.csv', pd.concat(frames, ignore_index=True), header)


def source_raster_figure():
    """A scanned or bitmap figure: calibrate, mask, extract, overlay."""
    image_path = SOURCES / '<paper>_p<page>.png'
    if not image_path.exists():
        ingest.render_pdf_pages(
            PAPERS / '<paper.pdf>', [0], SOURCES, dpi=500
        )  # or extract_pdf_images
    image = dg.load_image(image_path)
    ink = dg.dark_mask(image, 0.5)  # or dg.color_mask(image, rgb, tolerance, min_saturation)
    frame = dg.find_frame(ink, 0.5, skew_tolerance=4)
    ticks_x = dg.find_ticks(ink, frame, 'x', length=12)
    ticks_y = dg.find_ticks(ink, frame, 'y', length=12)
    print(frame, ticks_x, ticks_y)  # pair them with the printed values below
    calibration = dg.AxisCalibration(
        dg.Axis(['<x tick pixels>'], ['<values>'], 'linear'),  # piecewise=True for hand-drawn axes
        dg.Axis(['<y tick pixels>'], ['<values>'], 'log'),
    )
    ink = dg.erase_box(ink, (0, 0), (0, 0))  # labels / legend: rows, cols
    # markers: dg.filled_markers(ink, radius) | dg.ring_markers(ink, radius, width) | dg.marker_centroids(ink)
    # lines:   dg.trace_path(ink, seed, direction, ...) | dg.trace_curve(...)
    # dashes:  dg.link_dashes(dg.dash_segments(ink, bounds=...), seed, direction, max_link)
    # markers on lines: dg.assign_to_curves(markers, lines, max_distance)
    pixels = dg.trace_path(
        ink,
        (0.0, 0.0),
        (1, 0),
        bounds=(frame['left'], frame['right'], frame['top'], frame['bottom']),
    )
    data = calibration.to_data(pixels)
    dg.overlay_check(image, pixels, CHECKS / '<source>_fig<N>_overlay.png')
    residuals = dg.calibration_residuals(calibration)
    frame_out = make_dataset('<name>', '<quantity>', data[:, 0], data[:, 1], '<x unit>', '<y unit>', 'digitised', REF_A,
                             'Fig. <N>', y_rel_uncertainty=0.05, note='<traced line | marker centre>')  # fmt: skip
    header = [REF_A + ', Fig. <N>: <what, sample, conditions>.',
              'DIGITISED from <file, dpi>: <method>; calibration residuals x %.2f px, y %.2f px; <assumptions>.'
              % (np.abs(residuals['x']).max(), np.abs(residuals['y']).max()),
              'Check checks/<source>_fig<N>_overlay.png.']  # fmt: skip
    return write_dataset(DATA / '<source>_fig<N>.csv', frame_out, header)


if __name__ == '__main__':
    for function in (source_table, source_vector_figure, source_raster_figure):
        print('wrote', function())
