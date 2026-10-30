"""Inventory and extraction of experimental sources: PDFs, images, tables, decks.

PDF access uses PyMuPDF (``pymupdf``), imported lazily.  The decisive question for
every PDF figure is whether it is **vector** (lines drawn by the PDF: read the
coordinates exactly with :func:`pdf_vector_polylines`) or **raster** (a scan or an
embedded bitmap: render it with :func:`render_pdf_pages` or
:func:`extract_pdf_images` and digitise it with
:mod:`~PAOFLOW.agentic.thermo_optics.digitize`).  :func:`pdf_inventory` answers it
page by page.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def _open_pdf(pdf: str | Path):
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError('PDF ingestion needs PyMuPDF: pip install pymupdf') from error
    return pymupdf.open(str(pdf))


def pdf_inventory(pdf: str | Path, destination: str | Path | None = None) -> pd.DataFrame:
    """Page-by-page inventory of a PDF: text, captions, images and vector drawings.

    Parameters
    ----------
    pdf : str or Path
        The PDF file.
    destination : str or Path, optional
        Folder for ``<stem>_inventory.md`` (also the full text, ``<stem>_text.md``).

    Returns
    -------
    pandas.DataFrame
        One row per page: ``page`` (1-based), ``text_chars`` (0 for a scan),
        ``images``, ``drawings`` (vector paths), ``coloured_paths``, ``captions``.

    Notes
    -----
    A page with ``text_chars == 0`` is a scan: read it as an image (render it).
    Many coloured paths with hundreds of segments mean vector data curves.
    """
    document = _open_pdf(pdf)
    rows, texts = [], []
    for index, page in enumerate(document):
        text = page.get_text()
        drawings = page.get_drawings()
        coloured = [
            d for d in drawings
            if d.get('color') is not None and len(set(np.round(d['color'], 2))) > 1
            and len(d['items']) > 20
        ]  # fmt: skip
        captions = re.findall(r'(?:FIG\.|Fig\.|Figure|Table|TABLE)\s*\d+[^\n]{0,80}', text)
        rows.append({
            'page': index + 1, 'text_chars': len(text.strip()), 'images': len(page.get_images()),
            'drawings': len(drawings), 'coloured_paths': len(coloured),
            'captions': ' | '.join(captions)[:300],
        })  # fmt: skip
        texts.append('## page %d\n\n%s' % (index + 1, text))
    frame = pd.DataFrame(rows)
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        stem = Path(pdf).stem
        (destination / ('%s_text.md' % stem)).write_text('\n'.join(texts), encoding='utf-8')
        lines = ['# Inventory of %s' % Path(pdf).name, '', frame.to_markdown(index=False)]
        (destination / ('%s_inventory.md' % stem)).write_text('\n'.join(lines), encoding='utf-8')
    return frame


def render_pdf_pages(
    pdf: str | Path, pages: Sequence[int], destination: str | Path, dpi: int = 300
) -> list[Path]:
    """Render PDF pages (1-based) to PNG files ``<stem>_p<page>.png``."""
    document = _open_pdf(pdf)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    out = []
    for page in pages:
        path = destination / ('%s_p%d.png' % (Path(pdf).stem, page))
        document[page - 1].get_pixmap(dpi=dpi).save(str(path))
        out.append(path)
    return out


def extract_pdf_images(pdf: str | Path, page: int, destination: str | Path) -> list[Path]:
    """Save the embedded bitmaps of a page at their native resolution (1-based page)."""
    document = _open_pdf(pdf)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    out = []
    for number, info in enumerate(document[page - 1].get_images(full=True)):
        image = document.extract_image(info[0])
        path = destination / ('%s_p%d_img%d.%s' % (Path(pdf).stem, page, number, image['ext']))
        path.write_bytes(image['image'])
        out.append(path)
    return out


def image_placements(pdf: str | Path, page: int) -> list[dict]:
    """Where each embedded image sits on the page (PDF points), for mapping its pixels."""
    document = _open_pdf(pdf)
    page_object = document[page - 1]
    out = []
    for info in page_object.get_images(full=True):
        for rect in page_object.get_image_rects(info[0]):
            out.append({'xref': info[0], 'width_px': info[2], 'height_px': info[3],
                        'rect': (rect.x0, rect.y0, rect.x1, rect.y1)})  # fmt: skip
    return out


def pdf_words(pdf: str | Path, page: int, rect: Sequence[float] | None = None) -> pd.DataFrame:
    """Words of a page with their boxes (PDF points); optionally inside ``rect``.

    Tick labels and curve labels of vector figures come out with exact positions,
    the input of an exact axis calibration.
    """
    document = _open_pdf(pdf)
    words = document[page - 1].get_text('words')
    frame = pd.DataFrame(words, columns=['x0', 'y0', 'x1', 'y1', 'text', 'block', 'line', 'word'])
    frame['x'] = 0.5 * (frame['x0'] + frame['x1'])
    frame['y'] = 0.5 * (frame['y0'] + frame['y1'])
    if rect is not None:
        x0, y0, x1, y1 = rect
        frame = frame[
            (frame['x'] >= x0) & (frame['x'] <= x1) & (frame['y'] >= y0) & (frame['y'] <= y1)
        ]
    return frame.reset_index(drop=True)


def pdf_drawing_summary(pdf: str | Path, page: int) -> pd.DataFrame:
    """Vector drawings of a page grouped by stroke colour and dash pattern."""
    document = _open_pdf(pdf)
    rows = []
    for drawing in document[page - 1].get_drawings():
        colour = drawing.get('color')
        rect = drawing['rect']
        rows.append({
            'color': None if colour is None else tuple(round(c, 3) for c in colour),
            'dashes': drawing.get('dashes'), 'width': drawing.get('width'),
            'segments': len(drawing['items']),
            'kinds': ''.join(sorted(Counter(item[0] for item in drawing['items']))),
            'rect': (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)),
        })  # fmt: skip
    return pd.DataFrame(rows)


def pdf_vector_polylines(
    pdf: str | Path,
    page: int,
    color: Sequence[float],
    color_tolerance: float = 0.02,
    min_segments: int = 20,
    rect: Sequence[float] | None = None,
    join_tolerance: float = 0.05,
) -> list[np.ndarray]:
    """Continuous polylines of a given stroke colour on a PDF page (PDF points).

    Parameters
    ----------
    pdf : str or Path
        The PDF file.
    page : int
        1-based page.
    color : sequence of float
        Stroke RGB in [0, 1] (from :func:`pdf_drawing_summary`).
    color_tolerance : float, optional
        Per-channel tolerance on the colour.
    min_segments : int, optional
        Ignore drawings with fewer segments (legend swatches, ticks).
    rect : sequence of float, optional
        Keep only drawings whose box lies inside ``(x0, y0, x1, y1)``.
    join_tolerance : float, optional
        Consecutive segments closer than this (points) belong to one polyline.

    Returns
    -------
    list of ndarray, shape ``(npoints, 2)``
        Page coordinates ``(x, y)`` with y pointing **down**; map them with an
        :class:`~PAOFLOW.agentic.thermo_optics.digitize.AxisCalibration`.

    Notes
    -----
    Dashed curves are usually one continuous path with a dash pattern, so they
    come out whole.  Bezier segments (``'c'``) are sampled at their control end
    points.
    """
    document = _open_pdf(pdf)
    target = np.asarray(color, dtype=float)
    out = []
    for drawing in document[page - 1].get_drawings():
        colour = drawing.get('color')
        if colour is None or np.max(np.abs(np.asarray(colour) - target)) > color_tolerance:
            continue
        if len(drawing['items']) < min_segments:
            continue
        box = drawing['rect']
        if rect is not None and not (
            box.x0 >= rect[0] - 1 and box.y0 >= rect[1] - 1 and box.x1 <= rect[2] + 1
            and box.y1 <= rect[3] + 1
        ):  # fmt: skip
            continue
        current: list[tuple[float, float]] = []
        for item in drawing['items']:
            if item[0] not in ('l', 'c'):
                continue
            start, end = item[1], item[-1]
            if current and (
                abs(current[-1][0] - start.x) > join_tolerance
                or abs(current[-1][1] - start.y) > join_tolerance
            ):
                out.append(np.array(current))
                current = []
            if not current:
                current.append((start.x, start.y))
            current.append((end.x, end.y))
        if current:
            out.append(np.array(current))
    return out


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV/TXT/XLSX table into a DataFrame (``#`` comments skipped for text)."""
    path = Path(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path, **kwargs)
    if path.suffix.lower() in ('.tsv', '.txt', '.dat'):
        return pd.read_csv(path, sep=None, engine='python', comment='#', **kwargs)
    return pd.read_csv(path, comment='#', **kwargs)


def extract_slide_deck(deck: str | Path, destination: str | Path) -> Path:
    """Slide text and images of a pptx deck (see :func:`PAOFLOW.agentic.qo_analysis.experiment_data.extract_slide_deck`)."""
    from ..qo_analysis.experiment_data import extract_slide_deck as extract

    return extract(deck, destination)


def pdf_frame_ticks(
    pdf: str | Path,
    page: int,
    rect: Sequence[float] | None = None,
    color: Sequence[float] = (0, 0, 0),
) -> dict:
    """Plot frame and tick marks of a vector figure (PDF points), for an exact calibration.

    Parameters
    ----------
    pdf : str or Path
        The PDF file.
    page : int
        1-based page.
    rect : sequence of float, optional
        Region ``(x0, y0, x1, y1)`` containing the figure (when a page has several).
    color : sequence of float, optional
        Stroke colour of the axes.

    Returns
    -------
    dict
        ``frame`` ``(x0, y0, x1, y1)`` (the largest rectangle), ``x_major``,
        ``x_minor`` (tick x positions on the bottom edge), ``y_major``, ``y_minor``
        (tick y positions on the left edge).  Pair the major ticks with their
        printed labels (:func:`pdf_words`) to build an
        :class:`~PAOFLOW.agentic.thermo_optics.digitize.Axis`.
    """
    document = _open_pdf(pdf)
    target = np.asarray(color, dtype=float)
    rectangles, horizontal, vertical = [], [], []
    for drawing in document[page - 1].get_drawings():
        colour = drawing.get('color')
        if colour is None or np.max(np.abs(np.asarray(colour) - target)) > 0.05:
            continue
        for item in drawing['items']:
            if item[0] == 're':
                r = item[1]
                rectangles.append((r.x0, r.y0, r.x1, r.y1))
            elif item[0] == 'l':
                a, b = item[1], item[2]
                if abs(a.y - b.y) < 0.01:
                    horizontal.append((min(a.x, b.x), max(a.x, b.x), a.y))
                elif abs(a.x - b.x) < 0.01:
                    vertical.append((a.x, min(a.y, b.y), max(a.y, b.y)))
    if rect is not None:
        inside = lambda x, y: rect[0] - 1 <= x <= rect[2] + 1 and rect[1] - 1 <= y <= rect[3] + 1  # noqa: E731
        rectangles = [r for r in rectangles if inside(r[0], r[1]) and inside(r[2], r[3])]
        horizontal = [h for h in horizontal if inside(h[0], h[2])]
        vertical = [v for v in vertical if inside(v[0], v[1])]
    if not rectangles:
        raise ValueError('no frame rectangle found on page %d' % page)
    frame = max(rectangles, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    x0, y0, x1, y1 = frame

    def split(lengths, positions):
        if not positions:
            return np.array([]), np.array([])
        lengths, positions = np.asarray(lengths), np.asarray(positions)
        major = lengths > 0.75 * lengths.max()
        return np.unique(np.round(positions[major], 3)), np.unique(np.round(positions[~major], 3))

    # ticks: short segments touching the bottom / left edges (frame sides excluded)
    bottom = [
        (v[2] - v[1], v[0]) for v in vertical
        if (abs(v[2] - y1) < 0.5 or abs(v[1] - y1) < 0.5) and v[2] - v[1] < 0.2 * (y1 - y0)
    ]  # fmt: skip
    left = [
        (h[1] - h[0], h[2]) for h in horizontal
        if (abs(h[0] - x0) < 0.5 or abs(h[1] - x0) < 0.5) and h[1] - h[0] < 0.2 * (x1 - x0)
    ]  # fmt: skip
    x_major, x_minor = split([b[0] for b in bottom], [b[1] for b in bottom])
    y_major, y_minor = split([b[0] for b in left], [b[1] for b in left])
    return {'frame': frame, 'x_major': x_major, 'x_minor': x_minor, 'y_major': y_major,
            'y_minor': y_minor}  # fmt: skip
