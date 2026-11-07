"""Tolerant readers of the text tables PAOFLOW writes.

PAOFLOW writes whitespace columns with or without a ``#`` header (``write_file_row_col`` versus
``np.savetxt(header=...)``), tab-separated tables, comma-separated tables (pyskeaf), and band
blocks separated by blank lines (spin texture).  These readers accept all of them and never
evaluate anything.
"""

from __future__ import annotations

import re

import numpy as np

COMMENT = ('#', '!', '%')
_NUMBER = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?$|^[+-]?(?:nan|inf)$', re.I)


def _is_number(token: str) -> bool:
    return bool(_NUMBER.match(token))


def split_line(line: str) -> list[str]:
    """Fields of a data line (commas, tabs or blanks).

    Parameters
    ----------
    line : str
        The line.

    Returns
    -------
    list of str
    """
    return [token for token in re.split(r'[,\s]+', line.strip()) if token]


def read_columns(path: str, max_rows: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Numeric rows of a text table and its header lines.

    Lines starting with ``#``, ``!`` or ``%`` and non-numeric lines are header lines; rows with a
    different number of fields than the most common one are skipped (so blank-line separated
    blocks are concatenated).

    Parameters
    ----------
    path : str
        The file.
    max_rows : int, optional
        Stop after this many data rows.

    Returns
    -------
    data : numpy.ndarray, shape ``(nrows, ncolumns)``
        ``float`` values (Fortran ``D`` exponents accepted); empty ``(0, 0)`` if none.
    header : list of str
        The header lines without their comment marker.
    """
    rows: list[list[float]] = []
    header: list[str] = []
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(COMMENT):
                header.append(stripped.lstrip('#!% ').rstrip())
                continue
            tokens = split_line(stripped)
            if not tokens or not all(_is_number(t) for t in tokens):
                if not rows:
                    header.append(stripped)
                continue
            rows.append([float(t.replace('D', 'e').replace('d', 'e')) for t in tokens])
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        return np.zeros((0, 0)), header
    widths = [len(row) for row in rows]
    width = max(set(widths), key=widths.count)
    data = np.array([row for row in rows if len(row) == width], dtype=float)
    return data, header


def read_blocks(path: str) -> tuple[list[np.ndarray], list[str]]:
    """Blank-line separated numeric blocks (one per band of a spin-texture file).

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    blocks : list of numpy.ndarray
        One 2D array per block.
    header : list of str
        Header lines.
    """
    blocks: list[np.ndarray] = []
    header: list[str] = []
    current: list[list[float]] = []
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(COMMENT):
                header.append(stripped.lstrip('#!% '))
                continue
            if not stripped:
                if current:
                    blocks.append(np.array(current))
                    current = []
                continue
            tokens = split_line(stripped)
            if all(_is_number(t) for t in tokens):
                current.append([float(t) for t in tokens])
    if current:
        blocks.append(np.array(current))
    widths = {block.shape[1] for block in blocks if block.ndim == 2}
    if len(widths) > 1:
        width = max(widths, key=lambda w: sum(b.shape[1] == w for b in blocks))
        blocks = [b for b in blocks if b.shape[1] == width]
    return blocks, header


def header_parameters(header: list[str]) -> dict[str, float]:
    """``name=value`` pairs of header lines (``lambda=0.5767, w_log=724.11K, Tc_McM=14.644K``).

    Parameters
    ----------
    header : list of str
        Header lines.

    Returns
    -------
    dict
        Values as floats (units dropped), by name.
    """
    values: dict[str, float] = {}
    for line in header:
        for name, value in re.findall(
            r'([A-Za-z_*][\w*^]*)\s*=\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)', line
        ):
            try:
                values[name] = float(value)
            except ValueError:
                continue
    return values
