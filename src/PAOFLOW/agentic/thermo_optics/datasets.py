"""Canonical experimental (and calculated) datasets with provenance.

Every comparison reads one long-format table, whatever the original form of the
data.  One row is one point:

``dataset``, ``quantity``, ``x``, ``x_unit``, ``y``, ``y_unit``,
``temperature_k``, ``conditions``, ``provenance``, ``source``, ``locator``,
``y_rel_uncertainty``, ``note``.

``conditions`` holds the other experimental conditions as ``key=value``
pairs separated by ``;`` (e.g. ``thickness_m=1.77e-3;angle_deg=0;doping=n 15 ohm cm``).
``provenance`` is one of :data:`PROVENANCE`.  ``locator`` says where in the source
the point comes from (``Fig. 3``, ``Table 2``, ``slide 7``).  Free-form context
(how the data were obtained, the calibration, known caveats) goes into the
``#`` header lines of the CSV file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = (
    'dataset', 'quantity', 'x', 'x_unit', 'y', 'y_unit', 'temperature_k', 'conditions',
    'provenance', 'source', 'locator', 'y_rel_uncertainty', 'note',
)  # fmt: skip

#: Provenance classes, from the most to the least direct.
PROVENANCE = {
    'measured': 'raw data supplied by the experimentalists',
    'transcribed': 'numbers copied from a table or from text',
    'vector-extracted': 'coordinates read from the vector paths of a figure (exact rendering)',
    'digitised': 'coordinates read from a raster image of a figure',
    'calculated': 'model output',
}


def make_dataset(
    dataset: str,
    quantity: str,
    x: Sequence[float],
    y: Sequence[float],
    x_unit: str,
    y_unit: str,
    provenance: str,
    source: str,
    locator: str = '',
    temperature_k: float | Sequence[float] = np.nan,
    conditions: Mapping[str, object] | str = '',
    y_rel_uncertainty: float | Sequence[float] = np.nan,
    note: str = '',
) -> pd.DataFrame:
    """Rows of the canonical table for one curve or point set.

    Parameters
    ----------
    dataset : str
        Identifier of the curve (e.g. ``'Sato1967_fig3_793K'``).
    quantity : str
        Observable (e.g. ``'absorption_coefficient'``, ``'emissivity_normal'``).
    x, y : sequence of float
        Abscissa and values.
    x_unit, y_unit : str
        Units (see :mod:`~PAOFLOW.agentic.thermo_optics.units`).
    provenance : str
        One of :data:`PROVENANCE`.
    source : str
        Reference (e.g. ``'T. Sato, Jpn. J. Appl. Phys. 6, 339 (1967)'``).
    locator : str, optional
        Figure, table or slide.
    temperature_k : float or sequence, optional
        Temperature of each point (K).
    conditions : mapping or str, optional
        Other conditions (``{'thickness_m': 1.77e-3}``).
    y_rel_uncertainty : float or sequence, optional
        Relative uncertainty of ``y`` (digitisation or measurement).
    note : str, optional
        Short remark per row.

    Returns
    -------
    pandas.DataFrame
        Rows with :data:`COLUMNS`.

    Raises
    ------
    ValueError
        For an unknown provenance or mismatched lengths.
    """
    if provenance not in PROVENANCE:
        raise ValueError('provenance must be one of %s, got %r' % (sorted(PROVENANCE), provenance))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError('x and y have different lengths (%d, %d)' % (x.size, y.size))
    if isinstance(conditions, Mapping):
        conditions = format_conditions(conditions)
    frame = pd.DataFrame({
        'dataset': dataset, 'quantity': quantity, 'x': x, 'x_unit': x_unit, 'y': y,
        'y_unit': y_unit, 'temperature_k': np.broadcast_to(temperature_k, x.shape).astype(float),
        'conditions': conditions, 'provenance': provenance, 'source': source,
        'locator': locator,
        'y_rel_uncertainty': np.broadcast_to(y_rel_uncertainty, x.shape).astype(float),
        'note': note,
    })  # fmt: skip
    return frame[list(COLUMNS)]


def format_conditions(conditions: Mapping[str, object]) -> str:
    """``{'a': 1, 'b': 'x'}`` -> ``'a=1;b=x'``."""
    return ';'.join('%s=%s' % (key, value) for key, value in conditions.items())


def parse_conditions(text: str) -> dict[str, object]:
    """Inverse of :func:`format_conditions`; numeric values become floats."""
    out: dict[str, object] = {}
    if not isinstance(text, str) or not text.strip():
        return out
    for item in text.split(';'):
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        try:
            out[key.strip()] = float(value)
        except ValueError:
            out[key.strip()] = value.strip()
    return out


def write_dataset(path: str | Path, frame: pd.DataFrame, header: Sequence[str] = ()) -> Path:
    """Write a canonical table, with ``#`` provenance lines first.

    Parameters
    ----------
    path : str or Path
        Output CSV file.
    frame : pandas.DataFrame
        Rows with :data:`COLUMNS` (e.g. concatenated :func:`make_dataset` outputs).
    header : sequence of str, optional
        Provenance and caveat lines (written as ``# ...``).

    Returns
    -------
    Path
        The written file.
    """
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError('missing canonical columns: %s' % missing)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for line in header:
            handle.write('# %s\n' % line)
        frame[list(COLUMNS)].to_csv(handle, index=False, float_format='%.8g')
    return path


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read a canonical table (``#`` lines skipped) and check its columns and provenance."""
    # only the leading '#' lines are provenance: a '#' inside a field is data
    frame = pd.read_csv(path, skiprows=len(dataset_header_lines(path)))
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError('%s lacks canonical columns %s' % (path, missing))
    unknown = set(frame['provenance']) - set(PROVENANCE)
    if unknown:
        raise ValueError('%s has unknown provenance values %s' % (path, sorted(unknown)))
    frame['conditions'] = frame['conditions'].fillna('')
    frame['note'] = frame['note'].fillna('')
    frame['locator'] = frame['locator'].fillna('')
    return frame


def dataset_header_lines(path: str | Path) -> list[str]:
    """The ``#`` provenance lines of a canonical table (without the ``#``)."""
    lines = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.startswith('#'):
                break
            lines.append(line[1:].strip())
    return lines


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: quantity, units, temperature, point count, x range, provenance."""
    groups = frame.groupby('dataset', sort=False)
    return pd.DataFrame({
        'quantity': groups['quantity'].first(), 'x_unit': groups['x_unit'].first(),
        'y_unit': groups['y_unit'].first(), 'temperature_k': groups['temperature_k'].first(),
        'points': groups.size(), 'x_min': groups['x'].min(), 'x_max': groups['x'].max(),
        'provenance': groups['provenance'].first(), 'locator': groups['locator'].first(),
    })  # fmt: skip
