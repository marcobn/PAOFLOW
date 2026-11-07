"""Data model of the results layer: dataset kinds, references to files, loaded datasets.

A :class:`Kind` describes one type of PAOFLOW output (file-name pattern, folder, how to read it,
axes and units).  Discovery turns the files of a folder into :class:`DatasetRef` objects (one per
kind and group of files: the tensor components or spin channels of a quantity are series of the
same dataset); loading a reference gives a :class:`Dataset` of :class:`Series` that the figure
builders turn into Plotly specifications.  Nothing here imports a plotting library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Kind:
    """One type of PAOFLOW output.

    Parameters
    ----------
    name : str
        Identifier (``'dos'``, ``'optics'``, ``'mobility'``, ...).
    title : str
        Display title; ``{q}``-style fields are filled from the named groups of the pattern.
    category : str
        Group of the dataset tree (Electronic structure, Transport, Optics, ...).
    pattern : str
        Regular expression that the file name must match entirely; named groups label the files.
    family : str
        Loader and figure family: ``curve``, ``columns``, ``bands``, ``phonon_bands``,
        ``transport``, ``hall``, ``pdos``, ``texture``, ``weighted_bands``, ``table``, ``text``,
        ``map``, ``volume``, ``npz``, ``sscha``.
    x : tuple of (str, str)
        Label and unit of the x axis.
    y : tuple of (str, str)
        Label and unit of the y axis.
    series : tuple of str
        Named groups that become series of one dataset (the others split datasets).
    folder : str, optional
        Regular expression the folder (relative path) must match (``search``).
    exclude : str, optional
        Regular expression of file names to skip.
    zero : str, optional
        ``'x'`` (vertical line at x = 0, the Fermi level of energy axes) or ``'y'``.
    columns : tuple, optional
        Column groups of multi-column files: ``((title, unit, (column indices)), ...)``; the first
        group is shown by default.  Column 0 is the x axis.
    column_names : tuple of str, optional
        Names of the y columns (the series of the ``columns`` family).
    priority : int, optional
        Lower values are matched first.
    gpao : str, optional
        ``GPAO`` method that plots the same files (for the Python snippet).
    y_scale : dict, optional
        Unit and factor per named-group value (``{'q': {'sigma': ...}}``).
    describe : callable, optional
        ``describe(groups) -> (title, y_label, y_unit)`` for kinds whose labels depend on the file.
    """

    name: str
    title: str
    category: str
    pattern: str
    family: str
    x: tuple[str, str] = ('', '')
    y: tuple[str, str] = ('', '')
    series: tuple[str, ...] = ()
    folder: str | None = None
    exclude: str | None = None
    zero: str | None = None
    columns: tuple = ()
    column_names: tuple[str, ...] = ()
    priority: int = 50
    gpao: str | None = None
    describe: Callable[[dict], tuple[str, str, str]] | None = None


@dataclass
class DatasetRef:
    """A dataset found by discovery: a kind and the files of one group.

    Parameters
    ----------
    id : str
        Stable identifier (kind, folder and group).
    kind : str
        Kind name.
    title : str
        Display title.
    category : str
        Tree group.
    family : str
        Loader family.
    folder : str
        Folder relative to the scanned root.
    files : list of str
        Files relative to the scanned root.
    labels : list of str
        Series label of every file (same order).
    group : dict
        Named groups shared by the files.
    mtime : float
        Latest modification time of the files.
    size : int
        Total size in bytes.
    """

    id: str
    kind: str
    title: str
    category: str
    family: str
    folder: str
    files: list[str]
    labels: list[str]
    group: dict[str, str]
    mtime: float = 0.0
    size: int = 0

    def to_json(self) -> dict:
        """JSON-ready description.

        Returns
        -------
        dict
        """
        return {
            'id': self.id,
            'kind': self.kind,
            'title': self.title,
            'category': self.category,
            'family': self.family,
            'folder': self.folder,
            'files': self.files,
            'labels': self.labels,
            'group': self.group,
            'mtime': self.mtime,
            'size': self.size,
        }


@dataclass
class Series:
    """One curve (or band set) of a dataset.

    Parameters
    ----------
    name : str
        Legend label.
    x : numpy.ndarray
        Abscissae.
    y : numpy.ndarray
        Ordinates; 2D for band sets (one column per band).
    role : str, optional
        Colour role resolved by the front end: ``series-1`` ... ``series-8`` (categorical, in a
        fixed order that follows the entity), ``seq-<step>`` (ordered, e.g. temperatures),
        ``neutral``.
    dash : str, optional
        Line dash (``solid``, ``dash``, ``dot``) as a second encoding.
    mode : str, optional
        ``lines``, ``markers`` or ``lines+markers``.
    meta : dict, optional
        Extra information (file, temperature, component).
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    role: str = 'series-1'
    dash: str = 'solid'
    mode: str = 'lines'
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    """A loaded dataset, ready for a figure.

    Parameters
    ----------
    ref : DatasetRef
        Where it comes from.
    title : str
        Figure title.
    family : str
        Figure family.
    series : list of Series
        The curves.
    x_label, y_label : str
        Axis titles (units included).
    ticks : list of (float, str)
        Labelled ticks of the x axis (high-symmetry points).
    zero : str, optional
        Reference line (``'x'`` or ``'y'``).
    options : dict
        Selectable options: name -> {'label', 'choices', 'value'}.
    table : list of dict
        Scalar results (``{'name', 'value', 'unit'}``).
    text : str, optional
        Content of text-only datasets.
    surface : dict, optional
        Map data (``x``, ``y``, ``z`` 2D, ``z_label``, ``diverging``).
    mesh : dict, optional
        3D surface data (vertices, faces, values, edges).
    notes : list of str
        Remarks shown under the figure.
    """

    ref: DatasetRef
    title: str
    family: str
    series: list[Series] = field(default_factory=list)
    x_label: str = ''
    y_label: str = ''
    ticks: list[tuple[float, str]] = field(default_factory=list)
    zero: str | None = None
    options: dict[str, dict[str, Any]] = field(default_factory=dict)
    table: list[dict] = field(default_factory=list)
    text: str | None = None
    surface: dict | None = None
    mesh: dict | None = None
    notes: list[str] = field(default_factory=list)


def axis_title(label: str, unit: str) -> str:
    """``'label (unit)'`` (just the label without a unit).

    Parameters
    ----------
    label : str
        Quantity.
    unit : str
        Unit.

    Returns
    -------
    str
    """
    return '%s (%s)' % (label, unit) if unit else label


#: Categorical colour roles, in the fixed order of the palette.
CATEGORICAL = tuple('series-%d' % i for i in range(1, 9))

#: Ordered series shown at once at most: five steps of one hue keep visible lightness gaps
#: (adjacent ΔL >= 0.06 in both themes, checked with the palette validator).
MAX_ORDINAL = 5


def ordinal_roles(count: int) -> list[str]:
    """Colour roles of ``count`` ordered series (temperatures): one hue, light to dark.

    Parameters
    ----------
    count : int
        Number of series (at most :data:`MAX_ORDINAL`).

    Returns
    -------
    list of str
        ``ord-1`` ... ``ord-5``, evenly spread.
    """
    if count <= 1:
        return ['ord-3']
    steps = np.linspace(0, MAX_ORDINAL - 1, min(count, MAX_ORDINAL)).round().astype(int)
    return ['ord-%d' % (i + 1) for i in steps]


def spread(values: list, count: int = MAX_ORDINAL) -> list:
    """At most ``count`` evenly spread items of ``values`` (first and last kept).

    Parameters
    ----------
    values : list
        Ordered values.
    count : int, optional
        Items kept.

    Returns
    -------
    list
    """
    if len(values) <= count:
        return list(values)
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[i] for i in indices]
