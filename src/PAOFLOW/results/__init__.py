"""Results layer of PAOFLOW: find, read and draw the output files of a run.

Usable without the GUI (scripts, notebooks, reports)::

    from PAOFLOW.results import ResultsFolder

    results = ResultsFolder('Si/output')
    for ref in results.datasets:
        print(ref.category, ref.title, ref.files)
    dataset = results.load(results.find('bands').id)
    spec = results.figure(dataset)              # Plotly specification (dict)
    results.save(dataset, 'bands.pdf')          # matplotlib

In Jupyter, ``plotly.graph_objects.Figure(results.figure(dataset, theme='light'))`` shows the same
figure as the GUI.  The catalogue of output kinds is :data:`PAOFLOW.results.catalogue.KINDS`.
"""

from __future__ import annotations

import os

from .catalogue import KINDS
from .discovery import discover
from .figures import figure_spec
from .loaders import load
from .model import Dataset, DatasetRef

__all__ = ['KINDS', 'Dataset', 'DatasetRef', 'ResultsFolder', 'discover', 'figure_spec', 'load']


class ResultsFolder:
    """The datasets of a folder and helpers to load, draw and save them.

    Parameters
    ----------
    folder : str
        Results folder (an ``output`` folder, a run folder, a system folder).
    """

    def __init__(self, folder: str) -> None:
        self.folder = os.path.abspath(os.path.expanduser(folder))
        self.datasets: list[DatasetRef] = discover(self.folder)

    def refresh(self) -> list[DatasetRef]:
        """Scan the folder again.

        Returns
        -------
        list of DatasetRef
        """
        self.datasets = discover(self.folder)
        return self.datasets

    def find(self, kind: str, title: str | None = None) -> DatasetRef:
        """The first dataset of a kind (and title substring).

        Parameters
        ----------
        kind : str
            Kind name (``bands``, ``dos``, ``optics``, ...).
        title : str, optional
            Part of the title.

        Returns
        -------
        DatasetRef

        Raises
        ------
        KeyError
            When nothing matches.
        """
        for ref in self.datasets:
            if ref.kind == kind and (title is None or title.lower() in ref.title.lower()):
                return ref
        raise KeyError(
            'no %s dataset%s in %s' % (kind, ' "%s"' % title if title else '', self.folder)
        )

    def get(self, identifier: str) -> DatasetRef:
        """The dataset with this identifier.

        Parameters
        ----------
        identifier : str
            :attr:`DatasetRef.id`.

        Returns
        -------
        DatasetRef
        """
        for ref in self.datasets:
            if ref.id == identifier:
                return ref
        raise KeyError('no dataset %s in %s' % (identifier, self.folder))

    def load(self, identifier: str | DatasetRef, options: dict | None = None) -> Dataset:
        """Load a dataset.

        Parameters
        ----------
        identifier : str or DatasetRef
            Identifier or reference.
        options : dict, optional
            Selected options.

        Returns
        -------
        Dataset
        """
        ref = identifier if isinstance(identifier, DatasetRef) else self.get(identifier)
        return load(ref, self.folder, options)

    def figure(self, dataset: Dataset, theme: str | None = None) -> dict | None:
        """Plotly specification (colour roles resolved when ``theme`` is given).

        Parameters
        ----------
        dataset : Dataset
            Loaded dataset.
        theme : str, optional
            ``'light'`` or ``'dark'``.

        Returns
        -------
        dict or None
        """
        spec = figure_spec(dataset)
        if spec is not None and theme is not None:
            from .palette import resolve_spec

            spec = resolve_spec(spec, theme)
        return spec

    def save(self, dataset: Dataset, path: str) -> str:
        """Save a matplotlib rendering (format from the extension: pdf, png, svg).

        Parameters
        ----------
        dataset : Dataset
            Loaded dataset.
        path : str
            Output file.

        Returns
        -------
        str
            Absolute path.
        """
        from .export import matplotlib_figure

        fmt = os.path.splitext(path)[1].lstrip('.').lower() or 'pdf'
        with open(path, 'wb') as handle:
            handle.write(matplotlib_figure(dataset, fmt=fmt))
        return os.path.abspath(path)
