"""``sparse.log``: the sparse run's parameters and memory figures, kept off stdout.

Truncation, bond list, doubling projection, solve window and the mesh/path passes go to
``sparse.log`` in the output directory, so stdout matches a dense run (timings, warnings).
Format: section titles, indented ``key  value`` lines (:meth:`SparseLog.field`) present only
when they apply, and ``WARNING: ...`` lines naming the remedy (also on stdout). The log says
what was done; the why lives in the writers' docstrings. Only rank 0 writes; other ranks get a
silent :class:`SparseLog`, so call sites need no rank guard.
"""

from __future__ import annotations

from collections.abc import Iterable
from os.path import join
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

_ATTR = '_sparse_log'


class SparseLog:
    """Append-only text log for one sparse run.

    Parameters
    ----------
    path : str or None
        Destination, truncated on construction (one log per run); ``None`` discards all.

    Notes
    -----
    The file is reopened for every write so it is complete at every instant, e.g. after an
    out-of-memory kill during doubling; writes are rare, so the cost is negligible.
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        if path is not None:
            with open(path, 'w') as fh:
                fh.write('PAOFLOW sparse backend log\n')

    def _emit(self, text: str) -> None:
        """Append ``text`` verbatim, reopening the file for this write."""
        if self.path is not None:
            with open(self.path, 'a') as fh:
                fh.write(text)

    def header(self, title: str, fields: Iterable[tuple[str, Any]] = ()) -> None:
        """Write a :meth:`section` title followed by one :meth:`field` per ``(key, value)``."""
        self.section(title)
        for key, value in fields:
            self.field(key, value)

    def section(self, title: str) -> None:
        """Write a subsection title underlined to its own width."""
        self._emit('\n' + title + '\n' + '-' * len(title) + '\n')

    def write(self, text: str) -> None:
        """Write ``text`` as one line."""
        self._emit(f'{text}\n')

    def field(self, key: str, value: Any) -> None:
        """Write one indented ``key value`` line, key padded to 22 columns (or + 2 spaces)."""
        key = str(key)
        pad = f'{key:<22} ' if len(key) <= 22 else key + '  '
        self._emit(f'  {pad}{value}\n')

    def rows(self, pairs: Iterable[tuple[str, Any]]) -> None:
        """Write one :meth:`field` per ``(key, value)`` pair."""
        for key, value in pairs:
            self.field(key, value)

    def warning(self, text: str) -> None:
        """Print ``WARNING: text`` on stdout and append it to the log (rank 0 only)."""
        if self.path is not None:
            message = 'WARNING: ' + text
            print(message, flush=True)
            self._emit(f'  {message}\n')


def get_sparse_log(data_controller: DataController) -> SparseLog:
    """Return the run's log, creating (and truncating) it on first use.

    Cached on the ``data_controller``, which every stage holds, so all stages append to one
    file and it is truncated exactly once. Rank 0 gets the ``sparse.log`` writer (in
    ``attr['opath']``), other ranks a silent one.
    """
    log = getattr(data_controller, _ATTR, None)
    if log is None:
        path = join(data_controller.data_attributes['opath'], 'sparse.log')
        log = SparseLog(path if data_controller.rank == 0 else None)
        setattr(data_controller, _ATTR, log)
    return log
