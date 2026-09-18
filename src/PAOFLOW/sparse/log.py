"""Dedicated log file for the sparse backend.

Sparse diagnostics — bond-list statistics, truncation bounds, solver
dispatch (including whether a k-point solve is routed to the dense
branch), memory projections and progress — are written to ``sparse.log``
in the PAOFLOW output directory rather than to stdout.  That keeps the
standard output of a sparse run identical in kind to a dense one (module
timings and warnings only), while giving the sparse-specific numbers more
room than a single line each.

Only rank 0 writes; every other rank gets a no-op object, so call sites
need no rank guard.
"""

from __future__ import annotations

from collections.abc import Iterable
from os.path import join
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

_ATTR = '_sparse_log'


class _NullLog:
    """No-op logger handed to non-root ranks.

    Notes
    -----
    Implements the whole :class:`SparseLog` interface as empty methods so
    that call sites can log unconditionally: under MPI only rank 0 owns
    the file, and scattering ``if rank == 0`` guards through the backend
    would be both noisy and easy to forget.
    """

    def header(self, title: str, fields: Iterable[tuple[str, Any]] = ()) -> None:
        """Discard a section header and its fields."""

    def section(self, title: str) -> None:
        """Discard a subsection title."""

    def write(self, text: str) -> None:
        """Discard a free-form line."""

    def field(self, key: str, value: Any) -> None:
        """Discard a key/value line."""


class SparseLog:
    """Append-only text log for one sparse run.

    Parameters
    ----------
    path : str
        Destination file.  It is truncated on construction, so one
        :class:`SparseLog` corresponds to one run.

    Attributes
    ----------
    path : str
        The file being appended to.

    Notes
    -----
    The file is reopened and closed around every write rather than held
    open for the lifetime of the run.  A sparse run is long (hours of
    k-point loops) and its most interesting failure mode — an out-of-memory
    kill during doubling — gives no opportunity to flush buffers, so the
    log has to be complete and readable at every instant.  Writes are rare
    (a handful per property, plus progress lines every few percent of the
    k-loop), so the cost of reopening is negligible next to a single
    eigensolve.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, 'w') as fh:
            fh.write('PAOFLOW sparse backend log\n')

    def _emit(self, text: str) -> None:
        """Append ``text`` verbatim, reopening the file for this write."""
        with open(self.path, 'a') as fh:
            fh.write(text)

    def header(self, title: str, fields: Iterable[tuple[str, Any]] = ()) -> None:
        """Write a rule-delimited section header followed by ``fields``.

        Parameters
        ----------
        title : str
            Heading text, framed above and below by a line of ``=``.
        fields : iterable of (str, object)
            Key/value pairs emitted through :meth:`field`.
        """
        self._emit('\n' + '=' * 72 + '\n' + title + '\n' + '=' * 72 + '\n')
        for key, value in fields:
            self.field(key, value)

    def section(self, title: str) -> None:
        """Write a subsection title underlined to its own width."""
        self._emit('\n' + title + '\n' + '-' * len(title) + '\n')

    def write(self, text: str) -> None:
        """Write ``text`` as one line."""
        self._emit(f'{text}\n')

    def field(self, key: str, value: Any) -> None:
        """Write one indented ``key value`` line on a fixed 22-column key."""
        self._emit(f'  {key!s:<22} {value}\n')


def get_sparse_log(
    data_controller: DataController, fname: str = 'sparse.log'
) -> SparseLog | _NullLog:
    """Return the run's log, creating (and truncating) it on first use.

    Parameters
    ----------
    data_controller : DataController
        Run state; supplies the output directory and the MPI rank, and
        carries the cached log object.
    fname : str, optional
        Log file name inside the output directory.

    Returns
    -------
    SparseLog or _NullLog
        The shared logger on rank 0, a silent stand-in elsewhere.

    Notes
    -----
    The logger is cached as an attribute of the ``data_controller``, which
    is the one object every sparse stage already holds, so all stages of a
    run append to a single file instead of each truncating its own.  The
    caching also makes the truncation happen exactly once: the first caller
    of the run creates the file, every later caller gets the same handle.
    """
    log = getattr(data_controller, _ATTR, None)
    if log is None:
        if data_controller.rank == 0:
            log = SparseLog(join(data_controller.data_attributes['opath'], fname))
        else:
            log = _NullLog()
        setattr(data_controller, _ATTR, log)
    return log
