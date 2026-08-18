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

from os.path import join

_ATTR = '_sparse_log'


class _NullLog:
    """No-op logger handed to non-root ranks."""

    def header(self, title, fields=()):
        pass

    def section(self, title):
        pass

    def write(self, text):
        pass

    def field(self, key, value):
        pass


class SparseLog:
    """Append-only text log.  Reopened per write so the file is complete
    and readable even if the run is killed mid-property."""

    def __init__(self, path):
        self.path = path
        with open(path, 'w') as fh:
            fh.write('PAOFLOW sparse backend log\n')

    def _emit(self, text):
        with open(self.path, 'a') as fh:
            fh.write(text)

    def header(self, title, fields=()):
        self._emit('\n' + '=' * 72 + '\n' + title + '\n' + '=' * 72 + '\n')
        for key, value in fields:
            self.field(key, value)

    def section(self, title):
        self._emit('\n' + title + '\n' + '-' * len(title) + '\n')

    def write(self, text):
        self._emit('%s\n' % text)

    def field(self, key, value):
        self._emit('  %-22s %s\n' % (key, value))


def get_sparse_log(data_controller, fname='sparse.log'):
    """Return the run's :class:`SparseLog`, creating (and truncating) it on
    first use.  Cached on the ``data_controller`` so every sparse stage
    appends to one file."""
    log = getattr(data_controller, _ATTR, None)
    if log is None:
        if data_controller.rank == 0:
            log = SparseLog(join(data_controller.data_attributes['opath'], fname))
        else:
            log = _NullLog()
        setattr(data_controller, _ATTR, log)
    return log
