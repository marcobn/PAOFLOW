"""Shared orchestration helpers for the sparse PAOFLOW driver.

Keeps timing, exception, and sparsity-log formatting identical to the dense
:class:`PAOFLOW.PAOFLOW.PAOFLOW` so the sparse and dense user logs remain
directly comparable.  The driver class (:class:`PAOFLOW.SparsePAOFLOW`) mixes
these in and remains a thin sequence of physics actions.
"""

from time import time


class SparseOrchestrationMixin:
    """Timing / logging / exception helpers shared with the dense driver.

    Expects the host class to define ``self.comm``, ``self.rank``,
    ``self.data_controller``, ``self.start_time`` and ``self.reset_time``.
    """

    def report_module_time(self, mname):
        """Barrier, then print ``<mname> in: <pad> <secs>`` on rank 0.

        Identical formatting to :meth:`PAOFLOW.PAOFLOW.PAOFLOW.report_module_time`
        so dense and sparse timing lines line up.
        """
        self.comm.Barrier()
        if self.rank == 0:
            spaces = 35
            lmn = len(mname)
            if lmn > spaces:
                print('DEBUG: Please use a shorter module tag.')
                self.comm.Abort()
            lms = spaces - lmn
            dt = time() - self.reset_time
            print('%s in: %s %8.3f sec' % (mname, lms * ' ', dt), flush=True)
            self.reset_time = time()

    def sparse_log(self, line):
        """Print one sparsity/solver status line on rank 0 when verbose.

        Parameters
        ----------
        line : str
            A pre-formatted line from :mod:`PAOFLOW.sparse.stats`.
        """
        _, attr = self.data_controller.data_dicts()
        if self.rank == 0 and attr.get('verbose', False):
            print('  ' + line, flush=True)

    def _guard(self, tag, func):
        """Run ``func`` under the dense driver's exception policy.

        Mirrors the ``try/except report_exception + abort_on_exception`` pattern
        used throughout :class:`PAOFLOW.PAOFLOW.PAOFLOW`.
        """
        _, attr = self.data_controller.data_dicts()
        try:
            return func()
        except Exception as e:
            self.report_exception(tag)
            if attr.get('abort_on_exception', True):
                raise e
            return None
