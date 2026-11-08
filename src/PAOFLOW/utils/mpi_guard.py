"""Run a driver's ``main`` so that an error on one MPI rank stops every rank.

Under ``mpirun`` an exception raised on one rank ends that rank only. The other ranks wait for
it in the next collective operation (a barrier, a reduction, a broadcast) and spin there for
ever, keeping their cores busy, while the job looks alive (``paoflow-studio`` lists it as
running). :func:`run_main` prints the traceback of the failing rank and calls ``MPI_Abort``,
which ends the whole job with a non-zero status. Without MPI, or on a single rank, it runs
``main`` unchanged.

The drivers written by ``paoflow-gen`` end with::

    if __name__ == "__main__":
        from PAOFLOW.utils.mpi_guard import run_main

        run_main(main)
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Callable

#: Exit status of a job stopped by :func:`run_main`.
ABORT_STATUS = 1


def _parallel_world() -> Any:
    """``MPI.COMM_WORLD`` when MPI runs with more than one rank, else ``None``."""
    try:
        from mpi4py import MPI
    except ImportError:  # pragma: no cover - mpi4py is a core dependency
        return None
    if not MPI.Is_initialized() or MPI.Is_finalized():
        return None
    world = MPI.COMM_WORLD
    return world if world.Get_size() > 1 else None


def run_main(main: Callable[[], Any]) -> Any:
    """Call ``main()``; under MPI, an error on any rank aborts every rank.

    Parameters
    ----------
    main : callable
        The driver's entry point (no arguments).

    Returns
    -------
    object
        What ``main`` returns (when it returns).

    Raises
    ------
    BaseException
        Whatever ``main`` raises, when MPI runs on one rank or not at all. ``sys.exit()`` and
        ``sys.exit(0)`` are never turned into an abort.

    Notes
    -----
    On more than one rank the failing rank prints its rank and traceback (or the message of
    ``sys.exit('message')``) to standard error, flushes its output and calls
    ``MPI.COMM_WORLD.Abort(1)``. The MPI launcher then stops the other ranks, so the job ends
    instead of hanging in the next collective operation.
    """
    try:
        return main()
    except SystemExit as exc:
        if exc.code in (None, 0):
            raise
        world = _parallel_world()
        if world is None:
            raise
        message = exc.code if isinstance(exc.code, str) else 'exit status %s' % exc.code
        _abort(world, 'stopped: %s' % message)
    except BaseException:
        world = _parallel_world()
        if world is None:
            raise
        _abort(world, traceback.format_exc().rstrip())


def _abort(world: Any, text: str) -> None:
    rank = world.Get_rank()
    sys.stdout.flush()
    sys.stderr.write(
        '\n[rank %d of %d] %s\n[rank %d] aborting every MPI rank (an error on one rank would '
        'otherwise leave the others waiting for ever)\n' % (rank, world.Get_size(), text, rank)
    )
    sys.stderr.flush()
    world.Abort(ABORT_STATUS)
