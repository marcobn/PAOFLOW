"""Small helpers for MPI-launched pyskeaf jobs."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any


_RANK_ENV_VARS = (
    'OMPI_COMM_WORLD_RANK',
    'PMI_RANK',
    'PMIX_RANK',
    'SLURM_PROCID',
    'MV2_COMM_WORLD_RANK',
    'I_MPI_RANK',
)


def active_mpi_comm() -> Any | None:
    """Return ``MPI.COMM_WORLD`` when more than one MPI rank is active.

    Importing :mod:`mpi4py` is deliberately lazy so ordinary serial pyskeaf
    imports remain cheap.  A one-rank MPI launch follows the serial/joblib
    path, while a multi-rank Slurm or ``mpirun`` launch uses MPI collectives.
    """
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError) as error:
        if mpi_rank() is not None:
            raise RuntimeError(
                'pyskeaf detected an MPI launcher, but mpi4py could not initialize. '
                'Load the cluster MPI module before starting Python.'
            ) from error
        return None

    if MPI.Is_finalized():
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


def mpi_rank() -> int | None:
    """Return this process' MPI rank from common launcher env vars."""
    for name in _RANK_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def is_primary_process() -> bool:
    """True outside MPI, or for rank 0 inside an MPI-launched job."""
    rank = mpi_rank()
    return rank is None or rank == 0


def install_mpi_output_filters() -> None:
    """Suppress duplicate warnings and tracebacks from nonzero MPI ranks."""
    if is_primary_process():
        return

    warnings.showwarning = lambda *args, **kwargs: None

    def _quiet_excepthook(exc_type, exc_value, traceback):
        return None

    sys.excepthook = _quiet_excepthook
