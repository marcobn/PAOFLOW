"""Small helpers for MPI-launched pyskeaf jobs."""

from __future__ import annotations

import os
import sys
import warnings


_RANK_ENV_VARS = (
    'OMPI_COMM_WORLD_RANK',
    'PMI_RANK',
    'PMIX_RANK',
    'SLURM_PROCID',
    'MV2_COMM_WORLD_RANK',
    'I_MPI_RANK',
)


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
