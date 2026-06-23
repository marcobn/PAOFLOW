from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.transport.calculators.current import compute_current_vs_bias


def write_current_output(
    *,
    data: Mapping[str, Any],
    vgrid: NDArray[np.float64],
    currents: NDArray[np.float64],
) -> None:
    """Write current-vs-bias results to disk.

    Parameters
    ----------
    data : Mapping[str, Any]
        Current workflow input mapping. Must contain ``fileout``.
    vgrid : NDArray[np.float64]
        Bias grid in volts, shape ``(nV,)``.
    currents : NDArray[np.float64]
        Current values aligned with ``vgrid``, shape ``(nV,)``.

    Returns
    -------
    None
        Creates parent directories for ``fileout``, writes two-column data
        ``(V, I)`` with ``numpy.savetxt``, and logs the output location.
    """
    outpath = Path(str(data['fileout']))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(outpath, np.column_stack([vgrid, currents]))
    log.log_rank0(f'Saved current vs bias to {outpath}')


def run_current(
    *,
    data: Mapping[str, Any],
    egrid: NDArray[np.float64],
    transm: NDArray[np.float64],
    vgrid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute current-vs-bias and write output.

    Parameters
    ----------
    data : Mapping[str, Any]
        Current workflow input mapping. Required keys are ``mu_L``, ``mu_R``,
        ``sigma``, and ``fileout``.
    egrid : NDArray[np.float64]
        Energy grid for transmission data, shape ``(ne,)``.
    transm : NDArray[np.float64]
        Transmission array sampled on ``egrid``.
    vgrid : NDArray[np.float64]
        Bias grid in volts, shape ``(nV,)``.

    Returns
    -------
    NDArray[np.float64]
        Computed current values, shape ``(nV,)``.
    """
    currents = compute_current_vs_bias(
        egrid,
        transm,
        vgrid,
        data['mu_L'],
        data['mu_R'],
        data['sigma'],
    )
    write_current_output(data=data, vgrid=vgrid, currents=currents)
    return currents
