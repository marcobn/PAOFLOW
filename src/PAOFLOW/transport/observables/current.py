"""Current observable computed from energy-resolved transmission."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.transport.calculators.current import compute_current_vs_bias


def compute_current(
    *,
    energy_grid: NDArray[np.float64],
    transmission: NDArray[np.float64],
    bias_grid: NDArray[np.float64],
    mu_L: float,
    mu_R: float,
    sigma: float,
) -> NDArray[np.float64]:
    r"""Compute current as a downstream transport observable.

    Parameters
    ----------
    energy_grid : NDArray[np.float64]
        Energy mesh in eV with shape ``(ne,)`` where transmission is sampled.
    transmission : NDArray[np.float64]
        Transmission values :math:`T(E)` on ``energy_grid``, shape ``(ne,)``.
    bias_grid : NDArray[np.float64]
        Applied bias voltages in V, shape ``(nbias,)``.
    mu_L : float
        Left chemical-potential scaling coefficient used as ``mu_L * V``.
    mu_R : float
        Right chemical-potential scaling coefficient used as ``mu_R * V``.
    sigma : float
        Smearing width in eV for the Fermi-Dirac factors.

    Returns
    -------
    NDArray[np.float64]
        Current values aligned with ``bias_grid``, shape ``(nbias,)``.

    Notes
    -----
    This function preserves the existing current formula and numerical
    integration method by delegating to the existing current calculator.
    """
    return compute_current_vs_bias(
        egrid=energy_grid,
        transm=transmission,
        vgrid=bias_grid,
        mu_L=mu_L,
        mu_R=mu_R,
        sigma=sigma,
    )
