"""Transport observable results container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TransportResults:
    """Container for transport observable results.

    Attributes
    ----------
    transmission : NDArray[np.float64]
        Total and optional channel-resolved transmission.
        Shape: ``(1 + neigchn, ne)``.
    dos : NDArray[np.float64]
        Density of states.
        Shape: ``(ne,)``.
    transmission_k : NDArray[np.float64]
        K-resolved transmission.
        Shape: ``(1 + neigchn, nkpts_par, ne)``.
    dos_k : NDArray[np.float64]
        K-resolved density of states.
        Shape: ``(ne, nkpts_par)``.
    green_functions : NDArray[np.complex128] | None
        Optional conductor retarded Green's functions in real space.
        Shape: ``(ne, nrtot_par, dimC, dimC)``.
    self_energy_L : NDArray[np.complex128] | None
        Optional left lead self-energies in real space.
        Shape: ``(ne, nrtot_par, dimC, dimC)``.
    self_energy_R : NDArray[np.complex128] | None
        Optional right lead self-energies in real space.
        Shape: ``(ne, nrtot_par, dimC, dimC)``.
    energy_grid : NDArray[np.float64]
        Energy grid used for the calculation.
        Shape: ``(ne,)``.
    """

    transmission: NDArray[np.float64]
    dos: NDArray[np.float64]
    transmission_k: NDArray[np.float64]
    dos_k: NDArray[np.float64]
    energy_grid: NDArray[np.float64]
    green_functions: NDArray[np.complex128] | None = None
    self_energy_L: NDArray[np.complex128] | None = None
    self_energy_R: NDArray[np.complex128] | None = None
    bias_grid: NDArray[np.float64] | None = None
    current: NDArray[np.float64] | None = None
