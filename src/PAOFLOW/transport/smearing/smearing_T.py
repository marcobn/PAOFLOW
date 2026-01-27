import numpy as np
from typing import Optional, Callable, Tuple

from PAOFLOW.transport.utils.timing import timed_function


def initialize_smearing_grid(
    smearing_type: str,
    delta: float,
    delta_ratio: float,
    xmax: float,
    *,
    smearing_func: Callable[[float, str], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize the numerical smearing grid and compute g_smear values using FFT-based convolution.

    Parameters
    ----------
    `smearing_func` : callable
        A function f(x, type) returning the smearing function evaluated at `x` for a given type.
    `smearing_type` : str
        Type of smearing function, e.g., 'lorentzian', 'gaussian', etc.
    `delta` : float
        Broadening parameter used in FFT smearing.
    `delta_ratio` : float
        Ratio to determine resolution of numerical grid.
    `xmax` : float
        Half-width of the energy window.

    Returns
    -------
    `xgrid` : np.ndarray
        Energy grid points.
    `g_smear` : np.ndarray
        Smeared Green's function values over the grid.

    Notes
    -----
    This function constructs a numerically smeared Green's function `g_smear(x)`
    over an energy grid `xgrid` using FFT-based convolution:

        g_smear(x) = [1 / (x + i·δ_ratio)] ⋆ f_smear(x / δ)

    where:
    - `f_smear` is the chosen smearing function (e.g. Lorentzian, Gaussian)
    - `⋆` denotes convolution (implemented via FFT)
    - `delta` determines the width of the smearing function
    - `delta_ratio` controls the pole width for the convolution kernel
    - The final result is stored on an energy grid `xgrid ∈ [−xmax, xmax]`

    This is used only in the `'numerical'` smearing mode for computing
    Green's functions.

    """
    nx = int(2 * (2 * xmax / delta_ratio))
    dx = (2 * xmax) / nx
    xgrid = np.linspace(-xmax, xmax, nx, dtype=np.float64)

    eps_sx = 15.0
    eps_px = xmax + eps_sx
    Tmax = xmax + 2 * eps_sx
    nfft = int((Tmax / xmax) * nx) + 1

    fft_grid = np.linspace(
        -nfft // 2 * dx, (nfft // 2 - 1) * dx, nfft, dtype=np.float64
    )

    auxs_in = np.zeros(nfft, dtype=np.complex128)
    auxp_in = np.zeros(nfft, dtype=np.complex128)

    for i, x in enumerate(fft_grid):
        if -eps_sx <= x <= eps_sx:
            auxs_in[i] = (1.0 / delta) * smearing_func(x, smearing_type)
        if -eps_px <= x <= eps_px:
            auxp_in[i] = 1.0 / (x + 1j * delta_ratio)

    zero_index = np.searchsorted(fft_grid, 0.0)
    auxs_in = np.roll(auxs_in, -zero_index)

    auxs_out = np.fft.fft(auxs_in)
    auxp_out = np.fft.fft(auxp_in)

    auxp_out *= auxs_out * (2 * Tmax)

    auxp_in = np.fft.ifft(auxp_out)

    ix_start = np.searchsorted(fft_grid, -xmax)
    ix_end = ix_start + nx
    g_smear = auxp_in[ix_start:ix_end].copy()

    return xgrid, g_smear


class SmearingData:
    """
    Encapsulates smearing grid data and computes its memory usage.
    """

    def __init__(self, smearing_func: Callable[[float, str], float]):
        """
        Initialize SmearingData.

        Parameters
        ----------
        `smearing_func` : callable
            A function f(x, type) returning the smearing function evaluated at `x` for a given type.
        """
        self.xgrid: Optional[np.ndarray] = None
        self.g_smear: Optional[np.ndarray] = None
        self.smearing_func = smearing_func

    @timed_function("smearing_init")
    def initialize(
        self,
        smearing_type: str,
        delta: float,
        delta_ratio: float,
        xmax: float,
    ):
        """
        Initialize the smearing grid and compute `g_smear` values.
        """
        self.xgrid, self.g_smear = initialize_smearing_grid(
            smearing_type=smearing_type,
            delta=delta,
            delta_ratio=delta_ratio,
            xmax=xmax,
            smearing_func=self.smearing_func,
        )

    def memory_usage(self) -> float:
        """
        Compute memory usage in MB, following Fortran logic.
        """
        cost = 0.0
        if self.xgrid is not None and self.xgrid.size > 0:
            cost += self.xgrid.size * 8.0 / 1_000_000.0
        if self.g_smear is not None and self.g_smear.size > 0:
            cost += self.g_smear.size * 16.0 / 1_000_000.0
        return cost
