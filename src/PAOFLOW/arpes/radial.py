"""Radial dipole integrals of the plane-wave final state, tabulated and interpolated.

.. math::

    B^{l'}_{nl}(k) = i^{l'}\\int_0^\\infty dr\\, R_{nl}(r)\\, r^3\\, j_{l'}(kr)
    = i^{l'}\\int_0^\\infty dr\\, \\chi_{nl}(r)\\, r^2\\, j_{l'}(kr),

with :math:`\\chi = rR` the PAO radial tables (bohr), :math:`l' = l \\pm 1` and :math:`k` in
1/bohr (chinook ``radint_lib``; here Simpson's rule on the tabulated grid instead of an
adaptive integrator).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.special import spherical_jn

BOHR_ANGSTROM = 0.529177210903


def radial_integral(
    r_bohr: NDArray[np.float64],
    chi: NDArray[np.float64],
    lprime: int,
    k_bohr: NDArray[np.float64] | float,
) -> NDArray[np.complex128]:
    """``B^{l'}(k)`` of one radial function on a grid of ``k`` (1/bohr).

    Parameters
    ----------
    r_bohr : ndarray ``(nr,)``
        Radial grid (bohr), starting at or near zero.
    chi : ndarray ``(nr,)``
        ``r R(r)`` on the grid.
    lprime : int
        Final-state angular momentum.
    k_bohr : array_like
        Photoelectron momenta (1/bohr).

    Returns
    -------
    ndarray, complex
        ``i^l' int chi r^2 j_l'(k r) dr`` for every ``k``.
    """
    k = np.atleast_1d(np.asarray(k_bohr, dtype=float))
    r = np.asarray(r_bohr, dtype=float)
    integrand = spherical_jn(lprime, np.multiply.outer(k, r)) * (np.asarray(chi) * r**2)[None, :]
    return (1j**lprime) * simpson(integrand, x=r, axis=1)


class RadialIntegrals:
    """Tabulated ``B^{l'}(|k_f|)`` of every radial function of a basis, with interpolation.

    Parameters
    ----------
    functions : mapping ``key -> (r_bohr, chi, l)``
        Distinct radial functions (``chi = r R(r)``) with their angular momentum.
    k_max_bohr : float
        Largest photoelectron momentum to tabulate (1/bohr).
    nk : int, optional
        Number of tabulation points in ``[0, k_max]``.
    phase_shifts : mapping ``(key, lprime) -> complex``, optional
        Final-state scattering phase factors multiplying ``B^{l'}`` (chinook ``phase_shifts``).
    fixed : mapping ``(key, lprime) -> complex``, optional
        Constant values replacing the integrals (chinook ``'fixed'`` radial type).
    """

    def __init__(
        self,
        functions: Mapping[object, tuple[NDArray[np.float64], NDArray[np.float64], int]],
        k_max_bohr: float,
        nk: int = 400,
        phase_shifts: Mapping[tuple[object, int], complex] | None = None,
        fixed: Mapping[tuple[object, int], complex] | None = None,
    ) -> None:
        self.k_grid = np.linspace(0.0, max(float(k_max_bohr), 1e-6), int(nk))
        self.tables: dict[tuple[object, int], NDArray[np.complex128]] = {}
        phase_shifts = {} if phase_shifts is None else dict(phase_shifts)
        fixed = {} if fixed is None else dict(fixed)
        for key, (r, chi, l) in functions.items():
            for lprime in (l - 1, l + 1):
                if lprime < 0:
                    continue
                if (key, lprime) in fixed:
                    table = np.full(self.k_grid.size, complex(fixed[(key, lprime)]))
                else:
                    table = radial_integral(r, chi, lprime, self.k_grid)
                self.tables[(key, lprime)] = table * complex(phase_shifts.get((key, lprime), 1.0))

    def __call__(
        self, key: object, lprime: int, k_bohr: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Interpolated ``B^{l'}`` of radial function ``key`` at the momenta ``k_bohr``."""
        table = self.tables[(key, lprime)]
        k = np.asarray(k_bohr, dtype=float)
        return np.interp(k, self.k_grid, table.real) + 1j * np.interp(k, self.k_grid, table.imag)
