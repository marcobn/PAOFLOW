"""Adaptive smearing of the energy-conserving deltas of the electron-phonon drivers.

A delta function summed on a uniform k- or q-grid is replaced by a Gaussian whose width follows
the variation of its argument across one grid cell (Yates, Wang, Vanderbilt and Souza, Phys. Rev.
B 75, 195121 (2007); PAOFLOW's ``adaptive_smearing`` of the DOS and of the dielectric function):

.. math::

    \\sigma_{n\\mathbf k} = a\\,|\\nabla_{\\mathbf k}\\varepsilon_{n\\mathbf k}|\\,\\Delta k,
    \\qquad \\Delta k = \\Big(\\frac{8\\pi^3}{\\Omega N}\\Big)^{1/3},

with :math:`a` of order one (0.7 for Gaussians) and a floor that keeps the width finite at the
band extrema.  The electron-phonon deltas depend on two states and on the grid they are summed
over:

* scattering rates, summed over q at fixed :math:`n\\mathbf k`:
  :math:`\\delta(\\varepsilon_{m\\mathbf k+\\mathbf q} - \\varepsilon_{n\\mathbf k} \\mp
  \\omega_{\\mathbf q\\nu})` varies with q through :math:`\\mathbf v_{m\\mathbf k+\\mathbf q}`
  (the phonon group velocity, two orders of magnitude smaller, is neglected):
  :func:`state_widths` of the final state with the q spacing;
* phonon-assisted absorption, summed over k and q: :func:`pair_widths`,
  :math:`a\\sqrt{|\\mathbf v_{i\\mathbf k} - \\mathbf v_{j\\mathbf k+\\mathbf q}|^2\\Delta k^2
  + |\\mathbf v_{j\\mathbf k+\\mathbf q}|^2\\Delta q^2}`, the k-part being the width of the
  direct (vertical) transitions;
* the Fermi-surface double delta of the Eliashberg function, summed over k and q:
  :func:`state_widths` of each state with the k spacing, and a Fermi level consistent with the
  state-dependent widths (:func:`fermi_level_adaptive`).

EPW's ``adapt_smearing`` (``broadening`` in ``utilities.f90``) is the first case with the
standard deviation of a uniform distribution over the cell (``1/sqrt(12)`` per axis) and a
1 meV floor.  The widths are in the energy unit of the velocities (``energy x length``) times
the spacing (``1/length``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc


def grid_spacing(cell_volume: float, n_per_axis: int) -> float:
    """Linear size of a cell of a uniform ``n^3`` grid in reciprocal space.

    Parameters
    ----------
    cell_volume : float
        Volume of the real-space cell (bohr^3).
    n_per_axis : int
        Grid points per axis.

    Returns
    -------
    float
        :math:`(8\\pi^3 / (\\Omega N))^{1/3}` with :math:`N = n^3` (1/bohr), PAOFLOW's ``dk``.
    """
    return (8.0 * np.pi**3 / (float(cell_volume) * int(n_per_axis) ** 3)) ** (1.0 / 3.0)


def state_widths(
    velocity: NDArray[np.float64], spacing: float, afac: float, floor: float
) -> NDArray[np.float64]:
    """Width of the delta of one state, ``max(afac |v| spacing, floor)``.

    Parameters
    ----------
    velocity : ndarray ``(..., 3)``
        Band group velocities (energy x length).
    spacing : float
        Grid spacing of the summation variable (1/length), :func:`grid_spacing`.
    afac : float
        Prefactor :math:`a` (0.7 for Gaussians).
    floor : float
        Minimum width (energy).

    Returns
    -------
    ndarray ``(...)``
        Widths (energy).
    """
    speed = np.linalg.norm(np.asarray(velocity, dtype=float), axis=-1)
    return np.maximum(float(afac) * float(spacing) * speed, float(floor))


def pair_widths(
    velocity_k: NDArray[np.float64],
    velocity_kq: NDArray[np.float64],
    spacing_k: float,
    spacing_q: float,
    afac: float,
    floor: float,
) -> NDArray[np.float64]:
    """Width of ``delta(e_{j k+q} - e_{i k} - ...)`` summed over k and q.

    Parameters
    ----------
    velocity_k, velocity_kq : ndarray ``(..., 3)``, broadcastable
        Group velocities of the initial state at k and of the final state at k+q.
    spacing_k, spacing_q : float
        Spacings of the k- and q-grids (1/length); ``spacing_q = 0`` gives the width of a
        vertical transition summed over k alone.
    afac, floor : float
        As :func:`state_widths`.

    Returns
    -------
    ndarray
        :math:`a\\sqrt{|\\mathbf v_i - \\mathbf v_j|^2\\Delta k^2 + |\\mathbf v_j|^2\\Delta q^2}`,
        floored.
    """
    vk = np.asarray(velocity_k, dtype=float)
    vkq = np.asarray(velocity_kq, dtype=float)
    along_k = np.sum((vk - vkq) ** 2, axis=-1) * float(spacing_k) ** 2
    along_q = np.sum(vkq**2, axis=-1) * float(spacing_q) ** 2
    return np.maximum(float(afac) * np.sqrt(along_k + along_q), float(floor))


def fermi_level_adaptive(
    energies: NDArray[np.float64],
    widths: NDArray[np.float64],
    nelec: float,
    spin_degeneracy: float = 2.0,
    niter: int = 120,
) -> float:
    """Fermi level of ``nelec`` electrons with a Gaussian delta of state-dependent width.

    Parameters
    ----------
    energies, widths : ndarray ``(nk, nband)``
        Band energies and the widths of their deltas (same unit).
    nelec : float
        Electrons per cell.
    spin_degeneracy : float, optional
    niter : int, optional
        Bisection steps.

    Returns
    -------
    float
        :math:`\\mu` with :math:`\\frac{g_s}{N_k}\\sum_{n\\mathbf k}\\tfrac12\\,
        \\mathrm{erfc}\\big((\\varepsilon_{n\\mathbf k} - \\mu)/\\sigma_{n\\mathbf k}\\big) =
        n_{\\rm el}` (the occupation that integrates the Gaussian delta, QE ``efermig`` with
        ``ngauss = 0``).
    """
    e = np.asarray(energies, dtype=float)
    w = np.asarray(widths, dtype=float)
    nk = e.shape[0]
    lo, hi = float(e.min() - 10.0 * w.max()), float(e.max() + 10.0 * w.max())
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        occ = spin_degeneracy / nk * np.sum(0.5 * erfc((e - mid) / w))
        if occ < nelec:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def summarize_widths(widths: NDArray[np.float64], scale: float = 1.0) -> dict:
    """``{'min', 'median', 'max'}`` of the widths (times ``scale``), for reports."""
    w = np.asarray(widths, dtype=float).ravel() * float(scale)
    return {'min': float(w.min()), 'median': float(np.median(w)), 'max': float(w.max())}
