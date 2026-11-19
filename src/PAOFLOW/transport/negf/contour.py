"""Energy points and weights for the NEGF density matrix: the equilibrium contour and the bias window.

The equilibrium density of an open system is a real-axis integral of the retarded Green's function, sharp near the
real axis. Its integrand is analytic in the upper half plane except at the poles of the Fermi function,
:math:`z_n = \\mu + i\\omega_n`, :math:`\\omega_n = (2n+1)\\pi k_BT`. The integral is therefore deformed onto a contour
far from the real axis, where :math:`G(z)` is smooth, plus the residues of the poles it passes (Brandbyge et al.,
PRB 65, 165401 (2002); Ozaki, PRB 75, 035123 (2007)). The non-equilibrium part lives in the bias window and stays on
the real axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K


@dataclass(frozen=True)
class EnergyQuadrature:
    """Points and weights of an energy integral, :math:`\\int dE\\, g(E) \\approx \\sum_j w_j\\, g(z_j)`.

    Attributes
    ----------
    points : ndarray, shape ``(n,)``, complex
        Energies :math:`z_j` (eV); real for the bias window.
    weights : ndarray, shape ``(n,)``, complex
        Weights :math:`w_j` (eV), including the Fermi factor for the equilibrium contour.
    """

    points: NDArray[np.complex128]
    weights: NDArray[np.complex128]

    def __len__(self) -> int:
        return len(self.points)


def fermi_function(
    energy: NDArray[np.complexfloating] | NDArray[np.floating] | complex | float,
    chemical_potential: float,
    temperature_ev: float,
) -> NDArray[np.complex128]:
    """Fermi function :math:`1/(e^{(z-\\mu)/k_BT}+1)` at real or complex energies, without overflow.

    Parameters
    ----------
    energy : array_like or scalar
        Energies (eV), real or complex.
    chemical_potential : float
        :math:`\\mu` (eV).
    temperature_ev : float
        :math:`k_BT` (eV), positive.

    Returns
    -------
    ndarray, complex
        :math:`f(z)`; real for real energies.
    """
    x = (np.asarray(energy, dtype=complex) - chemical_potential) / temperature_ev
    positive = x.real > 0.0
    safe = np.where(positive, -x, x)
    exp_safe = np.exp(safe)
    return np.where(positive, exp_safe / (1.0 + exp_safe), 1.0 / (exp_safe + 1.0))


def equilibrium_contour(
    energy_bottom: float,
    chemical_potential: float,
    temperature_ev: float,
    arc_points: int = 30,
    line_points: int = 40,
    fermi_poles: int = 10,
    line_tail: float = 40.0,
    pole_margin: float = 10.0,
) -> EnergyQuadrature:
    """Contour quadrature of :math:`\\int dE\\, G(E)\\, f(E-\\mu)` for a function analytic in the upper half plane.

    Parameters
    ----------
    energy_bottom : float
        Real energy below every state of the system (eV); the contour starts there.
    chemical_potential : float
        :math:`\\mu` (eV).
    temperature_ev : float
        :math:`k_BT` (eV), positive.
    arc_points, line_points : int, optional
        Gauss-Legendre points on the arc and on the horizontal line.
    fermi_poles : int, optional
        Number :math:`N_p` of Fermi poles enclosed; the line runs at :math:`\\mathrm{Im}\\,z = 2N_p\\pi k_BT`.
    line_tail : float, optional
        The line ends at :math:`\\mu +` ``line_tail`` :math:`k_BT`, where :math:`f` is negligible.
    pole_margin : float, optional
        The arc meets the line at :math:`\\mathrm{Re}\\,z = \\mu -` ``pole_margin`` :math:`k_BT`.

    Returns
    -------
    EnergyQuadrature
        Points and weights with :math:`\\int_{-\\infty}^{\\infty} dE\\, G(E)f(E-\\mu) = \\sum_j w_j\\, G(z_j)`.

    Raises
    ------
    ValueError
        If the contour does not start below the arc's end or the temperature is not positive.

    Notes
    -----
    The real axis from ``energy_bottom`` to :math:`+\\infty` and the contour C (a circular arc from
    ``energy_bottom`` to :math:`\\mu - \\gamma + ih`, then the line :math:`x + ih`, :math:`h = 2N_p\\pi k_BT`)
    enclose the poles :math:`z_n = \\mu + i(2n+1)\\pi k_BT`, :math:`n < N_p`, of :math:`f`, each with residue
    :math:`-k_BT`:

    .. math::

        \\int dE\\, G(E) f(E-\\mu) = \\int_C dz\\, G(z) f(z-\\mu) - 2\\pi i\\, k_BT \\sum_{n<N_p} G(z_n).

    On the line :math:`e^{ih/k_BT} = 1`, so :math:`f` there is the real Fermi function. With ``G`` a retarded
    Green's function, :math:`-\\mathrm{Im}\\sum_j w_j G_{ii}(z_j)/\\pi` is the occupation of orbital ``i``.
    """
    if temperature_ev <= 0.0:
        raise ValueError('the contour needs a positive temperature')
    height = 2.0 * fermi_poles * np.pi * temperature_ev
    arc_end = chemical_potential - pole_margin * temperature_ev
    if energy_bottom >= arc_end:
        raise ValueError(
            f'energy_bottom ({energy_bottom:.3f} eV) must lie below mu - pole_margin k_B T ({arc_end:.3f} eV)'
        )
    centre = (arc_end**2 + height**2 - energy_bottom**2) / (2.0 * (arc_end - energy_bottom))
    radius = centre - energy_bottom
    angle_end = np.arctan2(height, arc_end - centre)

    nodes, node_weights = np.polynomial.legendre.leggauss(arc_points)
    theta = 0.5 * (np.pi - angle_end) * nodes + 0.5 * (np.pi + angle_end)
    theta_weights = 0.5 * (np.pi - angle_end) * node_weights
    arc = centre + radius * np.exp(1j * theta)
    # from energy_bottom (theta = pi) to the arc's end (theta = angle_end): minus the integral over theta
    arc_weights = -theta_weights * 1j * radius * np.exp(1j * theta)

    nodes, node_weights = np.polynomial.legendre.leggauss(line_points)
    line_end = chemical_potential + line_tail * temperature_ev
    x = 0.5 * (line_end - arc_end) * nodes + 0.5 * (line_end + arc_end)
    line = x + 1j * height
    line_weights = 0.5 * (line_end - arc_end) * node_weights.astype(complex)

    poles = chemical_potential + 1j * (2 * np.arange(fermi_poles) + 1) * np.pi * temperature_ev
    pole_weights = np.full(fermi_poles, -2j * np.pi * temperature_ev)

    points = np.concatenate([arc, line, poles])
    contour_weights = np.concatenate([arc_weights, line_weights]) * fermi_function(
        np.concatenate([arc, line]), chemical_potential, temperature_ev
    )
    return EnergyQuadrature(points=points, weights=np.concatenate([contour_weights, pole_weights]))


def bias_window(
    chemical_potential_left: float,
    chemical_potential_right: float,
    temperature_ev: float,
    spacing: float,
    tail: float = 10.0,
) -> EnergyQuadrature:
    """Real energies and trapezoid weights covering the bias window.

    Parameters
    ----------
    chemical_potential_left, chemical_potential_right : float
        :math:`\\mu_L`, :math:`\\mu_R` (eV).
    temperature_ev : float
        :math:`k_BT` (eV).
    spacing : float
        Largest energy step (eV); keep it below the narrowest feature of the transmission.
    tail : float, optional
        The window extends ``tail`` :math:`k_BT` beyond both chemical potentials.

    Returns
    -------
    EnergyQuadrature
        Real points (stored as complex) and trapezoid weights; empty when :math:`\\mu_L = \\mu_R`.
    """
    if chemical_potential_left == chemical_potential_right:
        return EnergyQuadrature(
            points=np.zeros(0, dtype=complex), weights=np.zeros(0, dtype=complex)
        )
    low = min(chemical_potential_left, chemical_potential_right) - tail * temperature_ev
    high = max(chemical_potential_left, chemical_potential_right) + tail * temperature_ev
    count = max(3, int(np.ceil((high - low) / spacing)) + 1)
    energies = np.linspace(low, high, count)
    weights = np.full(count, energies[1] - energies[0])
    weights[0] = weights[-1] = 0.5 * (energies[1] - energies[0])
    return EnergyQuadrature(points=energies.astype(complex), weights=weights.astype(complex))
