r"""Orbital magnetic moment of Bloch states and the orbital magnetization (modern theory).

For a non-degenerate band :math:`n` (Xiao, Chang and Niu, Rev. Mod. Phys. 82, 1959 (2010)),

.. math::

    \mathbf m_n(\mathbf k) = -\frac{ie}{2\hbar}\langle\nabla_{\mathbf k}u_n|\times[H(\mathbf k) - \varepsilon_n]
    |\nabla_{\mathbf k}u_n\rangle,\qquad
    m^z_n = \frac{e}{\hbar}\sum_{m\notin D(n)}\frac{\mathrm{Im}\,(V^x_{nm}V^y_{mn})}{\varepsilon_m - \varepsilon_n},

with :math:`V = \partial H/\partial\mathbf k` the velocity matrix (PAOFLOW's ``pksp``, with the
:math:`D_{nm}` term of the cell-centred gauge) and :math:`D(n)` the states degenerate with :math:`n`
(left out; inside a degenerate group the per-band values depend on the basis, their sum does not). The
moment is that of an electron (charge :math:`-e`): a state that circulates counter-clockwise about
:math:`\hat{\mathbf z}` has :math:`m^z < 0`, and its energy in a field is :math:`\varepsilon_n - \mathbf m_n
\cdot\mathbf B` to first order. With the Berry curvature :math:`\boldsymbol\Omega_n` (same velocities), the
orbital magnetization per cell is

.. math::

    \mathbf M(\mu) = \frac{g}{N_k}\sum_{n\mathbf k} f(\varepsilon_{n\mathbf k} - \mu)
    \Bigl[\mathbf m_{n\mathbf k} + \frac{e}{\hbar}(\mu - \varepsilon_{n\mathbf k})\,\boldsymbol\Omega_{n\mathbf k}\Bigr]

(Thonhauser, Ceresoli, Vanderbilt and Resta, Phys. Rev. Lett. 95, 137205 (2005); Xiao, Shi and Niu, Phys.
Rev. Lett. 95, 137204 (2005)), :math:`g` the spin degeneracy of the bands. Units: :math:`\mu_B` per cell;
:math:`(e/\hbar)\,\mathrm{eV\,bohr^2} = \mu_B\cdot\mathrm{eV\,bohr^2}/(\hbar^2/2m_e) = \mu_B/13.6057`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

RYDBERG_EV = (
    13.605693122994  # hbar^2 / (2 m_e bohr^2) in eV: (e/hbar) eV bohr^2 = mu_B / RYDBERG_EV
)
CYCLIC = ((1, 2), (2, 0), (0, 1))


def moment_and_curvature(
    energies: NDArray[np.float64],
    velocities: NDArray[np.complex128],
    degeneracy_tol: float = 1.0e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Orbital moment and Berry curvature of every band at one k-point.

    Parameters
    ----------
    energies : ndarray, shape ``(nb,)``
        Band energies (eV).
    velocities : ndarray, shape ``(3, nb, nb)``, complex
        :math:`\\pm\\hbar v^a_{nm}` in the band basis (eV bohr); the overall sign cancels.
    degeneracy_tol : float, optional
        Bands closer than this (eV) count as degenerate and are left out of each other's sums.

    Returns
    -------
    moment : ndarray, shape ``(nb, 3)``
        :math:`\\mathbf m_n` (:math:`\\mu_B`).
    curvature : ndarray, shape ``(nb, 3)``
        :math:`\\boldsymbol\\Omega_n = -2\\sum_{m\\notin D(n)}\\mathrm{Im}(\\mathbf V_{nm}\\times\\mathbf
        V_{mn})/(\\varepsilon_n - \\varepsilon_m)^2` component by component (bohr^2), the
        convention of :func:`PAOFLOW.response.kubo_kernels.berry_curvature`.
    """
    gap = energies[None, :] - energies[:, None]  # [n, m] = e_m - e_n
    outside = np.abs(gap) > degeneracy_tol
    inverse = np.where(outside, 1.0 / np.where(outside, gap, 1.0), 0.0)
    moment = np.zeros((energies.size, 3))
    curvature = np.zeros((energies.size, 3))
    for c, (a, b) in enumerate(CYCLIC):
        product = np.imag(velocities[a] * velocities[b].T)  # Im V^a_nm V^b_mn
        moment[:, c] = np.sum(product * inverse, axis=1) / RYDBERG_EV
        curvature[:, c] = -2.0 * np.sum(product * inverse**2, axis=1)
    return moment, curvature


def magnetization_density(
    energies: NDArray[np.float64],
    moment: NDArray[np.float64],
    curvature: NDArray[np.float64],
    chemical_potentials: NDArray[np.float64],
    temperature_ev: float = 0.0,
) -> NDArray[np.float64]:
    """Sum over the bands of one k-point of the modern-theory integrand.

    Parameters
    ----------
    energies : ndarray, shape ``(nb,)``
        Band energies (eV).
    moment, curvature : ndarray, shape ``(nb, 3)``
        From :func:`moment_and_curvature` (:math:`\\mu_B`, bohr^2).
    chemical_potentials : ndarray, shape ``(nmu,)``
        :math:`\\mu` (eV).
    temperature_ev : float, optional
        :math:`k_BT` (eV); 0 gives step occupations.

    Returns
    -------
    ndarray, shape ``(nmu, 3)``
        :math:`\\sum_n f_n[\\mathbf m_n + (\\mu - \\varepsilon_n)\\boldsymbol\\Omega_n/13.6057]`
        (:math:`\\mu_B`).
    """
    mu = np.asarray(chemical_potentials, dtype=float)[:, None]
    if temperature_ev > 0.0:
        x = np.clip((energies[None, :] - mu) / temperature_ev, -200.0, 200.0)
        occupation = 1.0 / (np.exp(x) + 1.0)
    else:
        occupation = (energies[None, :] <= mu).astype(float)
    weight = occupation[:, :, None]
    return np.sum(
        weight
        * (moment[None] + (mu - energies[None, :])[:, :, None] * curvature[None] / RYDBERG_EV),
        axis=1,
    )
