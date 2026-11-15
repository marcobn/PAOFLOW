"""Intraband (Drude) response of metals from scattering channels.

The intraband dielectric function of a metal follows from its plasma frequency
:math:`\\omega_p` and the relaxation rates of
:mod:`PAOFLOW.boltzmann.scattering_channels`, which combine by Matthiessen's
rule:

* :func:`drude_dielectric` -- simple Drude, frequency-independent rate;
* :func:`extended_drude_dielectric` -- extended (generalized) Drude with the
  frequency-dependent rate :math:`1/\\tau(\\omega, T)` and the mass enhancement
  :math:`\\lambda(\\omega, T)` of the electron-phonon channel (Allen 1971);
* :func:`dc_resistivity` -- :math:`\\rho(T) = 1/(\\varepsilon_0\\omega_p^2\\tau)`;
* :func:`plasma_frequency_tensor` -- :math:`\\omega_p^2` from band velocities at
  the Fermi surface on any k-grid.

Energies and frequencies in eV (:math:`\\hbar\\omega`), rates in 1/s,
temperatures in K.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..boltzmann.scattering_channels import (
    EPSILON0_SI,
    HBAR_EV_S,
    ScatteringChannel,
    matthiessen,
    total_mass_enhancement,
)

E2_EV_BOHR = 27.211386245988  # e^2 in eV bohr (one hartree times one bohr)


def drude_dielectric(
    omega_ev: ArrayLike, plasma_ev: float, rate_per_s: float | ArrayLike
) -> NDArray[np.complex128]:
    """Simple Drude intraband dielectric function.

    Parameters
    ----------
    omega_ev : array_like
        Photon energies (eV), positive.
    plasma_ev : float
        Plasma frequency :math:`\\hbar\\omega_p` (eV).
    rate_per_s : float or array_like
        Relaxation rate :math:`1/\\tau` (1/s), constant or per frequency.

    Returns
    -------
    ndarray, complex
        :math:`\\varepsilon_D(\\omega) = -\\omega_p^2 / [\\omega(\\omega + i/\\tau)]`.
    """
    omega = np.asarray(omega_ev, dtype=float)
    gamma = HBAR_EV_S * np.asarray(rate_per_s, dtype=float)
    return -(plasma_ev**2) / (omega * (omega + 1j * gamma))


def extended_drude_dielectric(
    omega_ev: ArrayLike,
    plasma_ev: float,
    channels: Sequence[ScatteringChannel],
    temperature_k: float,
) -> NDArray[np.complex128]:
    """Extended (generalized) Drude intraband dielectric function.

    Parameters
    ----------
    omega_ev : array_like
        Photon energies (eV), positive.
    plasma_ev : float
        Plasma frequency :math:`\\hbar\\omega_p` (eV).
    channels : sequence of ScatteringChannel
        Scattering channels (Matthiessen); the electron-phonon
        :class:`~PAOFLOW.boltzmann.scattering_channels.TransportSpectralChannel`
        supplies the frequency-dependent rate and the mass enhancement.
    temperature_k : float
        Temperature (K).

    Returns
    -------
    ndarray, complex
        Intraband :math:`\\varepsilon_D(\\omega, T)`.

    Notes
    -----
    With the memory function :math:`M(\\omega) = 1/\\tau(\\omega) - i\\omega\\lambda(\\omega)`
    (Allen 1971; Shulga, Dolgov and Maksimov 1991)

    .. math::

        \\varepsilon_D(\\omega, T) = -\\frac{\\omega_p^2}
            {\\omega\\left[\\omega\\,(1 + \\lambda(\\omega, T)) + i/\\tau(\\omega, T)\\right]},

    where :math:`1/\\tau(\\omega, T)` is the sum of the channels' optical rates and
    :math:`\\lambda(\\omega, T)` the sum of their mass enhancements (the
    electron-phonon one; constant rates have none).  For a frequency-independent
    rate it reduces to :func:`drude_dielectric`.  Above the phonon energies
    :math:`1/\\tau(\\omega)` exceeds the DC rate (Holstein absorption).
    """
    omega = np.asarray(omega_ev, dtype=float)
    gamma = HBAR_EV_S * matthiessen(channels, temperature_k, omega_ev=omega)
    lam = total_mass_enhancement(channels, omega, temperature_k)
    return -(plasma_ev**2) / (omega * (omega * (1.0 + lam) + 1j * gamma))


def dc_resistivity(
    plasma_ev: float,
    channels: Sequence[ScatteringChannel],
    temperature_k: float,
    energy_ev: float = 0.0,
) -> dict:
    """DC resistivity of a metal in the Drude (relaxation-time) model.

    Parameters
    ----------
    plasma_ev : float
        Plasma frequency :math:`\\hbar\\omega_p` (eV).
    channels : sequence of ScatteringChannel
        Scattering channels (Matthiessen).
    temperature_k : float
        Temperature (K).
    energy_ev : float, optional
        Carrier energy relative to the Fermi level (eV) for energy-resolved
        channels (default 0).

    Returns
    -------
    dict
        ``rho_ohm_m`` (total), ``rate_per_s`` (total) and ``channels``
        (``{name: (rate_per_s, rho_ohm_m)}``).

    Notes
    -----
    :math:`\\rho = 1/(\\varepsilon_0\\,\\omega_p^2\\,\\tau)` with
    :math:`1/\\tau = \\sum_c 1/\\tau_c`, so the channel resistivities add.
    """
    omega_p = float(plasma_ev) / HBAR_EV_S
    per_channel = {}
    total = 0.0
    for channel in channels:
        rate = float(np.asarray(channel.rate(temperature_k, energy_ev=np.array(energy_ev))))
        per_channel[channel.name] = (rate, rate / (EPSILON0_SI * omega_p**2))
        total += rate
    return {
        'rho_ohm_m': total / (EPSILON0_SI * omega_p**2),
        'rate_per_s': total,
        'channels': per_channel,
    }


def plasma_frequency_tensor(
    velocities_ev_bohr: ArrayLike,
    delta_per_ev: ArrayLike,
    cell_volume_bohr3: float,
    spin_degeneracy: float = 2.0,
) -> NDArray[np.float64]:
    """Plasma-frequency tensor from band velocities on a uniform k-grid.

    Parameters
    ----------
    velocities_ev_bohr : array_like, shape ``(nk, nband, 3)``
        Band velocities :math:`\\partial\\varepsilon/\\partial\\mathbf k` (eV bohr).
    delta_per_ev : array_like, shape ``(nk, nband)``
        Fermi-surface weights :math:`\\delta(\\varepsilon_{n\\mathbf k} - \\mu)`
        (1/eV), e.g. a Gaussian or :math:`-\\partial f/\\partial\\varepsilon`.
    cell_volume_bohr3 : float
        Unit-cell volume (bohr^3).
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it.

    Returns
    -------
    ndarray, shape ``(3, 3)``
        :math:`(\\hbar\\omega_p)^2_{\\alpha\\beta}` (eV^2).

    Notes
    -----
    .. math::

        (\\hbar\\omega_p)^2_{\\alpha\\beta} = \\frac{4\\pi e^2 g_s}{\\Omega N_k}
            \\sum_{n\\mathbf k} \\frac{\\partial\\varepsilon_{n\\mathbf k}}{\\partial k_\\alpha}
            \\frac{\\partial\\varepsilon_{n\\mathbf k}}{\\partial k_\\beta}
            \\delta(\\varepsilon_{n\\mathbf k} - \\mu),

    with :math:`e^2 = 27.2114` eV bohr; for free electrons it is
    :math:`4\\pi n e^2\\hbar^2/m`.
    """
    v = np.asarray(velocities_ev_bohr, dtype=float)
    delta = np.asarray(delta_per_ev, dtype=float)
    nk = v.shape[0]
    tensor = np.einsum('kna,knb,kn->ab', v, v, delta) / nk
    return 4.0 * np.pi * E2_EV_BOHR * spin_degeneracy / float(cell_volume_bohr3) * tensor
