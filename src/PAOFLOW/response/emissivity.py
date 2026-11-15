"""Thermal emissivity from a temperature-dependent dielectric function.

By Kirchhoff's law the emissivity of a body at uniform temperature equals its
absorptance.  :func:`emissivity_from_dielectric` turns
:math:`\\varepsilon(\\omega, T)` into the directional and hemispherical spectral
emissivity of a free-standing slab and of the opaque half-space, and into the
Planck-weighted totals at each temperature
(:mod:`PAOFLOW.response.do_epsilon` supplies the Fresnel, slab, hemispherical
and Planck integrals).  It is shared by

* semiconductors -- :func:`PAOFLOW.elphon.phonon_assisted_absorption.thermal_emissivity`
  (direct + phonon-assisted + free-carrier absorption), and
* metals -- :func:`metal_thermal_emissivity` (interband + extended Drude from
  :mod:`PAOFLOW.response.drude`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..boltzmann.scattering_channels import ScatteringChannel, matthiessen


def emissivity_from_dielectric(
    omega_ev: ArrayLike,
    eps1: ArrayLike,
    eps2: ArrayLike,
    temps_k: Sequence[float],
    thickness_m: float,
    angles_deg: Sequence[float] = (0.0, 30.0, 60.0),
    ntheta: int = 90,
) -> dict:
    """Slab and half-space emissivity spectra and totals at each temperature.

    Parameters
    ----------
    omega_ev : array_like, shape ``(nomega,)``
        Photon energies (eV), positive and increasing.
    eps1, eps2 : array_like, shape ``(ntemp, nomega)``
        Real (vacuum 1 included) and imaginary parts of the dielectric function
        at each temperature.
    temps_k : sequence of float
        Temperatures (K).
    thickness_m : float
        Slab thickness (m); ``inf`` gives the opaque half-space.
    angles_deg : sequence of float, optional
        Emission angles (degrees) of the directional spectra.
    ntheta : int, optional
        Polar-angle samples of the hemispherical integral.

    Returns
    -------
    dict
        ``omega_ev``, ``temps_k``, ``eps1``, ``eps2``, ``hemispherical_slab``,
        ``hemispherical_opaque`` ``(ntemp, nomega)``, ``directional_slab``
        ``(ntemp, nangle, nomega)``, ``total_slab``, ``total_opaque``,
        ``planck_coverage`` ``(ntemp,)``, ``thickness_m``, ``angles_deg``.

    Notes
    -----
    The directional emissivity is the slab absorptance
    (:func:`~PAOFLOW.response.do_epsilon.slab_directional_absorptance`), its
    hemispherical average
    :math:`2\\int_0^{\\pi/2}\\varepsilon(\\theta,\\omega)\\cos\\theta\\sin\\theta\\,d\\theta`,
    and the total the Planck-weighted average **at the same temperature**,
    over the photon-energy grid (``planck_coverage`` is the covered fraction of
    the blackbody spectrum).
    """
    from .do_epsilon import (
        planck_coverage,
        slab_directional_absorptance,
        spectral_hemispherical_emissivity,
        total_hemispherical_emissivity,
    )

    omega = np.asarray(omega_ev, dtype=float)
    temps = np.atleast_1d(np.asarray(temps_k, dtype=float))
    eps1 = np.atleast_2d(np.asarray(eps1, dtype=float))
    eps2 = np.atleast_2d(np.asarray(eps2, dtype=float))
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    ntemp, nomega = temps.size, omega.size
    out = {
        'omega_ev': omega,
        'temps_k': temps,
        'eps1': np.zeros((ntemp, nomega)),
        'eps2': np.zeros((ntemp, nomega)),
        'hemispherical_slab': np.zeros((ntemp, nomega)),
        'hemispherical_opaque': np.zeros((ntemp, nomega)),
        'directional_slab': np.zeros((ntemp, angles.size, nomega)),
        'total_slab': np.zeros(ntemp),
        'total_opaque': np.zeros(ntemp),
        'planck_coverage': np.zeros(ntemp),
        'thickness_m': float(thickness_m),
        'angles_deg': angles,
    }
    for it, temp in enumerate(temps):
        e1, e2 = eps1[it], eps2[it]
        slab = spectral_hemispherical_emissivity(e1, e2, ntheta, omega, thickness_m)
        opaque = spectral_hemispherical_emissivity(e1, e2, ntheta)
        out['eps1'][it], out['eps2'][it] = e1, e2
        out['hemispherical_slab'][it], out['hemispherical_opaque'][it] = slab, opaque
        for ia, angle in enumerate(angles):
            out['directional_slab'][it, ia] = slab_directional_absorptance(
                omega, e1, e2, np.deg2rad(angle), thickness_m
            )
        out['total_slab'][it] = total_hemispherical_emissivity(omega, slab, temp)
        out['total_opaque'][it] = total_hemispherical_emissivity(omega, opaque, temp)
        out['planck_coverage'][it] = planck_coverage(omega, temp)
    return out


def metal_thermal_emissivity(
    omega_ev: ArrayLike,
    temps_k: Sequence[float],
    plasma_ev: float,
    channels: Sequence[ScatteringChannel],
    interband: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
    thickness_m: float = np.inf,
    angles_deg: Sequence[float] = (0.0, 30.0, 60.0),
    ntheta: int = 90,
    extended: bool = True,
) -> dict:
    """Thermal emissivity of a metal: interband + (extended) Drude dielectric function.

    Parameters
    ----------
    omega_ev : array_like, shape ``(nomega,)``
        Photon energies (eV), positive.  Start near 0.005-0.01 eV to cover the
        blackbody spectrum.
    temps_k : sequence of float
        Temperatures (K).
    plasma_ev : float
        Intraband plasma frequency :math:`\\hbar\\omega_p` (eV), e.g.
        ``omega_plasma_ev`` of
        :func:`~PAOFLOW.elphon.do_pao_eph_dense_q.eliashberg_dense_q` with
        ``transport=True``.
    channels : sequence of ScatteringChannel
        Scattering channels (Matthiessen): typically
        :class:`~PAOFLOW.boltzmann.scattering_channels.TransportSpectralChannel`
        (electron-phonon), a
        :class:`~PAOFLOW.boltzmann.scattering_channels.ResidualResistivity` and a
        :class:`~PAOFLOW.boltzmann.scattering_channels.FermiLiquidRate`.
    interband : tuple of array_like, optional
        ``(omega_ev, eps1, eps2)`` of the interband dielectric function (eps1
        with the vacuum 1), e.g.
        :func:`~PAOFLOW.elphon.phonon_assisted_absorption.read_direct_dielectric`
        of a ``dielectric_tensor(intraband=False)`` run on the extended basis;
        components ``(ncomp, nomega_i)`` are averaged.  ``None`` uses
        :math:`\\varepsilon_{inter} = 1`.
    thickness_m : float, optional
        Film thickness (m); metals are opaque in practice (default ``inf``).
    angles_deg : sequence of float, optional
        Emission angles (degrees) of the directional spectra.
    ntheta : int, optional
        Polar-angle samples.
    extended : bool, optional
        Extended Drude (frequency-dependent rate and mass enhancement, default)
        or simple Drude with the DC rate.

    Returns
    -------
    dict
        As :func:`emissivity_from_dielectric`, plus ``plasma_ev``,
        ``dc_rate_per_s`` and ``rho_ohm_m`` ``(ntemp,)`` and ``drude``.

    Notes
    -----
    .. math::

        \\varepsilon(\\omega, T) = \\varepsilon_{inter}(\\omega)
            - \\frac{\\omega_p^2}{\\omega\\left[\\omega(1+\\lambda(\\omega,T))
            + i/\\tau(\\omega,T)\\right]}

    (:func:`~PAOFLOW.response.drude.extended_drude_dielectric`).  In the
    infrared the emissivity of a good metal approaches the Hagen-Rubens form
    :math:`\\varepsilon \\simeq 2\\sqrt{\\varepsilon_0\\omega/\\sigma_{dc}}`, so it
    grows with the resistivity, i.e. with temperature.
    """
    from ..boltzmann.scattering_channels import EPSILON0_SI, HBAR_EV_S
    from .drude import drude_dielectric, extended_drude_dielectric

    omega = np.asarray(omega_ev, dtype=float)
    temps = np.atleast_1d(np.asarray(temps_k, dtype=float))
    if interband is None:
        inter = np.ones(omega.size, dtype=complex)
    else:
        w_i, e1_i, e2_i = (np.asarray(a, dtype=float) for a in interband)
        if e1_i.ndim > 1:
            e1_i, e2_i = e1_i.mean(axis=0), e2_i.mean(axis=0)
        if w_i.min() > omega.min() + 1.0e-9 or w_i.max() < omega.max() - 1.0e-9:
            raise ValueError(
                'interband dielectric covers %.3f-%.3f eV, the photon-energy grid %.3f-%.3f eV'
                % (w_i.min(), w_i.max(), omega.min(), omega.max())
            )
        inter = np.interp(omega, w_i, e1_i) + 1j * np.interp(omega, w_i, e2_i)
    eps = np.empty((temps.size, omega.size), dtype=complex)
    dc_rate = np.empty(temps.size)
    for it, temp in enumerate(temps):
        dc_rate[it] = float(np.asarray(matthiessen(channels, temp, energy_ev=np.array(0.0))))
        if extended:
            intra = extended_drude_dielectric(omega, plasma_ev, channels, temp)
        else:
            intra = drude_dielectric(omega, plasma_ev, dc_rate[it])
        eps[it] = inter + intra
    out = emissivity_from_dielectric(
        omega, eps.real, eps.imag, temps, thickness_m, angles_deg, ntheta
    )
    omega_p = float(plasma_ev) / HBAR_EV_S
    out['plasma_ev'] = float(plasma_ev)
    out['dc_rate_per_s'] = dc_rate
    out['rho_ohm_m'] = dc_rate / (EPSILON0_SI * omega_p**2)
    out['drude'] = 'extended' if extended else 'simple'
    return out
