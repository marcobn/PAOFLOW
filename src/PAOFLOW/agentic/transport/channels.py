"""Bridge from the transport outputs of a system folder to PAOFLOW's Boltzmann transport.

The scattering channels of :mod:`PAOFLOW.boltzmann.scattering_channels` combine by
Matthiessen's rule inside ``PAOFLOW.transport(scattering_channels=[...])``.  These helpers
build the channel lists from the files a system folder writes, so that an agent can add the
channels a phonon-limited run lacks (residual resistivity, electron-electron scattering,
ionised impurities) without recomputing the electron-phonon vertex:

* metals: ``superconductivity/output/eliashberg.npz`` (``a2F_tr``, the plasma frequency);
* semiconductors: ``transport/<tag>/scattering_rates.npz`` (energy-resolved SERTA/MRTA rates
  aligned to the band edges of the run that uses them).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ...boltzmann.scattering_channels import (
    ConstantRate,
    EnergyResolvedRate,
    FermiLiquidRate,
    ResidualResistivity,
    ScatteringChannel,
    TransportSpectralChannel,
)


def metal_channels(
    eliashberg_npz: str | Path, rho0_uohm_cm: float = 0.0, b_ee_per_ev: float = 0.0
) -> list[ScatteringChannel]:
    """The channel list of ``transport/resistivity.py``: e-ph (alpha^2F_tr), residual, Fermi liquid.

    Parameters
    ----------
    eliashberg_npz : str or Path
        ``eliashberg.npz`` of an Eliashberg run with ``transport=True``.
    rho0_uohm_cm : float, optional
        Residual resistivity (micro-ohm cm; 0: none).
    b_ee_per_ev : float, optional
        Fermi-liquid coefficient of ``hbar/tau = b [(hbar w)^2 + (2 pi kT)^2]`` (0: none).

    Returns
    -------
    list of ScatteringChannel
        For ``PAOFLOW.transport(scattering_channels=...)`` or
        :func:`PAOFLOW.response.drude.dc_resistivity`.
    """
    path = str(eliashberg_npz)
    with np.load(path) as data:
        if 'a2F_tr' not in data.files:
            raise FileNotFoundError('%s has no alpha^2F_tr: run eliashberg.py --transport' % path)
        plasma = float(data['omega_plasma_ev'])
    channels: list[ScatteringChannel] = [TransportSpectralChannel.from_npz(path)]
    if rho0_uohm_cm > 0.0:
        channels.append(ResidualResistivity(rho0_uohm_cm * 1.0e-8, plasma))
    if b_ee_per_ev > 0.0:
        channels.append(FermiLiquidRate(b_ee_per_ev))
    return channels


def carrier_channels(
    scattering_rates_npz: str | Path,
    kind: str = 'mrta',
    band_edges: tuple[float, float] | None = None,
    extra: Sequence[ScatteringChannel] = (),
) -> list[ScatteringChannel]:
    """The energy-resolved e-ph rate of ``transport/mobility.py`` plus extra channels.

    Parameters
    ----------
    scattering_rates_npz : str or Path
        ``scattering_rates.npz`` (``energy_ev``, ``temps_k``, ``rate_serta``, ``rate_mrta``,
        the band edges of the run).
    kind : {'mrta', 'serta'}
    band_edges : (vbm_ev, cbm_ev), optional
        Band edges of the band structure the channel is used with; without them the channel
        aligns itself when bound to a PAOFLOW run (``bind``).
    extra : sequence of ScatteringChannel, optional
        Channels to add by Matthiessen's rule (e.g. :func:`impurity_channel`).

    Returns
    -------
    list of ScatteringChannel
    """
    channel = EnergyResolvedRate.from_npz(str(scattering_rates_npz), kind=kind)
    if band_edges is not None:
        channel.align_to(*band_edges)
    return [channel, *extra]


def impurity_rate_from_mobility(mobility_cm2_vs: float, effective_mass: float) -> float:
    """Constant rate :math:`1/\\tau = e/(m^* \\mu)` (1/s) behind a mobility ``mu`` (cm^2/Vs).

    Parameters
    ----------
    mobility_cm2_vs : float
        The mobility the channel alone would give (e.g. the ionised-impurity mobility of
        Brooks-Herring, or a measured low-temperature mobility).
    effective_mass : float
        Conductivity effective mass in units of the electron mass.
    """
    from scipy.constants import e, m_e

    return float(e / (effective_mass * m_e * mobility_cm2_vs * 1.0e-4))


def impurity_channel(
    mobility_cm2_vs: float, effective_mass: float, name: str = 'impurity'
) -> ConstantRate:
    """A constant-rate channel reproducing a mobility (:func:`impurity_rate_from_mobility`)."""
    return ConstantRate(impurity_rate_from_mobility(mobility_cm2_vs, effective_mass), name=name)
