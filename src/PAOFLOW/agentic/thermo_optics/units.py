"""Unit conversions for spectra and temperatures.

Photon abscissae are converted through the photon energy in eV:
``eV``, ``meV``, ``um`` (also ``micron``, ``μm``), ``nm``, ``cm-1`` (wavenumber),
``THz`` (frequency).  Temperatures: ``K``, ``C``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

HC_EV_UM = 1.23984198  # h c in eV um
EV_TO_CM1 = 8065.543937
EV_TO_THZ = 241.79892

_ALIASES = {
    'ev': 'eV', 'mev': 'meV', 'um': 'um', 'micron': 'um', 'microns': 'um', 'μm': 'um', 'µm': 'um',
    'nm': 'nm', 'cm-1': 'cm-1', 'cm^-1': 'cm-1', '1/cm': 'cm-1', 'thz': 'THz',
}  # fmt: skip


def canonical_unit(unit: str) -> str:
    """Normalised spelling of a photon unit (raises for an unknown one)."""
    key = str(unit).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            'unknown photon unit %r; known: %s' % (unit, sorted(set(_ALIASES.values())))
        )
    return _ALIASES[key]


def to_photon_energy_ev(values: ArrayLike, unit: str) -> NDArray[np.float64]:
    """Photon energy (eV) of values given in ``unit``.

    Parameters
    ----------
    values : array_like
        Abscissae.
    unit : str
        ``eV``, ``meV``, ``um``, ``nm``, ``cm-1`` or ``THz``.

    Returns
    -------
    ndarray
        Photon energies (eV).

    Notes
    -----
    :math:`E = hc/\\lambda`, :math:`E = hc\\,\\tilde\\nu`, :math:`E = h\\nu`.
    """
    values = np.asarray(values, dtype=float)
    unit = canonical_unit(unit)
    if unit == 'eV':
        return values
    if unit == 'meV':
        return values * 1.0e-3
    if unit == 'um':
        return HC_EV_UM / values
    if unit == 'nm':
        return HC_EV_UM * 1.0e3 / values
    if unit == 'cm-1':
        return values / EV_TO_CM1
    return values / EV_TO_THZ  # THz


def from_photon_energy_ev(energy_ev: ArrayLike, unit: str) -> NDArray[np.float64]:
    """Inverse of :func:`to_photon_energy_ev`."""
    energy_ev = np.asarray(energy_ev, dtype=float)
    unit = canonical_unit(unit)
    if unit == 'eV':
        return energy_ev
    if unit == 'meV':
        return energy_ev * 1.0e3
    if unit in ('um', 'nm'):
        return to_photon_energy_ev(energy_ev, unit)  # the map is its own inverse
    if unit == 'cm-1':
        return energy_ev * EV_TO_CM1
    return energy_ev * EV_TO_THZ


def convert_photon(values: ArrayLike, from_unit: str, to_unit: str) -> NDArray[np.float64]:
    """Convert photon abscissae between units."""
    return from_photon_energy_ev(to_photon_energy_ev(values, from_unit), to_unit)


def to_kelvin(values: ArrayLike, unit: str) -> NDArray[np.float64]:
    """Temperatures in K from ``K`` or ``C`` (``degC``)."""
    values = np.asarray(values, dtype=float)
    key = str(unit).strip().lower().replace('°', '').replace('deg', '')
    if key == 'k':
        return values
    if key == 'c':
        return values + 273.15
    raise ValueError('unknown temperature unit %r' % unit)
