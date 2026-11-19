"""Unit conversions shared by the linear-response (Kubo) routes.

The Brillouin-zone sums of the response modules are in eV and bohr (``dHksp`` in eV bohr, energies in
eV). One factor turns a sum :math:`N_k^{-1}\\sum_k(\\ldots)` into a conductivity:

- bulk: S/cm, :math:`10^8\\,a_0[\\text{\\AA}]\\,e^2/(\\hbar\\,\\Omega)` with the cell volume
  :math:`\\Omega` in bohr^3;
- sheet (``twoD``): units of :math:`e^2/\\hbar`, the area :math:`|a_1\\times a_2|` (bohr^2) replacing the
  volume. One :math:`e^2/\\hbar` is :math:`2\\pi` conductance quanta :math:`e^2/h`, i.e. 2.434e-4 S.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..utils.constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI

if TYPE_CHECKING:
    from ..DataController import DataController

#: Bohr radius in cm (the unit factor of the Rashba-Edelstein routes, per cell).
BOHR_TO_CM = 5.29177249e-9

#: :math:`e^2/\hbar` in siemens.
E2_OVER_HBAR_SIEMENS = ELECTRONVOLT_SI / H_OVER_TPI


def conductivity_conversion(data_controller: 'DataController', two_dimensional: bool) -> float:
    """Factor from a k-averaged :math:`\\langle A\\,v\\rangle/\\varepsilon` sum to a conductivity.

    Parameters
    ----------
    data_controller : DataController
        Reads ``a_vectors``, ``alat`` and ``omega``.
    two_dimensional : bool
        Sheet conductivity in units of :math:`e^2/\\hbar` instead of the bulk value in S/cm.

    Returns
    -------
    float
        The factor multiplying :math:`N_k^{-1}\\sum_k(\\ldots)`.
    """
    arry, attr = data_controller.data_dicts()
    if two_dimensional:
        area = np.linalg.norm(np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]))
        return 1.0 / (area * attr['alat'] ** 2)
    return 1.0e8 * ANGSTROM_AU * ELECTRONVOLT_SI**2 / (H_OVER_TPI * attr['omega'])


def sheet_unit(unit: str) -> str:
    """Header of a sheet (``twoD``) output: S/cm becomes e^2/hbar.

    Parameters
    ----------
    unit : str
        Header of the bulk output, e.g. ``'AH Conductivity [S/cm]'``.

    Returns
    -------
    str
        The same header with the sheet unit (unchanged when it has no S/cm).
    """
    return unit.replace('S/cm', 'e^2/hbar, sheet')
