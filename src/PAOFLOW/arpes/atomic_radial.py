"""Slater-type and hydrogenic radial functions for orbitals without a PAO radial table.

The Slater screening rules and the electron-configuration table ``electron_configs.txt`` are
adapted from chinook ``electron_configs.py`` (R. P. Day et al., npj Quantum Materials 4, 54
(2019), MIT licence, Copyright (c) 2018 Ryan Patrick Day).  Unlike chinook, the radii are in
bohr with the Bohr radius in the exponent, so the orbitals have their physical extent.
"""

from __future__ import annotations

import os
from math import factorial, gamma

import numpy as np
from numpy.typing import NDArray
from scipy.special import genlaguerre

ELEMENTS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As '
    'Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd '
    'Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am '
    'Cm Bk Cf Es Fm Md No Lr Rf'
).split()
#: Effective principal quantum numbers of Slater's rules.
SLATER_N_EFF = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2}
_L_LETTER = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
_L_NUMBER = {v: k for k, v in _L_LETTER.items()}
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'electron_configs.txt')


def atomic_number(element: str) -> int:
    """Atomic number of ``element`` (a chemical symbol, digits and suffixes ignored)."""
    symbol = ''.join(c for c in element if c.isalpha())[:2]
    if symbol not in ELEMENTS:
        symbol = symbol[:1]
    if symbol not in ELEMENTS:
        raise ValueError('unknown element %r' % element)
    return ELEMENTS.index(symbol) + 1


def electron_configuration(z: int) -> str:
    """Ground-state configuration string of the neutral atom ``z`` (e.g. ``'1s22s22p2'``)."""
    with open(_CONFIG_FILE) as fh:
        for line in fh:
            number, config = line.strip().split(',')
            if int(number) == int(z):
                return config
    raise ValueError('no electron configuration for Z = %d' % z)


def _parse_shells(config: str) -> list[tuple[int, int, int]]:
    """``[(n, l, occupation), ...]`` of a configuration string (chinook ``shield_split``)."""
    letters = [i for i, c in enumerate(config) if c.isalpha()]
    pieces = [config[a - 1 : b - 1] for a, b in zip(letters, letters[1:])]
    pieces.append(config[letters[-1] - 1 :])
    return [(int(p[0]), _L_NUMBER[p[1]], int(p[2:])) for p in pieces]


def slater_effective_charge(z: int, n: int, l: int) -> float:
    """Effective nuclear charge of the ``(n, l)`` shell of element ``z`` by Slater's rules.

    Notes
    -----
    Screening constants: 0.35 per other electron of the same group (0.30 in ``1s``); for
    ``s`` and ``p`` valence shells 0.85 per electron with ``n - 1`` and 1.00 for ``n - 2`` and
    below; for ``d`` and ``f`` shells 1.00 per electron of every lower group (chinook
    ``Z_eff``).
    """
    shells = _parse_shells(electron_configuration(z))
    group = {0: 0, 1: 0, 2: 2, 3: 3}  # s and p of the same n form one group
    occupation = 0
    for n_s, l_s, occ in shells:
        if n_s == n and group[l_s] == group[l]:
            occupation += occ
    if occupation == 0:
        raise ValueError('shell n=%d l=%d is empty in the configuration of Z=%d' % (n, l, z))
    screening = 0.0
    for n_s, l_s, occ in shells:
        if n_s == n and group[l_s] == group[l]:
            continue
        if l <= 1:
            if n_s <= n - 2:
                screening += 1.0 * occ
            elif n_s == n - 1:
                screening += 0.85 * occ
            elif n_s == n and group[l_s] < group[l]:
                screening += 0.35 * occ
        else:
            if n_s < n or (n_s == n and group[l_s] < group[l]):
                screening += 1.0 * occ
    screening += (0.30 if n == 1 else 0.35) * (occupation - 1)
    return float(z) - screening


def slater_radial(z: int, n: int, l: int, r_bohr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalised Slater-type radial function ``R(r)`` (bohr^-3/2) on ``r_bohr``.

    Notes
    -----
    .. math::

        R(r) = \\Big(\\frac{2\\zeta}{n^*}\\Big)^{n^*}\\sqrt{\\frac{2\\zeta}{\\Gamma(2n^*+1)}}\\,
        r^{n^*-1} e^{-\\zeta r / n^*}

    with :math:`\\zeta = Z_{\\rm eff}` (:func:`slater_effective_charge`) and :math:`n^*` the
    effective principal number (:data:`SLATER_N_EFF`); ``l`` enters through the screening only.
    The normalisation integrates :math:`R^2 r^2` to one (chinook's carries an extra
    :math:`1/\\sqrt{n^*}`).
    """
    n_eff = SLATER_N_EFF[int(n)]
    xi = slater_effective_charge(z, n, l) / n_eff
    norm = (2.0 * xi) ** n_eff * np.sqrt(2.0 * xi / gamma(2.0 * n_eff + 1.0))
    r = np.asarray(r_bohr, dtype=float)
    return norm * r ** (n_eff - 1.0) * np.exp(-xi * r)


def hydrogenic_radial(z: int, n: int, l: int, r_bohr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hydrogenic radial function ``R_nl(r)`` (bohr^-3/2) of nuclear charge ``z`` on ``r_bohr``."""
    r = np.asarray(r_bohr, dtype=float)
    rho = 2.0 * z * r / n
    norm = np.sqrt((2.0 * z / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l)))
    return norm * np.exp(-rho / 2.0) * rho**l * genlaguerre(n - l - 1, 2 * l + 1)(rho)


def radial_table(
    kind: str, element: str, n: int, l: int, r_bohr: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(r, r R(r))`` of an atomic radial function, in the convention of the PAO tables.

    Parameters
    ----------
    kind : {'slater', 'hydrogenic'}
    element : str
    n, l : int
    r_bohr : ndarray, optional
        Radial grid (default: 2000 points up to 40 bohr).
    """
    r = np.linspace(0.0, 40.0, 2000) if r_bohr is None else np.asarray(r_bohr, dtype=float)
    z = atomic_number(element)
    if kind == 'slater':
        radial = slater_radial(z, n, l, r)
    elif kind == 'hydrogenic':
        radial = hydrogenic_radial(z, n, l, r)
    else:
        raise ValueError("kind must be 'slater' or 'hydrogenic', got %r" % kind)
    return r, r * radial
