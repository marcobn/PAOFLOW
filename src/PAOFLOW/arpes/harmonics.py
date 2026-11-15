"""Real spherical harmonics in Quantum ESPRESSO order and the dipole Gaunt coefficients.

PAOFLOW's PAO basis uses QE's real cubic harmonics (``ylmr2``): within a shell the index
``m_qe = 1 .. 2l+1`` runs over ``m = 0, +1, -1, +2, -2, ...`` with a sign ``(-1)^|m|`` relative
to the standard tesseral harmonics of
:func:`PAOFLOW.hamiltonian._two_center.real_spherical_harmonic`.  The dipole operator
components are real ``l = 1`` harmonics, ``x/r = sqrt(4 pi / 3) Y_{1,+1}``, ``y/r = sqrt(4 pi / 3)
Y_{1,-1}``, ``z/r = sqrt(4 pi / 3) Y_{1,0}``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ..hamiltonian._two_center import real_gaunt_coefficient, real_spherical_harmonic

#: Cartesian dipole components as ``(l = 1, m)`` of the standard real harmonics.
DIPOLE_M = {'x': 1, 'y': -1, 'z': 0}
DIPOLE_NORM = np.sqrt(4.0 * np.pi / 3.0)


def qe_harmonic(l: int, m_qe: int) -> tuple[int, float]:
    """``(m, sign)``: the QE harmonic ``m_qe`` (1-based) of shell ``l`` equals ``sign Y_lm``."""
    if not 1 <= m_qe <= 2 * l + 1:
        raise ValueError('m_qe=%d outside 1..%d for l=%d' % (m_qe, 2 * l + 1, l))
    if m_qe == 1:
        return 0, 1.0
    magnitude = m_qe // 2
    m = magnitude if m_qe % 2 == 0 else -magnitude
    return m, (-1.0) ** magnitude


def real_harmonics(l: int, directions: NDArray[np.float64]) -> NDArray[np.float64]:
    """``Y_lm`` for ``m = -l .. l`` (last axis) on unit vectors ``directions (..., 3)``."""
    d = np.asarray(directions, dtype=float)
    return np.stack([real_spherical_harmonic(l, m, d) for m in range(-l, l + 1)], axis=-1)


@lru_cache(maxsize=16)
def dipole_gaunt_table(l: int, lprime: int) -> NDArray[np.float64]:
    """``G[alpha, m, m']`` = ``sqrt(4 pi / 3) int Y_lm (r_alpha / r) Y_l'm' dOmega``.

    Parameters
    ----------
    l, lprime : int
        Initial- and final-state angular momenta (``lprime = l +- 1`` is non-zero).

    Returns
    -------
    ndarray ``(3, 2l+1, 2l'+1)``
        Components ``x, y, z``; ``m`` and ``m'`` in the standard order ``-l .. l``.
    """
    table = np.zeros((3, 2 * l + 1, 2 * lprime + 1))
    for a, axis in enumerate('xyz'):
        for im, m in enumerate(range(-l, l + 1)):
            for imp, mp in enumerate(range(-lprime, lprime + 1)):
                table[a, im, imp] = DIPOLE_NORM * real_gaunt_coefficient(
                    l, m, 1, DIPOLE_M[axis], lprime, mp
                )
    return table


@lru_cache(maxsize=8)
def spinor_components(l: int) -> NDArray[np.complex128]:
    """``S[jm, sigma, m_qe]``: the spinor PAO functions of shell ``l`` on real harmonics and spin.

    PAOFLOW's fully-relativistic basis functions are the spinor harmonics
    :math:`\\chi_{j m_j}` (``assign_jm`` order: the ``j = l - 1/2`` block first, then
    ``j = l + 1/2``, each with increasing ``m_j``), built from complex harmonics
    (``calc_ylmg_so``, ``clebsch_jm_matrix``).  Here they are expanded on the real QE harmonics
    (``calc_ylmg_complex_0``), :math:`\\chi_{jm_j} = \\sum_{\\sigma, q} S[jm, \\sigma, q]\\,
    Y^{\\rm QE}_{lq}\\,|\\sigma\\rangle`.

    Returns
    -------
    ndarray ``(2(2l+1), 2, 2l+1)``, complex
        ``sigma = 0`` up, ``1`` down; ``m_qe`` 0-based (QE index minus one).
    """
    from ..projection.do_atwfc_proj import (
        _L_BLOCK_START_YLM,
        calc_ylmg_complex_0,
        clebsch_jm_matrix,
    )

    n = 2 * l + 1
    start = _L_BLOCK_START_YLM[l]
    # complex harmonic m (PAOFLOW local order 0, +1, -1, ...) on the real QE harmonics q
    T = np.zeros((n, n), dtype=complex)
    for q in range(n):
        real = np.zeros((1, 16))
        real[0, start + q] = 1.0
        T[:, q] = calc_ylmg_complex_0(real)[0, start : start + n]
    C = clebsch_jm_matrix(l)  # (jm, (m, sigma)) with column m + n sigma
    components = np.zeros((2 * n, 2, n), dtype=complex)
    for sigma in range(2):
        components[:, sigma, :] = C[:, sigma * n : (sigma + 1) * n] @ T
    return components
