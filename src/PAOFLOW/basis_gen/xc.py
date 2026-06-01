"""Exchange-correlation functionals for the radial pseudo-atom solver.

Spin-unpolarised only.  All inputs and outputs in Hartree atomic units.
"""

from __future__ import annotations

import numpy as np

# PW92 parameters (Perdew & Wang, PRB 45, 13244 (1992), Table I, paramagnetic).
_PW92_A = 0.031091
_PW92_a1 = 0.21370
_PW92_b1 = 7.5957
_PW92_b2 = 3.5876
_PW92_b3 = 1.6382
_PW92_b4 = 0.49294
_PW92_p = 1.0


def _lda_x(n):
    """LDA (Slater) exchange.  Returns (eps_x, v_x) for n > 0.

    eps_x(n) = -(3/4) * (3/pi)^(1/3) * n^(1/3)
    v_x      = (4/3) * eps_x
    """
    c = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    n13 = np.cbrt(n)
    eps_x = c * n13
    v_x = (4.0 / 3.0) * eps_x
    return eps_x, v_x


def _pw92_c(n):
    """PW92 correlation, paramagnetic.  Returns (eps_c, v_c) for n > 0."""
    rs = np.cbrt(3.0 / (4.0 * np.pi * n))

    rs_h = np.sqrt(rs)
    Q0 = -2.0 * _PW92_A * (1.0 + _PW92_a1 * rs)
    Q1 = 2.0 * _PW92_A * (
        _PW92_b1 * rs_h
        + _PW92_b2 * rs
        + _PW92_b3 * rs * rs_h
        + _PW92_b4 * rs ** (_PW92_p + 1.0)
    )
    # dQ1/drs
    Q1p = _PW92_A * (
        _PW92_b1 / rs_h
        + 2.0 * _PW92_b2
        + 3.0 * _PW92_b3 * rs_h
        + 2.0 * (_PW92_p + 1.0) * _PW92_b4 * rs ** _PW92_p
    )
    L = np.log(1.0 + 1.0 / Q1)
    eps_c = Q0 * L
    # d eps_c / drs
    dQ0 = -2.0 * _PW92_A * _PW92_a1
    deps_c_drs = dQ0 * L - Q0 * Q1p / (Q1 * (Q1 + 1.0))
    # v_c = eps_c - (rs/3) * d eps_c / drs
    v_c = eps_c - (rs / 3.0) * deps_c_drs
    return eps_c, v_c


def lda_pw92(n):
    """Spin-unpolarised LDA (Slater exchange + PW92 correlation).

    Parameters
    ----------
    n : ndarray
        Electron density (Bohr^-3).  Zero/negative entries return zero.

    Returns
    -------
    eps_xc : ndarray
        XC energy per particle (Hartree).
    v_xc : ndarray
        XC potential (Hartree).
    """
    n = np.asarray(n, dtype=float)
    eps = np.zeros_like(n)
    v = np.zeros_like(n)
    pos = n > 1e-30
    if not np.any(pos):
        return eps, v
    np_ = n[pos]
    ex, vx = _lda_x(np_)
    ec, vc = _pw92_c(np_)
    eps[pos] = ex + ec
    v[pos] = vx + vc
    return eps, v
