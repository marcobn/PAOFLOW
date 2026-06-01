"""Spherical Hartree potential and frozen-density effective potential.

For a spherically symmetric density n(r) the Hartree potential is

    V_H(r) = 4 pi [ (1/r) integral_0^r n(r') r'^2 dr' + integral_r^inf n(r') r' dr' ]

The UPF stores ``atrho(r) = 4 pi r^2 n(r)`` with int atrho dr = z_valence,
which makes the integrals cumulative trapezoidal sums of ``atrho`` and
``atrho/r`` respectively.
"""

from __future__ import annotations

import numpy as np

from .xc import lda_pw92


def hartree_radial(atrho, r, rab=None):
    """Compute V_H(r) from an atomic radial density.

    Parameters
    ----------
    atrho : ndarray, shape (N,)
        4 pi r^2 n(r), as stored in PP_RHOATOM.
    r : ndarray, shape (N,)
        Radial mesh (Bohr).
    rab : ndarray, optional
        Mesh integration weights (dr/di).  Defaults to centred differences
        on ``r``.

    Returns
    -------
    v_h : ndarray, shape (N,)
        Hartree potential in Hartree.
    """
    if rab is None:
        rab = np.gradient(r)

    # Cumulative integrals with the trapezoid rule using rab as the weight.
    # Inner : I1(r) = int_0^r atrho dr'
    # Outer : I2(r) = int_r^inf (atrho / r') dr'
    w = atrho * rab
    I1 = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]))))

    # atrho/r is finite at r=0 only if atrho ~ r^2 (which it is for valence);
    # but to be safe handle r[0] == 0:
    safe_r = np.where(r > 0.0, r, 1.0)
    g = atrho / safe_r * rab
    if r[0] == 0.0:
        g[0] = 0.0
    cum_g = np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]))))
    I2 = cum_g[-1] - cum_g

    v_h = np.zeros_like(r)
    nz = r > 0.0
    v_h[nz] = I1[nz] / r[nz] + I2[nz]
    if not nz[0]:
        v_h[0] = I2[0]  # finite r -> 0 limit (well-behaved n at origin)
    return v_h


def vxc_radial(atrho, r):
    """Spin-unpolarised LDA-PW92 V_xc(r) from atomic radial density.

    Parameters
    ----------
    atrho : ndarray
        4 pi r^2 n(r).
    r : ndarray
        Radial mesh (Bohr).

    Returns
    -------
    v_xc : ndarray
        XC potential in Hartree.
    """
    safe_r = np.where(r > 0.0, r, 1.0)
    n = atrho / (4.0 * np.pi * safe_r * safe_r)
    if r[0] == 0.0:
        # Extrapolate n(0) from n(r[1]) since atrho ~ r^2 makes the ratio
        # well-defined in the limit -> r[1] value is a good proxy.
        n[0] = n[1]
    _, v = lda_pw92(n)
    return v


def frozen_effective_potential(upf):
    """Return V_eff(r) = V_loc(r) + V_H(r) + V_xc(r) on the UPF mesh.

    The Hartree and XC contributions are evaluated from ``upf.atrho``
    (frozen at the pseudopotential-generation density).  All terms are
    in Hartree atomic units.

    Raises ``ValueError`` if ``upf.atrho`` is missing.
    """
    if getattr(upf, 'atrho', None) is None:
        raise ValueError(
            'UPF has no PP_RHOATOM block; frozen-density solver needs '
            'the atomic valence density.'
        )
    v_h = hartree_radial(upf.atrho, upf.r, upf.rab)
    v_xc = vxc_radial(upf.atrho, upf.r)
    return upf.vloc + v_h + v_xc
