"""Exchange-correlation functionals for the radial pseudo-atom solver.

Spin-unpolarised only.  All inputs and outputs in Hartree atomic units.

Two functionals are supported:

* LDA-PW92  -- Slater exchange + Perdew-Wang 1992 correlation.
* PBE       -- spin-unpolarised PBE GGA, for a spherical density on a
  user-supplied 1D radial mesh.

The :func:`select_functional` helper inspects a UPF and returns the
string ``'LDA'`` or ``'PBE'``.  Pseudos tagged with any GGA flavour
other than PBE fall back to PBE (the user-requested policy) with a
RuntimeWarning.
"""

from __future__ import annotations

import warnings

import numpy as np

# PW92 parameters (Perdew & Wang, PRB 45, 13244 (1992), Table I, paramagnetic).
_PW92_A = 0.031091
_PW92_a1 = 0.21370
_PW92_b1 = 7.5957
_PW92_b2 = 3.5876
_PW92_b3 = 1.6382
_PW92_b4 = 0.49294
_PW92_p = 1.0

# PBE constants (Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)).
_PBE_KAPPA = 0.804
_PBE_MU = 0.21951
_PBE_BETA = 0.066725
_PBE_GAMMA = (1.0 - np.log(2.0)) / np.pi**2  # ~ 0.0310907


def _lda_x(n):
    r"""LDA (Slater) exchange.  Returns ``(eps_x, v_x)`` for ``n > 0``.

    .. math::

       \varepsilon_x(n) = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3} n^{1/3}

    .. math::

       v_x = \frac{4}{3}\,\varepsilon_x
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
    Q1 = (
        2.0
        * _PW92_A
        * (
            _PW92_b1 * rs_h
            + _PW92_b2 * rs
            + _PW92_b3 * rs * rs_h
            + _PW92_b4 * rs ** (_PW92_p + 1.0)
        )
    )
    # dQ1/drs
    Q1p = _PW92_A * (
        _PW92_b1 / rs_h
        + 2.0 * _PW92_b2
        + 3.0 * _PW92_b3 * rs_h
        + 2.0 * (_PW92_p + 1.0) * _PW92_b4 * rs**_PW92_p
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


# ---------------------------------------------------------------------------
# PBE (spherical, frozen-density)
# ---------------------------------------------------------------------------


def _pbe_exchange_kernel(n, sigma):
    r"""Return :math:`(F_x,\, \partial F_x/\partial n,\, \partial F_x/\partial\sigma)` for PBE exchange.

    Here :math:`F_x` is the *energy density* :math:`n\,\varepsilon_x^{\mathrm{PBE}}(n, \sigma)`,
    with :math:`\sigma = |\nabla n|^2` and :math:`n > 0` elementwise.  Operates only on
    elements where :math:`n > 0`; callers must mask.
    """
    # LDA exchange energy density: A * n^{4/3}, A = -(3/4)(3/pi)^{1/3}
    A = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    n43 = n ** (4.0 / 3.0)
    e_x_lda = A * n43  # n * eps_x_LDA
    de_lda_dn = (4.0 / 3.0) * A * n ** (1.0 / 3.0)

    # s^2 = sigma / (4 k_F^2 n^2);  k_F = (3 pi^2 n)^{1/3}
    kF2 = (3.0 * np.pi * np.pi * n) ** (2.0 / 3.0)
    s2 = sigma / (4.0 * kF2 * n * n)
    # PBE enhancement F_x(s) = 1 + kappa*y/(1+y), y = mu s^2 / kappa
    y = _PBE_MU * s2 / _PBE_KAPPA
    Fx = 1.0 + _PBE_KAPPA * y / (1.0 + y)
    dFx_ds2 = _PBE_MU / (1.0 + y) ** 2

    # d(s2)/dn = -(8/3) s2 / n
    # d(s2)/dsigma = s2 / sigma (when sigma > 0); = 1/(4 kF^2 n^2) always
    ds2_dn = -(8.0 / 3.0) * s2 / n
    ds2_dsigma = 1.0 / (4.0 * kF2 * n * n)

    F = e_x_lda * Fx
    dF_dn = de_lda_dn * Fx + e_x_lda * dFx_ds2 * ds2_dn
    dF_dsigma = e_x_lda * dFx_ds2 * ds2_dsigma
    return F, dF_dn, dF_dsigma


def _pbe_correlation_kernel(n, sigma):
    r"""Return :math:`(F_c,\, \partial F_c/\partial n,\, \partial F_c/\partial\sigma)` for PBE correlation.

    :math:`F_c = n\,\varepsilon_c^{\mathrm{PBE}}(n, \sigma)`; spin-unpolarised
    (:math:`\phi = 1`, :math:`\zeta = 0`).  Operates only on elements where
    :math:`n > 0`; callers must mask.
    """
    # LDA correlation per particle and its drs derivative (PW92).
    rs = np.cbrt(3.0 / (4.0 * np.pi * n))
    rs_h = np.sqrt(rs)
    Q0 = -2.0 * _PW92_A * (1.0 + _PW92_a1 * rs)
    Q1 = (
        2.0
        * _PW92_A
        * (
            _PW92_b1 * rs_h
            + _PW92_b2 * rs
            + _PW92_b3 * rs * rs_h
            + _PW92_b4 * rs ** (_PW92_p + 1.0)
        )
    )
    Q1p = _PW92_A * (
        _PW92_b1 / rs_h
        + 2.0 * _PW92_b2
        + 3.0 * _PW92_b3 * rs_h
        + 2.0 * (_PW92_p + 1.0) * _PW92_b4 * rs**_PW92_p
    )
    L = np.log(1.0 + 1.0 / Q1)
    dQ0 = -2.0 * _PW92_A * _PW92_a1
    e_lda = Q0 * L  # eps_c_LDA
    de_lda_drs = dQ0 * L - Q0 * Q1p / (Q1 * (Q1 + 1.0))  # d eps_c / drs
    # drs/dn = -rs / (3 n)
    de_lda_dn = de_lda_drs * (-rs / (3.0 * n))

    # t^2 = sigma / (4 phi^2 ks^2 n^2);  phi = 1; ks = sqrt(4 kF / pi)
    kF = (3.0 * np.pi * np.pi * n) ** (1.0 / 3.0)
    ks2 = 4.0 * kF / np.pi
    t2 = sigma / (4.0 * ks2 * n * n)

    # A in PBE H functional
    expm = np.exp(-e_lda / _PBE_GAMMA) - 1.0
    # Guard tiny denominators (high-density limit -> A -> 0)
    expm = np.where(np.abs(expm) < 1e-30, 1e-30, expm)
    A_pbe = (_PBE_BETA / _PBE_GAMMA) / expm

    # H(rs, t) = gamma * ln[1 + (beta/gamma) t2 * (1 + A t2)/(1 + A t2 + A^2 t4)]
    At2 = A_pbe * t2
    num = 1.0 + At2
    den = 1.0 + At2 + At2 * At2
    arg = 1.0 + (_PBE_BETA / _PBE_GAMMA) * t2 * num / den
    H = _PBE_GAMMA * np.log(arg)
    eps_c = e_lda + H

    # ----- analytic derivatives of H -----
    # Let u = (beta/gamma) * t2 * num/den,  arg = 1 + u.
    # H = gamma * ln(arg);  dH/dX = gamma * (du/dX) / arg.
    # u = (beta/gamma) * t2 * num/den
    # dnum/d(At2) = 1; dden/d(At2) = 1 + 2 At2
    # d(num/den)/d(At2) = [den - num*(1 + 2 At2)] / den^2
    dnum_dAt2 = 1.0
    dden_dAt2 = 1.0 + 2.0 * At2
    dratio_dAt2 = (den * dnum_dAt2 - num * dden_dAt2) / (den * den)

    # u = (beta/gamma) * t2 * (num/den)
    # du/dt2 (at fixed A) = (beta/gamma) * [num/den + t2 * dratio_dAt2 * A]
    # du/dA  (at fixed t2) = (beta/gamma) * t2 * dratio_dAt2 * t2
    coef = _PBE_BETA / _PBE_GAMMA
    du_dt2_fixedA = coef * (num / den + t2 * dratio_dAt2 * A_pbe)
    du_dA_fixedt2 = coef * t2 * dratio_dAt2 * t2

    # A depends on e_lda: A = (beta/gamma)/[exp(-e_lda/gamma) - 1]
    # dA/de_lda = -(beta/gamma) * exp(-e_lda/gamma) * (-1/gamma) / expm^2
    #           = (beta / gamma^2) * exp(-e_lda/gamma) / expm^2
    expE = np.exp(-e_lda / _PBE_GAMMA)
    dA_de = (_PBE_BETA / (_PBE_GAMMA * _PBE_GAMMA)) * expE / (expm * expm)

    # dH/dn = gamma/arg * (du/dt2 * dt2/dn + du/dA * dA/de_lda * de_lda/dn) + de_lda/dn  -> for eps_c
    # but we want F_c = n * eps_c and its derivatives.
    # dt2/dn = -(7/3) * t2 / n   (since ks^2 ~ n^{1/3}, denominator ~ n^{2 + 1/3} = n^{7/3})
    dt2_dn = -(7.0 / 3.0) * t2 / n
    dt2_dsigma = 1.0 / (4.0 * ks2 * n * n)

    dH_dn = (_PBE_GAMMA / arg) * (du_dt2_fixedA * dt2_dn + du_dA_fixedt2 * dA_de * de_lda_dn)
    dH_dsigma = (_PBE_GAMMA / arg) * du_dt2_fixedA * dt2_dsigma

    F = n * eps_c
    dF_dn = eps_c + n * (de_lda_dn + dH_dn)
    dF_dsigma = n * dH_dsigma
    return F, dF_dn, dF_dsigma


def pbe(n, r, rab=None):
    r"""Spin-unpolarised PBE GGA for a spherical density on a radial mesh.

    Parameters
    ----------
    n : ndarray, shape (N,)
        Electron density n(r) (Bohr^-3).
    r : ndarray, shape (N,)
        Radial mesh (Bohr).  Need not be uniform; ``np.gradient`` is
        used with the actual ``r`` spacing for the derivative.
    rab : ndarray, optional
        Unused; kept for signature symmetry with the radial routines.

    Returns
    -------
    eps_xc : ndarray
        XC energy per particle (Hartree).
    v_xc : ndarray
        XC potential,

        .. math::

           v_{xc}(r) = \frac{\partial F}{\partial n}
                       - \frac{1}{r^2}\frac{d}{dr}\!\left[r^2 \cdot 2\frac{\partial F}{\partial\sigma}\frac{dn}{dr}\right].
    """
    n = np.asarray(n, dtype=float)
    r = np.asarray(r, dtype=float)
    eps = np.zeros_like(n)
    v = np.zeros_like(n)
    pos = n > 1e-12
    if not np.any(pos):
        return eps, v

    grad_n = np.gradient(n, r)
    sigma = grad_n * grad_n

    # Evaluate kernels only where n > 0 (avoid LDA/PBE singularities).
    F_x = np.zeros_like(n)
    dFx_dn = np.zeros_like(n)
    dFx_dsig = np.zeros_like(n)
    F_c = np.zeros_like(n)
    dFc_dn = np.zeros_like(n)
    dFc_dsig = np.zeros_like(n)
    F_x[pos], dFx_dn[pos], dFx_dsig[pos] = _pbe_exchange_kernel(n[pos], sigma[pos])
    F_c[pos], dFc_dn[pos], dFc_dsig[pos] = _pbe_correlation_kernel(n[pos], sigma[pos])

    F = F_x + F_c
    dF_dn = dFx_dn + dFc_dn
    dF_dsigma = dFx_dsig + dFc_dsig

    # Radial divergence: div(V_r r_hat) = (1/r^2) d(r^2 V_r)/dr
    Vr = 2.0 * dF_dsigma * grad_n
    r2 = r * r
    # Avoid the r=0 singularity: compute d(r^2 Vr)/dr and divide by r^2 with
    # a small floor for the first sample.
    flux = r2 * Vr
    dflux_dr = np.gradient(flux, r)
    safe_r2 = np.where(r2 > 0.0, r2, 1.0)
    div = dflux_dr / safe_r2
    if r[0] == 0.0:
        div[0] = div[1]

    v_xc = dF_dn - div
    eps_xc = np.zeros_like(n)
    eps_xc[pos] = F[pos] / n[pos]
    return eps_xc, v_xc


# ---------------------------------------------------------------------------
# Functional selection from the UPF header
# ---------------------------------------------------------------------------

_GGA_TOKENS = {'PBE', 'PBX', 'PBC', 'GGA', 'BLYP', 'BP', 'B3', 'B88'}
_LDA_ONLY_TOKENS = {'SLA', 'PW', 'PZ', 'LDA', 'NOGX', 'NOGC', 'VWN'}


def select_functional(upf):
    """Return ``'LDA'`` or ``'PBE'`` for the XC functional carried by ``upf``.

    Inspects ``upf.qexc`` (UPF ``PP_HEADER`` ``functional`` attribute).
    Tokens are matched case-insensitively:

    * any token in ``{PBE, PBX, PBC, GGA}`` -> ``'PBE'``.
    * tokens are a subset of ``{SLA, PW, PZ, LDA, NOGX, NOGC, VWN}`` -> ``'LDA'``.
    * any other GGA flavour (BLYP, BP, B3LYP, B88, ...) -> ``'PBE'``
      with a :class:`RuntimeWarning`.  This matches the user-requested
      "fall back to PBE for any other GGA" policy: the pseudopotential
      was built with gradient corrections, so a GGA-level frozen
      potential is a much closer match than LDA.
    """
    label = (getattr(upf, 'qexc', None) or '').upper().strip()
    if not label:
        warnings.warn(
            'UPF has no functional label; defaulting to LDA-PW92.',
            RuntimeWarning,
            stacklevel=2,
        )
        return 'LDA'
    tokens = set(label.replace('-', ' ').split())

    if tokens & {'PBE', 'PBX', 'PBC'}:
        return 'PBE'
    if tokens <= _LDA_ONLY_TOKENS:
        return 'LDA'
    if 'GGA' in tokens:
        return 'PBE'
    # Any other recognised GGA flavour -> PBE fallback.
    warnings.warn(
        f"UPF functional '{label}' is not LDA or PBE; using PBE as a "
        'fallback for the frozen-density solver.',
        RuntimeWarning,
        stacklevel=2,
    )
    return 'PBE'
