r"""Two-center integral primitives for atom-centered orbitals.

This file is internal to :mod:`PAOFLOW.hamiltonian.nonlocal_velocity` and
implements the radial / angular building blocks needed to assemble the
real-space overlap tables

.. math::

   S^{(\\beta\\varphi)}_{Ii,\\mu}(\\Delta\\mathbf{R})
       = \\langle\\beta_{I,i}(\\mathbf{r}) | \\varphi_\\mu(\\mathbf{r}-\\Delta\\mathbf{R})\\rangle,

between two real-space functions

.. math::
   f_A(\\mathbf{r}) = R_A(r)\\,Y_{l_A m_A}(\\hat{\\mathbf{r}}),
   \\qquad
   f_B(\\mathbf{r}) = R_B(r)\\,Y_{l_B m_B}(\\hat{\\mathbf{r}}),

separated by Cartesian displacement :math:`\\mathbf{R}`.  The spherical
harmonics are taken to be **real** (the convention used by Quantum ESPRESSO
projectors and PAOFLOW atomic orbitals).

Phase 3b scope:
    * Spherical-Bessel radial transform :math:`J_l(q) = \\int_0^\\infty R(r)\\,j_l(qr)\\,r^2\\,dr`.
    * Closed-form two-center :math:`s`--:math:`s` overlap.
    * Real spherical harmonics (Quantum-ESPRESSO convention) up to
      :math:`l=6` (covers all triangle outputs of two :math:`l\le 3`
      orbitals).
    * Real Gaunt coefficients
      :math:`G(l_A m_A; l_B m_B; LM) = \\int Y_{l_A m_A} Y_{l_B m_B} Y_{LM}\\,d\\Omega`
      via Gauss--Legendre :math:`\\times` uniform-:math:`\\phi` quadrature
      (cached by argument tuple).
    * General two-center overlap
      :math:`\\langle f_A | f_B(\\cdot - \\mathbf{R})\\rangle` for arbitrary
      :math:`(l_A,m_A,l_B,m_B)`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.integrate
import scipy.special

LMAX_REAL_YLM = 6
r"""Maximum supported angular momentum in :func:`real_spherical_harmonic`.

Must be at least :math:`2\,l_\mathrm{max}^\mathrm{orbital}` so that the
Gaunt triangle :math:`L \le l_A + l_B` is representable.  PAO basis sets
and UPF projectors go up to :math:`l=3` (f-channels), giving
:math:`L_\max = 6`.
"""


def radial_bessel_transform(
    r: np.ndarray,
    f: np.ndarray,
    l: int,
    q: np.ndarray,
) -> np.ndarray:
    r"""Spherical Bessel transform of a radial function.

    Evaluates

    .. math::
        J_l(q) = \int_0^\infty R(r)\,j_l(q\,r)\,r^2\,dr

    by Simpson's rule on the supplied radial mesh.

    Parameters
    ----------
    r : np.ndarray
        Radial mesh (Bohr).  Must be the same length as ``f``.
    f : np.ndarray
        Radial function :math:`R(r)` sampled on ``r``.
    l : int
        Angular momentum.
    q : np.ndarray
        Reciprocal-space points at which to evaluate :math:`J_l(q)`.

    Returns
    -------
    np.ndarray, shape ``(len(q),)``
        :math:`J_l(q)` on the supplied ``q`` grid.

    Notes
    -----
    With this convention, the 3D Fourier transform of
    :math:`f(\mathbf{r}) = R(r)\,Y_{lm}(\hat{\mathbf{r}})` is

    .. math::
        \tilde f(\mathbf{q}) = 4\pi\,(-i)^l\,Y_{lm}(\hat{\mathbf{q}})\,J_l(q),

    when the plane wave is expanded with real spherical harmonics as
    :math:`e^{-i\mathbf{q}\cdot\mathbf{r}} = 4\pi\sum_{LM}(-i)^L
    j_L(q r)\,Y_{LM}(\hat{\mathbf{q}})\,Y_{LM}(\hat{\mathbf{r}})`.
    """
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if r.shape != f.shape:
        raise ValueError(f'r and f must have matching shape, got {r.shape} vs {f.shape}.')
    qr = np.multiply.outer(q, r)  # (nq, nr)
    bess = scipy.special.spherical_jn(l, qr)
    integrand = bess * (f * r * r)
    return scipy.integrate.simpson(integrand, x=r, axis=1)


def two_center_overlap_ss(
    rA: np.ndarray,
    fA: np.ndarray,
    rB: np.ndarray,
    fB: np.ndarray,
    R: np.ndarray,
    *,
    q_max: float = 20.0,
    n_q: int = 600,
) -> float:
    r"""Two-center overlap of two :math:`s`-orbitals separated by ``R``.

    Each orbital has the form :math:`f(\mathbf{r}) = R(r)\,Y_{00}(\hat r)`
    with :math:`Y_{00} = 1/\sqrt{4\pi}`.  The overlap is

    .. math::
       \langle f_A | f_B(\cdot - \mathbf{R})\rangle
          = \tfrac{2}{\pi}\int_0^\infty q^2\,J_0^A(q)\,J_0^B(q)\,j_0(q R)\,dq,

    a special case of the general angular-momentum expansion
    (cf. e.g. Talman, *J. Comput. Phys.* **29**, 35 (1978)).  Independent
    of the orientation of :math:`\mathbf{R}` because both orbitals are
    spherically symmetric.

    Parameters
    ----------
    rA, fA : np.ndarray
        Radial mesh and :math:`R_A(r)` for orbital A.
    rB, fB : np.ndarray
        Radial mesh and :math:`R_B(r)` for orbital B.
    R : array_like, shape ``(3,)``
        Cartesian displacement (Bohr) :math:`\mathbf{R} = \tau_B - \tau_A`.
    q_max : float, optional
        Upper limit of the reciprocal-space integration grid.  Must be
        large enough to resolve both radial functions and ``j_0(qR)``;
        a few :math:`\pi / \min(\Delta r)` is typically safe.
    n_q : int, optional
        Number of Simpson nodes on ``[0, q_max]``.

    Returns
    -------
    float
        Value of the overlap integral.
    """
    Rnorm = float(np.linalg.norm(np.asarray(R, dtype=float)))
    q = np.linspace(0.0, q_max, n_q)
    JA = radial_bessel_transform(rA, fA, 0, q)
    JB = radial_bessel_transform(rB, fB, 0, q)
    j0 = scipy.special.spherical_jn(0, q * Rnorm)
    integrand = q * q * JA * JB * j0
    I0 = scipy.integrate.simpson(integrand, x=q)
    return (2.0 / np.pi) * float(I0)


# ---------------------------------------------------------------------------
# Real spherical harmonics (Quantum-ESPRESSO convention, l = 0..3).
# ---------------------------------------------------------------------------


def real_spherical_harmonic(l: int, m: int, vec: np.ndarray) -> np.ndarray:
    r"""Evaluate the real spherical harmonic :math:`Y_{lm}(\hat r)`.

    Quantum-ESPRESSO / tesseral convention.  Defined for :math:`m>0` as
    :math:`Y_{lm} = \sqrt{2}\,(-1)^m\,N_l^m\,P_l^m(\cos\theta)\,\cos(m\phi)`,
    for :math:`m<0` as the analogous :math:`\sin(|m|\phi)` form, and for
    :math:`m=0` as :math:`Y_{l0} = N_l^0\,P_l^0(\cos\theta)`, with
    :math:`N_l^m = \sqrt{\tfrac{2l+1}{4\pi}\,\tfrac{(l-m)!}{(l+m)!}}`.

    The :math:`(-1)^m` prefactor cancels the Condon--Shortley phase
    carried by ``scipy.special.lpmv`` so that, e.g.,
    :math:`Y_{1,+1} = +\sqrt{3/(4\pi)}\,x/r`.

    Parameters
    ----------
    l, m : int
        Angular momentum numbers; ``0 <= l <= LMAX_REAL_YLM``,
        ``-l <= m <= l``.
    vec : array_like, shape ``(..., 3)``
        One or more Cartesian directions.  Normalized internally; the
        zero vector returns 0 for :math:`l>0` and :math:`1/\sqrt{4\pi}`
        for :math:`l=0`.

    Returns
    -------
    np.ndarray, shape ``vec.shape[:-1]``
        :math:`Y_{lm}(\hat r)` values.
    """
    if not (0 <= l <= LMAX_REAL_YLM):
        raise ValueError(f'l={l} outside supported range [0, {LMAX_REAL_YLM}].')
    if not (-l <= m <= l):
        raise ValueError(f'm={m} outside [-{l}, {l}] for l={l}.')

    v = np.asarray(vec, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f'vec last axis must be 3, got shape {v.shape}.')
    r = np.linalg.norm(v, axis=-1)
    # Guard against r=0 (Y_lm is angular only — return 0 for the zero vec
    # except for the constant l=0 case which is non-singular).
    safe = np.where(r > 0, r, 1.0)
    z_over_r = v[..., 2] / safe
    # cos(theta) = z/r; phi from atan2(y, x).
    cos_t = z_over_r
    phi = np.arctan2(v[..., 1], v[..., 0])
    zero = r == 0

    am = abs(m)
    # Normalization N_l^m = sqrt((2l+1)/(4 pi) * (l-m)!/(l+m)!).
    # Compute (l-am)!/(l+am)! stably as a product.
    if am == 0:
        ratio = 1.0
    else:
        ratio = 1.0
        for k in range(l - am + 1, l + am + 1):
            ratio /= float(k)
    norm = np.sqrt((2 * l + 1) / (4.0 * np.pi) * ratio)

    # scipy.special.lpmv carries the Condon--Shortley phase (-1)^m.
    Plm = scipy.special.lpmv(am, l, cos_t)
    sign_qe = (-1.0) ** am  # cancel CS to recover the QE convention

    if m > 0:
        out = np.sqrt(2.0) * sign_qe * norm * Plm * np.cos(m * phi)
    elif m < 0:
        out = np.sqrt(2.0) * sign_qe * norm * Plm * np.sin(am * phi)
    else:
        out = norm * Plm  # m == 0: no CS phase to cancel

    if l == 0:
        # Y_00 is a non-singular constant; preserve at r=0.
        return np.asarray(out, dtype=float)
    return np.where(zero, 0.0, out)


# ---------------------------------------------------------------------------
# Real Gaunt coefficients (numerical quadrature, cached).
# ---------------------------------------------------------------------------


def _sphere_quadrature(n_theta: int = 32, n_phi: int = 65):
    r"""Build a (Gauss--Legendre :math:`\\times` uniform-:math:`\\phi`) rule.

    Returns
    -------
    dirs : np.ndarray, shape ``(n_theta*n_phi, 3)``
        Unit vectors on the sphere.
    weights : np.ndarray, shape ``(n_theta*n_phi,)``
        Quadrature weights such that
        :math:`\\int f\\,d\\Omega \\approx \\sum_i w_i f(\\hat r_i)`.
        Exact for products of three :math:`Y_{lm}` up to
        :math:`l_\\mathrm{max}` ~ ``n_theta`` (Legendre) and
        :math:`\\le (n_\\phi - 1)/2` in :math:`m`.
    """
    # Gauss-Legendre on cos(theta) in [-1, 1].
    nodes, w_cos = np.polynomial.legendre.leggauss(n_theta)
    cos_t = nodes
    sin_t = np.sqrt(1.0 - cos_t * cos_t)
    # Uniform in phi on [0, 2π); trapezoidal weight = 2π/n_phi.
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    w_phi = np.full(n_phi, 2.0 * np.pi / n_phi)

    # Outer product over (theta, phi).
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    x = np.outer(sin_t, cos_p)
    y = np.outer(sin_t, sin_p)
    z = np.outer(cos_t, np.ones(n_phi))
    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    weights = np.outer(w_cos, w_phi).reshape(-1)
    return dirs, weights


# Cache the quadrature points + ALL Y_lm values up to LMAX_REAL_YLM once.
def _sphere_cache():
    dirs, weights = _sphere_quadrature(n_theta=32, n_phi=65)
    lmax = LMAX_REAL_YLM
    # Index by (l, m) → array on the sphere.
    ylm = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm[(l, m)] = real_spherical_harmonic(l, m, dirs)
    return dirs, weights, ylm


_SPHERE_DIRS, _SPHERE_WEIGHTS, _SPHERE_YLM = _sphere_cache()


@lru_cache(maxsize=4096)
def real_gaunt_coefficient(lA: int, mA: int, lB: int, mB: int, L: int, M: int) -> float:
    r"""Real-Y_lm Gaunt coefficient
    :math:`G = \int Y_{l_A m_A} Y_{l_B m_B} Y_{LM}\,d\Omega`.

    Computed by spherical quadrature on the cached
    Gauss--Legendre :math:`\times` uniform-:math:`\phi` grid.  Values
    below :math:`10^{-12}` are clamped to zero (selection-rule cleanup).

    Parameters
    ----------
    lA, mA, lB, mB, L, M : int
        Angular momentum and azimuthal quantum numbers.  Each ``l``
        must be in :math:`[0, 2\,L_\\mathrm{max}^Y]` (so that the
        product :math:`Y_{l_A m_A} Y_{l_B m_B} Y_{LM}` is representable
        on the cached sphere); each :math:`|m| \le l`.

    Returns
    -------
    float
        Gaunt coefficient.  Zero unless
        :math:`|l_A - l_B| \le L \le l_A + l_B`, :math:`L` has the same
        parity as :math:`l_A + l_B`, and the :math:`m`-selection rule
        (implicit in the real-Y_lm Gaunt) is satisfied.
    """
    for l, m in ((lA, mA), (lB, mB), (L, M)):
        if not (0 <= l <= LMAX_REAL_YLM):
            raise ValueError(f'Gaunt: l={l} outside [0, {LMAX_REAL_YLM}].')
        if not (-l <= m <= l):
            raise ValueError(f'Gaunt: m={m} outside [-{l}, {l}].')
    # Selection rules: triangle and parity.
    if L < abs(lA - lB) or L > lA + lB:
        return 0.0
    if (lA + lB + L) % 2 != 0:
        return 0.0
    yA = _SPHERE_YLM[(lA, mA)]
    yB = _SPHERE_YLM[(lB, mB)]
    yL = _SPHERE_YLM[(L, M)]
    val = float(np.sum(_SPHERE_WEIGHTS * yA * yB * yL))
    if abs(val) < 1e-12:
        return 0.0
    return val


# ---------------------------------------------------------------------------
# Generalized two-center overlap.
# ---------------------------------------------------------------------------


def two_center_overlap(
    rA: np.ndarray,
    fA: np.ndarray,
    lA: int,
    mA: int,
    rB: np.ndarray,
    fB: np.ndarray,
    lB: int,
    mB: int,
    R: np.ndarray,
    *,
    q_max: float = 20.0,
    n_q: int = 600,
) -> float:
    r"""General two-center overlap of two atom-centered real Y_lm orbitals.

    Computes

    .. math::
       \langle f_A | f_B(\cdot - \mathbf{R})\rangle
          = 8 \sum_{L,M} i^{\,l_A - l_B - L}\,
              G(l_A m_A; l_B m_B; LM)\,
              Y_{LM}(\hat{\mathbf{R}})\,
              \int_0^\infty q^2\, J^A_{l_A}(q)\, J^B_{l_B}(q)\, j_L(qR)\,dq,

    with :math:`f_X(\mathbf{r}) = R_X(r)\,Y_{l_X m_X}(\hat r)` and
    :math:`J^X_l(q) = \int R_X(r)\,j_l(qr)\,r^2\,dr` (see
    :func:`radial_bessel_transform`).  The Gaunt selection rule
    (:math:`|l_A - l_B| \le L \le l_A + l_B`, same parity) makes
    :math:`i^{l_A-l_B-L}` purely real on the surviving terms.

    Parameters
    ----------
    rA, fA : np.ndarray
        Radial mesh and :math:`R_A(r)` for orbital A.
    lA, mA : int
        Angular momentum and azimuthal index for A.
    rB, fB : np.ndarray
        Radial mesh and :math:`R_B(r)` for orbital B.
    lB, mB : int
        Angular momentum and azimuthal index for B.
    R : array_like, shape ``(3,)``
        Cartesian displacement :math:`\tau_B - \tau_A` (Bohr).
    q_max, n_q : float, int
        Reciprocal-space Simpson grid (see :func:`two_center_overlap_ss`).

    Returns
    -------
    float
        Overlap integral value.
    """
    R_vec = np.asarray(R, dtype=float)
    Rnorm = float(np.linalg.norm(R_vec))

    q = np.linspace(0.0, q_max, n_q)
    JA = radial_bessel_transform(rA, fA, lA, q)
    JB = radial_bessel_transform(rB, fB, lB, q)

    # Triangle + parity selection for the L sum.
    L_min = abs(lA - lB)
    L_max = lA + lB
    parity = (lA + lB) % 2

    total = 0.0
    for L in range(L_min, L_max + 1):
        if L % 2 != parity:
            continue
        # i^{lA - lB - L} is real (parity guarantees the exponent is even).
        phase = complex(1j) ** (lA - lB - L)
        # Cast to real (imaginary part must vanish to roundoff).
        if abs(phase.imag) > 1e-10:
            raise RuntimeError(f'Internal: i^{lA - lB - L} not real (parity logic broke?)')
        phase = phase.real

        jL_qR = scipy.special.spherical_jn(L, q * Rnorm)
        integrand = q * q * JA * JB * jL_qR
        I_L = float(scipy.integrate.simpson(integrand, x=q))

        # Sum over M weighted by the Gaunt coefficient and Y_LM(R̂).
        for M in range(-L, L + 1):
            G = real_gaunt_coefficient(lA, mA, lB, mB, L, M)
            if G == 0.0:
                continue
            YLM = float(real_spherical_harmonic(L, M, R_vec[np.newaxis, :])[0])
            total += phase * G * YLM * I_L

    return 8.0 * total


# ---------------------------------------------------------------------------
# Precomputed-J variants: avoid recomputing the radial Bessel transform on
# every call when many pair displacements share the same radial channels.
# ---------------------------------------------------------------------------


def two_center_overlap_precomputed(
    JA: np.ndarray,
    JB: np.ndarray,
    lA: int,
    mA: int,
    lB: int,
    mB: int,
    R: np.ndarray,
    q_grid: np.ndarray,
) -> float:
    r"""Same as :func:`two_center_overlap` but with precomputed radial
    Bessel transforms ``JA = J^A_{l_A}(q_grid)`` and ``JB = J^B_{l_B}(q_grid)``.

    The reciprocal-space Simpson grid (``q_grid``) is supplied by the
    caller and must match the one used to build ``JA`` and ``JB``.
    """
    R_vec = np.asarray(R, dtype=float)
    Rnorm = float(np.linalg.norm(R_vec))

    L_min = abs(lA - lB)
    L_max = lA + lB
    parity = (lA + lB) % 2
    qq = q_grid * q_grid

    total = 0.0
    for L in range(L_min, L_max + 1):
        if L % 2 != parity:
            continue
        phase = complex(1j) ** (lA - lB - L)
        if abs(phase.imag) > 1e-10:
            raise RuntimeError(f'Internal: i^{lA - lB - L} not real (parity logic broke?)')
        phase = phase.real
        jL_qR = scipy.special.spherical_jn(L, q_grid * Rnorm)
        I_L = float(scipy.integrate.simpson(qq * JA * JB * jL_qR, x=q_grid))
        for M in range(-L, L + 1):
            G = real_gaunt_coefficient(lA, mA, lB, mB, L, M)
            if G == 0.0:
                continue
            YLM = float(real_spherical_harmonic(L, M, R_vec[np.newaxis, :])[0])
            total += phase * G * YLM * I_L
    return 8.0 * total


def two_center_dipole_overlap_precomputed(
    J_gA_by_Lp: dict,
    JB: np.ndarray,
    lA: int,
    mA: int,
    lB: int,
    mB: int,
    R: np.ndarray,
    alpha: int,
    q_grid: np.ndarray,
) -> float:
    r"""Same as :func:`two_center_dipole_overlap` but with precomputed
    radial Bessel transforms of the modified bra :math:`g_A(r) = r\,R_A(r)`.

    Parameters
    ----------
    J_gA_by_Lp : dict[int, np.ndarray]
        Mapping ``Lp -> J^{gA}_{Lp}(q_grid)`` for
        ``Lp \in {|l_A - 1|, l_A + 1}`` (parity allowed).  ``Lp = 0``
        for ``l_A = 1`` is included automatically by the caller.
    JB : np.ndarray
        :math:`J^B_{l_B}(q_grid)`.
    lA, mA, lB, mB, R, alpha, q_grid
        See :func:`two_center_dipole_overlap`.
    """
    if alpha not in (0, 1, 2):
        raise ValueError(f'alpha must be 0, 1, or 2 (got {alpha}).')
    m_alpha = _CART_TO_YLM_M[alpha]
    prefactor = np.sqrt(4.0 * np.pi / 3.0)
    Lp_min = abs(lA - 1)
    Lp_max = lA + 1
    parity = (lA + 1) % 2

    total = 0.0
    for Lp in range(Lp_min, Lp_max + 1):
        if Lp % 2 != parity:
            continue
        JgA = J_gA_by_Lp[Lp]
        for Mp in range(-Lp, Lp + 1):
            G_ang = real_gaunt_coefficient(1, m_alpha, lA, mA, Lp, Mp)
            if G_ang == 0.0:
                continue
            ov = two_center_overlap_precomputed(
                JgA,
                JB,
                Lp,
                Mp,
                lB,
                mB,
                R,
                q_grid,
            )
            total += prefactor * G_ang * ov
    return total


# Real-Y_lm convention: r̂_α = √(4π/3) Y_{1, m(α)}, with the (x, y, z) ↔ (+1, -1, 0)
# mapping used throughout this module.
_CART_TO_YLM_M = {0: 1, 1: -1, 2: 0}


def two_center_dipole_overlap(
    rA: np.ndarray,
    fA: np.ndarray,
    lA: int,
    mA: int,
    rB: np.ndarray,
    fB: np.ndarray,
    lB: int,
    mB: int,
    R: np.ndarray,
    alpha: int,
    *,
    q_max: float = 20.0,
    n_q: int = 600,
) -> float:
    r"""Intrinsic dipole-weighted overlap
    :math:`M_\alpha(\mathbf{R}) = \int \beta_A(\mathbf{r})\, r_\alpha\,
    \varphi_B(\mathbf{r} - \mathbf{R})\, d^3r`,
    with **origin at the A site** (i.e. ``r`` measured from :math:`\tau_A`).

    Useful for assembling :math:`\langle\beta_I | r_\alpha | \varphi_\mu\rangle`
    via

    .. math::
       \langle\beta_I|r_\alpha|\varphi_\mu\rangle
         = M_\alpha(\mathbf{R}) + (\tau_I)_\alpha\,
           \langle\beta_I|\varphi_\mu\rangle,

    where :math:`\mathbf{R} = \tau_\mu - \tau_I`.  The :math:`(\tau_I)_\alpha`
    c-number piece is added by the geometry-wiring layer; this routine only
    returns the intrinsic part.

    Derivation
    ----------
    Writing :math:`r_\alpha = r\,(\hat r)_\alpha = \sqrt{4\pi/3}\,r\,Y_{1,m_\alpha}(\hat r)`
    and using the real-:math:`Y_{lm}` product expansion
    :math:`Y_{1 m_\alpha}\,Y_{l_A m_A} = \sum_{L',M'} G(1 m_\alpha; l_A m_A; L'M')\, Y_{L'M'}`,
    the integral reduces to a finite sum of standard
    :func:`two_center_overlap` calls with the modified radial bra
    :math:`g_A(r) = r\,R_A(r)` and the bra angular index swept over
    :math:`(L', M')` with :math:`L' \in \{l_A - 1,\,l_A + 1\}` (parity rules).

    Parameters
    ----------
    rA, fA, lA, mA : np.ndarray, np.ndarray, int, int
        Bra (A-site) radial mesh, :math:`R_A(r)`, and angular indices.
    rB, fB, lB, mB : np.ndarray, np.ndarray, int, int
        Ket (B-site) radial mesh, :math:`R_B(r)`, and angular indices.
    R : array_like, shape ``(3,)``
        Cartesian displacement :math:`\tau_B - \tau_A` (Bohr).
    alpha : int
        Cartesian index of the dipole operator: 0 → :math:`x`,
        1 → :math:`y`, 2 → :math:`z`.
    q_max, n_q : float, int
        Reciprocal-space Simpson grid forwarded to
        :func:`two_center_overlap`.

    Returns
    -------
    float
        :math:`M_\alpha(\mathbf{R})`.
    """
    if alpha not in (0, 1, 2):
        raise ValueError(f'alpha must be 0, 1, or 2 (got {alpha}).')
    m_alpha = _CART_TO_YLM_M[alpha]

    # Modified bra radial: g_A(r) = r * R_A(r).  The angular factor (x_α/r)
    # carries no r-dependence, only Y_{1,m_α}.
    gA = rA * fA
    prefactor = np.sqrt(4.0 * np.pi / 3.0)

    # Sum over the parity-allowed L' from the Y_1 × Y_{lA} product.
    Lp_min = abs(lA - 1)
    Lp_max = lA + 1
    parity = (lA + 1) % 2  # L' must have opposite parity to lA.

    total = 0.0
    for Lp in range(Lp_min, Lp_max + 1):
        if Lp % 2 != parity:
            continue
        for Mp in range(-Lp, Lp + 1):
            G_ang = real_gaunt_coefficient(1, m_alpha, lA, mA, Lp, Mp)
            if G_ang == 0.0:
                continue
            ov = two_center_overlap(
                rA,
                gA,
                Lp,
                Mp,
                rB,
                fB,
                lB,
                mB,
                R,
                q_max=q_max,
                n_q=n_q,
            )
            total += prefactor * G_ang * ov
    return total
