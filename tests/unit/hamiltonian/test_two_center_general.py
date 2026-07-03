"""Unit tests for the generalized (any-l) two-center integral machinery.

Pins down :func:`PAOFLOW.hamiltonian._two_center.real_spherical_harmonic`,
:func:`real_gaunt_coefficient`, and the general
:func:`two_center_overlap` against three independent ground truths:

1. **Real-Y_lm orthonormality** on the cached spherical quadrature grid.
2. **Closed-form Gaunt values** and the parity/triangle selection rules.
3. **Brute-force 3D real-space quadrature** of two-center overlaps for
   several (l_A, m_A, l_B, m_B) combinations on Gaussian-times-polynomial
   test orbitals.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.hamiltonian._two_center import (
    LMAX_REAL_YLM,
    real_gaunt_coefficient,
    real_spherical_harmonic,
    two_center_overlap,
    two_center_overlap_ss,
)

# ----------------------------------------------------------------------
# 1) Real Y_lm orthonormality on the unit sphere.
# ----------------------------------------------------------------------


def test_real_ylm_orthonormality_grid():
    """∫ Y_lm Y_l'm' dΩ = δ_{ll'} δ_{mm'} on a dense Lebedev-like grid."""
    # Independent grid (do NOT reuse the cached one inside the module).
    n_theta, n_phi = 40, 81
    cos_t, w_cos = np.polynomial.legendre.leggauss(n_theta)
    sin_t = np.sqrt(1.0 - cos_t * cos_t)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    w_phi = np.full(n_phi, 2.0 * np.pi / n_phi)
    x = np.outer(sin_t, np.cos(phi))
    y = np.outer(sin_t, np.sin(phi))
    z = np.outer(cos_t, np.ones(n_phi))
    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    w = np.outer(w_cos, w_phi).reshape(-1)

    keys = [(l, m) for l in range(LMAX_REAL_YLM + 1) for m in range(-l, l + 1)]
    Y = {k: real_spherical_harmonic(k[0], k[1], dirs) for k in keys}

    for i, k1 in enumerate(keys):
        for k2 in keys[i:]:
            val = float(np.sum(w * Y[k1] * Y[k2]))
            expected = 1.0 if k1 == k2 else 0.0
            assert val == pytest.approx(
                expected, abs=1e-10
            ), f'<Y_{k1}|Y_{k2}> = {val}, expected {expected}'


# ----------------------------------------------------------------------
# 2) Real Gaunt coefficients.
# ----------------------------------------------------------------------


def test_gaunt_ss_to_s_closed_form():
    """G(0,0;0,0;0,0) = 1/√(4π)."""
    val = real_gaunt_coefficient(0, 0, 0, 0, 0, 0)
    assert val == pytest.approx(1.0 / np.sqrt(4.0 * np.pi), abs=1e-12)


@pytest.mark.parametrize('m', [-1, 0, 1])
def test_gaunt_pp_to_s_closed_form(m):
    """G(1,m;1,m;0,0) = 1/√(4π) for all m∈{-1,0,1}."""
    # <Y_lm | Y_lm> integrated against Y_00 = Y_00 × 1 = 1/√(4π).
    val = real_gaunt_coefficient(1, m, 1, m, 0, 0)
    assert val == pytest.approx(1.0 / np.sqrt(4.0 * np.pi), abs=1e-10)


@pytest.mark.parametrize('m', [-2, -1, 0, 1, 2])
def test_gaunt_dd_to_s_closed_form(m):
    """G(2,m;2,m;0,0) = 1/√(4π) for all m."""
    val = real_gaunt_coefficient(2, m, 2, m, 0, 0)
    assert val == pytest.approx(1.0 / np.sqrt(4.0 * np.pi), abs=1e-10)


@pytest.mark.parametrize(
    'lA,lB,L',
    [
        (0, 0, 1),  # parity-odd
        (1, 0, 0),  # triangle violated (|lA-lB|=1 > L=0)
        (1, 1, 1),  # parity-odd
        (2, 1, 0),  # triangle violated
        (2, 0, 3),  # triangle violated
        (1, 1, 3),  # triangle violated
    ],
)
def test_gaunt_selection_rules(lA, lB, L):
    """Gaunt vanishes outside the parity & triangle window."""
    for mA in range(-lA, lA + 1):
        for mB in range(-lB, lB + 1):
            for M in range(-L, L + 1):
                val = real_gaunt_coefficient(lA, mA, lB, mB, L, M)
                assert val == 0.0, f'Selection violated: G({lA},{mA};{lB},{mB};{L},{M})={val}'


# ----------------------------------------------------------------------
# 3) Generalized two-center overlap.
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def radial_mesh():
    return np.linspace(0.0, 8.0, 2001)


def _gaussian_ss(alpha, r):
    """Normalized 3D Gaussian written as R(r)·Y_00 → R(r) = √(4π)·N·e^{-αr²}."""
    N = (2.0 * alpha / np.pi) ** 0.75
    return np.sqrt(4.0 * np.pi) * N * np.exp(-alpha * r * r)


def test_general_overlap_reduces_to_ss(radial_mesh):
    """two_center_overlap with l=m=0 must match two_center_overlap_ss."""
    r = radial_mesh
    fA = _gaussian_ss(1.5, r)
    fB = _gaussian_ss(1.5, r)
    R = np.array([1.2, -0.6, 0.4])
    v_ss = two_center_overlap_ss(r, fA, r, fB, R)
    v_gen = two_center_overlap(r, fA, 0, 0, r, fB, 0, 0, R)
    assert v_gen == pytest.approx(v_ss, rel=1e-10, abs=1e-12)


# ---- Cartesian-polynomial test orbitals --------------------------------------
#
# Pick the radial part so that f(r) = polynomial(x,y,z) × e^{-α r²} in
# Cartesian — that makes brute-force quadrature trivial.
#
#   p_z = z e^{-αr²}        = R(r) Y_10(r̂),   R(r) = √(4π/3) · r · e^{-αr²}
#   p_x = x e^{-αr²}        = R(r) Y_11(r̂),   R(r) = √(4π/3) · r · e^{-αr²}
#   d_xy = xy e^{-αr²}      = R(r) Y_2,-2,    R(r) = √(4π/15) · r² · e^{-αr²}
#   d_z² ≡ (3z²-r²) e^{-αr²}= R(r) Y_20,     R(r) = √(16π/5)  · r² · e^{-αr²}
#
# Each helper returns both the radial samples and a 3D callable.


def _orbital_p(alpha, axis, r):
    """p-orbital along Cartesian ``axis`` ∈ {0,1,2} → (l=1, m, R(r), f3d)."""
    m_for_axis = {0: 1, 1: -1, 2: 0}[axis]  # x↔+1, y↔-1, z↔0
    R = np.sqrt(4.0 * np.pi / 3.0) * r * np.exp(-alpha * r * r)

    def f3d(x, y, z):
        coord = (x, y, z)[axis]
        return coord * np.exp(-alpha * (x * x + y * y + z * z))

    return 1, m_for_axis, R, f3d


def _orbital_dxy(alpha, r):
    R = np.sqrt(4.0 * np.pi / 15.0) * r * r * np.exp(-alpha * r * r)

    def f3d(x, y, z):
        return x * y * np.exp(-alpha * (x * x + y * y + z * z))

    return 2, -2, R, f3d


def _orbital_dz2(alpha, r):
    R = np.sqrt(16.0 * np.pi / 5.0) * r * r * np.exp(-alpha * r * r)

    def f3d(x, y, z):
        r2 = x * x + y * y + z * z
        return (3.0 * z * z - r2) * np.exp(-alpha * r2)

    return 2, 0, R, f3d


def _orbital_s(alpha, r):
    R = _gaussian_ss(alpha, r)

    def f3d(x, y, z):
        N = (2.0 * alpha / np.pi) ** 0.75
        return N * np.exp(-alpha * (x * x + y * y + z * z))

    return 0, 0, R, f3d


def _brute_force_overlap(fA_3d, fB_3d, R_vec, *, lim=6.0, n=121):
    """Brute-force ∫ fA(r) fB(r - R) d³r on a Cartesian grid."""
    ax = np.linspace(-lim, lim, n)
    dx = ax[1] - ax[0]
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    A = fA_3d(X, Y, Z)
    B = fB_3d(X - R_vec[0], Y - R_vec[1], Z - R_vec[2])
    return float(np.sum(A * B)) * dx**3


@pytest.mark.parametrize(
    'name, orbital_factory_A, orbital_factory_B, R',
    [
        # s ⊗ p_z along ẑ — non-trivial l-mixing, sharp directional dependence.
        (
            's_pz_along_z',
            lambda r: _orbital_s(1.2, r),
            lambda r: _orbital_p(1.0, axis=2, r=r),
            np.array([0.0, 0.0, 1.5]),
        ),
        # p_x ⊗ p_x along ẑ — σ vs π geometry mixing.
        (
            'px_px_along_z',
            lambda r: _orbital_p(1.0, 0, r),
            lambda r: _orbital_p(1.0, 0, r),
            np.array([0.0, 0.0, 1.4]),
        ),
        # p_z ⊗ p_z along ẑ — pure σ.
        (
            'pz_pz_along_z',
            lambda r: _orbital_p(1.0, 2, r),
            lambda r: _orbital_p(1.0, 2, r),
            np.array([0.0, 0.0, 1.4]),
        ),
        # p_x ⊗ p_x along x̂ — pure σ.
        (
            'px_px_along_x',
            lambda r: _orbital_p(1.0, 0, r),
            lambda r: _orbital_p(1.0, 0, r),
            np.array([1.4, 0.0, 0.0]),
        ),
        # d_xy ⊗ d_xy generic offset — tests l_A=l_B=2 with M=-2.
        (
            'dxy_dxy_generic',
            lambda r: _orbital_dxy(0.9, r),
            lambda r: _orbital_dxy(0.9, r),
            np.array([0.7, 0.5, -0.3]),
        ),
        # s ⊗ d_z² along ẑ — tests cross l=0,l=2.
        (
            's_dz2_along_z',
            lambda r: _orbital_s(0.8, r),
            lambda r: _orbital_dz2(0.7, r),
            np.array([0.0, 0.0, 1.2]),
        ),
        # p_z ⊗ d_z² generic — tests cross l=1,l=2 with parity-odd L=1,3.
        (
            'pz_dz2_generic',
            lambda r: _orbital_p(0.8, 2, r),
            lambda r: _orbital_dz2(0.7, r),
            np.array([0.4, 0.0, 0.9]),
        ),
    ],
)
def test_general_overlap_vs_brute_force(name, orbital_factory_A, orbital_factory_B, R, radial_mesh):
    r = radial_mesh
    lA, mA, RA, fA_3d = orbital_factory_A(r)
    lB, mB, RB, fB_3d = orbital_factory_B(r)

    expected = _brute_force_overlap(fA_3d, fB_3d, R)
    got = two_center_overlap(r, RA, lA, mA, r, RB, lB, mB, R, q_max=20.0, n_q=600)

    # Brute-force 3D quadrature on 121³ at lim=6 carries ~1% error itself.
    assert got == pytest.approx(
        expected, rel=2e-2, abs=2e-3
    ), f'[{name}] got={got:.6e}, expected={expected:.6e}'
