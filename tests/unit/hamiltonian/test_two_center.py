"""Unit tests for the two-center integral primitives (Phase 3b).

These tests pin down :mod:`PAOFLOW.hamiltonian._two_center` against three
independent ground truths:

1. **Analytic radial Fourier transform** of an s-Gaussian.
2. **Analytic two-center overlap** of two normalized 3D Gaussians.
3. **Brute-force 3D real-space quadrature** of the same integral.

All three must agree for the s--s case before higher-l (Gaunt-coefficient)
machinery is layered on top.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.hamiltonian._two_center import (
    radial_bessel_transform,
    two_center_overlap_ss,
)

# --- shared radial mesh ------------------------------------------------


@pytest.fixture(scope='module')
def radial_mesh():
    """Dense uniform mesh out to r_max = 8 Bohr.

    Gaussians decay so fast (α ≥ 0.5 a.u.⁻²) that 8 Bohr is plenty.
    """
    return np.linspace(0.0, 8.0, 2001)


# ----------------------------------------------------------------------
# 1) Radial Bessel transform vs analytic FT of an s-Gaussian.
# ----------------------------------------------------------------------


def test_radial_bessel_transform_s_gaussian(radial_mesh):
    r"""For :math:`R(r) = e^{-\alpha r^2}`, evaluating

    .. math::
       J_0(q) = \int_0^\infty e^{-\alpha r^2}\,\frac{\sin qr}{qr}\,r^2\,dr
              = \tfrac{1}{4}\sqrt{\pi/\alpha^3}\,\exp(-q^2/(4\alpha))

    by writing :math:`j_0(qr) = \sin(qr)/(qr)` and using
    :math:`\int_0^\infty r\,e^{-\alpha r^2}\sin(qr)\,dr
    = (q/(4\alpha))\sqrt{\pi/\alpha}\,e^{-q^2/(4\alpha)}`.
    """
    r = radial_mesh
    alpha = 1.5
    f = np.exp(-alpha * r * r)
    q = np.linspace(0.0, 8.0, 41)

    J0_num = radial_bessel_transform(r, f, 0, q)
    J0_ana = 0.25 * np.sqrt(np.pi / alpha**3) * np.exp(-q * q / (4.0 * alpha))

    np.testing.assert_allclose(J0_num, J0_ana, rtol=1e-6, atol=1e-9)


def test_radial_bessel_transform_l1_gaussian(radial_mesh):
    r"""For :math:`R(r) = r\,e^{-\alpha r^2}`, the :math:`l=1` transform
    has the closed form

    .. math::
       J_1(q) = \frac{q\,\sqrt{\pi}}{8\,\alpha^{5/2}}\,e^{-q^2/(4\alpha)}.

    (Verified by direct evaluation of the spherical-Bessel integral with
    :math:`j_1(x)=\sin x/x^2 - \cos x/x` and a Gaussian radial weight.)
    """
    r = radial_mesh
    alpha = 1.2
    f = r * np.exp(-alpha * r * r)
    q = np.linspace(0.0, 8.0, 41)

    J1_num = radial_bessel_transform(r, f, 1, q)
    J1_ana = q * np.sqrt(np.pi) / (8.0 * alpha**2.5) * np.exp(-q * q / (4.0 * alpha))

    np.testing.assert_allclose(J1_num, J1_ana, rtol=1e-6, atol=1e-9)


# ----------------------------------------------------------------------
# 2) Two-center s-s overlap vs analytic Gaussian--Gaussian product rule.
# ----------------------------------------------------------------------


def _gaussian_overlap_analytic(alpha, beta, R):
    """Closed-form ⟨g_α | g_β(·-R)⟩ for normalized 3D s-Gaussians.

    With g_α(r) = (2α/π)^{3/4} exp(-α r²), the standard result is

        ⟨g_α | g_β(·-R)⟩ = (2 √(αβ) / (α+β))^{3/2} exp(-αβ/(α+β) R²).
    """
    R2 = float(np.dot(R, R))
    pref = (2.0 * np.sqrt(alpha * beta) / (alpha + beta)) ** 1.5
    return pref * np.exp(-alpha * beta / (alpha + beta) * R2)


@pytest.mark.parametrize(
    'alpha,beta,R',
    [
        (1.5, 1.5, [0.0, 0.0, 0.0]),
        (1.5, 1.5, [1.0, 0.0, 0.0]),
        (1.5, 1.5, [0.6, 0.8, 0.0]),
        (1.0, 2.5, [1.2, -0.3, 0.7]),
        (2.0, 0.7, [0.0, 0.0, 2.0]),
    ],
)
def test_two_center_ss_overlap_matches_gaussian_analytic(radial_mesh, alpha, beta, R):
    r"""s-s two-center overlap matches the closed-form Gaussian product."""
    r = radial_mesh
    # g_α(r) = R_α(r) Y_00 with R_α(r) = √(4π)(2α/π)^{3/4} e^{-αr²}.
    norm_A = np.sqrt(4.0 * np.pi) * (2.0 * alpha / np.pi) ** 0.75
    norm_B = np.sqrt(4.0 * np.pi) * (2.0 * beta / np.pi) ** 0.75
    fA = norm_A * np.exp(-alpha * r * r)
    fB = norm_B * np.exp(-beta * r * r)

    R_vec = np.asarray(R, dtype=float)
    expected = _gaussian_overlap_analytic(alpha, beta, R_vec)
    got = two_center_overlap_ss(r, fA, r, fB, R_vec, q_max=20.0, n_q=601)
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-8)


# ----------------------------------------------------------------------
# 3) Two-center s-s overlap vs brute-force 3D quadrature.
# ----------------------------------------------------------------------


def _bruteforce_3d_overlap(fA_radial, fB_radial, R, *, lim=6.0, n=80):
    r"""Direct ⟨f_A | f_B(·-R)⟩ on a uniform 3D Cartesian grid.

    Each ``f_*_radial`` is a callable r → R(r) (already including the
    1/√(4π) inside the Y_00 prefactor: i.e. it returns R(r), and the
    function value at point r is R(|r|) × Y_00 = R(|r|)/√(4π)).
    """
    xs = np.linspace(-lim, lim, n)
    dx = xs[1] - xs[0]
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    rA = np.sqrt(X * X + Y * Y + Z * Z)
    rB = np.sqrt((X - R[0]) ** 2 + (Y - R[1]) ** 2 + (Z - R[2]) ** 2)
    Y00 = 1.0 / np.sqrt(4.0 * np.pi)
    A = fA_radial(rA) * Y00
    B = fB_radial(rB) * Y00
    return float(np.sum(A * B) * dx**3)


def test_two_center_ss_matches_bruteforce(radial_mesh):
    """Numerical s-s overlap matches a direct real-space quadrature."""
    r = radial_mesh
    alpha, beta = 1.8, 0.9
    R = np.array([0.7, 0.4, -0.5])

    norm_A = np.sqrt(4.0 * np.pi) * (2.0 * alpha / np.pi) ** 0.75
    norm_B = np.sqrt(4.0 * np.pi) * (2.0 * beta / np.pi) ** 0.75
    fA = norm_A * np.exp(-alpha * r * r)
    fB = norm_B * np.exp(-beta * r * r)

    spectral = two_center_overlap_ss(r, fA, r, fB, R, q_max=20.0, n_q=601)

    brute = _bruteforce_3d_overlap(
        lambda x: norm_A * np.exp(-alpha * x * x),
        lambda x: norm_B * np.exp(-beta * x * x),
        R,
        lim=6.0,
        n=120,
    )

    # Brute-force grid has ~1 % accuracy at n=120 for these widths; the
    # Bessel-transform result is the reference.
    assert spectral == pytest.approx(brute, rel=2e-2)


def test_two_center_ss_overlap_at_zero_separation(radial_mesh):
    r"""At :math:`\mathbf{R}=0` the formula reduces to
    :math:`\int R_A(r) R_B(r) r^2 dr` (the radial inner product)."""
    r = radial_mesh
    alpha, beta = 1.3, 2.1
    norm_A = np.sqrt(4.0 * np.pi) * (2.0 * alpha / np.pi) ** 0.75
    norm_B = np.sqrt(4.0 * np.pi) * (2.0 * beta / np.pi) ** 0.75
    fA = norm_A * np.exp(-alpha * r * r)
    fB = norm_B * np.exp(-beta * r * r)
    R0 = np.zeros(3)

    got = two_center_overlap_ss(r, fA, r, fB, R0, q_max=25.0, n_q=801)

    # Radial inner product (Y_00 contributes 1/√(4π) twice, area 4π
    # cancels — so this is just ∫ R_A R_B r² dr × 1):
    direct = float(np.trapezoid(fA * fB * r * r, r))
    # ... but our s-orbitals are R(r) Y_00, so the full real-space overlap is
    # (1/4π) ∫ R_A R_B r² dr × 4π (the angular integral of Y_00² is 1).
    # So ⟨f_A|f_B⟩ at R=0 = ∫ R_A R_B r² dr / (4π) × 4π = ∫ R_A R_B r² dr / 1.
    # Wait — Y_00² × 4π = (1/(4π)) × 4π = 1. So:
    expected = direct  # ∫ R_A R_B r² dr × ∫ Y_00² dΩ = ∫ R_A R_B r² dr.
    # And ⟨f_A|f_A⟩ for a normalized Gaussian is 1; let's also check that.
    assert got == pytest.approx(expected, rel=1e-5)

    fA_only = two_center_overlap_ss(r, fA, r, fA, R0, q_max=25.0, n_q=801)
    assert fA_only == pytest.approx(1.0, rel=1e-5)
