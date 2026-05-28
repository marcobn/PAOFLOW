"""Unit tests for the dipole-weighted two-center primitive
:func:`PAOFLOW.hamiltonian._two_center.two_center_dipole_overlap`.

This is the "intrinsic" integral

    M_α(R) = ∫ β_A(r) r_α φ_B(r - R) d³r

with the origin at the A site.  It is the heart of the ⟨β | r_α | φ⟩
matrix element needed for the non-local [V_NL, r] velocity correction.

Pinned against three independent ground truths:

1. **Analytic Gaussian closed form** for s ⊗ s along an axis:
       ⟨s_α | r_z | s_β(· - R ẑ)⟩ = (β R / (α + β)) × S_αβ(R).
2. **Gaunt selection rule** → off-axis components vanish for axial R.
3. **Brute-force 3D Cartesian quadrature** for several (l_A, l_B) cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.hamiltonian._two_center import (
    two_center_dipole_overlap,
    two_center_overlap_ss,
)

# --- shared mesh and helpers -------------------------------------------------


@pytest.fixture(scope='module')
def radial_mesh():
    return np.linspace(0.0, 8.0, 2001)


def _gaussian_ss_radial(alpha, r):
    """R(r) for a normalized s-Gaussian, written as f = R · Y_00."""
    N = (2.0 * alpha / np.pi) ** 0.75
    return np.sqrt(4.0 * np.pi) * N * np.exp(-alpha * r * r)


def _orbital_p(alpha, axis, r):
    """p-orbital: f(r) = x_axis · e^{-α r²}.  Returns (l, m, R(r), f3d)."""
    m_for_axis = {0: 1, 1: -1, 2: 0}[axis]
    R = np.sqrt(4.0 * np.pi / 3.0) * r * np.exp(-alpha * r * r)

    def f3d(x, y, z):
        coord = (x, y, z)[axis]
        return coord * np.exp(-alpha * (x * x + y * y + z * z))

    return 1, m_for_axis, R, f3d


def _orbital_s(alpha, r):
    R = _gaussian_ss_radial(alpha, r)

    def f3d(x, y, z):
        N = (2.0 * alpha / np.pi) ** 0.75
        return N * np.exp(-alpha * (x * x + y * y + z * z))

    return 0, 0, R, f3d


def _orbital_dxy(alpha, r):
    R = np.sqrt(4.0 * np.pi / 15.0) * r * r * np.exp(-alpha * r * r)

    def f3d(x, y, z):
        return x * y * np.exp(-alpha * (x * x + y * y + z * z))

    return 2, -2, R, f3d


def _brute_force_dipole(fA_3d, fB_3d, R_vec, alpha, *, lim=6.0, n=121):
    """∫ fA(r) r_α fB(r - R) d³r on a Cartesian grid."""
    ax = np.linspace(-lim, lim, n)
    dx = ax[1] - ax[0]
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    coord = (X, Y, Z)[alpha]
    A = fA_3d(X, Y, Z)
    B = fB_3d(X - R_vec[0], Y - R_vec[1], Z - R_vec[2])
    return float(np.sum(A * coord * B)) * dx**3


# ----------------------------------------------------------------------
# 1) s ⊗ s along ẑ vs analytic Gaussian closed form.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    'a,b,Rz',
    [
        (1.0, 1.0, 1.0),
        (1.2, 0.9, 1.3),
        (0.5, 2.5, 0.7),
        (2.0, 0.7, 1.8),
    ],
)
def test_dipole_ss_along_z_matches_gaussian_closed_form(a, b, Rz, radial_mesh):
    r = radial_mesh
    fA = _gaussian_ss_radial(a, r)
    fB = _gaussian_ss_radial(b, r)
    R = np.array([0.0, 0.0, Rz])
    S = two_center_overlap_ss(r, fA, r, fB, R)
    M_z = two_center_dipole_overlap(r, fA, 0, 0, r, fB, 0, 0, R, alpha=2)
    expected = b * Rz / (a + b) * S
    assert M_z == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_dipole_ss_selection_rules_axial_R(radial_mesh):
    """For R = (0,0,Rz), only M_z is non-zero by Gaunt selection."""
    r = radial_mesh
    fA = _gaussian_ss_radial(1.0, r)
    fB = _gaussian_ss_radial(1.0, r)
    R = np.array([0.0, 0.0, 1.5])
    M_x = two_center_dipole_overlap(r, fA, 0, 0, r, fB, 0, 0, R, alpha=0)
    M_y = two_center_dipole_overlap(r, fA, 0, 0, r, fB, 0, 0, R, alpha=1)
    assert M_x == 0.0
    assert M_y == 0.0


def test_dipole_ss_zero_at_zero_separation_for_equal_widths(radial_mesh):
    """For A = B centered at the origin, ⟨A | r_α | A⟩ = 0 by parity."""
    r = radial_mesh
    fA = _gaussian_ss_radial(1.0, r)
    R0 = np.zeros(3)
    for alpha in (0, 1, 2):
        M = two_center_dipole_overlap(
            r,
            fA,
            0,
            0,
            r,
            fA,
            0,
            0,
            R0,
            alpha=alpha,
            q_max=25.0,
            n_q=801,
        )
        assert M == pytest.approx(0.0, abs=1e-10)


# ----------------------------------------------------------------------
# 2) Generic (l_A, l_B) cases vs brute-force 3D quadrature.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    'name, factory_A, factory_B, R, alpha',
    [
        (
            's_s_generic_x',
            lambda r: _orbital_s(1.0, r),
            lambda r: _orbital_s(0.8, r),
            np.array([0.9, 0.4, -0.2]),
            0,
        ),
        (
            's_s_generic_z',
            lambda r: _orbital_s(1.0, r),
            lambda r: _orbital_s(0.8, r),
            np.array([0.9, 0.4, -0.2]),
            2,
        ),
        (
            's_pz_along_z_alpha_z',
            lambda r: _orbital_s(1.0, r),
            lambda r: _orbital_p(0.9, 2, r),
            np.array([0.0, 0.0, 1.3]),
            2,
        ),
        (
            's_pz_along_z_alpha_x',
            lambda r: _orbital_s(1.0, r),
            lambda r: _orbital_p(0.9, 2, r),
            np.array([0.0, 0.0, 1.3]),
            0,
        ),
        (
            'px_px_generic_alpha_x',
            lambda r: _orbital_p(0.9, 0, r),
            lambda r: _orbital_p(0.9, 0, r),
            np.array([0.6, 0.3, -0.4]),
            0,
        ),
        (
            'px_px_generic_alpha_z',
            lambda r: _orbital_p(0.9, 0, r),
            lambda r: _orbital_p(0.9, 0, r),
            np.array([0.6, 0.3, -0.4]),
            2,
        ),
        (
            'dxy_s_generic_alpha_y',
            lambda r: _orbital_dxy(0.8, r),
            lambda r: _orbital_s(1.0, r),
            np.array([0.5, 0.6, 0.4]),
            1,
        ),
        (
            'dxy_pz_generic_alpha_x',
            lambda r: _orbital_dxy(0.7, r),
            lambda r: _orbital_p(0.9, 2, r),
            np.array([0.4, -0.5, 0.7]),
            0,
        ),
    ],
)
def test_dipole_overlap_vs_brute_force(name, factory_A, factory_B, R, alpha, radial_mesh):
    r = radial_mesh
    lA, mA, RA, fA_3d = factory_A(r)
    lB, mB, RB, fB_3d = factory_B(r)

    got = two_center_dipole_overlap(
        r,
        RA,
        lA,
        mA,
        r,
        RB,
        lB,
        mB,
        R,
        alpha=alpha,
        q_max=20.0,
        n_q=600,
    )
    expected = _brute_force_dipole(fA_3d, fB_3d, R, alpha)

    # Brute-force 3D quadrature on 121³ at lim=6 carries ~1% error itself.
    assert got == pytest.approx(expected, rel=3e-2, abs=2e-3), (
        f'[{name}] got={got:.6e}, expected={expected:.6e}'
    )
