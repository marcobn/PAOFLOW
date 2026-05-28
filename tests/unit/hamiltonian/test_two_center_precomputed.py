"""Tests pinning the precomputed-J primitives against the originals.

:func:`two_center_overlap_precomputed` and
:func:`two_center_dipole_overlap_precomputed` must agree with
:func:`two_center_overlap` and :func:`two_center_dipole_overlap` to high
precision -- the precomputed variants exist purely as a caching
optimization for the real-space table builder.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.hamiltonian._two_center import (
    radial_bessel_transform,
    two_center_dipole_overlap,
    two_center_dipole_overlap_precomputed,
    two_center_overlap,
    two_center_overlap_precomputed,
)


def _radial_grid_and_funcs():
    r = np.linspace(0.0, 10.0, 1200)
    fA = np.exp(-0.6 * r**2)
    fB = np.exp(-0.4 * r**2)
    return r, fA, fB


@pytest.mark.parametrize(
    'lA,mA,lB,mB,R',
    [
        (0, 0, 0, 0, (0.0, 0.0, 1.7)),
        (1, 0, 0, 0, (0.0, 0.0, 1.3)),
        (1, +1, 1, +1, (1.0, 0.7, -0.5)),
        (1, -1, 1, -1, (1.0, -0.5, 0.2)),
        (2, 0, 0, 0, (0.5, 0.5, 0.5)),
        (2, +2, 2, +2, (0.3, -0.6, 0.9)),
        (2, +1, 1, 0, (1.0, 0.0, 0.7)),
    ],
)
def test_two_center_overlap_precomputed_matches_original(lA, mA, lB, mB, R):
    r, fA, fB = _radial_grid_and_funcs()
    q_max, n_q = 18.0, 500
    q_grid = np.linspace(0.0, q_max, n_q)
    JA = radial_bessel_transform(r, fA, lA, q_grid)
    JB = radial_bessel_transform(r, fB, lB, q_grid)
    expected = two_center_overlap(
        r,
        fA,
        lA,
        mA,
        r,
        fB,
        lB,
        mB,
        np.asarray(R),
        q_max=q_max,
        n_q=n_q,
    )
    got = two_center_overlap_precomputed(
        JA,
        JB,
        lA,
        mA,
        lB,
        mB,
        np.asarray(R),
        q_grid,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    'lA,mA,lB,mB,alpha,R',
    [
        (0, 0, 1, 0, 2, (0.0, 0.0, 1.5)),
        (0, 0, 1, +1, 0, (1.0, 0.0, 0.5)),
        (0, 0, 1, -1, 1, (0.3, 1.2, -0.4)),
        (1, 0, 0, 0, 2, (0.0, 0.0, 1.0)),
        (1, +1, 2, +1, 0, (0.7, -0.2, 0.5)),
        (2, 0, 1, 0, 2, (0.5, 0.5, 0.5)),
    ],
)
def test_two_center_dipole_overlap_precomputed_matches_original(lA, mA, lB, mB, alpha, R):
    r, fA, fB = _radial_grid_and_funcs()
    q_max, n_q = 18.0, 500
    q_grid = np.linspace(0.0, q_max, n_q)

    # Modified-bra J's: g_A(r) = r * f_A(r), evaluated at L' in {|lA-1|, lA+1}
    # with parity (lA + 1) % 2.
    gA = r * fA
    parity = (lA + 1) % 2
    J_gA_by_Lp = {}
    for Lp in (abs(lA - 1), lA + 1):
        if Lp % 2 != parity:
            continue
        J_gA_by_Lp[Lp] = radial_bessel_transform(r, gA, Lp, q_grid)
    JB = radial_bessel_transform(r, fB, lB, q_grid)

    expected = two_center_dipole_overlap(
        r,
        fA,
        lA,
        mA,
        r,
        fB,
        lB,
        mB,
        np.asarray(R),
        alpha,
        q_max=q_max,
        n_q=n_q,
    )
    got = two_center_dipole_overlap_precomputed(
        J_gA_by_Lp,
        JB,
        lA,
        mA,
        lB,
        mB,
        np.asarray(R),
        alpha,
        q_grid,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_dipole_precomputed_rejects_bad_alpha():
    r, fA, fB = _radial_grid_and_funcs()
    q_grid = np.linspace(0.0, 18.0, 400)
    JB = radial_bessel_transform(r, fB, 0, q_grid)
    with pytest.raises(ValueError, match='alpha'):
        two_center_dipole_overlap_precomputed(
            {1: np.zeros_like(q_grid)},
            JB,
            0,
            0,
            0,
            0,
            np.array([0.0, 0.0, 1.0]),
            7,
            q_grid,
        )
