"""Unit tests for the electron-phonon symmetry-expansion core."""

import numpy as np
import pytest

from PAOFLOW.elphon.symmetry import (
    cartesian_derivatives_from_directional,
    frac_to_cartesian_rotation,
    site_symmetry_rotations,
)


def test_frac_to_cartesian_identity():
    a = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]]) * 7.63
    R = frac_to_cartesian_rotation(np.eye(3), a)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_frac_to_cartesian_is_orthogonal_for_lattice_symmetry():
    # A fractional rotation that is a symmetry of the fcc lattice maps to an
    # orthogonal Cartesian rotation.
    a = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]]) * 7.63
    # 90-degree-like fractional permutation that is an fcc symmetry.
    W = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    R = frac_to_cartesian_rotation(W, a)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)  # orthogonal
    assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-10


def test_site_symmetry_all_fix_atom_at_origin():
    # Every operation fixes an atom at the origin (single-atom cell).
    sym = np.array([np.eye(3), [[0, 1, 0], [1, 0, 0], [0, 0, 1]], -np.eye(3)], dtype=float)
    shifts = np.zeros((3, 3))
    pos = np.zeros((1, 3))
    keep = site_symmetry_rotations(sym, shifts, pos, 0)
    np.testing.assert_array_equal(keep, [0, 1, 2])


def test_site_symmetry_filters_non_fixing_ops():
    # Two atoms; a swap operation does not fix atom 0.
    sym = np.array([np.eye(3), np.eye(3)], dtype=float)
    shifts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])  # second op shifts atom off itself
    pos = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    keep = site_symmetry_rotations(sym, shifts, pos, 0)
    np.testing.assert_array_equal(keep, [0])


def test_cartesian_solve_recovers_exact_with_three_axes():
    rng = np.random.default_rng(0)
    D_true = rng.standard_normal((3, 2, 2))  # (alpha, tensor...)
    dirs = np.eye(3)  # x, y, z
    responses = np.einsum('na,axy->nxy', dirs, D_true)
    D = cartesian_derivatives_from_directional(dirs, responses)
    np.testing.assert_allclose(D, D_true, atol=1e-12)


def test_cartesian_solve_overdetermined_star():
    rng = np.random.default_rng(1)
    D_true = rng.standard_normal((3, 3, 3))
    # A redundant star of directions (more than 3), still linear in D.
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [-1, 0, 1]], float)
    responses = np.einsum('na,axy->nxy', dirs, D_true)
    D = cartesian_derivatives_from_directional(dirs, responses)
    np.testing.assert_allclose(D, D_true, atol=1e-10)


def test_cartesian_solve_rejects_degenerate_directions():
    dirs = np.array([[1, 0, 0], [2, 0, 0], [-1, 0, 0]], float)  # all along x
    responses = np.zeros((3, 2))
    with pytest.raises(ValueError):
        cartesian_derivatives_from_directional(dirs, responses)
