"""Unit tests for the electron-phonon symmetry-expansion core."""

import numpy as np
import pytest

from PAOFLOW.elphon.symmetry import (
    build_dv_symmetry_operators,
    cartesian_derivatives_from_directional,
    expand_directional_responses,
    frac_to_cartesian_rotation,
    rotate_dV_under_symmetry,
    site_symmetry_rotations,
)


def _cubic_proper_rotations():
    """The 24 proper rotations of the cubic point group as integer matrices."""
    from itertools import permutations, product

    mats = []
    for perm in permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for signs in product((1, -1), repeat=3):
            M = P * np.array(signs)[:, None]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    return np.array(mats)


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


# ---------------------------------------------------------------------------
# Wigner-D / dV rotation machinery (simple-cubic, single s orbital: the
# orbital rotation is trivial so the plumbing -- atom permutation, phase,
# k-remap, Cartesian direction handling -- is exercised directly).
# ---------------------------------------------------------------------------


def _cubic_s_operators(ngrid=(3, 3, 3)):
    a_vectors = np.eye(3)  # simple cubic, alat = 1
    tau = np.zeros((1, 3))
    sym_rot = _cubic_proper_rotations()
    equiv_atom = np.zeros((sym_rot.shape[0], 1), dtype=int)  # atom fixed at origin
    shells = {'X': [0]}  # single s orbital
    atoms = ['X']
    return build_dv_symmetry_operators(
        a_vectors, tau, 1.0, sym_rot, equiv_atom, shells, atoms, ngrid
    )


def test_build_dv_symmetry_operators_structure():
    ops = _cubic_s_operators()
    n = ops['symop'].shape[0]
    assert ops['U'].shape == (n, 1, 1)  # single s orbital
    # s-orbital rotation is trivial (identity) for every proper rotation.
    np.testing.assert_allclose(np.abs(ops['U']), 1.0, atol=1e-8)
    # Cartesian rotations are orthogonal.
    for R in ops['symop_cart']:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)


def test_rotate_dV_identity_is_noop():
    ops = _cubic_s_operators()
    rng = np.random.default_rng(3)
    dV = rng.standard_normal((1, 1, 3, 3, 3, 1))
    out = rotate_dV_under_symmetry(dV, 0, ops)  # op 0 is the identity
    np.testing.assert_allclose(out, dV, atol=1e-10)


def test_rotate_dV_s_orbital_symmetric_field_invariant():
    # A scalar hopping H(R) depending only on |R| is invariant under every
    # cubic rotation: rotate_dV must return it unchanged.
    ops = _cubic_s_operators()
    n1, n2, n3 = ops['ngrid']
    f = np.fft.fftfreq(n1, 1.0 / n1).astype(int)
    R2 = (f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2).astype(float)
    field = np.exp(-R2)[None, None, :, :, :, None]  # (1,1,n,n,n,1)
    for isym in range(ops['symop'].shape[0]):
        out = rotate_dV_under_symmetry(field, isym, ops)
        np.testing.assert_allclose(out, field, atol=1e-8)


def test_expand_directional_generates_independent_star():
    ops = _cubic_s_operators()
    rng = np.random.default_rng(4)
    dV0 = rng.standard_normal((1, 1, 3, 3, 3, 1))
    directional = [{'sc_atom': 0, 'displacement': [1.0, 0.0, 0.0], 'dV': dV0}]
    expanded = expand_directional_responses(directional, ops)
    dirs = np.array([e['displacement'] for e in expanded], dtype=float)
    # The cubic site group turns a single x displacement into x, y, z (>=3 rank).
    assert np.linalg.matrix_rank(dirs, tol=1e-8) == 3
    # Every generated entry displaces the same (fixed) atom.
    assert all(e['sc_atom'] == 0 for e in expanded)


def test_expand_then_cartesian_solve_recovers_full_tensor():
    # For the s-orbital model with trivial orbital rotation, rotating dV along a
    # cubic axis just relabels the direction, so the least-squares Cartesian
    # solve over the generated star recovers the (diagonal) response tensor.
    ops = _cubic_s_operators()
    n1, n2, n3 = ops['ngrid']
    f = np.fft.fftfreq(n1, 1.0 / n1).astype(int)
    R2 = (f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2).astype(float)
    dV0 = np.exp(-R2)[None, None, :, :, :, None]  # symmetric field along "x"
    directional = [{'sc_atom': 0, 'displacement': [1.0, 0.0, 0.0], 'dV': dV0}]
    expanded = expand_directional_responses(directional, ops)
    dirs = np.array([e['displacement'] for e in expanded], dtype=float)
    resp = np.array([e['dV'] for e in expanded])
    D = cartesian_derivatives_from_directional(dirs, resp)
    assert D.shape[0] == 3
    # x-component equals the measured symmetric field; y, z equal it too (the
    # field is isotropic), consistent with rotating a symmetric hopping.
    np.testing.assert_allclose(D[0], dV0, atol=1e-8)
