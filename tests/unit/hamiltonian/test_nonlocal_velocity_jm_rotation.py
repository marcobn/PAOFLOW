"""Phase C tests for build_jm_transformation_matrix.

These checks verify that the global ``T`` assembled from the per-shell
Clebsch-Gordan and tesseral->Y_lm blocks is unitary, has the expected
block-diagonal structure in (atom, shell), and behaves correctly on
synthetic Pt-like and multi-atom (GaAs-like) bases.
"""

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    build_jm_transformation_matrix,
    rotate_dp_to_jm,
)
from PAOFLOW.projection.do_atwfc_proj import (
    clebsch_jm_matrix,
    tesseral_to_ylm_matrix,
)


def _make_shell(atom, label, l, *, dft_so):
    """Construct a sequence of basis dicts for one shell."""
    if dft_so:
        n = 2 * (2 * l + 1)
    else:
        n = 2 * l + 1
    return [{'atom': atom, 'label': label, 'l': l, 'm': m + 1} for m in range(n)]


def _make_basis_pair(shells):
    """shells: list of (atom, label, l). Returns (basis_rel, basis_scalar)."""
    rel, scalar = [], []
    for atom, label, l in shells:
        rel.extend(_make_shell(atom, label, l, dft_so=True))
        scalar.extend(_make_shell(atom, label, l, dft_so=False))
    return rel, scalar


# ----------------------------------------------------------------------
# Basic shapes and unitarity
# ----------------------------------------------------------------------


def test_T_shape_and_unitarity_pt_like():
    # Pt-ish: 1 atom with s + p + d (n_s = 9, n_r = 18).
    shells = [('Pt', '5S', 0), ('Pt', '5P', 1), ('Pt', '5D', 2)]
    rel, sca = _make_basis_pair(shells)
    T = build_jm_transformation_matrix(rel, sca)

    n_s = len(sca)
    n_r = len(rel)
    assert n_s == 9
    assert n_r == 18
    assert T.shape == (n_r, 2 * n_s)

    # T must be unitary on its rows (T @ T^H == I_{n_r}) since n_r = 2 n_s.
    eye = T @ T.conj().T
    np.testing.assert_allclose(eye, np.eye(n_r), atol=1e-12)

    # Likewise T^H @ T == I_{2 n_s}.
    eye2 = T.conj().T @ T
    np.testing.assert_allclose(eye2, np.eye(2 * n_s), atol=1e-12)


def test_T_shape_and_unitarity_pt_extended():
    # Extended Pt basis (matches Pt_REL production layout).
    labels = ['5S', '5P', '5D', '6S', '7S', '8S', '6P', '7P', '8P', '6D', '7D']
    ls = [0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2]
    shells = [('Pt', lab, l) for lab, l in zip(labels, ls)]
    rel, sca = _make_basis_pair(shells)
    T = build_jm_transformation_matrix(rel, sca)

    assert len(sca) == 31
    assert len(rel) == 62
    assert T.shape == (62, 62)

    np.testing.assert_allclose(T @ T.conj().T, np.eye(62), atol=1e-12)


def test_T_block_diagonal_in_shells():
    # GaAs-like: 2 atoms, s + p each. Verify T is shell-block-diagonal.
    shells = [
        ('Ga', '4S', 0),
        ('Ga', '4P', 1),
        ('As', '4S', 0),
        ('As', '4P', 1),
    ]
    rel, sca = _make_basis_pair(shells)
    n_s = len(sca)
    T = build_jm_transformation_matrix(rel, sca)

    # Build expected zero-mask: nonzero only within each shell's rows
    # paired with that shell's (up-col, down-col) blocks.
    r_ptr = 0
    s_ptr = 0
    mask = np.zeros_like(T, dtype=bool)
    for _atom, _label, l in shells:
        n_m = 2 * l + 1
        shell_r = 2 * n_m
        rows = slice(r_ptr, r_ptr + shell_r)
        mask[rows, s_ptr : s_ptr + n_m] = True
        mask[rows, n_s + s_ptr : n_s + s_ptr + n_m] = True
        r_ptr += shell_r
        s_ptr += n_m

    # Outside the per-shell blocks T must be exactly zero.
    assert np.all(T[~mask] == 0)


# ----------------------------------------------------------------------
# Per-shell content matches Phase B helpers
# ----------------------------------------------------------------------


@pytest.mark.parametrize('l', [0, 1, 2, 3])
def test_T_shell_content_matches_C_and_U(l):
    n_m = 2 * l + 1
    shells = [('X', 'AA', l)]
    rel, sca = _make_basis_pair(shells)
    n_s = len(sca)
    T = build_jm_transformation_matrix(rel, sca)

    U = tesseral_to_ylm_matrix(l)
    C = clebsch_jm_matrix(l)
    U_spin = np.zeros((2 * n_m, 2 * n_m), dtype=complex)
    U_spin[:n_m, :n_m] = U
    U_spin[n_m:, n_m:] = U
    T_shell_expected = C @ U_spin

    # Up-spin block:
    np.testing.assert_allclose(T[:, :n_m], T_shell_expected[:, :n_m], atol=1e-14)
    # Down-spin block (global col offset n_s):
    np.testing.assert_allclose(T[:, n_s : n_s + n_m], T_shell_expected[:, n_m:], atol=1e-14)


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------


def test_mismatched_shell_raises():
    rel, sca = _make_basis_pair([('Pt', '5S', 0), ('Pt', '5P', 1)])
    # Swap the scalar shell ordering -> mismatch at first shell.
    sca_bad = sca[1:] + sca[:1]
    with pytest.raises(RuntimeError, match='Scalar shell'):
        build_jm_transformation_matrix(rel, sca_bad)


def test_size_mismatch_raises():
    rel, sca = _make_basis_pair([('Pt', '5S', 0)])
    with pytest.raises(RuntimeError, match='n_r=. must equal'):
        build_jm_transformation_matrix(rel, sca + sca)


def test_rel_shell_truncated_raises():
    # Drop a single entry from the rel s-shell to break the 2(2l+1) contract.
    rel, sca = _make_basis_pair([('Pt', '5S', 0), ('Pt', '5P', 1)])
    rel_bad = rel[:1] + rel[2:]  # 1+6 = 7 entries, scalar still has 4
    with pytest.raises(RuntimeError):
        build_jm_transformation_matrix(rel_bad, sca[:-1])


# ----------------------------------------------------------------------
# rotate_dp_to_jm
# ----------------------------------------------------------------------


def _random_hermitian(n, rng):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return 0.5 * (A + A.conj().T)


def test_rotate_dp_preserves_hermiticity_and_trace():
    shells = [('Pt', '5S', 0), ('Pt', '5P', 1), ('Pt', '5D', 2)]
    rel, sca = _make_basis_pair(shells)
    T = build_jm_transformation_matrix(rel, sca)
    n_s = len(sca)
    n_r = len(rel)
    rng = np.random.default_rng(20260528)
    dp = np.stack([_random_hermitian(n_s, rng) for _ in range(3)])  # (3, n_s, n_s)

    dp_rel = rotate_dp_to_jm(dp, T)
    assert dp_rel.shape == (3, n_r, n_r)

    # Hermiticity per alpha.
    for a in range(3):
        np.testing.assert_allclose(dp_rel[a], dp_rel[a].conj().T, atol=1e-12)

    # The block-diagonal spinor extension doubles the trace.
    for a in range(3):
        np.testing.assert_allclose(np.trace(dp_rel[a]), 2.0 * np.trace(dp[a]), atol=1e-12)


def test_rotate_dp_matches_explicit_block_diagonal():
    # Independent reference: build the explicit (2 n_s, 2 n_s) block-diag
    # spinor matrix and rotate via T directly.
    shells = [('Ga', '4S', 0), ('Ga', '4P', 1), ('As', '4P', 1)]
    rel, sca = _make_basis_pair(shells)
    T = build_jm_transformation_matrix(rel, sca)
    n_s = len(sca)
    rng = np.random.default_rng(1)
    dp = _random_hermitian(n_s, rng)  # (n_s, n_s)

    dp_spinor = np.zeros((2 * n_s, 2 * n_s), dtype=complex)
    dp_spinor[:n_s, :n_s] = dp
    dp_spinor[n_s:, n_s:] = dp
    expected = T @ dp_spinor @ T.conj().T

    got = rotate_dp_to_jm(dp, T)
    np.testing.assert_allclose(got, expected, atol=1e-13)


def test_rotate_dp_broadcasts_over_leading_axes():
    # Mimic (nk, 3, n_s, n_s) layout.
    shells = [('Pt', '5S', 0), ('Pt', '5D', 2)]
    rel, sca = _make_basis_pair(shells)
    T = build_jm_transformation_matrix(rel, sca)
    n_s = len(sca)
    n_r = len(rel)
    nk = 4

    rng = np.random.default_rng(7)
    dp = np.stack(
        [np.stack([_random_hermitian(n_s, rng) for _ in range(3)]) for _ in range(nk)]
    )  # (nk, 3, n_s, n_s)

    dp_rel = rotate_dp_to_jm(dp, T)
    assert dp_rel.shape == (nk, 3, n_r, n_r)

    # Compare against a per-k, per-alpha explicit loop.
    expected = np.zeros((nk, 3, n_r, n_r), dtype=complex)
    for k in range(nk):
        for a in range(3):
            dp_spinor = np.zeros((2 * n_s, 2 * n_s), dtype=complex)
            dp_spinor[:n_s, :n_s] = dp[k, a]
            dp_spinor[n_s:, n_s:] = dp[k, a]
            expected[k, a] = T @ dp_spinor @ T.conj().T
    np.testing.assert_allclose(dp_rel, expected, atol=1e-13)


def test_rotate_dp_shape_mismatch_raises():
    shells = [('Pt', '5S', 0), ('Pt', '5P', 1)]
    rel, sca = _make_basis_pair(shells)
    T = build_jm_transformation_matrix(rel, sca)
    bad = np.zeros((3, 5, 5), dtype=complex)  # wrong n_s
    with pytest.raises(ValueError, match='n_s='):
        rotate_dp_to_jm(bad, T)
