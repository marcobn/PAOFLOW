"""Degenerate-group helpers of the sparse mesh pass.

- ``kpoint.degenerate_groups`` lists each state in at most one group, where
  ``get_degeneracies`` repeats a group whose cluster spans two rounded
  energies (a near-degenerate pair with an exact copy, as on a folded cell).
- ``solver._orthonormalize_degenerate`` repairs the same groups, also a pair
  whose energies round to different 5-decimal values.
- ``mesh._invariant_subspaces`` splits a degenerate group into the copies
  that the three velocity blocks never couple, and leaves a genuine
  multiplet whole.
- ``mesh._band_diagonal_velocities`` gives each hidden copy the velocities it
  has on its own.
"""

import numpy as np
import pytest
from scipy.linalg import block_diag

from PAOFLOW.sparse.kpoint import degenerate_groups
from PAOFLOW.sparse.mesh import _band_diagonal_velocities, _commute, _invariant_subspaces
from PAOFLOW.sparse.solver import _orthonormalize_degenerate
from PAOFLOW.spectrum.do_eigh import get_degeneracies


def _herm(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _hidden_copies(rng):
    """Two non-commuting 2x2 velocity sectors (k and its time-reversed -k)
    in a basis that mixes them."""
    sectors = [_herm(rng, 2) for _ in range(3)]
    B = np.array([block_diag(b, -b.conj()) for b in sectors])
    U, _ = np.linalg.qr(_herm(rng, 4))
    return sectors, U, np.einsum('ia,lij,jb->lab', U.conj(), B, U)


def test_degenerate_groups_merges_repeated_clusters():
    E = np.array([-1.0, -0.70502, -0.70502, -0.70501, -0.70501, 0.3, 0.3, 1.0])
    raw = get_degeneracies(E[None, :, None], len(E))[0][0]
    assert len(raw) > len({tuple(g) for g in raw}), 'the input must trigger the repeat'
    groups = degenerate_groups(E, len(E))
    assert [list(g) for g in groups] == [[1, 2, 3, 4], [5, 6]]


def test_orthonormalize_covers_pairs_split_by_rounding():
    E = np.array([-1.0, 0.4550049, 0.4550051, 2.0])
    assert np.around(E[1], 5) != np.around(E[2], 5), 'the pair must round apart'
    assert [list(g) for g in degenerate_groups(E, len(E))] == [[1, 2]]
    rng = np.random.default_rng(3)
    V, _ = np.linalg.qr(rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4)))
    V[:, 2] += 0.3 * V[:, 1]  # oblique within the pair, as ARPACK returns it
    V = _orthonormalize_degenerate(E, V)
    assert np.allclose(V.conj().T @ V, np.eye(4), atol=1e-12)


def test_invariant_subspaces_split_hidden_copies():
    rng = np.random.default_rng(1)
    sectors, _, B = _hidden_copies(rng)
    assert not _commute(B)
    Q, parts = _invariant_subspaces(B)
    assert sorted(len(p) for p in parts) == [2, 2]
    for p in parts:
        Bs = np.einsum('ia,lij,jb->lab', Q[:, p].conj(), B, Q[:, p])
        got = np.linalg.eigvalsh(Bs[0])
        expected = [np.linalg.eigvalsh(sectors[0]), np.linalg.eigvalsh(-sectors[0].conj())]
        assert min(np.abs(got - e).max() for e in expected) < 1e-10


def test_invariant_subspaces_fall_back_to_gesvd(monkeypatch):
    """When numpy's gesdd does not converge, the gesvd retry finds the same sectors."""
    _, _, B = _hidden_copies(np.random.default_rng(1))
    Q, parts = _invariant_subspaces(B)

    def no_convergence(*args, **kwargs):
        raise np.linalg.LinAlgError('SVD did not converge')

    monkeypatch.setattr(np.linalg, 'svd', no_convergence)
    Q2, parts2 = _invariant_subspaces(B)
    assert [list(p) for p in parts2] == [list(p) for p in parts]
    for p in parts:
        # the same subspaces, whatever basis gesvd picks inside each
        P, P2 = Q[:, p] @ Q[:, p].conj().T, Q2[:, p] @ Q2[:, p].conj().T
        assert np.allclose(P, P2, atol=1e-10)


def test_invariant_subspaces_name_non_finite_blocks():
    _, _, B = _hidden_copies(np.random.default_rng(1))
    B[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN or inf'):
        _invariant_subspaces(B)


def test_invariant_subspaces_keep_a_multiplet_whole():
    rng = np.random.default_rng(2)
    _, parts = _invariant_subspaces(np.array([_herm(rng, 3) for _ in range(3)]))
    assert [len(p) for p in parts] == [3]


def test_band_velocities_of_hidden_copies():
    """Four degenerate states that are two copies: each copy takes the
    per-direction rule on its own, not across both."""
    rng = np.random.default_rng(3)
    sectors, U, _ = _hidden_copies(rng)
    n = 6
    E = np.array([-2.0, 0.5, 0.5, 0.5, 0.5, 3.0])
    V = np.eye(n, dtype=complex)
    V[:, 1:5] = V[:, 1:5] @ U
    dhk = [block_diag(rng.standard_normal(), b, -b.conj(), rng.standard_normal()) for b in sectors]
    dhk = [0.5 * (d + d.conj().T) for d in dhk]
    vel, resolved = _band_diagonal_velocities(E, V, dhk)
    ((D, R, parts),) = resolved
    assert list(D) == [1, 2, 3, 4] and sorted(len(p) for p in parts) == [2, 2]
    expected = sorted(
        np.round(np.r_[np.linalg.eigvalsh(b), np.linalg.eigvalsh(-b.conj())], 8).tolist()
        for b in sectors
    )
    got = sorted(np.round(np.sort(vel[l, 1:5]), 8).tolist() for l in range(3))
    # per direction, the multiset of the group's velocities is the union of the copies'
    assert np.allclose(sorted(map(sorted, got)), sorted(map(sorted, expected)))
    # and each part holds one copy: its velocities are that copy's eigenvalues
    for p in parts:
        Vp = (V[:, D] @ R)[:, p - D[0]]
        for l, b in enumerate(sectors):
            Bp = Vp.conj().T @ dhk[l] @ Vp
            assert any(
                np.allclose(np.linalg.eigvalsh(Bp), np.linalg.eigvalsh(c), atol=1e-10)
                for c in (b, -b.conj())
            )
