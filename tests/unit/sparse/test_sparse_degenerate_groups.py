"""Degenerate-group helpers of the sparse mesh pass.

- ``kpoint.degenerate_groups`` lists each state in at most one group, where
  ``get_degeneracies`` repeats a group whose cluster spans two rounded
  energies (a near-degenerate pair with an exact copy, as on a folded cell).
- ``mesh._invariant_subspaces`` splits a degenerate group into the copies
  that the three velocity blocks never couple, and leaves a genuine
  multiplet whole.
- ``mesh._band_diagonal_velocities`` gives each hidden copy the velocities it
  has on its own.
"""

import numpy as np
from scipy.linalg import block_diag

from PAOFLOW.sparse.kpoint import degenerate_groups
from PAOFLOW.sparse.mesh import _band_diagonal_velocities, _commute, _invariant_subspaces
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
