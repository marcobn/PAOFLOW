"""Parity: ``band_curvature_k`` (all six components at once) against
``band_curvature_k_ij`` (one component per call, the dense pipeline's kernel).

The folded case copies one block twice, as ``doubling_Hamiltonian`` does, so
every band is doubly degenerate; with ``dH/dk_x`` and ``dH/dk_z`` copied too,
the rotated velocities stay degenerate and the subspace ``eigh`` branch runs.
An odd ``bnd`` puts a degenerate pair across the window edge, which
``get_degeneracies`` drops from both the energy and the velocity groups.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from PAOFLOW.hamiltonian.do_d2Hd2k import IJ_PAIRS
from PAOFLOW.sparse.curvature import band_curvature_k
from PAOFLOW.spectrum.do_band_curvature import band_curvature_k_ij, velocity_groups
from PAOFLOW.spectrum.do_eigh import get_degeneracies


def _herm(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _operators(rng, n, folded):
    if not folded:
        return _herm(rng, n), [_herm(rng, n) for _ in range(3)], [_herm(rng, n) for _ in range(6)]
    copy = lambda a: np.kron(np.eye(2), a)  # noqa: E731
    h = copy(_herm(rng, n // 2))
    dh = [copy(_herm(rng, n // 2)), _herm(rng, n), copy(_herm(rng, n // 2))]
    return h, dh, [_herm(rng, n) for _ in range(6)]


@pytest.mark.parametrize('sparse', [False, True], ids=['dense', 'sparse'])
@pytest.mark.parametrize('folded', [False, True], ids=['generic', 'folded'])
@pytest.mark.parametrize('bnd', [9, 12, 24])
def test_band_curvature_k_matches_per_component(folded, sparse, bnd):
    rng = np.random.default_rng(7)
    n = 24
    h, dh, d2h = _operators(rng, n, folded)
    E, V = np.linalg.eigh(h)
    degen = get_degeneracies(E[None, :, None], bnd)[0][0]
    if folded:
        assert degen, 'the folded case must have degenerate groups'
    if sparse:
        dh = [csr_matrix(a) for a in dh]
        d2h = [csr_matrix(a) for a in d2h]

    expected = np.array(
        [
            band_curvature_k_ij(E, dh[i], dh[j], d2h[ij], V, degen, bnd)
            for ij, (i, j) in enumerate(IJ_PAIRS)
        ]
    )
    got = band_curvature_k(E, dh, d2h, V, degen, bnd, degen)
    assert got.shape == (6, bnd)
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8 * np.abs(expected).max())


def testvelocity_groups_need_not_be_adjacent_or_share_energy_across_groups():
    # a folded cluster ordered (a, b, b, a) along v, and a state of another energy at the same v
    vel = np.array([0.3, -0.1, -0.1, 0.3, 0.3])
    groups = velocity_groups(vel, [np.arange(4)])
    assert sorted(g.tolist() for g in groups) == [[0, 3], [1, 2]]


def _copy_curvature(h, dh, d2h):
    """Non-degenerate curvature of one copy: E, velocities (3, n) and the six components."""
    e, v = np.linalg.eigh(h)
    p = [v.conj().T @ a @ v for a in dh]
    de = e[:, None] - e[None, :]
    np.fill_diagonal(de, np.inf)
    c = [
        np.diag(v.conj().T @ t @ v).real + np.sum(2 * (p[i] * p[j].T).real / de, axis=1)
        for (i, j), t in zip(IJ_PAIRS, d2h)
    ]
    return e, np.array([np.diag(a).real for a in p]), np.array(c)


def test_folded_mirror_copies_keep_their_own_mixed_curvature():
    # two copies k and k' = (kx, -ky, kz), as folding puts them: equal E and v_x, opposite v_y,
    # in the sparse joint basis; sorting d2E/dkx dky inside the v_x group swapped the copies
    from PAOFLOW.sparse.kpoint import degenerate_groups
    from PAOFLOW.sparse.mesh import _band_diagonal_velocities

    rng = np.random.default_rng(1)
    nb = 4
    h0, dh1, d2h1 = (
        _herm(rng, nb),
        [_herm(rng, nb) for _ in range(3)],
        [_herm(rng, nb) for _ in IJ_PAIRS],
    )
    s = [1, -1, 1]
    dh2 = [s[i] * a for i, a in enumerate(dh1)]
    d2h2 = [s[i] * s[j] * a for (i, j), a in zip(IJ_PAIRS, d2h1)]
    z = np.zeros((nb, nb))
    bd = lambda a, b: np.block([[a, z], [z, b]])  # noqa: E731
    dh = [bd(a, b) for a, b in zip(dh1, dh2)]
    d2h = [bd(a, b) for a, b in zip(d2h1, d2h2)]

    E, V = np.linalg.eigh(bd(h0, h0))
    vel, resolved = _band_diagonal_velocities(E, V, dh)
    for D, R, _ in resolved:
        V[:, D] = V[:, D] @ R
    degen = [p for _, _, parts in resolved for p in parts]
    got = band_curvature_k(E, dh, d2h, V, degen, len(E), degenerate_groups(E, len(E)))

    copies = [_copy_curvature(h0, dh1, d2h1), _copy_curvature(h0, dh2, d2h2)]
    expected = np.empty_like(got)
    for n in range(len(E)):
        hits = [
            c[:, m]
            for e, v, c in copies
            for m in range(nb)
            if abs(E[n] - e[m]) < 1e-8 and np.allclose(vel[:, n], v[:, m])
        ]
        assert len(hits) == 1
        expected[:, n] = hits[0]
    np.testing.assert_allclose(got, expected, atol=1e-8)
