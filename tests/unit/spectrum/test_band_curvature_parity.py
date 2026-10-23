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
from PAOFLOW.spectrum.do_band_curvature import band_curvature_k, band_curvature_k_ij
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
    got = band_curvature_k(E, dh, d2h, V, degen, bnd)
    assert got.shape == (6, bnd)
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8 * np.abs(expected).max())
