"""solve_lowest: correctness against a dense reference (test-side only)."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from PAOFLOW.sparse.solver import gershgorin_lower, solve_lowest


def _random_sparse_hermitian(rng, n, band=8, noise=0.05):
    """Banded Hermitian matrix with a bit of long-range noise, CSR."""
    A = np.zeros((n, n), dtype=complex)
    for d in range(band):
        v = rng.standard_normal(n - d) + 1j * rng.standard_normal(n - d)
        A += np.diag(v, k=d)
    mask = rng.random((n, n)) < noise
    A[mask] += rng.standard_normal(mask.sum()) + 1j * rng.standard_normal(mask.sum())
    A = 0.5 * (A + A.conj().T)
    return csr_matrix(A), A


def test_matches_dense_eigh():
    rng = np.random.default_rng(3)
    H, A = _random_sparse_hermitian(rng, 200)
    nev = 24
    E, V = solve_lowest(H, nev)
    E_ref = np.linalg.eigvalsh(A)[:nev]
    assert np.allclose(E, E_ref, atol=1e-9)
    # residuals ||H v - e v|| validate eigenvectors without fixing the gauge
    R = H @ V - V * E[None, :]
    assert np.abs(R).max() < 1e-8


def test_gershgorin_bound_below_spectrum():
    rng = np.random.default_rng(5)
    H, A = _random_sparse_hermitian(rng, 120)
    lo = gershgorin_lower(H)
    assert lo <= np.linalg.eigvalsh(A)[0] + 1e-12


def test_degenerate_cluster_straddling_nev():
    """An exact multiplet split by the nev cut must still converge and
    return the correct lowest eigenvalues (the guard absorbs the cluster)."""
    rng = np.random.default_rng(7)
    n = 150
    evals = np.sort(rng.standard_normal(n)) * 5.0
    evals[10:16] = evals[10]  # 6-fold degenerate cluster
    Q = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    A = (Q * evals[None, :]) @ Q.conj().T
    A = 0.5 * (A + A.conj().T)
    H = csr_matrix(A)
    nev = 13  # cuts through the middle of the cluster
    E, V = solve_lowest(H, nev)
    assert np.allclose(E, np.linalg.eigvalsh(A)[:nev], atol=1e-8)


def test_warm_start_accepts_v0():
    rng = np.random.default_rng(9)
    H, A = _random_sparse_hermitian(rng, 100)
    nev = 10
    E1, V1 = solve_lowest(H, nev)
    E2, _ = solve_lowest(H, nev, v0=V1[:, 0])
    assert np.allclose(E1, E2, atol=1e-9)


def test_nev_too_close_to_n_raises():
    rng = np.random.default_rng(11)
    H, _ = _random_sparse_hermitian(rng, 30)
    with pytest.raises(NotImplementedError):
        solve_lowest(H, 28, guard=4)
