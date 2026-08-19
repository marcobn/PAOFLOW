"""solve_lowest: correctness against a dense reference (test-side only).

Both dispatch branches are exercised.  Where a test is about the ARPACK
mechanism specifically it forces ``hk_solver='sparse'``, since ``'auto'``
now routes large-``nev`` cases to the dense branch.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from PAOFLOW.sparse.solver import gershgorin_lower, select_hk_solver, solve_lowest


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


def test_nev_too_close_to_n_raises_when_sparse_is_forced():
    """The hard stop still exists, but it is now a property of the ARPACK
    branch rather than of solve_lowest: k >= n-1 leaves no Krylov room.
    Under 'auto' the same case is simply dispatched to the dense branch
    (see test_nev_too_close_to_n_dispatches_dense)."""
    rng = np.random.default_rng(11)
    H, _ = _random_sparse_hermitian(rng, 30)
    with pytest.raises(NotImplementedError):
        solve_lowest(H, 28, guard=4, hk_solver='sparse')


def test_nev_too_close_to_n_dispatches_dense():
    """Same (nev, n) as above: 'auto' must solve it, not raise."""
    rng = np.random.default_rng(11)
    H, A = _random_sparse_hermitian(rng, 30)
    assert select_hk_solver(30, 28)[0] == 'dense'
    E, V = solve_lowest(H, 28, guard=4)
    assert np.allclose(E, np.linalg.eigvalsh(A)[:28], atol=1e-10)
    assert V.shape == (30, 28)


@pytest.mark.parametrize('mult', [8, 16, 32])
def test_sparse_kernel_returns_full_multiplicity(mult):
    """Exact multiplets must come back with their true multiplicity.

    A Krylov space from a single start vector holds at most one vector per
    degenerate eigenspace.  At scipy's default ncv = 2k+1 ARPACK returns
    too few copies of a high-multiplicity eigenvalue and never raises --
    every band above the multiplet is then shifted.  Cell folding makes
    8-64-fold multiplets routine, so this pins the widened ncv in
    solve_lowest.  Regression: at 2k+1 this test fails at every mult.
    """
    rng = np.random.default_rng(mult)
    n, nev = 200, 48
    evals = np.sort(rng.standard_normal(n)) * 4.0
    evals[8 : 8 + mult] = evals[8]  # multiplet well inside the window
    Q = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    A = 0.5 * ((Q * evals[None, :]) @ Q.conj().T + ((Q * evals[None, :]) @ Q.conj().T).conj().T)
    H = csr_matrix(A)
    ref = np.linalg.eigvalsh(A)[:nev]
    for hk_solver in ('sparse', 'dense'):
        E, _ = solve_lowest(H, nev, hk_solver=hk_solver)
        assert int(np.isclose(E, evals[8], atol=1e-7).sum()) == mult, hk_solver
        assert np.allclose(E, ref, atol=1e-8), hk_solver


def test_select_hk_solver_rule():
    # small nev fraction -> iterative
    assert select_hk_solver(1000, 50)[0] == 'sparse'
    # past the 1/8 ratio, still small enough for a dense scratch
    assert select_hk_solver(1000, 500)[0] == 'dense'
    # no Krylov room at all
    assert select_hk_solver(100, 97)[0] == 'dense'
    # past the ratio and past dense_n_max: loud, naming both exits
    with pytest.raises(NotImplementedError, match='energy_window'):
        select_hk_solver(9216, 4608)
    # explicit choices are honoured verbatim, including ones auto would refuse
    assert select_hk_solver(9216, 4608, hk_solver='dense')[0] == 'dense'
    assert select_hk_solver(9216, 4608, hk_solver='sparse')[0] == 'sparse'
    with pytest.raises(ValueError):
        select_hk_solver(100, 10, hk_solver='lobpcg')


@pytest.mark.parametrize('n', [200, 400])
@pytest.mark.parametrize('frac', [0.12, 0.5])
def test_dense_matches_sparse(n, frac):
    """The two branches must agree to well inside the physics floor."""
    rng = np.random.default_rng(101 + n)
    H, A = _random_sparse_hermitian(rng, n)
    nev = max(4, int(frac * n))
    E_ref = np.linalg.eigvalsh(A)[:nev]

    E_d, V_d = solve_lowest(H, nev, hk_solver='dense')
    E_a, V_a = solve_lowest(H, nev, hk_solver='sparse')
    assert np.allclose(E_d, E_ref, atol=1e-9)
    assert np.allclose(E_d, E_a, atol=1e-9)
    for V in (V_d, V_a):
        R = H @ V - V * E_d[None, :]
        assert np.abs(R).max() < 1e-8


def test_dense_gauge_invariants_match_sparse():
    """Eigenvectors are gauge-dependent (and the gauge differs between the
    branches inside degenerate clusters), so pin the quantities the
    pipeline actually consumes: band-diagonal expectation values on
    non-degenerate bands, and per-orbital weight sums on degenerate
    groups -- the PDOS-relevant, gauge-invariant combinations."""
    rng = np.random.default_rng(77)
    n = 240
    evals = np.sort(rng.standard_normal(n)) * 4.0
    evals[20:28] = evals[20]  # exact 8-fold multiplet, as cell folding makes
    Q = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    A = (Q * evals[None, :]) @ Q.conj().T
    A = 0.5 * (A + A.conj().T)
    H = csr_matrix(A)
    nev = 40
    E_d, V_d = solve_lowest(H, nev, hk_solver='dense')
    E_a, V_a = solve_lowest(H, nev, hk_solver='sparse')
    assert np.allclose(E_d, E_a, atol=1e-8)

    # an observable with the structure of dH/dk
    O = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    O = 0.5 * (O + O.conj().T)
    diag_d = np.einsum('an,an->n', np.conj(V_d), O @ V_d).real
    diag_a = np.einsum('an,an->n', np.conj(V_a), O @ V_a).real
    nondeg = np.abs(np.diff(np.r_[-np.inf, E_d])) > 1e-6
    nondeg &= np.abs(np.diff(np.r_[E_d, np.inf])) > 1e-6
    assert np.allclose(diag_d[nondeg], diag_a[nondeg], atol=1e-7)

    # on the multiplet only the subspace projector is gauge-free
    D = slice(20, 28)
    assert np.allclose(
        (np.abs(V_d[:, D]) ** 2).sum(axis=1), (np.abs(V_a[:, D]) ** 2).sum(axis=1), atol=1e-7
    )
    assert np.isclose(diag_d[D].sum(), diag_a[D].sum(), atol=1e-7)


@pytest.mark.parametrize('hk_solver', ['sparse', 'dense'])
def test_degenerate_block_is_orthonormal(hk_solver):
    """Both branches must return an orthonormal basis on a multiplet.

    ARPACK converges each multiplet vector as an eigenvector but leaves
    the block oblique, which silently corrupts everything computed from a
    degenerate group -- PDOS weights |V_mn|^2 and the perturb_split block
    V_D^dag dH V_D both assume orthonormality.  Regression: without the
    group QR in solve_lowest this fails for 'sparse' at 6e-1.
    """
    rng = np.random.default_rng(202)
    n, mult = 200, 8
    evals = np.sort(rng.standard_normal(n)) * 4.0
    evals[10 : 10 + mult] = evals[10]
    Q = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    A = (Q * evals[None, :]) @ Q.conj().T
    A = 0.5 * (A + A.conj().T)
    H = csr_matrix(A)
    nev = 40
    E, V = solve_lowest(H, nev, hk_solver=hk_solver)
    B = V[:, 10 : 10 + mult]
    assert np.abs(B.conj().T @ B - np.eye(mult)).max() < 1e-10
    # and the block still spans the true eigenspace
    P_true = Q[:, 10 : 10 + mult] @ Q[:, 10 : 10 + mult].conj().T
    assert np.abs(B @ B.conj().T - P_true).max() < 1e-9
    # the whole returned block is orthonormal, not just the multiplet
    assert np.abs(V.conj().T @ V - np.eye(nev)).max() < 1e-9


def test_dense_ignores_v0_and_returns_contiguous():
    """v0/tol/sigma/guard are ARPACK-only; the dense branch must accept and
    ignore them, and hand back a C-contiguous block (zheevr does not)."""
    rng = np.random.default_rng(13)
    H, A = _random_sparse_hermitian(rng, 120)
    nev = 60
    E, V = solve_lowest(
        H, nev, hk_solver='dense', v0=rng.standard_normal(120), tol=1e-3, sigma=-99.0
    )
    assert V.flags['C_CONTIGUOUS']
    assert np.allclose(E, np.linalg.eigvalsh(A)[:nev], atol=1e-10)


def test_count_below_matches_dense_count():
    from PAOFLOW.sparse.solver import count_below

    rng = np.random.default_rng(19)
    H, A = _random_sparse_hermitian(rng, 150)
    ref = np.linalg.eigvalsh(A)
    # midpoints between eigenvalues: an ehi placed exactly on an eigenvalue is
    # a coin flip at the last bit (LAPACK's range is half-open and its own
    # eigenvalue differs from numpy's in the final ulp), and a real energy
    # window never lands there
    probes = [ref[0] - 1.0, 0.5 * (ref[36] + ref[37]), 0.5 * (ref[99] + ref[100]), ref[-1] + 1.0]
    for ehi in probes:
        assert count_below(H, ehi) == int((ref <= ehi).sum())
