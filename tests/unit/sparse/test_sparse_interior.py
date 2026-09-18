"""Interior-window eigensolver: every state in [elo, ehi], neither end counted.

The dense branch (``zheevr`` ``subset_by_value``) is exact by construction, so
it doubles as the reference for the shift-invert path.  Every correctness test
here therefore compares the two branches on the same matrix rather than
against hand-computed numbers.

Degenerate multiplets are tested explicitly: the intended applications (folded
supercells, moire flat-band manifolds) put multiplets *inside* the window by
design, which is exactly where an un-orthogonalized ARPACK block would corrupt
anything quadratic in the eigenvectors.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from PAOFLOW.sparse.solver import count_in_window, solve_interior


def _hermitian(n, seed=0, density=0.05):
    """Reproducible sparse Hermitian test matrix with a spread spectrum."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A[np.abs(A) < np.quantile(np.abs(A), 1.0 - density)] = 0.0
    A = A + A.conj().T
    A += np.diag(np.linspace(-10.0, 10.0, n))
    return csr_matrix(A)


def _spectrum(H):
    return np.linalg.eigvalsh(H.toarray())


@pytest.mark.parametrize('n', [200, 400])
@pytest.mark.parametrize('width', [0.5, 2.0, 6.0])
def test_sparse_matches_dense(n, width):
    """The shift-invert path must return the same window as subset_by_value."""
    H = _hermitian(n, seed=n)
    ev = _spectrum(H)
    mid = float(np.median(ev))
    elo, ehi = mid - width, mid + width

    E_d, V_d = solve_interior(H, elo, ehi, hk_solver='dense')
    E_s, V_s = solve_interior(H, elo, ehi, hk_solver='sparse', k0=4)

    assert len(E_s) == len(E_d), 'window count differs: %d vs %d' % (len(E_s), len(E_d))
    assert np.allclose(E_s, E_d, atol=1e-9)
    # eigenvector gauges differ; compare the gauge-invariant projector
    assert np.allclose(V_s @ V_s.conj().T, V_d @ V_d.conj().T, atol=1e-8)


def test_count_matches_a_full_diagonalization():
    H = _hermitian(300, seed=3)
    ev = _spectrum(H)
    elo, ehi = -1.5, 2.5
    expected = int(((ev >= elo) & (ev <= ehi)).sum())
    assert count_in_window(H, elo, ehi) == expected
    E, _ = solve_interior(H, elo, ehi, hk_solver='sparse', k0=2)
    assert len(E) == expected


def test_k0_is_only_a_hint_never_a_truncation():
    """A hopelessly small k0 must grow, not silently return fewer states."""
    H = _hermitian(300, seed=7)
    elo, ehi = -2.0, 2.0
    ref, _ = solve_interior(H, elo, ehi, hk_solver='dense')
    for k0 in (1, 2, 5, 200):
        E, _ = solve_interior(H, elo, ehi, hk_solver='sparse', k0=k0)
        assert len(E) == len(ref), 'k0=%d truncated the window' % k0
        assert np.allclose(E, ref, atol=1e-9)


def test_empty_window_returns_empty_not_an_error():
    """A window inside a gap is legitimate and must come back correctly shaped."""
    n = 200
    ev = np.concatenate([np.linspace(-10, -3, n // 2), np.linspace(3, 10, n // 2)])
    H = csr_matrix(diags(ev))
    for solver in ('dense', 'sparse'):
        E, V = solve_interior(H, -1.0, 1.0, hk_solver=solver, k0=4)
        assert E.shape == (0,), solver
        assert V.shape == (n, 0), solver


@pytest.mark.parametrize('mult', [2, 4, 8])
def test_degenerate_multiplet_inside_window(mult):
    """Full multiplicity, and an orthonormal block -- the moire flat-band case."""
    n = 240
    ev = np.linspace(-12.0, 12.0, n)
    ev[n // 2 : n // 2 + mult] = 0.37  # exact multiplet inside the window
    rng = np.random.default_rng(11)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    H = csr_matrix(Q @ np.diag(ev) @ Q.conj().T)

    E, V = solve_interior(H, -0.5, 1.0, hk_solver='sparse', k0=4)
    assert int(np.isclose(E, 0.37, atol=1e-7).sum()) == mult
    gram = V.conj().T @ V
    assert np.allclose(gram, np.eye(V.shape[1]), atol=1e-8), 'block is oblique'


def test_window_covering_everything_falls_back_under_auto_and_raises_under_sparse():
    H = _hermitian(80, seed=5)
    lo, hi = float(_spectrum(H)[0]) - 1.0, float(_spectrum(H)[-1]) + 1.0
    E, _ = solve_interior(H, lo, hi, hk_solver='auto', interior_dense_n=0, k0=4)
    assert len(E) == 80
    with pytest.raises(RuntimeError, match='nearly the whole spectrum'):
        solve_interior(H, lo, hi, hk_solver='sparse', k0=4)


def test_bad_window_and_bad_solver_raise():
    H = _hermitian(60, seed=1)
    with pytest.raises(ValueError, match='need ehi > elo'):
        solve_interior(H, 1.0, 1.0)
    with pytest.raises(ValueError, match="must be 'auto'"):
        solve_interior(H, -1.0, 1.0, hk_solver='lobpcg')


def test_auto_uses_dense_below_the_threshold():
    """Small n must not pay for an splu; result identical either way."""
    H = _hermitian(100, seed=2)
    E_a, _ = solve_interior(H, -1.0, 1.0, hk_solver='auto', interior_dense_n=512)
    E_d, _ = solve_interior(H, -1.0, 1.0, hk_solver='dense')
    assert np.allclose(E_a, E_d, atol=1e-12)


def test_solve_lowest_is_untouched_by_the_new_path():
    """The interior solver is additive: the from-the-bottom path is unchanged."""
    from PAOFLOW.sparse.solver import solve_lowest

    H = _hermitian(200, seed=9)
    E, V = solve_lowest(H, 12, hk_solver='sparse')
    ref = _spectrum(H)[:12]
    assert np.allclose(E, ref, atol=1e-9)
    assert V.shape == (200, 12)


def test_ladder_extrapolates_instead_of_doubling_blindly():
    """A window holding far more states than k0 must still resolve.

    Blind doubling needs log2(m/k0) rounds and exhausts max_attempts on a
    problem that is not hard; the ladder instead measures the level density
    from the failed attempt and jumps.  With k0=4 and ~300 states in the
    window, doubling would reach only 128 in 6 attempts.
    """
    n = 900
    ev = np.linspace(-5.0, 5.0, n)  # uniform, dense spectrum
    rng = np.random.default_rng(23)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    H = csr_matrix(Q @ np.diag(ev) @ Q.conj().T)

    elo, ehi = -1.7, 1.7
    expected = int(((ev >= elo) & (ev <= ehi)).sum())
    assert expected > 250, 'test premise: window must hold many states'

    E, V = solve_interior(H, elo, ehi, hk_solver='sparse', k0=4)
    assert len(E) == expected
    assert np.allclose(E, ev[(ev >= elo) & (ev <= ehi)], atol=1e-7)
