"""Band curvature at degenerate states against finite differences of the eigenvalues.

Taylor models ``H(k) = H0 + k.V + k.W.k / 2`` (so ``dH = V`` and ``d2H = W`` exactly at k=0),
scrambled by a random unitary so LAPACK's basis inside a degenerate level is arbitrary. The
cases are those of ``benchmarks/band_curvature_toy_models.ipynb``: equal velocities at
different energies (mirror plane), a threefold level with zero velocity and Kramers pairs.
"""

import numpy as np
import pytest

from PAOFLOW.hamiltonian.do_d2Hd2k import IJ_PAIRS
from PAOFLOW.sparse.curvature import band_curvature_k
from PAOFLOW.spectrum.do_band_curvature import band_curvature_k_ij
from PAOFLOW.spectrum.do_eigh import get_degeneracies

rng = np.random.default_rng(7)


def _herm(n, s=1.0):
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return s * (x + x.conj().T) / 2


def _unitary(n):
    return np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]


def _W(n, s=0.5, sym=lambda a, b, x: x):
    W = np.zeros((3, 3, n, n), complex)
    for a in range(3):
        for b in range(a, 3):
            W[a, b] = W[b, a] = sym(a, b, _herm(n, s))
    return W


def _scramble(H0, V, W):
    U = _unitary(len(H0))
    r = lambda a: U @ a @ U.conj().T  # noqa: E731
    return r(H0), np.array([r(v) for v in V]), np.array([[r(w) for w in row] for row in W])


def _mirror():
    # H0, V_y, V_z even and V_x odd under P: v_x = 0 for every band at k = 0
    n = 6
    P = np.diag([1, 1, 1, -1, -1, -1]).astype(complex)
    even = lambda x: (x + P @ x @ P) / 2  # noqa: E731
    odd = lambda x: (x - P @ x @ P) / 2  # noqa: E731
    H0 = np.zeros((n, n), complex)
    for sl, levels in ((slice(0, 3), [-1.8, -0.3, 1.2]), (slice(3, 6), [-1.0, 0.5, 2.0])):
        u = _unitary(3)
        H0[sl, sl] = u @ np.diag(levels) @ u.conj().T
    V = [odd(_herm(n, 0.6)), even(_herm(n, 0.5)), even(_herm(n, 0.5))]
    W = _W(n, sym=lambda a, b, x: odd(x) if (a == 0) != (b == 0) else even(x))
    return _scramble(H0, V, W)


def _triplet():
    # s1, (px, py, pz) at 0 with v = 0 in every direction, s2: heavy/light-hole-like
    n = 5
    H0 = np.diag([-2.0, 0, 0, 0, 1.5]).astype(complex)
    V = np.zeros((3, n, n), complex)
    for a in range(3):
        V[a, 0, 1 + a] = V[a, 1 + a, 0] = 1.2
        V[a, 4, 1 + a] = V[a, 1 + a, 4] = 0.4
    V[:, 0, 0] = [0.3, 0.2, 0.1]
    V[:, 4, 4] = [-0.5, -0.4, 0.2]
    W = np.zeros((3, 3, n, n), complex)
    for a in range(3):
        W[a, a, 1:4, 1:4] = np.diag([-1.0 if c == a else -0.3 for c in range(3)])
        W[a, a, 0, 0], W[a, a, 4, 4] = 0.8, 1.2
        for b in range(3):
            if a != b:
                W[a, b, 1 + a, 1 + b] = W[a, b, 1 + b, 1 + a] = 0.4
    return _scramble(H0, V, W)


def _kramers():
    # PT symmetry with SOC: Theta = (1 x i sigma_y) K, every level a pair with equal velocity
    T = np.kron(np.eye(4), np.array([[0, 1], [-1, 0]]))
    pt = lambda x: (x + T @ x.conj() @ T.T) / 2  # noqa: E731
    H0 = pt(_herm(8)) + np.kron(np.diag([-2, -0.5, 0.6, 2.0]), np.eye(2))
    V = [pt(_herm(8, 0.5)) for _ in range(3)]
    return _scramble(H0, V, _W(8, sym=lambda a, b, x: pt(x)))


def _fd(H0, V, W, i, h=1e-3):
    """Forward 4-point second derivative of the sorted eigenvalues along k_i (O(h^2)); forward,
    so the order inside a split level is the velocity order the kernels use."""
    f = [np.linalg.eigvalsh(H0 + s * h * V[i] + 0.5 * (s * h) ** 2 * W[i, i]) for s in range(4)]
    return (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / h**2


@pytest.mark.parametrize('model', [_mirror, _triplet, _kramers])
def test_curvature_matches_finite_differences(model):
    H0, V, W = model()
    E, U = np.linalg.eigh(H0)
    n = len(E)
    for bnd in (n, n - 1):
        degen = get_degeneracies(E[None, :, None], bnd)[0][0]
        d2H = [W[i, j] for i, j in IJ_PAIRS]
        sparse = band_curvature_k(E, V, d2H, U, degen, bnd, degen)
        for ij, (i, j) in enumerate(IJ_PAIRS[:3]):
            fd = _fd(H0, V, W, i)[:bnd]
            dense = band_curvature_k_ij(E, V[i], V[j], W[i, j], U, degen, bnd)
            # a level cut by bnd keeps its non-degenerate value; compare whole levels only
            keep = np.ones(bnd, bool)
            if bnd < n and abs(E[bnd] - E[bnd - 1]) < 1e-5:
                keep[np.abs(E[:bnd] - E[bnd]) < 1e-5] = False
            np.testing.assert_allclose(dense[keep], fd[keep], atol=2e-3)
            np.testing.assert_allclose(sparse[ij][keep], fd[keep], atol=2e-3)
