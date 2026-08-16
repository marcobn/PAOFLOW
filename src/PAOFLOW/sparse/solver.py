"""Iterative lowest-eigenpair solver for sparse Bloch Hamiltonians.

Strictly sparse: the matrix is never densified (``.toarray()`` is never
called), per the sparse-backend contract.  The primary method is ARPACK
``eigsh`` in shift-invert mode with the shift placed strictly below the
spectrum via a Gershgorin lower bound, which turns the lowest eigenvalues
into the largest (and well separated) eigenvalues of ``(H - sigma)^-1``
— the regime where Lanczos converges fast.  The inverse is applied with a
sparse LU factorization (``splu`` inside scipy), not a dense one.

Failure handling is loud: a retry ladder (larger Krylov space, more
iterations, plain ``which='SA'``) is attempted and exhaustion raises
``RuntimeError`` with diagnostics.  There is no silent dense fallback.
"""

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def gershgorin_lower(H):
    """Rigorous lower bound on the spectrum of a Hermitian sparse matrix:
    ``min_i (Re H_ii - sum_{j != i} |H_ij|)``."""
    H = H.tocsr()
    diag = H.diagonal().real
    abs_row_sums = np.abs(H).sum(axis=1).ravel()
    return float((diag - (abs_row_sums - np.abs(diag))).min())


def solve_lowest(H, nev, sigma=None, v0=None, tol=0.0, guard=4):
    """Lowest ``nev`` eigenpairs of a Hermitian sparse matrix, iteratively.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    nev : int
        Number of lowest eigenpairs wanted.
    sigma : float or None
        Shift for shift-invert.  Default: ``gershgorin_lower(H) - 1.0``,
        strictly below the spectrum (never inside a cluster).
    v0 : np.ndarray or None
        Warm-start vector for the Lanczos iteration (e.g. the lowest
        eigenvector of a neighbouring k-point).
    tol : float
        ARPACK relative tolerance; 0.0 means machine precision.
    guard : int
        Extra pairs computed and discarded, so a degenerate cluster split
        exactly at ``nev`` does not stall convergence.

    Returns
    -------
    (E, V) : eigenvalues ``(nev,)`` ascending, eigenvectors ``(n, nev)``.
    """
    n = H.shape[0]
    k = nev + guard
    if k >= n - 1:
        raise NotImplementedError(
            'solve_lowest: nev + guard = %d is too close to the matrix size n = %d '
            'for iterative solution (ARPACK needs k < n - 1). Reduce the number of '
            'requested bands or increase the basis.' % (k, n)
        )
    if sigma is None:
        sigma = gershgorin_lower(H) - 1.0

    def _sorted_lowest(E, V):
        # ARPACK's return order follows the transformed problem and is not
        # guaranteed ascending; sort before dropping the guard pairs, or
        # degenerate copies land in the discarded tail.
        order = np.argsort(E)[:nev]
        return E[order], V[:, order]

    attempts = []
    ncv = None  # ARPACK default: min(n, max(2k + 1, 20))
    maxiter = None  # ARPACK default: n * 10
    for attempt in range(3):
        try:
            E, V = eigsh(
                H,
                k=k,
                sigma=sigma,
                which='LM',
                mode='normal',
                v0=v0,
                tol=tol,
                ncv=ncv,
                maxiter=maxiter,
            )
            return _sorted_lowest(E, V)
        except ArpackNoConvergence as err:
            attempts.append(
                'shift-invert(sigma=%.3f, ncv=%s, maxiter=%s): %s' % (sigma, ncv, maxiter, err)
            )
            ncv = min(n, max(4 * k + 1, 40) * (attempt + 1))
            maxiter = 40 * n * (attempt + 1)

    try:
        E, V = eigsh(
            H, k=k, which='SA', v0=v0, tol=tol, ncv=min(n, max(4 * k + 1, 40)), maxiter=100 * n
        )
        return _sorted_lowest(E, V)
    except ArpackNoConvergence as err:
        attempts.append('SA: %s' % err)

    raise RuntimeError(
        'solve_lowest failed to converge %d eigenpairs of an n=%d sparse matrix.\n'
        'Attempts:\n  %s' % (k, n, '\n  '.join(attempts))
    )
