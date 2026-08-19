"""Lowest-eigenpair solver for sparse Bloch Hamiltonians.

Two per-k-point kernels behind one entry point, chosen by
:func:`select_hk_solver` from ``(n, nev)`` alone.  Both consume the same
sparse ``H(k)`` assembled from the bond list, so the choice is only *how
one k-point's matrix is diagonalized* — it never changes how ``H(R)`` is
stored.  ``hk_solver='dense'`` does not put the run back on the dense
PAOFLOW pipeline: there is no ``O(nk * n^2)`` tensor either way.

``hk_solver='sparse'`` — ARPACK shift-invert (``eigsh``, shift below the
spectrum via a Gershgorin bound, inverse applied with a sparse ``splu``)
whenever the requested fraction of the spectrum is small.  This is the
regime Lanczos is built for and the matrix is never densified.

``hk_solver='dense'`` — ``scipy.linalg.eigh``, ``driver='evr'``,
``subset_by_index``, when ``nev`` approaches ``n``.  This is a deliberate, measured exception
to the backend's "strictly iterative" rule — see the memory contract in
``sparse/README.md``.  The reason is that the rule stops buying anything
in this regime: scipy's default Krylov dimension is
``ncv = min(n, max(2k+1, 20))``, so at ``k = nev + guard`` above roughly
``n/2`` it *degenerates to ``ncv = n``* and ARPACK internally allocates a
dense ``(n, n)`` Arnoldi basis plus ``O(n*ncv^2)`` reorthogonalization —
the same per-k memory class as a dense ``eigh``, at 13-66x the cost.  The
``'dense'`` kernel allocates one ``(n, n)`` scratch matrix that is
discarded before the next k-point; no global ``O(nk*n^2)`` tensor is ever
formed, which is the contract that actually matters.

Above ``dense_n_max`` the ``(n, n)`` scratch stops being cheap, so the
mixed regime raises ``NotImplementedError`` naming both exits rather than
silently swapping tens of GB.  Failure stays loud everywhere: the ARPACK
retry ladder ends in ``RuntimeError`` with diagnostics, and there is no
silent fallback in either direction.

Driver choice for the ``'dense'`` kernel is ``evr`` (zheevr, MRRR).  ``evd``
(divide and conquer) has no subset support and always computes all ``n``
vectors; ``evx`` resolves clusters by inverse iteration with explicit
reorthogonalization, whereas at ``N``-fold cell folding multiplets of
8-64 exactly degenerate bands are the *normal* case and ``evr`` returns
an orthonormal cluster basis by construction.
"""

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

DENSE_RATIO = 0.125
DENSE_N_MAX = 4096


def gershgorin_lower(H):
    """Rigorous lower bound on the spectrum of a Hermitian sparse matrix:
    ``min_i (Re H_ii - sum_{j != i} |H_ij|)``."""
    H = H.tocsr()
    diag = H.diagonal().real
    abs_row_sums = np.abs(H).sum(axis=1).ravel()
    return float((diag - (abs_row_sums - np.abs(diag))).min())


def select_hk_solver(
    n, nev, guard=4, hk_solver='auto', dense_ratio=DENSE_RATIO, dense_n_max=DENSE_N_MAX
):
    """Pick the per-k kernel, 'sparse' or 'dense', for an ``(n, nev)`` solve.

    Deterministic in ``(n, nev, guard)`` only — never in the matrix
    values — so it can be hoisted out of a k-loop and reported once.
    First match wins:

    ====================================== ==========================
    condition                              outcome
    ====================================== ==========================
    ``hk_solver`` is 'sparse' or 'dense'   honoured verbatim
    ``nev + guard >= n - 1``               dense (no Krylov room)
    ``nev + guard > ratio*n``, small n     dense
    ``nev + guard > ratio*n``, large n     NotImplementedError
    otherwise                              sparse
    ====================================== ==========================

    Returns ``(hk_solver, reason)``.
    """
    k = nev + guard
    if hk_solver in ('sparse', 'dense'):
        return hk_solver, "forced by hk_solver='%s'" % hk_solver
    if hk_solver != 'auto':
        raise ValueError(
            "select_hk_solver: hk_solver must be 'auto', 'sparse' or 'dense', got %r" % (hk_solver,)
        )
    if k >= n - 1:
        return 'dense', 'nev+guard=%d leaves no Krylov room in n=%d (ARPACK needs k < n-1)' % (k, n)
    if k > dense_ratio * n:
        if n <= dense_n_max:
            return 'dense', (
                'nev+guard=%d is %.1f%% of n=%d (> %.1f%%), where ARPACK degenerates to a '
                'dense (n,n) Arnoldi basis anyway' % (k, 100.0 * k / n, n, 100.0 * dense_ratio)
            )
        raise NotImplementedError(
            'solve_lowest: nev+guard = %d of n = %d (%.1f%%) is past the iterative regime '
            '(> %.1f%%), but n exceeds dense_n_max = %d, where the per-k (n,n) scratch '
            'matrix would be %.2f GB.\n'
            'Two exits: (a) reduce nev with SparsePAOFLOW.energy_window() so the solve '
            "returns to the hk_solver='sparse' regime, or (b) move to a distributed "
            'eigensolver (ELPA/SLEPc) with a distributed bond list. There is no silent '
            'dense fallback.'
            % (k, n, 100.0 * k / n, 100.0 * dense_ratio, dense_n_max, 16.0 * n * n / 1024**3)
        )
    return 'sparse', 'nev+guard=%d is %.1f%% of n=%d' % (k, 100.0 * k / n, n)


def describe_hk_solver(n, nev, guard=4, hk_solver='auto', **kw):
    """Multi-line report of the dispatch, for printing once per k-loop.

    States what the choice does *and does not* change: 'dense' names the
    per-k-point kernel, never the storage model, and the line says so
    explicitly because 'DENSE' in a log emitted by the sparse backend
    otherwise reads as 'this run fell back to dense PAOFLOW'.

    Carries the ``nev`` creep warning that the ``nev + guard >= n - 1``
    hard stop used to enforce: the dense kernel removes the error, so the
    intent has to survive as a warning.
    """
    chosen, reason = select_hk_solver(n, nev, guard=guard, hk_solver=hk_solver, **kw)
    if chosen == 'dense':
        line = (
            'H(k) solver: dense (LAPACK zheevr)\n'
            '  H(R) stays a bond list; one (%d,%d) scratch = %.1f MB per k-point,\n'
            '  freed before the next. No O(nk*n^2) tensor is formed.\n'
            '  n=%d, nev=%d; %s' % (n, n, 16.0 * n * n / 1024**2, n, nev, reason)
        )
    else:
        line = (
            'H(k) solver: sparse (ARPACK shift-invert Lanczos)\n'
            '  H(k) is never densified; only the Krylov basis is stored.\n'
            '  n=%d, nev=%d; %s' % (n, nev, reason)
        )
    if nev > 0.5 * n:
        line += (
            '\n  WARNING: nev/n = %.2f. The eigenvector block is O(n^2/2) and this run is '
            'past the size where an iterative solve helps. Size nev from an energy window '
            '(SparsePAOFLOW.energy_window) before growing the cell further.' % (nev / n)
        )
    return line


def _orthonormalize_degenerate(E, V, decimals=5):
    """Re-orthonormalize the eigenvector block on each degenerate group.

    ARPACK converges each vector of a degenerate multiplet to machine
    precision *as an eigenvector*, but does not orthogonalize them
    against each other: the returned block spans the right eigenspace
    while being noticeably oblique (measured singular values 0.89-1.10
    across an 8-fold multiplet, projector error 6e-3).  Everything the
    pipeline computes from a degenerate group assumes an orthonormal
    basis -- PDOS orbital weights ``|V_mn|^2``, and the ``perturb_split``
    group block ``V_D^dag (dH/dk) V_D`` -- so the obliquity would show up
    as a small, silent, k-dependent error exactly where cell folding puts
    its multiplets.

    Any orthonormal basis of the span is an equally valid eigenbasis
    (the gauge inside a degenerate subspace is free), so a QR of each
    group restores orthonormality without changing any gauge-invariant
    quantity.  The ``'dense'`` ``evr`` kernel returns such a basis by
    construction and does not need this.

    Grouping uses the same 5-decimal rounding convention as
    ``spectrum.do_eigh.get_degeneracies``, so sparse and dense agree on
    what counts as degenerate.
    """
    Er = np.around(E, decimals=decimals)
    edges = np.flatnonzero(np.r_[True, Er[1:] != Er[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a > 1:
            Q, R = np.linalg.qr(V[:, a:b])
            if np.abs(np.diag(R)).min() < 1e-8:
                raise RuntimeError(
                    'solve_lowest: the %d-fold degenerate group at E = %.6f eV came back '
                    'rank-deficient from ARPACK, so it does not span its eigenspace and '
                    "cannot be repaired. Raise ncv, or use hk_solver='dense' at this size."
                    % (b - a, E[a])
                )
            V[:, a:b] = Q
    return V


def _solve_dense(H, nev):
    """Lowest ``nev`` eigenpairs by ``zheevr`` on a per-k dense scratch.

    The ``(n, n)`` array is local to this call and freed on return; the
    caller keeps only the ``(n, nev)`` eigenvector block it already owns.
    """
    A = H.toarray()
    E, V = scipy.linalg.eigh(
        A, subset_by_index=[0, nev - 1], driver='evr', check_finite=False, overwrite_a=True
    )
    return E, np.ascontiguousarray(V)  # evr's V is not C-contiguous


def solve_lowest(
    H,
    nev,
    sigma=None,
    v0=None,
    tol=0.0,
    guard=4,
    hk_solver='auto',
    dense_ratio=DENSE_RATIO,
    dense_n_max=DENSE_N_MAX,
):
    """Lowest ``nev`` eigenpairs of a Hermitian sparse matrix.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    nev : int
        Number of lowest eigenpairs wanted.
    sigma : float or None
        ARPACK shift for shift-invert.  Default: ``gershgorin_lower(H) -
        1.0``, strictly below the spectrum (never inside a cluster).
    v0 : np.ndarray or None
        Warm-start vector for the Lanczos iteration (e.g. the lowest
        eigenvector of a neighbouring k-point).
    tol : float
        ARPACK relative tolerance; 0.0 means machine precision.
    guard : int
        Extra pairs computed and discarded on the ``'sparse'`` path, so a
        degenerate cluster split exactly at ``nev`` does not stall
        convergence.
    hk_solver : {'auto', 'sparse', 'dense'}
        Per-k kernel, see :func:`select_hk_solver`.  ``'sparse'`` and
        ``'dense'`` force a kernel, for A/B validation.  Neither changes
        how ``H(R)`` is stored.

    ``sigma``, ``v0``, ``tol`` and ``guard`` are ARPACK-only and are
    ignored on the ``'dense'`` kernel: ``evr`` returns exactly the requested
    index range with an orthonormal basis on each degenerate cluster, so
    it needs neither a guard nor the ``_sorted_lowest`` fixup.  The two
    branches agree to ~1e-9 eV but are **not** bit-identical (Householder
    tridiagonalization vs an splu solve chain), and their eigenvector
    gauges differ inside degenerate subspaces — compare gauge-invariant
    quantities, never eigenvectors elementwise.

    Returns
    -------
    (E, V) : eigenvalues ``(nev,)`` ascending, eigenvectors ``(n, nev)``.
    """
    n = H.shape[0]
    chosen, _ = select_hk_solver(
        n, nev, guard=guard, hk_solver=hk_solver, dense_ratio=dense_ratio, dense_n_max=dense_n_max
    )
    if chosen == 'dense':
        return _solve_dense(H, nev)

    k = nev + guard
    if k >= n - 1:
        # only reachable with hk_solver='sparse' forced; 'auto' sends this to dense
        raise NotImplementedError(
            'solve_lowest: nev + guard = %d is too close to the matrix size n = %d for '
            'iterative solution (ARPACK needs k < n - 1). Drop the explicit '
            "hk_solver='sparse' to dispatch this size to the dense kernel." % (k, n)
        )
    if sigma is None:
        sigma = gershgorin_lower(H) - 1.0

    def _sorted_lowest(E, V):
        order = np.argsort(E)[:nev]
        return E[order], _orthonormalize_degenerate(E[order], V[:, order])

    attempts = []
    ncv0 = min(n, max(4 * k + 1, 40))
    ncv = ncv0
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
            ncv = min(n, ncv0 * (attempt + 2))
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


def count_below(H, ehi, dense_n_max=DENSE_N_MAX):
    """Number of eigenvalues of ``H`` at or below ``ehi``.

    One tridiagonal reduction with ``zheevr``'s ``subset_by_value`` and no
    eigenvectors — the probe used to size an energy window.  Densifies,
    so it carries the same size guard as the ``'dense'`` kernel.
    """
    n = H.shape[0]
    if n > dense_n_max:
        raise NotImplementedError(
            'count_below: the energy-window probe densifies H(k), which at n = %d would need '
            '%.2f GB (dense_n_max = %d). Pass an explicit nev to energy_window() to skip the '
            'probe, or move to a distributed eigensolver.'
            % (n, 16.0 * n * n / 1024**3, dense_n_max)
        )
    A = H.toarray()
    E = scipy.linalg.eigvalsh(
        A,
        subset_by_value=[-np.inf, float(ehi)],
        driver='evr',
        check_finite=False,
        overwrite_a=True,
    )
    return len(E)
