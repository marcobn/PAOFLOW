"""Lowest-eigenpair solver for sparse Bloch Hamiltonians.

Two per-k kernels, chosen by :func:`select_hk_solver` from ``(n, nev)`` alone. ``'dense'`` names
how one k-point's ``H(k)`` is diagonalized, never the storage model: ``H(R)`` stays a bond list,
the one ``(n, n)`` scratch is freed before the next k, and no ``O(nk n^2)`` tensor is formed.

- ``'sparse'``: ARPACK shift-invert (``eigsh``, shift below the spectrum via Gershgorin, ``splu``
  inverse) when ``nev`` is a small fraction of ``n``; never densifies H(k).
- ``'dense'``: ``eigh(driver='evr', subset_by_index)`` when ``nev`` approaches ``n``, a measured
  exception to the "strictly iterative" rule (``sparse/README.md``). scipy's
  ``ncv = min(n, max(2k+1, 20))`` degenerates to ``ncv = n`` for ``k = nev + guard`` above ~n/2,
  so ARPACK allocates a dense ``(n, n)`` basis anyway, at 13-66x the cost of ``eigh``.

``evr`` (MRRR) because ``evd`` has no subset support and ``evx`` resolves clusters by inverse
iteration, while 8-64-fold exact multiplets are normal under N-fold folding; ``evr`` returns an
orthonormal cluster basis by construction. Past ``dense_n_max`` the mixed regime raises
``NotImplementedError`` naming both exits; ARPACK failures end in ``RuntimeError``. No silent
fallback in either direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg
from scipy.sparse import spmatrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

DENSE_RATIO = 0.125
DENSE_N_MAX = 4096
# extra pairs ARPACK converges beyond nev, so a multiplet cut at nev does not stall it
GUARD = 4


def gershgorin_lower(H: spmatrix) -> float:
    """Gershgorin lower bound :math:`\\min_i(\\mathrm{Re}\\,H_{ii} - \\sum_{j\\ne i}|H_{ij}|)`
    on the spectrum of a Hermitian sparse ``H``.

    One pass, no iteration: a shift-invert origin below the spectrum at every k, so it never lands
    in a cluster where ``H - sigma`` would be near-singular.
    """
    H = H.tocsr()
    diag = H.diagonal().real
    abs_row_sums = np.abs(H).sum(axis=1).ravel()
    return float((diag - (abs_row_sums - np.abs(diag))).min())


def select_hk_solver(
    n: int,
    nev: int,
    hk_solver: str = 'auto',
    dense_ratio: float = DENSE_RATIO,
    dense_n_max: int = DENSE_N_MAX,
) -> tuple[str, str]:
    """Pick the per-k kernel, 'sparse' or 'dense', for an ``(n, nev)`` solve.

    Parameters
    ----------
    n : int
        Matrix dimension (number of orbitals).
    nev : int
        Number of lowest eigenpairs wanted.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        ``'sparse'``/``'dense'`` are honoured verbatim, for A/B validation.
    dense_ratio : float, optional
        Fraction of the spectrum above which an iterative solve stops paying.
    dense_n_max : int, optional
        Largest ``n`` with an affordable ``(n, n)`` per-k scratch.

    Returns
    -------
    (hk_solver, reason) : tuple of str
        Chosen kernel and a justification, logged once per k-loop.

    Raises
    ------
    ValueError
        Unknown ``hk_solver``.
    NotImplementedError
        Past the iterative regime and ``n > dense_n_max``; the message names both exits.

    Notes
    -----
    First match: forced kernel; ``nev + guard >= n - 1`` -> dense (no Krylov room);
    ``nev + guard > ratio*n`` -> dense if ``n <= dense_n_max`` else raise; otherwise sparse.
    Depends on ``(n, nev)`` only, never on matrix values, so the kernel cannot change partway
    through a mesh and make two halves of one property incomparable.
    """
    k = nev + GUARD
    if hk_solver in ('sparse', 'dense'):
        return hk_solver, f"forced by hk_solver='{hk_solver}'"
    if hk_solver != 'auto':
        raise ValueError(
            f"select_hk_solver: hk_solver must be 'auto', 'sparse' or 'dense', got {hk_solver!r}"
        )
    if k >= n - 1:
        return 'dense', f'nev+guard={k} leaves no Krylov room in n={n} (ARPACK needs k < n-1)'
    if k > dense_ratio * n:
        if n <= dense_n_max:
            return 'dense', (
                f'nev+guard={k} is {100.0 * k / n:.1f}% of n={n} (> {100.0 * dense_ratio:.1f}%), '
                'where ARPACK degenerates to a dense (n,n) Arnoldi basis anyway'
            )
        raise NotImplementedError(
            f'solve_lowest: nev+guard = {k} of n = {n} ({100.0 * k / n:.1f}%) is past the '
            f'iterative regime (> {100.0 * dense_ratio:.1f}%), but n exceeds dense_n_max = '
            f'{dense_n_max}, where the per-k (n,n) scratch matrix would be '
            f'{16.0 * n * n / 1024**3:.2f} GB.\n'
            "Two exits: (a) reduce nev (narrower property ranges, 'nev' or 'interior_window' in "
            "sparse_config) so the solve returns to the hk_solver='sparse' regime, or (b) move "
            'to a distributed eigensolver (ELPA/SLEPc) with a distributed bond list. There is '
            'no silent dense fallback.'
        )
    return 'sparse', f'nev+guard={k} is {100.0 * k / n:.1f}% of n={n}'


def hk_solver_fields(n: int, nev: int, hk_solver: str = 'auto', **kw: Any) -> list[tuple[str, str]]:
    """``(key, value)`` log fields of the per-k dispatch, written once per k-loop.

    Chosen kernel, solved fraction ``nev/n`` and, for ``'dense'``, the per-k scratch size.
    Arguments are forwarded to :func:`select_hk_solver`.
    """
    chosen, _ = select_hk_solver(n, nev, hk_solver=hk_solver, **kw)
    if chosen == 'dense':
        fields = [
            ('H(k) solver', f'dense (LAPACK, one {n} x {n} matrix per k-point)'),
            ('per-k scratch (MB)', f'{16.0 * n * n / 1024**2:.1f}'),
        ]
    else:
        fields = [('H(k) solver', 'sparse (ARPACK shift-invert)')]
    fields.append(('solved fraction nev/n', f'{nev / n:.2f}'))
    return fields


def creep_warning(n: int, nev: int) -> str | None:
    """Warning when ``nev > n/2``, else ``None``: the eigenvector block alone is then
    quadratic in ``n``, outside the regime this backend serves (the dense kernel still runs).
    """
    if nev <= 0.5 * n:
        return None
    return (
        f'nev/n = {nev / n:.2f}: an iterative solve brings nothing here; narrow the property '
        "ranges, give 'nev', or use 'interior_window' in sparse_config before growing the cell"
    )


def _orthonormalize_degenerate(E: np.ndarray, V: np.ndarray) -> np.ndarray:
    """QR-orthonormalize ``V`` (n, m) in place on each degenerate group of ascending ``E``.

    ARPACK returns degenerate multiplets as valid eigenvectors that are oblique to each other
    (singular values 0.89-1.10 on an 8-fold multiplet, projector error 6e-3), silently biasing
    PDOS weights :math:`|V_{mn}|^2` and the ``perturb_split`` block
    :math:`V_D^\\dagger (\\partial H/\\partial k) V_D`. QR keeps the span, so gauge-invariant
    quantities are unchanged. A rank-deficient group (span itself wrong) raises ``RuntimeError``.
    The groups are those of :func:`~PAOFLOW.sparse.kpoint.degenerate_groups`, which
    ``perturb_split`` later rotates, so every group it treats as one is orthonormal.
    """
    from .kpoint import degenerate_groups

    for g in degenerate_groups(E, len(E)):
        a, b = int(g[0]), int(g[-1]) + 1
        if b - a > 1:
            Q, R = np.linalg.qr(V[:, a:b])
            if np.abs(np.diag(R)).min() < 1e-8:
                raise RuntimeError(
                    f'solve_lowest: the {b - a}-fold degenerate group at E = {E[a]:.6f} eV came '
                    'back rank-deficient from ARPACK, so it does not span its eigenspace and '
                    "cannot be repaired. Raise ncv, or use hk_solver='dense' at this size."
                )
            V[:, a:b] = Q
    return V


def _solve_dense(H: spmatrix, nev: int) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``nev`` pairs, ``E (nev,)`` and C-contiguous ``V (n, nev)``, by ``zheevr`` on a
    throwaway ``(n, n)`` scratch freed on return; the other ``n - nev`` vectors are never formed.
    """
    A = H.toarray()
    E, V = scipy.linalg.eigh(
        A, subset_by_index=[0, nev - 1], driver='evr', check_finite=False, overwrite_a=True
    )
    return E, np.ascontiguousarray(V)


def solve_lowest(
    H: spmatrix,
    nev: int,
    v0: np.ndarray | None = None,
    hk_solver: str = 'auto',
    dense_ratio: float = DENSE_RATIO,
    dense_n_max: int = DENSE_N_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``nev`` eigenpairs of a Hermitian sparse matrix.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    nev : int
        Number of lowest eigenpairs wanted.
    v0 : np.ndarray or None, optional
        ARPACK warm start (e.g. a neighbouring k's lowest vector); ignored by ``'dense'``.
    hk_solver, dense_ratio, dense_n_max
        Kernel dispatch, see :func:`select_hk_solver`.

    Returns
    -------
    (E, V) : tuple of np.ndarray
        Eigenvalues ``(nev,)`` ascending and eigenvectors ``(n, nev)``.

    Raises
    ------
    NotImplementedError
        ``hk_solver='sparse'`` forced with no Krylov room.
    RuntimeError
        The whole retry ladder failed; the message lists every attempt.

    Notes
    -----
    Shift-invert at :func:`gershgorin_lower` minus 1 maps the lowest states to the largest of
    :math:`(H - \\sigma)^{-1}` (applied by sparse LU). ``GUARD`` extra pairs, because cutting a
    folded-cell multiplet at ``nev`` stalls convergence; the surplus is dropped after sorting.
    Retry ladder: grow ``ncv``/``maxiter`` twice, then one ``which='SA'`` solve without LU, then
    raise. The dense and sparse kernels agree to ~1e-9 eV but not bitwise, and their gauges differ
    inside degenerate subspaces: compare gauge-invariant quantities only.
    """
    n = H.shape[0]
    chosen, _ = select_hk_solver(
        n, nev, hk_solver=hk_solver, dense_ratio=dense_ratio, dense_n_max=dense_n_max
    )
    if chosen == 'dense':
        return _solve_dense(H, nev)

    k = nev + GUARD
    if k >= n - 1:
        raise NotImplementedError(
            f'solve_lowest: nev + guard = {k} is too close to the matrix size n = {n} for '
            'iterative solution (ARPACK needs k < n - 1). Drop the explicit '
            "hk_solver='sparse' to dispatch this size to the dense kernel."
        )
    sigma = gershgorin_lower(H) - 1.0

    def _sorted_lowest(E: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep the ``nev`` lowest of the ``k`` converged pairs, in order."""
        order = np.argsort(E)[:nev]
        return E[order], _orthonormalize_degenerate(E[order], V[:, order])

    attempts = []
    ncv0 = min(n, max(4 * k + 1, 40))
    ncv = ncv0
    maxiter = None
    for attempt in range(3):
        try:
            E, V = eigsh(
                H,
                k=k,
                sigma=sigma,
                which='LM',
                mode='normal',
                v0=v0,
                ncv=ncv,
                maxiter=maxiter,
            )
            return _sorted_lowest(E, V)
        except ArpackNoConvergence as err:
            attempts.append(f'shift-invert(sigma={sigma:.3f}, ncv={ncv}, maxiter={maxiter}): {err}')
            ncv = min(n, ncv0 * (attempt + 2))
            maxiter = 40 * n * (attempt + 1)

    try:
        E, V = eigsh(H, k=k, which='SA', v0=v0, ncv=min(n, max(4 * k + 1, 40)), maxiter=100 * n)
        return _sorted_lowest(E, V)
    except ArpackNoConvergence as err:
        attempts.append(f'SA: {err}')

    attempt_lines = '\n  '.join(attempts)
    raise RuntimeError(
        f'solve_lowest failed to converge {k} eigenpairs of an n={n} sparse matrix.\n'
        f'Attempts:\n  {attempt_lines}'
    )


INTERIOR_DENSE_N = 512
INTERIOR_K0 = 16
INTERIOR_GROWTH = 2.0
INTERIOR_MAX_ATTEMPTS = 6


def _solve_interior_dense(H: spmatrix, elo: float, ehi: float) -> tuple[np.ndarray, np.ndarray]:
    """Every pair in ``[elo, ehi]``, ``E (m,)`` and ``V (n, m)``, by ``zheevr`` ``subset_by_value``.

    Small-``n`` branch of :func:`solve_interior` and the A/B reference for its sparse path.
    """
    A = H.toarray()
    E, V = scipy.linalg.eigh(
        A,
        subset_by_value=[float(elo), float(ehi)],
        driver='evr',
        check_finite=False,
        overwrite_a=True,
    )
    return E, np.ascontiguousarray(V)


def solve_interior(
    H: spmatrix,
    elo: float,
    ehi: float,
    k0: int | None = None,
    hk_solver: str = 'auto',
    dense_n_max: int = DENSE_N_MAX,
    interior_dense_n: int = INTERIOR_DENSE_N,
) -> tuple[np.ndarray, np.ndarray]:
    """All eigenpairs of a Hermitian sparse ``H`` with ``elo <= E <= ehi``.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    elo, ehi : float
        Window bounds in eV, ``ehi > elo``.  Both are inclusive.
    k0 : int or None, optional
        Starting Krylov request (e.g. a neighbouring k's count); a bad one costs time only.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        ``'auto'`` goes dense below ``interior_dense_n`` (``splu`` costs more than a subset solve)
        and when the window holds nearly the whole spectrum; ``'sparse'`` raises there instead.
    dense_n_max : int, optional
        Largest affordable ``(n, n)`` scratch when dense is reached by fallback, not by request.
    interior_dense_n : int, optional
        Below this ``n``, ``'auto'`` goes straight to the dense branch.

    Returns
    -------
    (E, V) : tuple of np.ndarray
        ``E (m,)`` ascending and ``V (n, m)``; ``m`` varies with k and is 0 in a gap (arrays
        still correctly shaped). Not padded: callers needing rectangular arrays pad themselves.

    Raises
    ------
    ValueError
        Empty or inverted window, or unknown ``hk_solver``.
    NotImplementedError
        Dense branch needed past ``dense_n_max``.
    RuntimeError
        Attempts exhausted, or forced sparse on a window holding nearly the whole spectrum.

    Notes
    -----
    Cost tracks the states *in* the window, not below it (transport, Fermi-surface and flat-band
    properties need only a few :math:`k_BT` around :math:`E_F`); that fraction is invariant under
    N-fold folding. Shift-invert at the window centre returns the ``k`` states nearest it; the set
    is accepted once it reaches past both edges. Otherwise ``k`` is extrapolated from the measured
    level density ``k/span`` times the window width, not doubled, which would burn
    ``log2(m/k0)`` attempts. Degenerate groups are re-orthonormalized as in :func:`solve_lowest`.
    """
    n = H.shape[0]
    elo = float(elo)
    ehi = float(ehi)
    if not ehi > elo:
        raise ValueError(f'solve_interior: need ehi > elo, got [{elo:g}, {ehi:g}]')
    if hk_solver not in ('auto', 'sparse', 'dense'):
        raise ValueError(
            f"solve_interior: hk_solver must be 'auto', 'sparse' or 'dense', got {hk_solver!r}"
        )

    def _dense(reason: str) -> tuple[np.ndarray, np.ndarray]:
        """Take the dense branch, unless its scratch matrix is unaffordable."""
        if n > dense_n_max and hk_solver != 'dense':
            raise NotImplementedError(
                f'solve_interior: {reason}, but n = {n} exceeds dense_n_max = {dense_n_max}, '
                f'where the (n,n) scratch would be {16.0 * n * n / 1024**3:.2f} GB. Narrow the '
                "window, or pass hk_solver='dense' explicitly if that allocation is affordable "
                'here.'
            )
        return _solve_interior_dense(H, elo, ehi)

    if hk_solver == 'dense':
        return _solve_interior_dense(H, elo, ehi)
    if hk_solver == 'auto' and n <= interior_dense_n:
        return _solve_interior_dense(H, elo, ehi)
    if n < 5:
        return _dense(f'n = {n} is too small for a Krylov solve')

    sigma = 0.5 * (elo + ehi)
    k = max(1, int(INTERIOR_K0 if k0 is None else k0))
    kmax = n - 2
    attempts = []

    for _ in range(INTERIOR_MAX_ATTEMPTS):
        k = min(k, kmax)
        try:
            E, V = eigsh(
                H,
                k=k,
                sigma=sigma,
                which='LM',
                mode='normal',
                ncv=min(n, max(2 * k + 1, 20)),
            )
        except ArpackNoConvergence as err:
            attempts.append(f'shift-invert(sigma={sigma:.4f}, k={k}): {err}')
            k = int(np.ceil(k * INTERIOR_GROWTH))
            continue

        order = np.argsort(E)
        E = E[order]
        V = V[:, order]
        if E[0] < elo and E[-1] > ehi:
            mask = (E >= elo) & (E <= ehi)
            Ew = E[mask]
            Vw = np.ascontiguousarray(V[:, mask])
            return Ew, _orthonormalize_degenerate(Ew, Vw)

        if k >= kmax:
            reason = f'the window holds nearly the whole spectrum (k reached n-2 = {kmax})'
            if hk_solver == 'sparse':
                raise RuntimeError(
                    f'solve_interior: {reason}, so shift-invert cannot bracket '
                    f'[{elo:g}, {ehi:g}]. Drop the explicit '
                    "hk_solver='sparse' to dispatch this to the dense branch."
                )
            return _dense(reason)
        attempts.append(
            f'k={k} returned [{E[0]:.4f}, {E[-1]:.4f}], which does not straddle [{elo:g}, {ehi:g}]'
        )
        span = float(E[-1] - E[0])
        est = k * 2.0
        if span > 0.0:
            est = 1.3 * (k / span) * (ehi - elo) + 8.0
        k = int(np.ceil(max(est, k * INTERIOR_GROWTH)))

    attempt_lines = '\n  '.join(attempts)
    raise RuntimeError(
        f'solve_interior failed to bracket the window [{elo:g}, {ehi:g}] of an n={n} matrix in '
        f'{INTERIOR_MAX_ATTEMPTS} attempts (last k={k}).\nAttempts:\n  {attempt_lines}\n'
        'Either the window holds a large fraction of the spectrum -- in which case use '
        "hk_solver='dense' -- or ARPACK is not converging; pass a larger k0."
    )


def solve_k(
    hk: spmatrix,
    nsolve: int,
    interior: tuple[float, float] | None,
    warm: Any,
    hk_solver: str,
    dense_ratio: float = DENSE_RATIO,
    dense_n_max: int = DENSE_N_MAX,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """One k-point of a pass: the ``nsolve`` lowest states, or every state in ``interior``.

    ``warm`` is the third value the previous k-point returned (``None`` at the first): its
    lowest eigenvector as ``v0``, or twice its interior count as ``k0``.
    """
    if interior is None:
        E, V = solve_lowest(
            hk,
            nsolve,
            v0=warm,
            hk_solver=hk_solver,
            dense_ratio=dense_ratio,
            dense_n_max=dense_n_max,
        )
        return E, V, np.ascontiguousarray(V[:, 0])
    E, V = solve_interior(
        hk, interior[0], interior[1], k0=warm, hk_solver=hk_solver, dense_n_max=dense_n_max
    )
    return E, V, max(8, 2 * len(E))
