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

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg
from scipy.sparse import spmatrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

DENSE_RATIO = 0.125
DENSE_N_MAX = 4096


def gershgorin_lower(H: spmatrix) -> float:
    """Rigorous lower bound on the spectrum of a Hermitian sparse matrix.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian

    Returns
    -------
    float
        A value guaranteed to be at or below the smallest eigenvalue.

    Notes
    -----
    Gershgorin's theorem places every eigenvalue of a matrix inside at
    least one disc centred on a diagonal entry with radius the sum of the
    absolute off-diagonal entries in that row.  For a Hermitian matrix the
    eigenvalues are real, so the discs collapse to intervals and the lowest
    eigenvalue cannot fall below

    .. math::

        \\min_i \\Big( \\mathrm{Re}\\, H_{ii}
            - \\sum_{j \\neq i} |H_{ij}| \\Big).

    The bound is cheap (one pass over the stored entries) and needs no
    iteration, which is what makes it usable as a shift-invert origin at
    every k-point: placing the shift strictly below the spectrum guarantees
    it never lands inside a cluster of eigenvalues, where the shifted
    inverse would be near-singular.
    """
    H = H.tocsr()
    diag = H.diagonal().real
    abs_row_sums = np.abs(H).sum(axis=1).ravel()
    return float((diag - (abs_row_sums - np.abs(diag))).min())


def select_hk_solver(
    n: int,
    nev: int,
    guard: int = 4,
    hk_solver: str = 'auto',
    dense_ratio: float = DENSE_RATIO,
    dense_n_max: int = DENSE_N_MAX,
) -> tuple[str, str]:
    """Pick the per-k kernel, 'sparse' or 'dense', for an ``(n, nev)`` solve.

    Parameters
    ----------
    n : int
        Matrix dimension, i.e. the number of orbitals.
    nev : int
        Number of lowest eigenpairs wanted.
    guard : int, optional
        Extra pairs the sparse kernel requests on top of ``nev``; counted
        here because it is what ARPACK actually has to converge.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        ``'auto'`` decides from the table below; the other two are honoured
        verbatim, for A/B validation.
    dense_ratio : float, optional
        Fraction of the spectrum above which an iterative solve stops
        paying for itself.
    dense_n_max : int, optional
        Largest ``n`` for which an ``(n, n)`` per-k scratch matrix is
        considered affordable.

    Returns
    -------
    (hk_solver, reason) : tuple of str
        The chosen kernel and a human-readable justification, logged once
        per k-loop.

    Raises
    ------
    ValueError
        If ``hk_solver`` is not one of the three accepted values.
    NotImplementedError
        If the solve is past the iterative regime *and* too large for a
        dense scratch matrix.  The message names both ways out.

    Notes
    -----
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

    The decision is deterministic in ``(n, nev, guard)`` only — never in
    the matrix values — so it can be hoisted out of a k-loop and reported
    once.  That is a property worth preserving: a dispatch that depended on
    the spectrum could silently change kernel partway through a mesh and
    make two halves of one property incomparable.
    """
    k = nev + guard
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
            'Two exits: (a) reduce nev with SparsePAOFLOW.energy_window() so the solve '
            "returns to the hk_solver='sparse' regime, or (b) move to a distributed "
            'eigensolver (ELPA/SLEPc) with a distributed bond list. There is no silent '
            'dense fallback.'
        )
    return 'sparse', f'nev+guard={k} is {100.0 * k / n:.1f}% of n={n}'


def describe_hk_solver(n: int, nev: int, guard: int = 4, hk_solver: str = 'auto', **kw: Any) -> str:
    """Multi-line report of the dispatch, for printing once per k-loop.

    Parameters
    ----------
    n : int
        Matrix dimension.
    nev : int
        Number of lowest eigenpairs wanted.
    guard : int, optional
        Extra pairs requested by the sparse kernel.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        Requested kernel, passed through to :func:`select_hk_solver`.
    **kw
        Further keyword arguments forwarded to :func:`select_hk_solver`
        (``dense_ratio``, ``dense_n_max``).

    Returns
    -------
    str
        Two or three lines naming the chosen kernel, its memory
        consequence, and the reason for the choice.

    Notes
    -----
    The report states what the choice does *and does not* change: 'dense'
    names the per-k-point kernel, never the storage model, and the line
    says so explicitly because the word 'dense' in a log emitted by the
    sparse backend otherwise reads as "this run fell back to dense
    PAOFLOW".

    It also carries the ``nev`` creep warning that the ``nev + guard >=
    n - 1`` hard stop used to enforce.  The dense kernel removes the error,
    so the intent has to survive as a warning: past ``nev`` of about half
    the spectrum the eigenvector block alone is quadratic in ``n`` and the
    run has left the regime this backend exists to serve, even though it
    still completes.
    """
    chosen, reason = select_hk_solver(n, nev, guard=guard, hk_solver=hk_solver, **kw)
    if chosen == 'dense':
        line = (
            'H(k) solver: dense (LAPACK zheevr)\n'
            f'  H(R) stays a bond list; one ({n},{n}) scratch = '
            f'{16.0 * n * n / 1024**2:.1f} MB per k-point,\n'
            '  freed before the next. No O(nk*n^2) tensor is formed.\n'
            f'  n={n}, nev={nev}; {reason}'
        )
    else:
        line = (
            'H(k) solver: sparse (ARPACK shift-invert Lanczos)\n'
            '  H(k) is never densified; only the Krylov basis is stored.\n'
            f'  n={n}, nev={nev}; {reason}'
        )
    if nev > 0.5 * n:
        line += (
            f'\n  WARNING: nev/n = {nev / n:.2f}. The eigenvector block is O(n^2/2) and this '
            'run is past the size where an iterative solve helps. Size nev from an energy '
            'window (SparsePAOFLOW.energy_window) before growing the cell further.'
        )
    return line


def _orthonormalize_degenerate(E: np.ndarray, V: np.ndarray, decimals: int = 5) -> np.ndarray:
    """Re-orthonormalize the eigenvector block on each degenerate group.

    Parameters
    ----------
    E : np.ndarray, shape (m,)
        Eigenvalues, ascending; used only to find the degenerate groups.
    V : np.ndarray, shape (n, m)
        Matching eigenvector block, modified in place on degenerate groups.
    decimals : int, optional
        Rounding applied to ``E`` before grouping.

    Returns
    -------
    np.ndarray, shape (n, m)
        The same array, with each degenerate group replaced by an
        orthonormal basis of its span.

    Raises
    ------
    RuntimeError
        If a group came back rank-deficient, meaning it does not span its
        eigenspace and cannot be repaired by re-orthonormalization.

    Notes
    -----
    ARPACK converges each vector of a degenerate multiplet to machine
    precision *as an eigenvector*, but does not orthogonalize them against
    each other: the returned block spans the right eigenspace while being
    noticeably oblique (measured singular values 0.89-1.10 across an 8-fold
    multiplet, projector error 6e-3).  Everything the pipeline computes
    from a degenerate group assumes an orthonormal basis — PDOS orbital
    weights :math:`|V_{mn}|^2`, and the ``perturb_split`` group block
    :math:`V_D^\\dagger (\\partial H/\\partial k) V_D` — so the obliquity
    would show up as a small, silent, k-dependent error exactly where cell
    folding puts its multiplets.

    Any orthonormal basis of the span is an equally valid eigenbasis (the
    gauge inside a degenerate subspace is free), so a QR factorization of
    each group restores orthonormality without changing any
    gauge-invariant quantity.  A vanishing diagonal element of the QR
    triangular factor means the group's vectors were linearly dependent, so
    the span itself is wrong and no gauge choice can fix it — hence the
    hard failure rather than a silent repair.  The ``'dense'`` ``evr``
    kernel returns an orthonormal cluster basis by construction and does
    not need any of this.

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
                    f'solve_lowest: the {b - a}-fold degenerate group at E = {E[a]:.6f} eV came '
                    'back rank-deficient from ARPACK, so it does not span its eigenspace and '
                    "cannot be repaired. Raise ncv, or use hk_solver='dense' at this size."
                )
            V[:, a:b] = Q
    return V


def _solve_dense(H: spmatrix, nev: int) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``nev`` eigenpairs by ``zheevr`` on a per-k dense scratch.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    nev : int
        Number of lowest eigenpairs wanted.

    Returns
    -------
    (E, V) : tuple of np.ndarray
        Eigenvalues ``(nev,)`` ascending and eigenvectors ``(n, nev)``,
        C-contiguous.

    Notes
    -----
    The ``(n, n)`` array is local to this call and freed on return; the
    caller keeps only the ``(n, nev)`` eigenvector block it already owns.
    ``evr`` computes an index subset of the spectrum, so it never forms the
    other ``n - nev`` eigenvectors, and it is asked to overwrite its input
    since that input is the throwaway scratch.  Its output is returned in
    Fortran order, so it is copied into C order for the callers.
    """
    A = H.toarray()
    E, V = scipy.linalg.eigh(
        A, subset_by_index=[0, nev - 1], driver='evr', check_finite=False, overwrite_a=True
    )
    return E, np.ascontiguousarray(V)


def solve_lowest(
    H: spmatrix,
    nev: int,
    sigma: float | None = None,
    v0: np.ndarray | None = None,
    tol: float = 0.0,
    guard: int = 4,
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
    sigma : float or None, optional
        ARPACK shift for shift-invert.  Default: ``gershgorin_lower(H) -
        1.0``, strictly below the spectrum (never inside a cluster).
    v0 : np.ndarray or None, optional
        Warm-start vector for the Lanczos iteration (e.g. the lowest
        eigenvector of a neighbouring k-point).
    tol : float, optional
        ARPACK relative tolerance; 0.0 means machine precision.
    guard : int, optional
        Extra pairs computed and discarded on the ``'sparse'`` path, so a
        degenerate cluster split exactly at ``nev`` does not stall
        convergence.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        Per-k kernel, see :func:`select_hk_solver`.  ``'sparse'`` and
        ``'dense'`` force a kernel, for A/B validation.  Neither changes
        how ``H(R)`` is stored.
    dense_ratio : float, optional
        Fraction of the spectrum above which the dense kernel is chosen.
    dense_n_max : int, optional
        Largest ``n`` for which the dense kernel's scratch is affordable.

    Returns
    -------
    (E, V) : tuple of np.ndarray
        Eigenvalues ``(nev,)`` ascending and eigenvectors ``(n, nev)``.

    Raises
    ------
    NotImplementedError
        If ``hk_solver='sparse'`` is forced at a size that leaves ARPACK no
        Krylov room.
    RuntimeError
        If every attempt in the retry ladder failed to converge.  The
        message lists what was tried.

    Notes
    -----
    Lanczos-type methods converge fastest on the *outermost* eigenvalues of
    a matrix, and the states wanted here are at the bottom of the spectrum,
    interior to nothing but also not extremal in magnitude.  Shift-invert
    fixes that: the solver is handed the operator :math:`(H - \\sigma)^{-1}`
    with :math:`\\sigma` placed just below the whole spectrum, which maps
    the lowest eigenvalues of :math:`H` to the largest of the transformed
    operator, where the iteration converges quickly.  The inverse is never
    formed — it is applied as a sparse LU solve.  The shift comes from a
    Gershgorin bound (:func:`gershgorin_lower`) minus one, so it is
    guaranteed to sit outside the spectrum and never inside a cluster,
    where the shifted matrix would be near-singular.

    A few extra pairs beyond ``nev`` are requested (``guard``).  Degenerate
    multiplets are common in folded cells, and asking for a count that cuts
    a multiplet in half makes convergence markedly harder; the surplus is
    discarded after sorting.

    Non-convergence is answered with a retry ladder rather than a fallback:
    the Krylov subspace and iteration limit are grown a few times, then one
    attempt is made without shift-invert (targeting the algebraically
    smallest eigenvalues directly, slower to converge but not dependent on
    the LU factorization), and if all of that fails the call raises with the
    full history.  There is no silent degradation of the answer.

    ``sigma``, ``v0``, ``tol`` and ``guard`` are ARPACK-only and are
    ignored on the ``'dense'`` kernel: ``evr`` returns exactly the requested
    index range with an orthonormal basis on each degenerate cluster, so
    it needs neither a guard nor the sort-and-orthonormalize fixup.  The two
    branches agree to ~1e-9 eV but are **not** bit-identical (Householder
    tridiagonalization vs an splu solve chain), and their eigenvector
    gauges differ inside degenerate subspaces — compare gauge-invariant
    quantities, never eigenvectors elementwise.
    """
    n = H.shape[0]
    chosen, _ = select_hk_solver(
        n, nev, guard=guard, hk_solver=hk_solver, dense_ratio=dense_ratio, dense_n_max=dense_n_max
    )
    if chosen == 'dense':
        return _solve_dense(H, nev)

    k = nev + guard
    if k >= n - 1:
        raise NotImplementedError(
            f'solve_lowest: nev + guard = {k} is too close to the matrix size n = {n} for '
            'iterative solution (ARPACK needs k < n - 1). Drop the explicit '
            "hk_solver='sparse' to dispatch this size to the dense kernel."
        )
    if sigma is None:
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
                tol=tol,
                ncv=ncv,
                maxiter=maxiter,
            )
            return _sorted_lowest(E, V)
        except ArpackNoConvergence as err:
            attempts.append(f'shift-invert(sigma={sigma:.3f}, ncv={ncv}, maxiter={maxiter}): {err}')
            ncv = min(n, ncv0 * (attempt + 2))
            maxiter = 40 * n * (attempt + 1)

    try:
        E, V = eigsh(
            H, k=k, which='SA', v0=v0, tol=tol, ncv=min(n, max(4 * k + 1, 40)), maxiter=100 * n
        )
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


def count_in_window(H: spmatrix, elo: float, ehi: float, dense_n_max: int = DENSE_N_MAX) -> int:
    """Number of eigenvalues of ``H`` in ``[elo, ehi]``.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    elo, ehi : float
        Window bounds in eV, both inclusive.
    dense_n_max : int, optional
        Largest ``n`` for which the ``(n, n)`` scratch is affordable.

    Returns
    -------
    int
        The number of states in the window.

    Raises
    ------
    NotImplementedError
        If ``n`` exceeds ``dense_n_max``.  The message names the way out.

    Notes
    -----
    Companion to :func:`count_below` for sizing an interior window, and the
    reference the sparse path is validated against.  Counting states in an
    interval needs no eigenvectors, so this is one tridiagonal reduction
    followed by a value-subset eigenvalue query — cheap in time, but it
    densifies, so it carries the same size guard as the dense kernel.
    """
    n = H.shape[0]
    if n > dense_n_max:
        raise NotImplementedError(
            f'count_in_window: the interior-window probe densifies H(k), which at n = {n} '
            f'would need {16.0 * n * n / 1024**3:.2f} GB (dense_n_max = {dense_n_max}). Pass an '
            'explicit k0 to solve_interior() to skip the probe.'
        )
    A = H.toarray()
    E = scipy.linalg.eigvalsh(
        A,
        subset_by_value=[float(elo), float(ehi)],
        driver='evr',
        check_finite=False,
        overwrite_a=True,
    )
    return len(E)


def _solve_interior_dense(H: spmatrix, elo: float, ehi: float) -> tuple[np.ndarray, np.ndarray]:
    """Every eigenpair in ``[elo, ehi]`` by ``zheevr``'s ``subset_by_value``.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    elo, ehi : float
        Window bounds in eV, both inclusive.

    Returns
    -------
    (E, V) : tuple of np.ndarray
        Eigenvalues ``(m,)`` ascending and eigenvectors ``(n, m)``, where
        ``m`` is the number of states in the window.

    Notes
    -----
    Exact by construction and orthonormal on degenerate clusters, so it is
    both the small-``n`` branch of :func:`solve_interior` and the A/B
    reference its sparse path is validated against.
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
    tol: float = 0.0,
    hk_solver: str = 'auto',
    dense_n_max: int = DENSE_N_MAX,
    interior_dense_n: int = INTERIOR_DENSE_N,
    max_attempts: int = INTERIOR_MAX_ATTEMPTS,
) -> tuple[np.ndarray, np.ndarray]:
    """All eigenpairs of a Hermitian sparse ``H`` with ``elo <= E <= ehi``.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    elo, ehi : float
        Window bounds in eV, ``ehi > elo``.  Both are inclusive.
    k0 : int or None, optional
        Starting Krylov request.  ARPACK is asked for the ``k0`` eigenvalues
        nearest the window centre; if that set does not straddle *both* edges
        of the window, ``k`` grows and the solve repeats.  A good ``k0`` (say
        the count from a neighbouring k-point) removes the retries; a bad one
        only costs time, never correctness — a failed attempt measures the
        local level density, so the next ``k`` is extrapolated rather than
        merely doubled.
    tol : float, optional
        ARPACK relative tolerance; 0.0 means machine precision.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        ``'dense'`` uses ``subset_by_value`` directly; ``'sparse'`` forces
        shift-invert; ``'auto'`` takes dense below ``interior_dense_n``, where
        the ``splu`` factorization costs more than a direct subset solve, and
        also falls back to dense if the window turns out to hold so much of
        the spectrum that the Krylov request approaches ``n``.  Under
        ``'sparse'`` that situation raises instead — no silent fallback.
    dense_n_max : int, optional
        Largest ``n`` for which an ``(n, n)`` scratch is affordable when the
        dense branch is reached by fallback rather than by request.
    interior_dense_n : int, optional
        Below this ``n``, ``'auto'`` goes straight to the dense branch.
    max_attempts : int, optional
        How many Krylov sizes to try before giving up.

    Returns
    -------
    (E, V) : tuple of np.ndarray
        Eigenvalues ``(m,)`` ascending and eigenvectors ``(n, m)``, where
        ``m`` is the number of states in the window.  ``m`` is **not** known
        in advance and is legitimately 0 when the window lies in a gap; both
        arrays come back correctly shaped in that case.

    Raises
    ------
    ValueError
        If the window is empty or inverted, or ``hk_solver`` is unknown.
    NotImplementedError
        If the dense branch is needed at an ``n`` past ``dense_n_max``.
    RuntimeError
        If the Krylov ladder ran out of attempts, or if a forced sparse
        solve met a window holding nearly the whole spectrum.

    Notes
    -----
    :func:`solve_lowest` answers "the ``nev`` lowest states", which forces
    every property to carry the whole valence manifold even when its physics
    only needs a few :math:`k_BT` around :math:`E_F` — Boltzmann transport,
    Fermi-surface quantities, flat bands in a moire cell.  This function
    answers "every state in ``[elo, ehi]``" instead, so the cost tracks the
    number of states *in* the window rather than the number below it.

    The mechanism is again shift-invert, but with the shift placed at the
    centre of the window: the transformed operator's largest eigenvalues are
    then those of :math:`H` closest to the window centre, which is exactly
    the set wanted.  Two properties make this the regime ARPACK was built
    for.  The count in a narrow interior window is typically a fraction of a
    percent of the spectrum, and — unlike the from-the-bottom case — that
    fraction is set by the material's basis richness rather than by where
    :math:`E_F` sits.  Under N-fold folding both the count and ``n`` scale
    by N, so the fraction is scale-invariant here too: a supercell neither
    helps nor hurts.

    The number of states in the window is not known in advance, so the
    solver asks for ``k`` states nearest the centre and then checks whether
    that set reaches *past both edges*.  If it does, every in-window state
    is inside the returned set and the rest are discarded; if it does not,
    states may lie beyond the last one converged and ``k`` must grow.  The
    growth is an extrapolation rather than a blind doubling: the ``k``
    values just returned span some energy range, so ``k`` divided by that
    span estimates the local level density, and multiplying by the window
    width estimates the ``k`` needed to bracket it.  Blind doubling would
    need ``log2(m/k0)`` rounds to reach a window holding ``m`` states,
    exhausting the attempt budget on a problem that is not actually hard.

    The returned count is k-dependent by construction (a band crossing the
    window edge is in at one k-point and out at the next).  Callers that
    need rectangular arrays must pad; that is deliberately not done here, so
    this function stays a pure eigensolver.

    Degenerate groups are re-orthonormalized exactly as in
    :func:`solve_lowest` — required here, since a folded supercell or a
    moire flat-band manifold puts multiplets inside the window by design.
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

    for _ in range(max_attempts):
        k = min(k, kmax)
        try:
            E, V = eigsh(
                H,
                k=k,
                sigma=sigma,
                which='LM',
                mode='normal',
                tol=tol,
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
        f'{max_attempts} attempts (last k={k}).\nAttempts:\n  {attempt_lines}\n'
        'Either the window holds a large fraction of the spectrum -- in which case use '
        "hk_solver='dense' -- or ARPACK is not converging; raise tol or pass a larger k0."
    )


def count_below(H: spmatrix, ehi: float, dense_n_max: int = DENSE_N_MAX) -> int:
    """Number of eigenvalues of ``H`` at or below ``ehi``.

    Parameters
    ----------
    H : scipy.sparse matrix, shape (n, n), Hermitian
    ehi : float
        Upper bound in eV, inclusive.
    dense_n_max : int, optional
        Largest ``n`` for which the ``(n, n)`` scratch is affordable.

    Returns
    -------
    int
        The number of states at or below ``ehi``.

    Raises
    ------
    NotImplementedError
        If ``n`` exceeds ``dense_n_max``.  The message names the two ways
        out.

    Notes
    -----
    This is the probe used to size an energy window: it answers how many
    bands a from-the-bottom solve must compute to reach a given energy at a
    given k-point.  One tridiagonal reduction with ``zheevr``'s
    ``subset_by_value`` and no eigenvectors — cheap, but it densifies, so it
    carries the same size guard as the ``'dense'`` kernel.
    """
    n = H.shape[0]
    if n > dense_n_max:
        raise NotImplementedError(
            f'count_below: the energy-window probe densifies H(k), which at n = {n} would need '
            f'{16.0 * n * n / 1024**3:.2f} GB (dense_n_max = {dense_n_max}). Pass an explicit '
            'nev to energy_window() to skip the probe, or move to a distributed eigensolver.'
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
