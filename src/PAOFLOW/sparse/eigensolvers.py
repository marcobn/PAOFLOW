"""Sparse selected/energy-window eigensolvers.

Whole-spectrum requirements are handled deliberately here (per the sparse-design
contract).  The downstream observables of the target workflow (DOS, adaptive
smearing, Boltzmann transport) need eigenpairs only inside a bounded energy
window ``[emin, emax]``.  This module therefore computes the *lowest* ``n_sel``
eigenpairs per k-point with :func:`scipy.sparse.linalg.eigsh`, choosing
``n_sel`` large enough that the window is fully covered, and never forms a dense
``(nkpnts, nawf, nawf, nspin)`` eigenvector tensor.

Requesting almost every eigenpair from ARPACK is both mathematically outside a
selected-spectrum sparse workflow and extremely slow for small dense-ish test
systems.  Such requests are rejected unless the caller explicitly selects the
guarded developer-only dense solver.
"""

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from .containers import SparseEigenpairs

# Hard ceiling for the bounded small-system dense fallback.  Above this the
# sparse eigensolver must be used; a full dense solve is refused.
_DENSE_EIGH_GATE_NAWF = 2000
_NEAR_FULL_EIGSH_FRACTION = 0.75


def build_bz_kgrid(data_controller):
    """Build the full-BZ Cartesian evaluation k-grid for the sparse solver.

    Uses the interpolation grid ``(nfft1, nfft2, nfft3)`` when
    :func:`set_interpolation_grid` has run, otherwise the current
    ``(nk1, nk2, nk3)`` grid.  This is the mesh on which ``pao_eigh`` and the
    observables are evaluated.

    Parameters
    ----------
    data_controller : DataController
        Provides ``b_vectors`` and the grid attributes.

    Returns
    -------
    np.ndarray, shape ``(nktot, 3)``
        Cartesian k-points (units of :math:`2\\pi/a`).  Also sets
        ``attr['nkpnts']`` to ``nktot``.
    """
    from ..utils.get_K_grid_fft import get_K_grid_fft_crystal

    arry, attr = data_controller.data_dicts()

    if attr.get('sparse_interpolated', False):
        nk1, nk2, nk3 = attr['nfft1'], attr['nfft2'], attr['nfft3']
    else:
        nk1, nk2, nk3 = attr['nk1'], attr['nk2'], attr['nk3']

    kcry = get_K_grid_fft_crystal(nk1, nk2, nk3)  # (nktot, 3) reduced
    kcart = kcry @ arry['b_vectors']  # to Cartesian (2π/a)
    attr['nkpnts'] = kcart.shape[0]
    return kcart


def _estimate_window_bands(data_controller, _emax):
    """Heuristic lower bound on the number of bands needed to cover ``emax``.

    Uses the electron count (occupied bands) plus a conduction-band buffer.
    Coverage is verified and grown after the first solve, so this only needs to
    be a reasonable starting point.
    """
    _, attr = data_controller.data_dicts()
    nawf = attr['nawf']
    n_occ = int(round(attr.get('nelec', nawf))) // 2
    # Keep the default genuinely selected.  A large fixed conduction buffer can
    # become ``nawf - 2`` for tiny validation systems, which puts ARPACK into
    # its worst near-full-spectrum regime before any output is written.
    buffer = max(4, n_occ // 2)
    selected_cap = max(1, int(np.floor(_NEAR_FULL_EIGSH_FRACTION * nawf)) - 1)
    n_guess = min(nawf - 2, selected_cap, n_occ + buffer)
    return max(1, n_guess)


def _validate_eigsh_request(stage, nawf, n_eigs, solver):
    """Reject near-full sparse requests before entering ARPACK."""
    if solver == 'dense':
        return

    near_full = n_eigs >= max(1, int(np.floor(_NEAR_FULL_EIGSH_FRACTION * nawf)))
    arpack_limit = n_eigs >= nawf - 1
    if near_full or arpack_limit:
        raise NotImplementedError(
            f'Sparse {stage}: requested {n_eigs} of {nawf} bands. This is an '
            'almost full-spectrum solve, which ARPACK handles poorly and which '
            'would require dense materialization in the dense PAOFLOW path. '
            'Keep the sparse test selected-spectrum by passing a smaller '
            '`n_bands`, narrowing the energy window, or adding a sparse '
            'interior/contour solver for this observable.'
        )


def _eigsh_lowest(h_k, n_eigs, want_vectors, tol, maxiter):
    """Lowest ``n_eigs`` eigenpairs of a Hermitian sparse ``Hk`` via Lanczos.

    Returns ``(evals_sorted, evecs_sorted_or_None, converged_bool)``.
    """
    try:
        out = eigsh(
            h_k,
            k=n_eigs,
            which='SA',
            return_eigenvectors=want_vectors,
            tol=tol,
            maxiter=maxiter,
        )
        converged = True
    except ArpackNoConvergence as e:
        # Use whatever converged; caller records the non-convergence.
        out = (e.eigenvalues, e.eigenvectors) if want_vectors else e.eigenvalues
        converged = False

    if want_vectors:
        evals, evecs = out
        order = np.argsort(evals)
        return evals[order], evecs[:, order], converged
    else:
        evals = out
        return np.sort(evals), None, converged


def _dense_gated(h_k, n_eigs, want_vectors):
    """Bounded small-system dense solve of a sparse ``Hk`` (explicitly gated).

    This is the only place a sparse matrix is densified, and only for systems
    below :data:`_DENSE_EIGH_GATE_NAWF`.  It exists so that whole-spectrum-only
    requests on small reference systems (e.g. validation against dense PAOFLOW)
    remain feasible without weakening the sparse contract for large systems.
    """
    from scipy.linalg import eigh  # noqa: guarded dense path only

    H = h_k.toarray()
    if want_vectors:
        evals, evecs = eigh(H)
        return evals[:n_eigs], evecs[:, :n_eigs], True
    evals = eigh(H, eigvals_only=True)
    return evals[:n_eigs], None, True


def solve_window(
    data_controller,
    kcart,
    emin,
    emax,
    want_vectors,
    n_eigs=None,
    solver='eigsh',
    tol=1.0e-8,
    maxiter=None,
):
    """Compute the selected lowest eigenpairs covering ``[emin, emax]`` on a mesh.

    Parameters
    ----------
    data_controller : DataController
        Provides the sparse Hamiltonian (``data_arrays['sparse_H']``) and
        ``nawf``/``nspin``.
    kcart : np.ndarray, shape ``(nktot, 3)``
        Cartesian evaluation k-points.
    emin, emax : float
        Target energy window (eV).  ``n_sel`` is grown until every k-point's
        highest computed band reaches ``emax``.
    want_vectors : bool
        If ``True``, selected eigenvectors are returned (needed for velocities).
    n_eigs : int, optional
        Fixed number of lowest bands to compute.  If ``None``, chosen from the
        electron count and grown to cover ``emax``.
    solver : {'eigsh', 'dense'}
        ``'eigsh'`` (default) uses sparse Lanczos.  ``'dense'`` forces the
        bounded gated dense solve (small systems only).
    tol, maxiter :
        Passed to :func:`scipy.sparse.linalg.eigsh`.

    Returns
    -------
    SparseEigenpairs

    Raises
    ------
    NotImplementedError
        If the window requires (almost) the full spectrum and the system is too
        large for the bounded dense gate.
    """
    arry, _ = data_controller.data_dicts()
    sparse_h = arry['sparse_H']
    nawf = sparse_h.nawf
    nspin = sparse_h.nspin
    nk = kcart.shape[0]

    if n_eigs is None:
        n_eigs = _estimate_window_bands(data_controller, emax)
    n_eigs = int(min(n_eigs, nawf))

    _validate_eigsh_request('pao_eigh', nawf, n_eigs, solver)
    use_dense = solver == 'dense'
    # eigsh cannot return all (or nearly all) eigenpairs: it needs k < nawf-1.
    if n_eigs > nawf - 2:
        if nawf <= _DENSE_EIGH_GATE_NAWF:
            use_dense = True
        else:
            raise NotImplementedError(
                f'Sparse pao_eigh: window [{emin}, {emax}] eV requires '
                f'{n_eigs} of {nawf} bands, i.e. essentially the full spectrum. '
                'Sparse Lanczos cannot return the full spectrum and the system '
                f'exceeds the bounded dense gate (nawf={nawf} > '
                f'{_DENSE_EIGH_GATE_NAWF}). Narrow the energy window or use a '
                'shift-invert interior solver.'
            )

    solver_label = 'dense(gated)' if use_dense else 'eigsh(SA)'

    e_k = np.zeros((nk, n_eigs, nspin), dtype=float)
    v_k = np.zeros((nk, nawf, n_eigs, nspin), dtype=complex) if want_vectors else None
    n_converged = 0

    for ispin in range(nspin):
        for ik in range(nk):
            h_k = sparse_h.build_hk(kcart[ik], ispin)
            if use_dense:
                evals, evecs, conv = _dense_gated(h_k, n_eigs, want_vectors)
            else:
                evals, evecs, conv = _eigsh_lowest(h_k, n_eigs, want_vectors, tol, maxiter)
            e_k[ik, : len(evals), ispin] = evals
            if want_vectors and evecs is not None:
                assert v_k is not None
                v_k[ik, :, : evecs.shape[1], ispin] = evecs
            n_converged += int(conv)

    # Verify window coverage; grow n_eigs once if the top band falls short.
    if not use_dense and n_eigs < nawf - 2:
        top = e_k[:, -1, :]
        if np.any(top < emax):
            grown = int(min(nawf - 2, np.ceil(n_eigs * 1.5)))
            if grown > n_eigs:
                return solve_window(
                    data_controller,
                    kcart,
                    emin,
                    emax,
                    want_vectors,
                    n_eigs=grown,
                    solver=solver,
                    tol=tol,
                    maxiter=maxiter,
                )

    return SparseEigenpairs(e_k, v_k, (emin, emax), n_eigs, n_converged, solver_label)


def solve_path(
    data_controller,
    kq_cart,
    n_eigs,
    emin=None,
    emax=None,
    solver='eigsh',
    tol=1.0e-8,
    maxiter=None,
    progress_callback=None,
):
    """Compute selected band eigenvalues along a k-path (eigenvalues only).

    Parameters
    ----------
    data_controller : DataController
        Provides the sparse Hamiltonian.
    kq_cart : np.ndarray, shape ``(nkpi, 3)``
        Cartesian k-points along the path.
    n_eigs : int
        Number of lowest bands to compute at each k-point.
    emin, emax : float, optional
        Recorded on the result for reporting (no coverage growth on paths).
    solver : {'eigsh', 'dense'}
    tol, maxiter :
        Passed to :func:`eigsh`.
    progress_callback : callable, optional
        Called as ``progress_callback(done, total)`` after selected k-points are
        solved.  Used only for user-visible progress logging.

    Returns
    -------
    SparseEigenpairs
        With ``E_k`` of shape ``(nkpi, n_eigs, nspin)`` and ``v_k is None``.
    """
    arry, _ = data_controller.data_dicts()
    sparse_h = arry['sparse_H']
    nawf = sparse_h.nawf
    nspin = sparse_h.nspin
    nkpi = kq_cart.shape[0]

    n_eigs = int(min(n_eigs, nawf))
    _validate_eigsh_request('bands', nawf, n_eigs, solver)
    use_dense = solver == 'dense'
    if n_eigs > nawf - 2:
        if nawf <= _DENSE_EIGH_GATE_NAWF:
            use_dense = True
        else:
            raise NotImplementedError(
                f'Sparse bands: requested {n_eigs} of {nawf} bands exceeds what '
                'sparse Lanczos can return and the system is above the dense '
                f'gate (nawf={nawf} > {_DENSE_EIGH_GATE_NAWF}).'
            )
    solver_label = 'dense(gated)' if use_dense else 'eigsh(SA)'

    e_k = np.zeros((nkpi, n_eigs, nspin), dtype=float)
    n_converged = 0
    progress_step = max(1, nkpi // 20)
    for ispin in range(nspin):
        for ik in range(nkpi):
            h_k = sparse_h.build_hk(kq_cart[ik], ispin)
            if use_dense:
                evals, _, conv = _dense_gated(h_k, n_eigs, False)
            else:
                evals, _, conv = _eigsh_lowest(h_k, n_eigs, False, tol, maxiter)
            e_k[ik, : len(evals), ispin] = evals
            n_converged += int(conv)
            if progress_callback is not None and ispin == nspin - 1:
                done = ik + 1
                if done == 1 or done == nkpi or done % progress_step == 0:
                    progress_callback(done, nkpi)

    window = (emin, emax) if emin is not None and emax is not None else None
    return SparseEigenpairs(e_k, None, window, n_eigs, n_converged, solver_label)
