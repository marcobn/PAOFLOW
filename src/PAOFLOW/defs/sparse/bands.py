from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from scipy import linalg as dense_linalg
from scipy import sparse
from scipy.sparse import linalg as spla

if TYPE_CHECKING:
    from ...DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# TODO Figure out a better algorithm for selecting candidate states for target='near_fermi' than blindly increasing the number of requested states until enough are found in the window. This can lead to many more candidate states than needed, which increases the cost of the sparse eigensolver. A more efficient approach would be to use a root-finding method to adjust the shift sigma or the number of candidates until just enough states are found in the window.


def _band_options(
    attributes: dict,
) -> tuple[int, bool, str, float | None, float, float | None, dict[str, int]]:
    r"""Read the sparse band-selection controls.

    Parameters
    ----------
    attributes : dict
        PAOFLOW run attributes. The sparse band controls are interpreted as
        mathematical instructions for which part of the spectrum of
        :math:`H(k)` should be computed.

    Returns
    -------
    tuple[int, bool, str, float or None, float, float or None, dict[str, int]]
        Number of requested eigenvalues, whether to return the corresponding
        eigenvectors, the spectral target, the optional target energy, the
        tolerance used by the iterative sparse eigensolver, and the optional
        Fermi-window half-width. The final dictionary contains near-Fermi
        candidate controls.

    Notes
    -----
    Sparse diagonalization computes only a selected part of the spectrum

    .. math::

        H(k) u_n(k) = E_n(k) u_n(k),

    because storing all vectors :math:`u_n(k)` for all :math:`n` is dense in the
    PAO dimension. Supported targets are ``'lowest'``, ``'highest'``,
    ``'near_fermi'``, and ``'near_energy'``.
    """
    nawf = int(attributes['nawf'])
    nbands = int(attributes['sparse_bands_nbands'])
    return_eigenvectors = bool(attributes['sparse_bands_return_eigenvectors'])
    target = str(attributes['sparse_bands_target']).lower()
    tol = float(attributes['sparse_bands_tol'])

    if target not in {'lowest', 'highest', 'near_fermi', 'near_energy'}:
        raise ValueError(
            "sparse_bands_target must be one of 'lowest', 'highest', "
            "'near_fermi', or 'near_energy'."
        )

    if nbands < 1:
        raise ValueError('sparse_bands_nbands must be at least 1.')
    if nbands >= nawf - 1:
        raise ValueError(
            'sparse_bands_nbands must be smaller than nawf - 1 for sparse ARPACK '
            'diagonalization. Requesting the full spectrum is a dense-output problem.'
        )

    fermi_window = None
    sigma = None
    if target == 'near_energy':
        if 'sparse_bands_sigma' not in attributes:
            raise KeyError("sparse_bands_sigma is required when sparse_bands_target='near_energy'.")
        sigma = float(attributes['sparse_bands_sigma'])
    elif target == 'near_fermi':
        sigma = 0.0
        fermi_window = float(attributes.get('sparse_bands_fermi_window', 6.0))
        if fermi_window <= 0.0:
            raise ValueError('sparse_bands_fermi_window must be positive.')

    near_fermi_controls = {
        'initial': int(attributes.get('sparse_bands_near_fermi_initial', 0)),
        'step': int(attributes.get('sparse_bands_near_fermi_step', 0)),
        'max_candidates': int(attributes.get('sparse_bands_near_fermi_max_candidates', 0)),
    }

    return nbands, return_eigenvectors, target, sigma, tol, fermi_window, near_fermi_controls


def _reduce_timing(local_value: float) -> tuple[float, float]:
    """Return MPI sum and max timing values for one local metric."""
    if size == 1:
        return local_value, local_value
    total = comm.reduce(local_value, op=MPI.SUM, root=0)
    maxv = comm.reduce(local_value, op=MPI.MAX, root=0)
    if rank == 0:
        assert total is not None and maxv is not None
        return float(total), float(maxv)
    return 0.0, 0.0


def _select_dense_eigensolution(
    eigvals: np.ndarray,
    eigvecs: np.ndarray | None,
    *,
    nbands: int,
    target: str,
    sigma: float | None,
    fermi_window: float | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Select the requested band window from a dense Hermitian spectrum.

    Parameters
    ----------
    eigvals : numpy.ndarray
        Eigenvalues :math:`E_n(k)` for one Hamiltonian block, ordered or
        unordered.
    eigvecs : numpy.ndarray or None
        Eigenvectors :math:`u_n(k)` stored by column. ``None`` means only band
        energies are requested.
    nbands : int
        Number of bands to keep in the output.
    target : {'lowest', 'highest', 'near_fermi', 'near_energy'}
        Spectral region of interest.
    sigma : float or None
        Reference energy for ``'near_fermi'`` and ``'near_energy'``.
    fermi_window : float or None
        Half-width around ``sigma`` used to require both occupied and empty
        states for ``'near_fermi'``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        Selected eigenvalues and, when available, matching eigenvectors.

    Notes
    -----
    The selection is purely spectral. For ``'near_fermi'`` it returns
    ``nbands // 2`` states below :math:`E_F` and the remaining states above
    :math:`E_F`, matching the sparse ARPACK path.
    """
    eigvals_real = np.real(np.asarray(eigvals))

    if target == 'lowest':
        order = np.argsort(eigvals_real)[:nbands]
    elif target == 'highest':
        order = np.argsort(eigvals_real)[-nbands:]
        order = order[np.argsort(eigvals_real[order])]
    elif target == 'near_energy':
        if sigma is None:
            raise ValueError('sigma must be provided for target=near_energy.')
        order = np.argsort(np.abs(eigvals_real - float(sigma)))[:nbands]
        order = order[np.argsort(eigvals_real[order])]
    elif target == 'near_fermi':
        if sigma is None or fermi_window is None:
            raise ValueError('sigma and fermi_window are required for target=near_fermi.')
        in_window = np.flatnonzero(np.abs(eigvals_real - float(sigma)) <= float(fermi_window))
        n_below = nbands // 2
        n_above = nbands - n_below
        below = in_window[eigvals_real[in_window] < float(sigma)]
        above = in_window[eigvals_real[in_window] >= float(sigma)]
        if below.size < n_below or above.size < n_above:
            raise ValueError(
                f'dense target=near_fermi found {below.size} states below and {above.size} states above '
                f'sigma={float(sigma):.6g} inside '
                f'[{float(sigma) - float(fermi_window):.6g}, {float(sigma) + float(fermi_window):.6g}] eV, but '
                f'sparse_bands_nbands={nbands} requires {n_below} below and {n_above} above. '
                'Increase fermi_window or reduce nbands.'
            )
        selected_below = below[np.argsort(eigvals_real[below])[-n_below:]] if n_below else below[:0]
        selected_above = above[np.argsort(eigvals_real[above])[:n_above]] if n_above else above[:0]
        order = np.concatenate((selected_below, selected_above))
        order = order[np.argsort(eigvals_real[order])]
    else:
        raise ValueError(f'Unsupported dense band target: {target!r}')

    selected_vals = np.asarray(eigvals)[order]
    selected_vecs = None if eigvecs is None else np.asarray(eigvecs)[:, order]
    return selected_vals, selected_vecs


def _solve_block_dense_local(
    h_k: sparse.spmatrix,
    *,
    nbands: int,
    return_eigenvectors: bool,
    target: str,
    sigma: float | None,
    fermi_window: float | None,
    diagnostics: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Solve one Hamiltonian block by intentionally densifying that block only.

    Parameters
    ----------
    h_k : scipy.sparse.spmatrix
        Sparse assembled Hamiltonian block :math:`H(k)` for one k-point and spin.
        This object is converted to one dense matrix locally and then discarded
        by the caller after the eigensolve.
    nbands : int
        Number of band energies :math:`E_n(k)` to return.
    return_eigenvectors : bool
        If ``True``, return the selected eigenvectors :math:`u_n(k)` as columns.
    target : {'lowest', 'highest', 'near_fermi', 'near_energy'}
        Spectral region requested.
    sigma : float or None
        Reference energy. For ``'near_fermi'`` this is the Fermi level
        :math:`E_F`; for ``'near_energy'`` it is the target energy.
    fermi_window : float or None
        Energy half-window around :math:`E_F` for ``'near_fermi'``.
    diagnostics : dict or None, optional
        Mutable per-block diagnostic record updated with dense matrix density,
        conversion time, and LAPACK time.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        Selected eigenvalues and optionally selected eigenvectors.

    Notes
    -----
    This is an explicitly opt-in escape hatch for cases where the Fourier sum

    .. math::

        H(k) = \sum_R e^{2\pi i k\cdot R} H(R)

    is structurally dense even though :math:`H(R)` is sparse. It violates the
    strict sparse-at-all-times rule by calling ``toarray()`` on a single
    assembled :math:`H(k)` block, but it does **not** materialize dense
    Hamiltonians for multiple k-points or a dense real-space tensor.
    """
    h_csr = sparse.csr_matrix(h_k)
    h_csr = 0.5 * (h_csr + h_csr.getH())
    dimension = int(h_csr.shape[0])
    density = float(h_csr.nnz) / float(max(1, dimension * dimension))

    if diagnostics is not None:
        diagnostics.update(
            {
                'backend': 'dense_local',
                'dimension': dimension,
                'nnz': int(h_csr.nnz),
                'density': density,
                'target': str(target),
                'requested_nbands': int(nbands),
                'sigma': None if sigma is None else float(sigma),
            }
        )

    t0 = perf_counter()
    h_dense = h_csr.toarray()
    h_dense = 0.5 * (h_dense + h_dense.conj().T)
    conversion_seconds = perf_counter() - t0

    t0 = perf_counter()
    if return_eigenvectors:
        eigvals, eigvecs = dense_linalg.eigh(
            h_dense,
            overwrite_a=True,
            check_finite=False,
            driver='evr',
        )
    else:
        eigvals = dense_linalg.eigvalsh(
            h_dense,
            overwrite_a=True,
            check_finite=False,
            driver='evr',
        )
        eigvecs = None
    lapack_seconds = perf_counter() - t0

    if diagnostics is not None:
        diagnostics['dense_conversion_seconds'] = float(conversion_seconds)
        diagnostics['dense_lapack_seconds'] = float(lapack_seconds)
        diagnostics['seconds'] = float(conversion_seconds + lapack_seconds)

    return _select_dense_eigensolution(
        eigvals,
        eigvecs,
        nbands=nbands,
        target=target,
        sigma=sigma,
        fermi_window=fermi_window,
    )


def _solve_block_with_backend(
    h_k: sparse.spmatrix,
    *,
    nbands: int,
    return_eigenvectors: bool,
    target: str,
    sigma: float | None,
    tol: float,
    fermi_window: float | None,
    near_fermi_controls: dict[str, int] | None,
    dense_local_enabled: bool,
    dense_density_threshold: float,
    diagnostics: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Choose the sparse ARPACK path or the opt-in local dense backend.

    Parameters
    ----------
    h_k : scipy.sparse.spmatrix
        Assembled Hamiltonian block :math:`H(k)` for one k-point and spin.
    nbands : int
        Number of bands to return.
    return_eigenvectors : bool
        Whether selected eigenvectors are needed.
    target, sigma, tol, fermi_window, near_fermi_controls
        Spectral-selection and solver controls used by the sparse band path.
    dense_local_enabled : bool
        Enables deliberate local densification for blocks whose density exceeds
        ``dense_density_threshold``.
    dense_density_threshold : float
        Minimum structural density :math:`\mathrm{nnz}(H)/N^2` required before
        using the dense backend.
    diagnostics : dict or None, optional
        Per-block diagnostic record.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        Selected eigenvalues and optionally eigenvectors.
    """
    if dense_density_threshold < 0.0 or dense_density_threshold > 1.0:
        raise ValueError('dense_density_threshold must lie in [0, 1].')

    h_csr = sparse.csr_matrix(h_k)
    dimension = int(h_csr.shape[0])
    density = float(h_csr.nnz) / float(max(1, dimension * dimension))
    use_dense = bool(dense_local_enabled) and density >= float(dense_density_threshold)

    if use_dense:
        return _solve_block_dense_local(
            h_csr,
            nbands=nbands,
            return_eigenvectors=return_eigenvectors,
            target=target,
            sigma=sigma,
            fermi_window=fermi_window,
            diagnostics=diagnostics,
        )

    if diagnostics is not None:
        diagnostics['backend'] = 'sparse_arpack'
    return _solve_block(
        h_csr,
        nbands=nbands,
        return_eigenvectors=return_eigenvectors,
        target=target,
        sigma=sigma,
        tol=tol,
        fermi_window=fermi_window,
        near_fermi_controls=near_fermi_controls,
        diagnostics=diagnostics,
    )


def _solve_block(
    h_k: sparse.spmatrix,
    *,
    nbands: int,
    return_eigenvectors: bool,
    target: str,
    sigma: float | None,
    tol: float,
    fermi_window: float | None,
    near_fermi_controls: dict[str, int] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Solve a selected sparse eigenproblem for one k-point and spin.

    Parameters
    ----------
    h_k : scipy.sparse.spmatrix
        Sparse Hamiltonian block :math:`H(k)` in the PAO basis for one spin
        channel.
    nbands : int
        Number of eigenvalues :math:`E_n(k)` to compute.
    return_eigenvectors : bool
        If ``True``, also return the selected wave-function coefficients
        :math:`u_n(k)` in the PAO basis.
    target : {'lowest', 'highest', 'near_fermi', 'near_energy'}
        Spectral region to compute. ``'lowest'`` and ``'highest'`` select by
        real eigenvalue. ``'near_fermi'`` selects valence and conduction states around
        ``sigma`` within a fixed energy window. ``'near_energy'`` selects the
        eigenvalues closest to ``sigma``.
    sigma : float or None
        Target energy for shift-invert selection, used for
        ``'near_fermi'`` and ``'near_energy'``.
    tol : float
        Iterative solver tolerance passed to ARPACK.
    fermi_window : float or None
        Half-width of the accepted energy window around ``sigma`` for
        ``target='near_fermi'``.
    near_fermi_controls : dict or None, optional
        Candidate controls used by ``target='near_fermi'`` with keys
        ``initial``, ``step``, and ``max_candidates``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        Selected eigenvalues and, optionally, the associated eigenvectors. The
        eigenvectors have shape ``(nawf, nbands)``.

    Notes
    -----
    This routine never forms a dense Hamiltonian. It solves

    .. math::

        H(k) u_n(k) = E_n(k) u_n(k)

    with a sparse Hermitian Krylov method. For interior states, ARPACK applies the
    shift-invert operator :math:`(H(k)-\sigma I)^{-1}` using sparse linear
    algebra rather than forming dense inverse matrices.
    """
    h_csr = sparse.csr_matrix(h_k)
    h_csr = 0.5 * (h_csr + h_csr.getH())

    max_candidate_nbands = h_csr.shape[0] - 2
    if max_candidate_nbands < nbands:
        raise ValueError(
            'sparse_bands_nbands must be smaller than the Hamiltonian dimension minus 1 '
            'for sparse ARPACK diagonalization.'
        )

    def run_eigsh(
        *, k: int, which: str, sigma_arg: float | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        kwargs: dict[str, int | str | float] = {'k': int(k), 'which': str(which), 'tol': float(tol)}
        if sigma_arg is not None:
            kwargs['sigma'] = float(sigma_arg)
        if return_eigenvectors:
            eigvals_local, eigvecs_local = spla.eigsh(h_csr, return_eigenvectors=True, **kwargs)
            return np.asarray(eigvals_local), np.asarray(eigvecs_local)
        eigvals_local = spla.eigsh(h_csr, return_eigenvectors=False, **kwargs)
        return np.asarray(eigvals_local), None

    if target == 'lowest':
        eigvals, eigvecs = run_eigsh(k=nbands, which='SA')
    elif target == 'highest':
        eigvals, eigvecs = run_eigsh(k=nbands, which='LA')
    else:
        if sigma is None:
            raise ValueError('sigma must be provided for interior sparse band selection.')

        if target == 'near_fermi':
            if fermi_window is None:
                raise ValueError('fermi_window is required for target=near_fermi.')
            controls = near_fermi_controls or {}
            default_initial = min(max(2 * nbands, nbands + 8), max_candidate_nbands)
            default_step = max(4, nbands // 2)
            default_max = min(max(5 * nbands, nbands + 32), max_candidate_nbands)

            candidate_nbands = int(controls.get('initial', 0))
            growth_step = int(controls.get('step', 0))
            max_candidates = int(controls.get('max_candidates', 0))

            if candidate_nbands < nbands:
                candidate_nbands = default_initial
            if growth_step < 1:
                growth_step = default_step
            if max_candidates < nbands:
                max_candidates = default_max

            candidate_nbands = min(candidate_nbands, max_candidate_nbands)
            max_candidates = min(max_candidates, max_candidate_nbands)
            if max_candidates < candidate_nbands:
                max_candidates = candidate_nbands
        else:
            candidate_nbands = nbands

        while True:
            trial_eigvals, trial_eigvecs = run_eigsh(
                k=candidate_nbands, which='LM', sigma_arg=sigma
            )
            if target != 'near_fermi':
                eigvals = trial_eigvals
                eigvecs = trial_eigvecs
                break

            in_window = np.abs(np.real(trial_eigvals) - sigma) <= fermi_window
            window_eigs = np.real(trial_eigvals)[in_window]
            n_below = nbands // 2
            n_above = nbands - n_below
            below_count = int(np.count_nonzero(window_eigs < sigma))
            above_count = int(np.count_nonzero(window_eigs >= sigma))
            if (
                below_count >= n_below and above_count >= n_above
            ) or candidate_nbands >= max_candidate_nbands:
                eigvals = trial_eigvals
                eigvecs = trial_eigvecs
                break

            next_candidate_nbands = min(candidate_nbands + growth_step, max_candidates)
            if next_candidate_nbands == candidate_nbands:
                eigvals = trial_eigvals
                eigvecs = trial_eigvecs
                break
            candidate_nbands = next_candidate_nbands

    if target == 'near_fermi':
        if fermi_window is None or sigma is None:
            raise ValueError('fermi_window and sigma are required for target=near_fermi.')
        in_window = np.flatnonzero(np.abs(np.real(eigvals) - sigma) <= fermi_window)
        n_below = nbands // 2
        n_above = nbands - n_below

        eigvals_real = np.real(eigvals)
        below = in_window[eigvals_real[in_window] < sigma]
        above = in_window[eigvals_real[in_window] >= sigma]

        if below.size < n_below or above.size < n_above:
            raise ValueError(
                f'target=near_fermi found {below.size} states below and {above.size} states above '
                f'sigma={sigma:.6g} inside '
                f'[{sigma - fermi_window:.6g}, {sigma + fermi_window:.6g}] eV after '
                f'requesting {candidate_nbands} candidate states, but '
                f'sparse_bands_nbands={nbands} requires {n_below} below and {n_above} above. '
                'Increase fermi_window or reduce nbands.'
            )

        selected_below = below[np.argsort(eigvals_real[below])[-n_below:]] if n_below else below[:0]
        selected_above = above[np.argsort(eigvals_real[above])[:n_above]] if n_above else above[:0]
        selected = np.concatenate((selected_below, selected_above))
        order = selected[np.argsort(eigvals_real[selected])]
    else:
        order = np.argsort(np.real(eigvals))[:nbands]

    if return_eigenvectors:
        assert eigvecs is not None
        return eigvals[order], eigvecs[:, order]

    return eigvals[order], None


def _normalise_kpath(kgrid: np.ndarray) -> np.ndarray:
    r"""Return a band-path k-grid in ``(nkpnts, 3)`` form.

    Parameters
    ----------
    kgrid : numpy.ndarray
        Band-path coordinates generated by PAOFLOW. The dense interpolation
        helper stores these as ``arrays['kq']`` with shape ``(3, nkpnts)``,
        while sparse Fourier assembly is simpler with one row per k-point.

    Returns
    -------
    numpy.ndarray
        K-point coordinates with shape ``(nkpnts, 3)``. Each row is the
        reciprocal-space coordinate entering
        :math:`\exp[2\pi i k\cdot R]` after conversion to reciprocal Cartesian coordinates.
    """
    if kgrid.ndim != 2:
        raise ValueError('k-path array must be rank 2.')
    if kgrid.shape[0] == 3:
        return np.asarray(kgrid.T, dtype=float)
    if kgrid.shape[1] == 3:
        return np.asarray(kgrid, dtype=float)
    raise ValueError('k-path array must have one dimension of length 3.')


def _coerce_hrs_container(sparse_hrs_obj: object, attributes: dict):
    r"""Return sparse real-space Hamiltonian blocks as a ``SparseHRs`` object.

    Parameters
    ----------
    sparse_hrs_obj : object
        Stored sparse real-space Hamiltonian container. Supported forms are a
        ``SparseHRs`` instance, a mapping keyed by ``(ir, ispin)``, or a mapping
        keyed by ``(i, j, k, ispin)``.
    attributes : dict
        Runtime attributes providing ``nk1``, ``nk2``, ``nk3``, ``nspin``, and
        ``nawf`` needed to validate and normalize block indexing.

    Returns
    -------
    SparseHRs
        Normalized sparse real-space Hamiltonian container.

    Notes
    -----
    Sparse doubling may store ``arrays['SparseHRs']`` as a plain dictionary of
    sparse blocks keyed by explicit grid coordinates. Interpolated sparse bands
    assemble :math:`H(k)` through ``SparseHRs._assemble_weighted_block``,
    which expects flattened real-space indexing. This helper converts supported
    sparse mappings into the canonical ``SparseHRs`` representation without any
    dense reconstruction.
    """
    from .get_hr import SparseHRs

    if isinstance(sparse_hrs_obj, SparseHRs):
        return sparse_hrs_obj

    if not isinstance(sparse_hrs_obj, Mapping):
        raise TypeError('SparseHRs must be a SparseHRs object or a mapping of sparse blocks.')

    nk1 = int(attributes['nk1'])
    nk2 = int(attributes['nk2'])
    nk3 = int(attributes['nk3'])
    nspin = int(attributes['nspin'])
    nawf = int(attributes['nawf'])

    nrtot = nk1 * nk2 * nk3
    normalized_blocks: dict[tuple[int, int], sparse.csr_matrix] = {}

    for key, block in sparse_hrs_obj.items():
        if not sparse.issparse(block):
            raise TypeError(f'SparseHRs block at key {key!r} is not a scipy sparse matrix.')

        if not isinstance(key, tuple):
            raise TypeError(f'Unsupported SparseHRs key type {type(key)!r}; expected tuple.')

        if len(key) == 2:
            ir, ispin = (int(v) for v in key)
        elif len(key) == 4:
            i, j, k, ispin = (int(v) for v in key)
            if i < 0 or i >= nk1 or j < 0 or j >= nk2 or k < 0 or k >= nk3:
                raise ValueError(f'SparseHRs grid index out of range for key {key!r}.')
            ir = k + j * nk3 + i * nk2 * nk3
        else:
            raise TypeError(
                f'Unsupported SparseHRs key shape {key!r}; expected (ir, ispin) or (i, j, k, ispin).'
            )

        if ir < 0 or ir >= nrtot:
            raise ValueError(f'SparseHRs real-space index ir={ir} out of range [0, {nrtot}).')
        if ispin < 0 or ispin >= nspin:
            raise ValueError(f'SparseHRs spin index ispin={ispin} out of range [0, {nspin}).')

        csr_block = block.tocsr()
        if csr_block.shape != (nawf, nawf):
            raise ValueError(
                f'SparseHRs block at key {key!r} has shape {csr_block.shape}, expected ({nawf}, {nawf}).'
            )
        normalized_blocks[(ir, ispin)] = csr_block

    return SparseHRs(
        nawf=nawf,
        nk1=nk1,
        nk2=nk2,
        nk3=nk3,
        nspin=nspin,
        blocks=normalized_blocks,
    )


def _print_solver_backend_summary(records: list[dict[str, object]], *, context: str) -> None:
    r"""Print a compact summary of sparse and local dense band solvers.

    Parameters
    ----------
    records : list of dict
        Per-k-point eigensolver diagnostics. Each record represents one spin
        block Hamiltonian :math:`H_s(k)`.
    context : str
        Label describing whether the blocks came from the interpolated path or
        the original NSCF mesh.

    Returns
    -------
    None
        Writes a rank-0 diagnostic summary.
    """
    if not records:
        return

    dense_records = [item for item in records if item.get('backend') == 'dense_local']
    sparse_records = [item for item in records if item.get('backend') != 'dense_local']
    densities = np.asarray([float(item.get('density', 0.0)) for item in records], dtype=float)
    total_times = np.asarray(
        [float(item.get('total_solver_seconds', item.get('seconds', 0.0))) for item in records],
        dtype=float,
    )
    print(
        f'Sparse bands backend summary ({context}): '
        f'blocks={len(records)}, dense-local={len(dense_records)}, sparse-arpack={len(sparse_records)}, '
        f'density avg/max={float(np.mean(densities)):.3e}/{float(np.max(densities)):.3e}, '
        f'solver seconds avg/max={float(np.mean(total_times)):.3f}/{float(np.max(total_times)):.3f}'
    )

    if dense_records:
        dense_times = np.asarray(
            [
                float(item.get('total_solver_seconds', item.get('seconds', 0.0)))
                for item in dense_records
            ],
            dtype=float,
        )
        conversion_times = np.asarray(
            [float(item.get('dense_conversion_seconds', 0.0)) for item in dense_records],
            dtype=float,
        )
        lapack_times = np.asarray(
            [float(item.get('dense_lapack_seconds', 0.0)) for item in dense_records],
            dtype=float,
        )
        print(
            '  dense-local timing: '
            f'conversion avg/max={float(np.mean(conversion_times)):.3f}/{float(np.max(conversion_times)):.3f}s, '
            f'LAPACK avg/max={float(np.mean(lapack_times)):.3f}/{float(np.max(lapack_times)):.3f}s, '
            f'total avg/max={float(np.mean(dense_times)):.3f}/{float(np.max(dense_times)):.3f}s'
        )

    slowest = sorted(
        records,
        key=lambda item: float(item.get('total_solver_seconds', item.get('seconds', 0.0))),
        reverse=True,
    )[:5]
    for item in slowest:
        print(
            '  slow eigensolve: '
            f'backend={item.get("backend", "sparse_arpack")}, '
            f'block={item.get("block_idx")}, ik={item.get("ik")}, spin={item.get("ispin")}, '
            f'seconds={float(item.get("total_solver_seconds", item.get("seconds", 0.0))):.3f}, '
            f'nnz={int(item.get("nnz", 0))}, density={float(item.get("density", 0.0)):.3e}'
        )


def bands_calc_interpolated(
    data_controller: DataController,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Diagonalize selected bands on an interpolated sparse k-path.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding ``arrays['SparseHRs']`` and the interpolated
        path ``arrays['kq']``. The real-space Hamiltonian blocks represent
        :math:`H(R)` on the FFT real-space grid.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        On rank 0, selected eigenvalues with shape
        ``(nkpnts, sparse_bands_nbands, nspin)``. If requested, selected
        eigenvectors have shape ``(nkpnts, nawf, sparse_bands_nbands, nspin)``.
        Non-root ranks return empty arrays with matching trailing dimensions,
        or ``None`` for eigenvectors.

    Notes
    -----
    Each path Hamiltonian is assembled directly from sparse real-space blocks,

    .. math::

        H(k) = \sum_R e^{2\pi i k\cdot R} H(R),

    using ``SparseHRs._assemble_weighted_block``. The routine does not
    build ``HRs``, ``Hksp``, dense Fourier matrices, or a stored collection of
    interpolated ``H(k)`` blocks. One sparse :math:`H(k)` block is built,
    diagonalized, and discarded.
    """
    from .communication import load_balancing_kpoints

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    if 'SparseHRs' not in arrays:
        raise KeyError('SparseHRs')
    if 'kq' not in arrays:
        raise KeyError('kq')

    sparse_hrs = _coerce_hrs_container(arrays['SparseHRs'], attributes)
    arrays['SparseHRs'] = sparse_hrs

    kpoints = _normalise_kpath(np.asarray(arrays['kq'], dtype=float))
    if 'b_vectors' not in arrays:
        raise KeyError('b_vectors')
    kpoints = kpoints @ np.asarray(arrays['b_vectors'], dtype=float)
    nkpnts = int(kpoints.shape[0])
    attributes['nkpnts'] = nkpnts

    nspin = int(attributes['nspin'])
    nawf = int(attributes['nawf'])
    (
        nbands,
        return_eigenvectors,
        target,
        sigma,
        tol,
        fermi_window,
        near_fermi_controls,
    ) = _band_options(attributes)
    threshold = float(attributes.get('sparse_threshold', 0.0))
    dense_local_enabled = bool(attributes.get('sparse_bands_dense_local', False))
    dense_density_threshold = float(attributes.get('sparse_bands_dense_density_threshold', 0.5))
    a_vectors = np.asarray(arrays['a_vectors'], dtype=float)
    r_cart = sparse_hrs.compute_R_cart(a_vectors)
    coo_caches = {ispin: sparse_hrs.collect_spin_coo_cache(ispin) for ispin in range(nspin)}

    if sparse_hrs.nspin != nspin:
        raise ValueError(
            f'SparseHRs.nspin={sparse_hrs.nspin} does not match attributes nspin={nspin}.'
        )
    if sparse_hrs.nawf != nawf:
        raise ValueError(f'SparseHRs.nawf={sparse_hrs.nawf} does not match attributes nawf={nawf}.')

    start_block, stop_block = load_balancing_kpoints(nkpnts, nspin, rank, size)

    local_assembly_time = 0.0
    local_eigh_time = 0.0
    local_payload: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    local_solver_diagnostics: list[dict[str, object]] = []
    for block_idx in range(int(start_block), int(stop_block)):
        ik = int(block_idx) // nspin
        ispin = int(block_idx) % nspin
        weights = np.exp(2.0j * np.pi * np.dot(r_cart, kpoints[ik]))

        t0 = perf_counter()
        h_k = sparse_hrs._assemble_weighted_block(
            weights=weights,
            ispin=ispin,
            threshold=threshold,
            coo_cache=coo_caches[ispin],
        )
        local_assembly_time += perf_counter() - t0

        solver_diagnostics: dict[str, object] = {
            'block_idx': int(block_idx),
            'ik': int(ik),
            'ispin': int(ispin),
        }
        t0 = perf_counter()
        eigvals, eigvecs = _solve_block_with_backend(
            h_k,
            nbands=nbands,
            return_eigenvectors=return_eigenvectors,
            target=target,
            sigma=sigma,
            tol=tol,
            fermi_window=fermi_window,
            near_fermi_controls=near_fermi_controls,
            dense_local_enabled=dense_local_enabled,
            dense_density_threshold=dense_density_threshold,
            diagnostics=solver_diagnostics,
        )
        elapsed_eigh = perf_counter() - t0
        solver_diagnostics['total_solver_seconds'] = float(elapsed_eigh)
        local_solver_diagnostics.append(solver_diagnostics)
        local_eigh_time += elapsed_eigh
        local_payload.append((int(block_idx), eigvals, eigvecs))

    assembly_sum, assembly_max = _reduce_timing(local_assembly_time)
    eigh_sum, eigh_max = _reduce_timing(local_eigh_time)
    if rank == 0:
        attributes['sparse_bands_timing'] = {
            'interpolated': True,
            'assembly_seconds_sum': float(assembly_sum),
            'assembly_seconds_max_rank': float(assembly_max),
            'eigh_seconds_sum': float(eigh_sum),
            'eigh_seconds_max_rank': float(eigh_max),
            'timing_mode': 'mpi_sum_and_max',
        }
        if bool(attributes.get('sparse_bands_profile_timing', False)) or bool(
            attributes.get('verbose', False)
        ):
            print(
                'Sparse bands timing (interpolated): '
                f'assembly max-rank={assembly_max:.3f}s, '
                f'eigsolve max-rank={eigh_max:.3f}s'
            )

    if size == 1:
        gathered_solver_diagnostics = [local_solver_diagnostics]
    else:
        gathered_solver_diagnostics = comm.gather(local_solver_diagnostics, root=0)

    if rank == 0 and (
        bool(attributes.get('sparse_bands_profile_timing', False))
        or bool(attributes.get('verbose', False))
    ):
        assert gathered_solver_diagnostics is not None
        flat_solver_diagnostics = [
            item for rank_items in gathered_solver_diagnostics for item in rank_items
        ]
        _print_solver_backend_summary(flat_solver_diagnostics, context='interpolated')

    if size == 1:
        gathered_payload = [local_payload]
    else:
        gathered_payload = comm.gather(local_payload, root=0)

    if rank == 0:
        assert gathered_payload is not None
        all_imaginary_parts = [
            np.max(np.abs(np.imag(eigvals)))
            for rank_payload in gathered_payload
            for _, eigvals, _ in rank_payload
            if eigvals.size
        ]
        max_imaginary = max(all_imaginary_parts, default=0.0)
        real_tol = float(attributes['sparse_bands_real_tol'])
        eig_dtype = float if max_imaginary <= real_tol else complex

        e_k = np.empty((nkpnts, nbands, nspin), dtype=eig_dtype)
        v_k = (
            np.empty((nkpnts, nawf, nbands, nspin), dtype=complex) if return_eigenvectors else None
        )

        for rank_payload in gathered_payload:
            for block_idx, eigvals, eigvecs in rank_payload:
                ik = int(block_idx) // nspin
                ispin = int(block_idx) % nspin
                e_k[ik, :, ispin] = np.real(eigvals) if eig_dtype is float else eigvals
                if return_eigenvectors:
                    if eigvecs is None:
                        raise RuntimeError(
                            'Sparse eigensolver did not return requested eigenvectors.'
                        )
                    assert v_k is not None
                    v_k[ik, :, :, ispin] = eigvecs

        return e_k, v_k

    empty_e = np.empty((0, nbands, nspin), dtype=float)
    empty_v = np.empty((0, nawf, nbands, nspin), dtype=complex) if return_eigenvectors else None
    return empty_e, empty_v


def bands_calc(
    data_controller: DataController,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Diagonalize selected sparse Hamiltonian blocks on the NSCF k-mesh.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding the distributed sparse blocks
        ``Hks_sparse[ik * nspin + ispin]`` for the active NSCF mesh.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        On rank 0, returns the selected eigenvalues ``E_k(k, n, s)`` with shape
        ``(nkpnts, sparse_bands_nbands, nspin)``. If requested, also returns the
        selected eigenvectors ``v_k(k, i, n, s)`` with shape
        ``(nkpnts, nawf, sparse_bands_nbands, nspin)``. Other ranks return empty
        arrays with matching trailing dimensions, or ``None`` for eigenvectors.

    Notes
    -----
    For each k-point and spin channel, this routine solves only the selected
    part of

    .. math::

        H(k) u_n(k) = E_n(k) u_n(k).

    This is not a dense-equivalent full diagonalization. A full set of
    eigenvectors has dense memory scaling in the PAO basis and is intentionally
    rejected by the sparse workflow.
    """
    from .communication import load_balancing_kpoints

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    hks_sparse = arrays['Hks_sparse']

    if not isinstance(hks_sparse, dict):
        raise TypeError(
            'Hks_sparse must be dict[int, scipy.sparse.spmatrix] in sparse bands paths.'
        )

    nkpnts = int(attributes['nkpnts'])
    nspin = int(attributes['nspin'])
    nawf = int(attributes['nawf'])
    (
        nbands,
        return_eigenvectors,
        target,
        sigma,
        tol,
        fermi_window,
        near_fermi_controls,
    ) = _band_options(attributes)

    dense_local_enabled = bool(attributes.get('sparse_bands_dense_local', False))
    dense_density_threshold = float(attributes.get('sparse_bands_dense_density_threshold', 0.5))

    if 'sparse_kpoint_block_range' in attributes:
        start_block, stop_block = attributes['sparse_kpoint_block_range']
    else:
        start_block, stop_block = load_balancing_kpoints(nkpnts, nspin, rank, size)

    local_eigh_time = 0.0
    local_payload: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    local_solver_diagnostics: list[dict[str, object]] = []
    for block_idx in range(int(start_block), int(stop_block)):
        if block_idx not in hks_sparse:
            continue
        h_k = hks_sparse[block_idx]
        if not sparse.issparse(h_k):
            raise TypeError(f'Hks_sparse[{block_idx}] must be a scipy sparse matrix.')

        solver_diagnostics: dict[str, object] = {
            'block_idx': int(block_idx),
            'ik': int(block_idx) // nspin,
            'ispin': int(block_idx) % nspin,
        }
        t0 = perf_counter()
        eigvals, eigvecs = _solve_block_with_backend(
            h_k,
            nbands=nbands,
            return_eigenvectors=return_eigenvectors,
            target=target,
            sigma=sigma,
            tol=tol,
            fermi_window=fermi_window,
            near_fermi_controls=near_fermi_controls,
            dense_local_enabled=dense_local_enabled,
            dense_density_threshold=dense_density_threshold,
            diagnostics=solver_diagnostics,
        )
        elapsed_eigh = perf_counter() - t0
        solver_diagnostics['total_solver_seconds'] = float(elapsed_eigh)
        local_solver_diagnostics.append(solver_diagnostics)
        local_eigh_time += elapsed_eigh
        local_payload.append((int(block_idx), eigvals, eigvecs))

    eigh_sum, eigh_max = _reduce_timing(local_eigh_time)
    if rank == 0:
        attributes['sparse_bands_timing'] = {
            'interpolated': False,
            'assembly_seconds_sum': 0.0,
            'assembly_seconds_max_rank': 0.0,
            'eigh_seconds_sum': float(eigh_sum),
            'eigh_seconds_max_rank': float(eigh_max),
            'timing_mode': 'mpi_sum_and_max',
        }
        if bool(attributes.get('sparse_bands_profile_timing', False)) or bool(
            attributes.get('verbose', False)
        ):
            print(f'Sparse bands timing (nscf): eigsolve max-rank={eigh_max:.3f}s')

    if size == 1:
        gathered_solver_diagnostics = [local_solver_diagnostics]
    else:
        gathered_solver_diagnostics = comm.gather(local_solver_diagnostics, root=0)

    if rank == 0 and (
        bool(attributes.get('sparse_bands_profile_timing', False))
        or bool(attributes.get('verbose', False))
    ):
        assert gathered_solver_diagnostics is not None
        flat_solver_diagnostics = [
            item for rank_items in gathered_solver_diagnostics for item in rank_items
        ]
        _print_solver_backend_summary(flat_solver_diagnostics, context='nscf')

    if size == 1:
        gathered_payload = [local_payload]
    else:
        gathered_payload = comm.gather(local_payload, root=0)

    if rank == 0:
        assert gathered_payload is not None
        all_imaginary_parts = [
            np.max(np.abs(np.imag(eigvals)))
            for rank_payload in gathered_payload
            for _, eigvals, _ in rank_payload
            if eigvals.size
        ]
        max_imaginary = max(all_imaginary_parts, default=0.0)
        real_tol = float(attributes['sparse_bands_real_tol'])
        eig_dtype = float if max_imaginary <= real_tol else complex

        e_k = np.empty((nkpnts, nbands, nspin), dtype=eig_dtype)
        v_k = (
            np.empty((nkpnts, nawf, nbands, nspin), dtype=complex) if return_eigenvectors else None
        )

        for rank_payload in gathered_payload:
            for block_idx, eigvals, eigvecs in rank_payload:
                ik = int(block_idx) // nspin
                ispin = int(block_idx) % nspin
                e_k[ik, :, ispin] = np.real(eigvals) if eig_dtype is float else eigvals
                if return_eigenvectors:
                    if eigvecs is None:
                        raise RuntimeError(
                            'Sparse eigensolver did not return requested eigenvectors.'
                        )
                    assert v_k is not None
                    v_k[ik, :, :, ispin] = eigvecs

        return e_k, v_k

    empty_e = np.empty((0, nbands, nspin), dtype=float)
    empty_v = np.empty((0, nawf, nbands, nspin), dtype=complex) if return_eigenvectors else None
    return empty_e, empty_v


def do_bands(data_controller: DataController) -> None:
    r"""Compute selected sparse band energies for the active NSCF mesh.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding sparse ``H(k)`` blocks on the NSCF mesh.

    Returns
    -------
    None
        Stores the selected band energies in ``arrays['E_k']`` and, when
        requested, the selected eigenvectors in ``arrays['v_k']``.

    Notes
    -----
    Sparse mode supports direct diagonalization of stored NSCF Hamiltonian
    blocks and interpolated band paths from sparse real-space blocks. For
    interpolation, each path Hamiltonian is assembled as

    .. math::

        H(k) = \sum_R e^{2\pi i k\cdot R} H(R),

    then immediately diagonalized with the selected sparse eigensolver. The
    routine does not materialize dense ``HRs`` or ``Hksp`` tensors.
    """
    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    use_interpolation = (
        'ibrav' in attributes or 'band_path' in attributes or 'high_sym_points' in arrays
    )

    if use_interpolation:
        if 'SparseHRs' not in arrays:
            raise KeyError('SparseHRs')
        from PAOFLOW.defs.kpnts_interpolation_mesh import kpnts_interpolation_mesh

        kpnts_interpolation_mesh(data_controller)
    elif 'Hks_sparse' not in arrays:
        raise KeyError('Hks_sparse')

    if use_interpolation:
        e_k, v_k = bands_calc_interpolated(data_controller)
    else:
        e_k, v_k = bands_calc(data_controller)
    arrays['E_k'] = e_k
    if v_k is not None:
        arrays['v_k'] = v_k
    else:
        arrays.pop('v_k', None)
