from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh

if TYPE_CHECKING:
    from ..DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _validate_sparse_block(block: csr_matrix, *, name: str) -> csr_matrix:
    r"""Return one Hamiltonian block in CSR form without densifying it.

    Parameters
    ----------
    block : scipy.sparse.csr_matrix
        Sparse Hamiltonian matrix for one k-point and one spin channel.  The
        matrix represents :math:`H_{ij}(k)`, where ``i`` and ``j`` label PAO
        basis functions.
    name : str
        Human-readable block label used in error messages.

    Returns
    -------
    scipy.sparse.csr_matrix
        The same Hamiltonian represented in compressed sparse row format.

    Notes
    -----
    The eigensolver acts on the sparse matrix equation

    .. math::

        H(k) u_n(k) = E_n(k) u_n(k).

    No dense matrix with shape :math:`N_{\mathrm{PAO}} \times
    N_{\mathrm{PAO}}` is formed here.
    """
    if not issparse(block):
        raise TypeError(f'{name} must be a scipy sparse matrix.')

    block = block.tocsr()
    nrow, ncol = block.shape
    if int(nrow) != int(ncol):
        raise ValueError(f'{name} must be square, got shape {block.shape}.')

    return block


def _target_to_eigsh_parameters(
    target: str,
    sigma: float | None,
) -> tuple[str, float | None]:
    r"""Map the requested spectral region onto ARPACK parameters.

    Parameters
    ----------
    target : {'lowest', 'highest', 'near_fermi', 'near_energy'}
        Spectral region requested from :math:`H(k)`.  ``near_fermi`` assumes the
        PAOFLOW convention that the Fermi level has already been shifted to
        :math:`E_F = 0`.
    sigma : float or None
        Target energy for ``near_energy``.  It is ignored for ``lowest`` and
        ``highest``.

    Returns
    -------
    tuple[str, float or None]
        ``which`` and ``sigma`` values passed to ``scipy.sparse.linalg.eigsh``.

    Notes
    -----
    For edge states of the spectrum, ARPACK solves the sparse Hermitian problem
    directly.  For interior states near an energy :math:`\sigma`, ARPACK uses
    shift-invert mode and seeks eigenvalues of :math:`(H - \sigma I)^{-1}`.
    The factorization is sparse, but may still have fill-in.
    """
    target = str(target).lower()

    if target == 'lowest':
        return 'SA', None
    if target == 'highest':
        return 'LA', None
    if target == 'near_fermi':
        return 'LM', 0.0
    if target == 'near_energy':
        if sigma is None:
            raise ValueError("target='near_energy' requires sigma.")
        return 'LM', float(sigma)

    raise ValueError(
        "Sparse pao_eigh target must be one of 'lowest', 'highest', 'near_fermi', or 'near_energy'."
    )


def _selected_sparse_eigh(
    block: csr_matrix,
    *,
    nbands: int,
    target: str,
    sigma: float | None,
    tol: float,
    return_eigenvectors: bool,
    real_tol: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Solve a selected sparse Hermitian eigenproblem.

    Parameters
    ----------
    block : scipy.sparse.csr_matrix
        Sparse Hamiltonian :math:`H(k)` for one k-point and spin channel.
    nbands : int
        Number of selected eigenvalues :math:`E_n(k)` requested.
    target : {'lowest', 'highest', 'near_fermi', 'near_energy'}
        Spectral region to compute.
    sigma : float or None
        Target energy used when ``target='near_energy'``.
    tol : float
        ARPACK convergence tolerance.
    return_eigenvectors : bool
        If ``True``, return the selected eigenvectors :math:`u_n(k)`.  If
        ``False``, only eigenvalues are returned.
    real_tol : float
        Maximum allowed absolute imaginary part of returned eigenvalues before
        raising an error.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        Selected eigenvalues with shape ``(nbands,)`` and, when requested,
        selected eigenvectors with shape ``(nawf, nbands)``.

    Notes
    -----
    The method computes only a small subspace of the PAO Hamiltonian,

    .. math::

        H(k) u_n(k) = E_n(k) u_n(k), \qquad n \in \mathcal{S},

    where :math:`\mathcal{S}` is the selected band set.  It does not attempt to
    reconstruct the complete dense eigensystem.
    """
    block = _validate_sparse_block(block, name='H(k) block')
    n = int(block.shape[0])
    nbands = int(nbands)

    if nbands < 1:
        raise ValueError('nbands must be at least 1 for sparse pao_eigh.')
    if nbands >= n:
        raise ValueError(
            f'Sparse selected-spectrum pao_eigh requires nbands < nawf; got nbands={nbands}, '
            f'nawf={n}. Full-spectrum diagonalization is intentionally not provided here.'
        )
    if nbands >= n - 1:
        raise ValueError(
            f'Sparse selected-spectrum pao_eigh requires nbands < nawf - 1 for ARPACK; '
            f'got nbands={nbands}, nawf={n}.'
        )

    which, mapped_sigma = _target_to_eigsh_parameters(target, sigma)

    if return_eigenvectors:
        eigvals, eigvecs = eigsh(
            block,
            k=nbands,
            which=which,
            sigma=mapped_sigma,
            tol=float(tol),
            return_eigenvectors=True,
        )
    else:
        eigvals = eigsh(
            block,
            k=nbands,
            which=which,
            sigma=mapped_sigma,
            tol=float(tol),
            return_eigenvectors=False,
        )
        eigvecs = None

    imag_part = np.max(np.abs(np.imag(eigvals))) if eigvals.size else 0.0
    if imag_part > float(real_tol):
        raise ValueError(
            f'Sparse eigensolver returned eigenvalues with imaginary parts up to {imag_part}.'
        )

    eigvals = np.real(eigvals)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]

    if not return_eigenvectors:
        return eigvals, None

    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def do_pao_eigh(data_controller: DataController) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Diagonalize selected sparse Hamiltonian states on the stored k mesh.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with sparse ``H(k)`` blocks and metadata for the
        current PAO basis, k-point grid, and spin channels.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        ``(E_k_sparse, v_k_sparse)`` for the local k-point window owned by the
        current rank.  ``E_k_sparse`` has shape ``(nkpnts_local, nbands,
        nspin)``.  ``v_k_sparse`` has shape ``(nkpnts_local, nawf, nbands,
        nspin)`` only when selected eigenvectors are explicitly requested;
        otherwise it is ``None``.

    Notes
    -----
    The physical problem solved at each k-point is

    .. math::

        H(k) u_n(k) = E_n(k) u_n(k),

    but only a selected band set :math:`n \in \mathcal{S}` is computed.  This
    selected-spectrum contract is deliberately different from dense PAOFLOW's
    full ``E_k``/``v_k`` contract.  Sparse results are therefore returned as
    ``E_k_sparse`` and optionally ``v_k_sparse`` by the sparse frontend.
    """

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    if 'Hks_sparse' not in arrays:
        raise KeyError('Hks_sparse')

    hks_sparse = arrays['Hks_sparse']
    nspin = int(attributes['nspin'])
    nawf = int(attributes['nawf'])

    if not isinstance(hks_sparse, dict):
        raise TypeError('Hks_sparse must be dict[int, csr_matrix] in sparse eigensolver paths.')

    target = str(attributes.get('sparse_eigh_target', 'near_fermi')).lower()
    nbands = int(
        attributes.get('sparse_eigh_nbands', min(max(1, int(attributes['bnd'])), nawf - 2))
    )
    return_eigenvectors = bool(attributes.get('sparse_eigh_return_eigenvectors', False))
    sigma = attributes.get('sparse_eigh_sigma', None)
    tol = float(attributes.get('sparse_eigh_tol', 0.0))
    real_tol = float(attributes.get('sparse_eigh_real_tol', 1.0e-10))

    start_block, stop_block = attributes['sparse_kpoint_block_range']
    start_block = int(start_block)
    stop_block = int(stop_block)

    start_kpoint = start_block // nspin
    stop_kpoint = stop_block // nspin
    local_nkpnts = int(stop_kpoint - start_kpoint)

    e_k_local = np.empty((local_nkpnts, nbands, nspin), dtype=float)
    v_k_local = None
    if return_eigenvectors:
        v_k_local = np.empty((local_nkpnts, nawf, nbands, nspin), dtype=complex)

    e_k_local.fill(np.nan)
    if v_k_local is not None:
        v_k_local.fill(np.nan + 1j * np.nan)

    for block_idx in range(start_block, stop_block):
        block = hks_sparse.get(block_idx)
        if block is None:
            raise KeyError(f'Missing Hks_sparse block {block_idx}.')

        eigvals, eigvecs = _selected_sparse_eigh(
            block,
            nbands=nbands,
            target=target,
            sigma=sigma,
            tol=tol,
            return_eigenvectors=return_eigenvectors,
            real_tol=real_tol,
        )
        ik = block_idx // nspin
        ispin = block_idx % nspin
        ik_local = ik - start_kpoint
        e_k_local[ik_local, :, ispin] = eigvals
        if v_k_local is not None and eigvecs is not None:
            v_k_local[ik_local, :, :, ispin] = eigvecs

    return e_k_local, v_k_local
