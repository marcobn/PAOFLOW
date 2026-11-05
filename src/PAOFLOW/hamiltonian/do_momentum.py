from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# Fixed generic direction: the common basis of a degenerate subspace diagonalises dH/dk along it.
# Irrational, unequal components, so that no crystal symmetry maps it onto itself.
_GENERIC_DIRECTION = np.array([1.0, np.sqrt(2.0) - 1.0, np.pi - 3.0])
_GENERIC_DIRECTION /= np.linalg.norm(_GENERIC_DIRECTION)


def momentum_in_eigenbasis(dhk: np.ndarray, v_k: np.ndarray, degen: list) -> np.ndarray:
    r"""Momentum matrix of one k-point and spin in the Bloch eigenbasis, degenerate subspaces fixed.

    Parameters
    ----------
    dhk : np.ndarray, shape ``(3, nawf, nawf)``
        :math:`\partial H/\partial k_l` in the PAO basis.
    v_k : np.ndarray, shape ``(nawf, nbnd)``
        Eigenvectors (columns) at this k-point.
    degen : list of array_like
        Index sets of the degenerate subspaces (consecutive bands).

    Returns
    -------
    np.ndarray, shape ``(3, nbnd, nbnd)``
        :math:`p^l_{nm}`, Hermitian.

    Notes
    -----
    Two requirements meet in a degenerate subspace :math:`D`:

    * the elements between :math:`D` and the other bands enter products of two directions,
      :math:`\sum_{n\in D,\,m} p^a_{nm} p^b_{mn}`, which are invariant only if :math:`p^a` and
      :math:`p^b` use the same basis of :math:`D`. That basis is the eigenbasis of
      :math:`\hat g\cdot\partial H/\partial\mathbf k` restricted to :math:`D`, with
      :math:`\hat g` a fixed generic direction;
    * the diagonal holds the band velocities of degenerate perturbation theory: the eigenvalues of
      :math:`p^l` restricted to :math:`D`, for each direction :math:`l` separately. The block of
      :math:`D` is replaced by their diagonal matrix (ascending order), as before.

    Rotating :math:`D` separately for each direction (the former rule) made the off-diagonal
    tensor components depend on the basis at degenerate k-points (a cubic crystal had
    :math:`\varepsilon_{xy}\neq 0`). Diagonal components :math:`aa` are unchanged by the common
    rotation: :math:`\sum_{n\in D}|p^a_{nm}|^2` is invariant.
    """
    vk = np.array(v_k, copy=True)
    pk = vk.conj().T @ dhk @ vk
    if len(degen) == 0:
        return 0.5 * (pk + np.conj(np.swapaxes(pk, 1, 2)))
    blocks = sorted({(int(d[0]), int(d[-1]) + 1) for d in degen})  # each subspace once
    for lo, hi in blocks:
        generic = np.einsum('l,lnm->nm', _GENERIC_DIRECTION, pk[:, lo:hi, lo:hi])
        _, weight = np.linalg.eigh(0.5 * (generic + generic.conj().T))
        vk[:, lo:hi] = vk[:, lo:hi] @ weight
    pk = vk.conj().T @ dhk @ vk
    pk = 0.5 * (pk + np.conj(np.swapaxes(pk, 1, 2)))
    for lo, hi in blocks:
        for l in range(3):
            pk[l, lo:hi, lo:hi] = np.diag(np.linalg.eigvalsh(pk[l, lo:hi, lo:hi]))
    return pk


def momentum_k(dhk: Sequence[Any], v_k: np.ndarray, degen: list[np.ndarray]) -> np.ndarray:
    """Momentum matrix elements in the Bloch eigenstate basis at one k-point.

    Parameters
    ----------
    dhk : sequence of np.ndarray or scipy.sparse matrix
        The three Cartesian derivatives ``dH/dk_l``, each ``(nawf, nawf)``.
    v_k : np.ndarray, shape ``(nawf, m)``
        Eigenvectors at this k-point (columns).
    degen : list of np.ndarray
        Degenerate band groups at this k-point, as from
        :func:`~PAOFLOW.spectrum.do_eigh.get_degeneracies`.

    Returns
    -------
    np.ndarray, shape ``(3, m, m)``, complex
        :math:`p^l_{nm} = \\langle n | \\partial H/\\partial k_l | m \\rangle`,
        each direction projected by :func:`perturb_split` and Hermitized.

    Notes
    -----
    The per-k body of :func:`do_momentum`, which loops over it; the sparse
    backend calls it with CSR operators.
    """
    from ..utils.perturb_split import perturb_split

    m = v_k.shape[1]
    pk = np.zeros((3, m, m), dtype=complex)
    for l in range(3):
        pk[l], _ = perturb_split(dhk[l], dhk[l], v_k, degen)
        #  impose hermiticity
        pk[l] = (pk[l] + np.conj(pk[l].T)) / 2.0
    return pk


def do_momentum(data_controller):
    r"""Compute momentum matrix elements in the Bloch eigenstate basis.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``dHksp`` (shape ``(nktot, 3, nawf, nawf, nspin)``),
        ``v_k`` (shape ``(nktot, nawf, bnd, nspin)``),
        ``degen`` (nested list of degenerate subspace indices).

    Returns
    -------
    None
        Adds the following key to ``data_controller.data_arrays``:

        - ``pksp`` : np.ndarray, shape ``(nktot, 3, nawf, nawf, nspin)``,
          complex — the momentum matrix elements
          :math:`\langle n\mathbf{k} | \hat{p}_l | m\mathbf{k} \rangle`
          for each Cartesian direction :math:`l`, projected onto the Bloch
          eigenstate basis and Hermitian-symmetrised.

    Notes
    -----
    For each k-point and Cartesian direction :math:`l`,

    .. math::

        p^l_{nm}(\mathbf{k}) =
            \langle n\mathbf{k} | \partial H / \partial k_l | m\mathbf{k} \rangle ,
        \qquad p^l \leftarrow \frac{p^l + (p^l)^\dagger}{2}.

    The three directions share one basis of every degenerate subspace, and the diagonal block of a
    subspace holds the band velocities of each direction: see :func:`momentum_in_eigenbasis`.
    """
    arry, attr = data_controller.data_dicts()

    nktot, _, nawf, nawf, nspin = arry['dHksp'].shape

    arry['pksp'] = np.zeros_like(arry['dHksp'])

    for ispin in range(nspin):
        for ik in range(nktot):
            arry['pksp'][ik, :, :, :, ispin] = momentum_in_eigenbasis(
                arry['dHksp'][ik, :, :, :, ispin],
                arry['v_k'][ik, :, :, ispin],
                arry['degen'][ispin][ik],
            )
