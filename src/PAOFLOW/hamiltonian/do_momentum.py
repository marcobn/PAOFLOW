from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


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
    """Compute momentum matrix elements in the Bloch eigenstate basis.

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
          :math:`\\langle n\\mathbf{k} | \\hat{p}_l | m\\mathbf{k} \\rangle`
          for each Cartesian direction :math:`l`, projected onto the Bloch
          eigenstate basis and Hermitian-symmetrised.

    Notes
    -----
    For each k-point and Cartesian direction :math:`l`, the momentum
    operator matrix element is computed in the Bloch eigenstate basis via
    :func:`perturb_split`:

    .. math::

        p^l_{nm}(\\mathbf{k}) =
            \\langle n\\mathbf{k} | \\partial H / \\partial k_l | m\\mathbf{k} \\rangle

    Hermitian symmetry is then enforced as

    .. math::

        p^l \\leftarrow \\frac{p^l + (p^l)^\\dagger}{2}

    Degenerate subspaces at each k-point are handled by :func:`perturb_split`
    to ensure a well-defined basis.
    """
    arry, attr = data_controller.data_dicts()

    nktot, _, nawf, nawf, nspin = arry['dHksp'].shape

    arry['pksp'] = np.zeros_like(arry['dHksp'])

    for ispin in range(nspin):
        for ik in range(nktot):
            arry['pksp'][ik, :, :, :, ispin] = momentum_k(
                arry['dHksp'][ik, :, :, :, ispin],
                arry['v_k'][ik, :, :, ispin],
                arry['degen'][ispin][ik],
            )

    # for ispin in range(nspin):
    #   for ik in range(nktot):
    #     for l in range(3):
    #        arry['pksp'][ik,l,:,:,ispin] = arry['v_k'][ik,:,:,ispin]@arry['dHksp'][ik,l,:,:,ispin]@ \
    #                                       np.conj(arry['v_k'][ik,:,:,ispin]).T
