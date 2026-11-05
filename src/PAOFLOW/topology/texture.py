"""Shared pieces of the spin and orbital texture kernels.

:func:`texture_k` is the per-k body (band-diagonal expectation values of
an operator), :func:`fermi_window_bands` selects the bands crossing the
Fermi window, and :func:`write_texture` gathers and writes the result.
``do_spin_texture`` and ``do_orbital_texture`` are these three in
sequence; the sparse backend calls the same functions on values it
streams from the mesh pass.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

_KINDS = {
    # stem of the k-path text file, stem of the per-band npz, npz key,
    # whether the k-path energy column is written with the spin index
    'spin': ('spin-texture-bands', 'spin_text_band_', 'spinband', False),
    'orbital': ('orbital-texture-bands', 'orbital_text_band_', 'orbitalband', True),
}


def texture_k(v_k: np.ndarray, operator: Any) -> np.ndarray:
    """Band-diagonal expectation values :math:`\\langle n|O_l|n\\rangle` at one k-point.

    Parameters
    ----------
    v_k : np.ndarray, shape ``(nawf, m)``
        Eigenvectors at this k-point (columns).
    operator : sequence of three np.ndarray or scipy.sparse matrices
        The Cartesian components ``O_l``, each ``(nawf, nawf)``.

    Returns
    -------
    np.ndarray, shape ``(3, m)``, complex
        The diagonal of :math:`V^\\dagger O_l V` per direction.

    Notes
    -----
    Plain projection, without degenerate-subspace rotation, as the dense
    texture kernels have always done: inside a degenerate group the
    diagonal depends on the eigensolver's gauge.  For an ndarray operator
    the full product is formed (and its diagonal taken) so the dense
    kernels stay bit-identical.
    """
    from scipy.sparse import issparse

    out = np.empty((3, v_k.shape[1]), dtype=complex)
    for l in range(3):
        if issparse(operator[l]):
            out[l] = np.einsum('an,an->n', np.conj(v_k), np.asarray(operator[l] @ v_k))
        else:
            out[l] = np.diagonal(np.conj(v_k.T).dot(operator[l]).dot(v_k))
    return out


def fermi_window_bands(data_controller: DataController, E_k_full: np.ndarray | None) -> list[int]:
    """Bands whose energy range overlaps ``[fermi_dw, fermi_up]``.

    Parameters
    ----------
    data_controller : DataController
        Supplies ``fermi_up`` and ``fermi_dw``; receives ``arrays['ind_plot']``.
    E_k_full : np.ndarray or None, shape ``(nkpnts, nbands, nspin)``
        Gathered eigenvalues on rank 0, ``None`` elsewhere.

    Returns
    -------
    list of int
        The band indices, on every rank (collective).
    """
    arrays, attributes = data_controller.data_dicts()
    fermi_up, fermi_dw = attributes['fermi_up'], attributes['fermi_dw']

    ind_plot = []
    if rank == 0:
        for ib in range(E_k_full.shape[1]):
            E_k_min = np.amin(E_k_full[:, ib, 0])
            E_k_max = np.amax(E_k_full[:, ib, 0])
            btwUp = E_k_min < fermi_up and E_k_max > fermi_up
            btwDwn = E_k_min < fermi_dw and E_k_max > fermi_dw
            btwUaD = E_k_min > fermi_dw and E_k_max < fermi_up
            if btwUp or btwDwn or btwUaD:
                ind_plot.append(ib)

    ind_plot = comm.bcast(ind_plot)
    arrays['ind_plot'] = ind_plot
    return ind_plot


def write_texture(
    data_controller: DataController,
    kind: str,
    E_k_full: np.ndarray | None,
    txtaux: np.ndarray,
    ind_plot: list[int],
) -> np.ndarray | None:
    """Gather the texture of the selected bands and write it.

    Parameters
    ----------
    data_controller : DataController
        Supplies the mesh and ``opath``.
    kind : {'spin', 'orbital'}
        Selects the file names and the npz key.
    E_k_full : np.ndarray or None
        Gathered eigenvalues on rank 0, ``None`` elsewhere.
    txtaux : np.ndarray, shape ``(nk_local, 3, len(ind_plot))``
        This rank's expectation values of the selected bands.
    ind_plot : list of int
        Selected band indices.

    Returns
    -------
    np.ndarray or None
        The gathered texture on rank 0 (``(nkpnts, 3, nb)`` on a k-path,
        ``(nk1, nk2, nk3, 3, nb)`` on the mesh), ``None`` elsewhere.  The
        caller stores it as ``sktxt``/``oktxt``.

    Notes
    -----
    On a k-path (``kq`` present with one point per eigenvalue row) the
    rank-0 file is ``<kind>-texture-bands.dat``; on the mesh, one
    ``<kind>_text_band_<ib>.npz`` per selected band.
    """
    from ..utils.communication import gather_full

    arrays, attributes = data_controller.data_dicts()
    path_stem, npz_stem, npz_key, energy_with_spin = _KINDS[kind]
    nk1, nk2, nk3 = attributes['nk1'], attributes['nk2'], attributes['nk3']
    icount = len(ind_plot)

    txt = gather_full(np.ascontiguousarray(txtaux), attributes['npool'])

    if rank == 0:
        if 'kq' in arrays and E_k_full.shape[0] == arrays['kq'].shape[1]:
            f = open(os.path.join(attributes['opath'], path_stem + '.dat'), 'w')
            for ik in range(E_k_full.shape[0]):
                for ib in range(icount):
                    idx = ind_plot[ib]
                    energy = E_k_full[ik, idx, 0] if energy_with_spin else E_k_full[ik, idx]
                    f.write(
                        '\t'.join(
                            ['%d' % ik]
                            + ['% 5.8f' % energy]
                            + ['% 5.8f' % j for j in txt[ik, :, ib].real]
                        )
                        + '\n'
                    )
                f.write('\n')
            f.close()
        else:
            txt = np.reshape(txt, (nk1, nk2, nk3, 3, icount), order='C')
            for ib in range(icount):
                np.savez(
                    os.path.join(attributes['opath'], npz_stem + str(ib)),
                    **{npz_key: txt[:, :, :, :, ib]},
                )
    return txt
