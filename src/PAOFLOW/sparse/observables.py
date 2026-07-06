"""Sparse observables: band velocities, adaptive smearing, and DOS glue.

These stages consume only *band-diagonal* spectral quantities, which is exactly
why the sparse path meets the dense pipeline cleanly here.  Band velocities are
obtained by the Hellmann–Feynman theorem on the selected eigenvectors — the
diagonal of the momentum operator in the eigenbasis — so no dense
``(nkpnts, nawf, nawf, nspin)`` momentum tensor (``pksp``) is ever formed.  The
resulting ``E_k``/``velkp``/``deltakp`` arrays carry only the selected bands and
slot directly into the existing DOS and Boltzmann kernels, which index bands as
``[:bnd]``.
"""

import numpy as np


def store_eigenpairs(data_controller, eigpairs):
    """Publish selected eigenpairs into ``data_arrays`` for downstream kernels.

    Parameters
    ----------
    data_controller : DataController
    eigpairs : SparseEigenpairs
        Result of :func:`solve_window` (must carry eigenvectors).

    Returns
    -------
    None
        Sets ``E_k`` and (transiently) ``sparse_v_k`` in ``data_arrays`` and
        sets ``attr['bnd']`` to the number of selected bands so that DOS and
        transport operate on exactly the computed window.
    """
    arry, attr = data_controller.data_dicts()
    arry['E_k'] = eigpairs.E_k
    arry['sparse_v_k'] = eigpairs.v_k
    attr['bnd'] = eigpairs.n_sel
    attr['nkpnts'] = eigpairs.E_k.shape[0]


def compute_velocities(data_controller, kcart):
    """Band-diagonal group velocities via Hellmann–Feynman on selected states.

    .. math::

        v_{n}^{l}(\\mathbf{k}) = \\mathrm{Re}\\,
            \\langle \\psi_{n\\mathbf{k}} |
            \\partial H/\\partial k_l | \\psi_{n\\mathbf{k}} \\rangle

    evaluated only for the selected bands, with ``dH/dk_l`` assembled sparse
    (:meth:`SparseHamiltonian.build_dHk`).  The sparse-times-dense product
    ``dHk_l @ V`` is ``(nawf, n_sel)`` — bounded, never ``(nawf, nawf)``.

    Parameters
    ----------
    data_controller : DataController
        Provides ``sparse_H``, ``sparse_v_k`` (selected eigenvectors), ``bnd``.
    kcart : np.ndarray, shape ``(nkpnts, 3)``
        Cartesian k-points matching the eigenvector mesh.

    Returns
    -------
    None
        Sets ``data_arrays['velkp']`` of shape ``(nkpnts, 3, n_sel, nspin)``
        (real) and ``data_arrays['velkp_bare']`` (an alias used by the adaptive
        smearing helper).
    """
    arry, attr = data_controller.data_dicts()
    sparse_h = arry['sparse_H']
    v_k = arry['sparse_v_k']
    nk, nawf, n_sel, nspin = v_k.shape

    velkp = np.zeros((nk, 3, n_sel, nspin), dtype=float)
    for ispin in range(nspin):
        for ik in range(nk):
            V = v_k[ik, :, :, ispin]  # (nawf, n_sel)
            dHk = sparse_h.build_dHk(kcart[ik], ispin)  # 3 x csr(nawf, nawf)
            for l in range(3):
                dHV = dHk[l] @ V  # (nawf, n_sel) dense, bounded
                # Diagonal <n|dH/dk|n> without forming the n_sel x n_sel matrix.
                velkp[ik, l, :, ispin] = np.real(np.einsum('an,an->n', np.conj(V), dHV))

    arry['velkp'] = velkp
    # The velocities are already the bare group velocities (no non-local
    # correction in the sparse path); expose under the key the smearing helper
    # and adaptive DOS expect.
    arry['velkp_bare'] = velkp


def adaptive_smearing(data_controller, smearing, afac=None):
    """Yates adaptive smearing widths from the band-diagonal velocities.

    Mirrors :func:`PAOFLOW.spectrum.do_adaptive_smearing.do_adaptive_smearing`
    but for the *selected* bands only, and omits the dense interband
    ``deltakp2`` (an ``(nkpnts, nawf, nawf, nspin)`` array) which is not used by
    DOS or transport.

    Parameters
    ----------
    data_controller : DataController
        Provides ``velkp`` (band velocities), ``omega``, ``nkpnts``, ``bnd``.
    smearing : str
        ``'gauss'`` or ``'m-p'``.
    afac : float, optional
        Prefactor; defaults to ``1.0`` for ``'m-p'`` and ``0.7`` otherwise.

    Returns
    -------
    None
        Sets ``data_arrays['deltakp']`` of shape ``(nkpnts, n_sel, nspin)``.
    """
    from numpy.linalg import norm

    arry, attr = data_controller.data_dicts()

    velkp = arry['velkp']  # (nk, 3, n_sel, nspin)
    nk, _, n_sel, nspin = velkp.shape

    dk = (8.0 * np.pi**3 / attr['omega'] / attr['nkpnts']) ** (1.0 / 3.0)
    if afac is None:
        afac = 1.0 if smearing == 'm-p' else 0.7

    deltakp = np.zeros((nk, n_sel, nspin), dtype=float)
    for n in range(n_sel):
        deltakp[:, n, :] = norm(np.real(velkp[:, :, n, :]), axis=1)
    deltakp *= afac * dk

    arry['deltakp'] = deltakp
