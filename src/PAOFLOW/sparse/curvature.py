"""Band curvature of one k-point of the sparse mesh pass.

:func:`band_curvature_k` computes the six components that the dense
:func:`~PAOFLOW.spectrum.do_band_curvature.band_curvature_k_ij` computes one per call,
sharing the work between them. It assumes the sparse eigenbasis: folded clusters already
rotated into the joint ``dH/dk_l`` eigenbasis by :mod:`PAOFLOW.sparse.mesh`, so the dense
driver does not call it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def band_curvature_k(
    E: np.ndarray,
    dH: Sequence[Any],
    d2H: Sequence[Any],
    v_k: np.ndarray,
    degen: list[np.ndarray],
    bnd: int,
    energy_groups: list[np.ndarray],
) -> np.ndarray:
    """All six components :math:`d^2E_n/dk_i dk_j` of the band curvature at one k-point.

    The same result as :func:`~PAOFLOW.spectrum.do_band_curvature.band_curvature_k_ij` for
    every ``(i, j)`` of ``IJ_PAIRS``, up to rounding, with the work shared between the
    components, except in energy groups that ``degen`` leaves out.  There ``v_k`` must be a
    joint eigenbasis of the three ``dH/dk_l`` (:func:`~PAOFLOW.sparse.mesh._joint_basis`),
    and a velocity group needs equal ``v_x``, ``v_y`` and ``v_z``, not only equal ``v_i``.  Otherwise the subspace ``eigh``
    would sort ``d2E/dk_i dk_j`` across folded copies of different primitive k, which share
    ``E`` and ``v_i`` but not ``v_j``, and hand each copy the other's value.

    Parameters
    ----------
    E : np.ndarray, shape ``(nawf,)``
        Every eigenvalue at this k-point (eV), ascending.
    dH : sequence of 3 arrays or scipy.sparse matrices, shape ``(nawf, nawf)``
        ``dH/dk_x``, ``dH/dk_y``, ``dH/dk_z`` in the PAO basis.
    d2H : sequence of 6 arrays or scipy.sparse matrices, shape ``(nawf, nawf)``
        :math:`d^2H/dk_i dk_j` in the order of ``IJ_PAIRS``; each is read once.
    v_k : np.ndarray, shape ``(nawf, nawf)``
        Eigenvectors matching ``E`` (columns).
    degen : list of np.ndarray
        Degenerate band groups at this k-point, all below ``bnd``, in which ``dH/dk_i`` is
        diagonalized per direction (the sparse ``KPoint.degen``).
    bnd : int
        Number of bands whose curvature is returned.
    energy_groups : list of np.ndarray
        Every degenerate energy group below ``bnd``, including those ``degen`` leaves out
        (:func:`~PAOFLOW.sparse.kpoint.degenerate_groups`).  The velocity groups that take
        the subspace ``eigh`` are searched within them, by ``v_i`` inside a ``degen`` part
        and by all three velocity components elsewhere.

    Returns
    -------
    np.ndarray, shape ``(6, bnd)``

    Notes
    -----
    Called once per ``(i, j)``, :func:`band_curvature_k_ij` rotates
    ``dH/dk_i`` into the eigenbasis four times and ``dH/dk_j`` and
    ``d2H/dk_i dk_j`` twice each: 60 ``(nawf, nawf, nawf)`` products per
    k-point.  Every quantity it uses, though, needs only the first ``bnd``
    columns of the rotated operators (the curvature rows are ``< bnd``, the
    operators are Hermitian, and the degenerate groups lie below ``bnd``).

    So each ``dH/dk_l V`` is formed once, kept as its first
    ``bnd`` columns, and rotated for each direction ``i`` block by block
    within the degenerate groups.  Each ``d2H/dk_i dk_j`` is applied to the
    rotated ``bnd`` window bands only.  That is 3 full products and
    9 ``(nawf, nawf, bnd)`` ones per k-point.
    """
    from numpy.linalg import eigh

    from ..hamiltonian.do_d2Hd2k import IJ_PAIRS
    from ..spectrum.do_band_curvature import velocity_groups

    vb = v_k[:, :bnd]
    cols = []
    for op in dH:
        av = np.asarray(op @ v_k)
        cols.append(np.conj(v_k.T) @ av[:, :bnd])  # (V^+ dH V)[:, :bnd]
        av = None

    def rotate(a, blocks):
        """``W^+ a W`` restricted to the first ``bnd`` columns."""
        a = a.copy()
        for ll, ul, w in blocks:
            a[:, ll:ul] = a[:, ll:ul] @ w
            a[ll:ul, :] = np.conj(w.T) @ a[ll:ul, :]
        return a

    # rows m, columns n < bnd: E_n - E_m, with the near-degenerate pairs dropped
    E_temp = E[None, :bnd] - E[:, None]
    E_temp[np.abs(E_temp) < 1.0e-3] = np.inf

    # outside degen, V is a joint eigenbasis of the three dH/dk_l (folded copies of different
    # primitive k): a state there is fixed by all three velocities, so groups must share them.
    # Inside a degen part only v_i is defined; the part mean stands in for the other two.
    v_all = np.array([np.diagonal(a).real for a in cols])
    for g in degen:
        v_all[:, g] = v_all[:, g].mean(axis=1, keepdims=True)

    # per direction i: the rotation that diagonalizes dH/dk_i in each
    # degenerate group, and the groups in which its diagonal stays degenerate
    rot = []
    for i in range(3):
        blocks = [
            (g[0], g[-1] + 1, eigh(cols[i][g[0] : g[-1] + 1, g[0] : g[-1] + 1])[1]) for g in degen
        ]
        p_i = rotate(cols[i], blocks)
        key = v_all.copy()
        key[i] = np.diagonal(p_i).real
        vel_degen = velocity_groups(key, energy_groups)
        rot.append((blocks, p_i, vel_degen))

    curvature = np.zeros((len(IJ_PAIRS), bnd), dtype=float)
    for ij, (i, j) in enumerate(IJ_PAIRS):
        blocks, p_i, vel_degen = rot[i]
        p_j = p_i if j == i else rotate(cols[j], blocks)
        vt = vb.copy()
        for ll, ul, w in blocks:
            vt[:, ll:ul] = vt[:, ll:ul] @ w
        t = np.asarray(d2H[ij] @ vt)

        c = np.einsum('na,na->a', np.conj(vt), t).real
        c += np.sum((2.0 * np.conj(p_i) * p_j).real / E_temp, axis=0)

        # second order perturbation for degeneracies of E and dE/dk_i
        # (groups need not be contiguous, hence index arrays); the push sums over every row m
        for g in vel_degen:
            push = np.conj(p_i[:, g]).T @ (p_j[:, g] / E_temp[:, g[0]][:, None])
            block = np.conj(vt[:, g].T) @ t[:, g] + push + np.conj(push.T)
            c[g], _ = eigh(block)
        curvature[ij] = c
    return curvature
