from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _project(v_k: np.ndarray, operator: Any) -> np.ndarray:
    """``v_k^dagger operator v_k`` for a dense or ``scipy.sparse`` operator.

    Notes
    -----
    The dense association order ``(v^dagger A) v`` is kept for ndarrays so
    :func:`do_band_curvature` stays bit-identical.
    """
    from scipy.sparse import issparse

    if issparse(operator):
        return np.conj(v_k.T) @ np.asarray(operator @ v_k)
    return np.conj(v_k.T).dot(operator).dot(v_k)


def band_curvature_k_ij(
    E: np.ndarray,
    dH_i: Any,
    dH_j: Any,
    d2H_ij: Any,
    v_k: np.ndarray,
    degen: list[np.ndarray],
    bnd: int,
) -> np.ndarray:
    """One component :math:`d^2E_n/dk_i dk_j` of the band curvature at one k-point.

    Parameters
    ----------
    E : np.ndarray, shape ``(nawf,)``
        Every eigenvalue at this k-point (eV), ascending.  The interband sum
        runs over all of them, so a truncated spectrum gives a wrong result.
    dH_i, dH_j : np.ndarray or scipy.sparse matrix, shape ``(nawf, nawf)``
        ``dH/dk_i`` and ``dH/dk_j`` in the PAO basis.
    d2H_ij : np.ndarray or scipy.sparse matrix, shape ``(nawf, nawf)``
        :math:`d^2H/dk_i dk_j` in the PAO basis.
    v_k : np.ndarray, shape ``(nawf, nawf)``
        Eigenvectors matching ``E`` (columns).
    degen : list of np.ndarray
        Degenerate band groups at this k-point, from
        :func:`~PAOFLOW.spectrum.do_eigh.get_degeneracies` with ``bnd``.  The velocity
        groups that take the subspace ``eigh`` are searched within them.
    bnd : int
        Number of bands whose curvature is returned.

    Returns
    -------
    np.ndarray, shape ``(bnd,)``
        The curvature of the lowest ``bnd`` bands, in the units of
        :func:`do_band_curvature`.

    Notes
    -----
    The three steps of :func:`do_band_curvature` for one ``(i, j)`` and one
    k-point: the rotated diagonal of :math:`d^2H/dk_i dk_j`, the interband
    second-order sum over every other band, and the eigenvalues of the
    subspace blocks in which the velocity along ``i`` stays degenerate.
    """
    from numpy.linalg import eigh

    from ..utils.perturb_split import perturb_split

    v_aux, tksp, dvec = perturb_split(dH_i, d2H_ij, v_k, degen, return_v_k=True)
    vel_degen = _velocity_groups(v_aux.diagonal().real, degen)

    curvature = np.array(tksp.diagonal()[:bnd].real, dtype=float)

    # tksp_ij = <psi|d2Hd2k_ij|psi>
    # ij component of second derivative of the energy is:
    # tksp_ij + sum_i( (pksp_i*pksp_j.T + pksp_j*pksp_i.T)/(E_i-E_j) )
    E_temp = ((E - E[:, None])[:, :]).T
    E_temp[np.where(np.abs(E_temp) < 1.0e-3)] = np.inf

    # to avoid a zero in the denominator when E_i=E_j
    v_rot = dvec if dvec.size else v_k

    pksp_i = _project(v_rot, dH_i)
    pksp_j = _project(v_rot, dH_j)

    # this is where d2Ed2k becomes the actual curvature tensor
    curvature += np.sum((((pksp_i * pksp_j.T + pksp_j * pksp_i.T) / E_temp).real), axis=1)[:bnd]

    # second order perturbation for degeneracies of E and dEdk: the push sums over every band m,
    # the inf entries of E_temp drop the group itself and bands within 1 meV
    for g in vel_degen:
        push = pksp_i[g, :] @ (pksp_j[:, g] / E_temp[g[0], :][:, None])
        curvature[g], _ = eigh(tksp[np.ix_(g, g)] + push + push.conj().T)
    return curvature


def _velocity_groups(vel: np.ndarray, energy_groups: list[np.ndarray]) -> list[np.ndarray]:
    """Sets of two or more states that share an energy group and a velocity (within 1e-5).

    Unlike ``get_degeneracies`` the members need not be index neighbours: a folded cluster
    ordered by ``v_x`` can hold equal ``v_y`` in the order ``(a, b, b, a)``.
    """
    groups = []
    for e in energy_groups:
        order = e[np.argsort(vel[e], kind='stable')]
        cuts = np.flatnonzero(np.diff(vel[order]) > 1.0e-5) + 1
        groups += [np.sort(g) for g in np.split(order, cuts) if len(g) > 1]
    return groups


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

    The same result as :func:`band_curvature_k_ij` for every ``(i, j)`` of
    ``IJ_PAIRS``, up to rounding, with the work shared between the components.

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
        the subspace ``eigh`` are searched within them.

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

    # per direction i: the rotation that diagonalizes dH/dk_i in each
    # degenerate group, and the groups in which its diagonal stays degenerate
    rot = []
    for i in range(3):
        blocks = [
            (g[0], g[-1] + 1, eigh(cols[i][g[0] : g[-1] + 1, g[0] : g[-1] + 1])[1]) for g in degen
        ]
        p_i = rotate(cols[i], blocks)
        vel_degen = _velocity_groups(np.diagonal(p_i).real, energy_groups)
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


def do_band_curvature(data_controller):
    """Compute the full band curvature (inverse effective mass) tensor :math:`d^2E/dk_i dk_j`.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``Hksp``, ``Rfft``, ``E_k``, ``dHksp``, ``Dnm``,
        ``v_k``, ``degen``.
        Required attributes: ``bnd``, ``nawf``, ``alat``, ``npool``.

    Returns
    -------
    None
        Adds the following key to ``data_controller.data_arrays``:

        - ``d2Ed2k`` : np.ndarray, shape ``(6, nkpnts, bnd, nspin)`` —
          the six unique components of the curvature tensor (in units
          of :math:`\\hbar^2 / (\\text{eV} \\cdot \\text{Bohr}^2)`).
          Component ordering: ``xx, yy, zz, xy, xz, yz``.

    Notes
    -----
    The curvature tensor is computed in three steps.

    First, the diagonal matrix elements of :math:`d^2H/dk_i dk_j` in the
    Bloch eigenstate basis are obtained from :func:`do_d2Hd2k_ij`, rotated
    by :func:`perturb_split` with ``dH/dk_i`` resolving the degeneracies.

    Second, the second-order energy correction due to off-diagonal
    (inter-band) coupling is added via second-order perturbation theory:

    .. math::

        \\frac{d^2 E_n}{dk_i dk_j} = \\langle n | \\partial^2_{k_i k_j} H | n \\rangle
            + \\sum_{m \\neq n}
              \\frac{\\langle n | \\partial_{k_i} H | m \\rangle
                     \\langle m | \\partial_{k_j} H | n \\rangle
                     + (i \\leftrightarrow j)}
                    {E_n - E_m}

    Degenerate subspaces are handled by :func:`perturb_split` and the
    modified eigenvector set it returns is reused here.
    Pairs with :math:`|E_n - E_m| < 10^{-3}` eV are excluded to avoid
    numerical divergences.

    Third, within each degenerate energy group, the states whose rotated
    velocity along ``i`` is still equal (within :math:`10^{-5}`, adjacent in
    index or not) form a group :math:`g`.  Their diagonal elements are
    overwritten by the eigenvalues of

    .. math::

        M_{ab} = \\langle a | \\partial^2_{k_i k_j} H | b \\rangle
            + P_{ab} + P^{*}_{ba},
        \\qquad
        P_{ab} = \\sum_{|E_m - E_g| > 10^{-3}}
            \\frac{\\langle a | \\partial_{k_i} H | m \\rangle
                   \\langle m | \\partial_{k_j} H | b \\rangle}{E_g - E_m},
        \\qquad a, b \\in g,

    with :math:`m` running over every band.  The first-order term does not
    fix the basis in such a group, so its curvatures are the eigenvalues of
    the second-order matrix.  For a Kramers pair :math:`M` is proportional to
    the identity and the result equals the non-degenerate formula.

    The loop is ``(i, j)`` outer and k inner, so only one
    :math:`d^2H/dk_i dk_j` is held at a time; the per-k work is
    :func:`band_curvature_k_ij`.  The sparse backend holds all six at one
    k-point and calls :func:`band_curvature_k`, which shares the work
    between them.
    """

    from ..hamiltonian.do_d2Hd2k import IJ_PAIRS, do_d2Hd2k_ij
    from ..utils.communication import scatter_full

    ary, attr = data_controller.data_dicts()
    bnd = attr['bnd']
    E_k = ary['E_k']
    v_kp = ary['v_k']
    dHksp = ary['dHksp']

    Dnm = scatter_full(
        np.reshape(ary['Dnm'], (attr['nawf'] * attr['nawf'], 3), order='C'), attr['npool']
    )

    # d2Ed2k is only the 6 unique components of the curvature
    # (inverse effective mass ) tensor. This is one to save memory.
    d2Ed2k = np.zeros((6, v_kp.shape[0], bnd, v_kp.shape[3]), dtype=float, order='C')

    for ij, (ipol, jpol) in enumerate(IJ_PAIRS):
        d2Hksp = do_d2Hd2k_ij(
            ary['Hksp'], Dnm, ary['Rfft'], attr['alat'], attr['npool'], ipol, jpol
        )
        for ispin in range(d2Ed2k.shape[3]):
            for ik in range(d2Ed2k.shape[1]):
                d2Ed2k[ij, ik, :, ispin] = band_curvature_k_ij(
                    E_k[ik, :, ispin],
                    dHksp[ik, ipol, :, :, ispin],
                    dHksp[ik, jpol, :, :, ispin],
                    d2Hksp[:, :, ik, ispin],
                    v_kp[ik, :, :, ispin],
                    ary['degen'][ispin][ik],
                    bnd,
                )
        d2Hksp = None

    ary['d2Ed2k'] = d2Ed2k
