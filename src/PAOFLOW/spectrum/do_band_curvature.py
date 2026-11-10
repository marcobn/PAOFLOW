from __future__ import annotations

from typing import Any

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
    vel_degen = velocity_groups(v_aux.diagonal().real, degen)

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


def velocity_groups(vel: np.ndarray, energy_groups: list[np.ndarray]) -> list[np.ndarray]:
    """Sets of two or more states that share an energy group and every row of ``vel`` (within 1e-5).

    ``vel`` is ``(n,)`` or ``(ncomp, n)``. Unlike ``get_degeneracies`` the members need not be
    index neighbours: a folded cluster ordered by ``v_x`` can hold equal ``v_y`` in the order
    ``(a, b, b, a)``.
    """
    groups = [np.asarray(e) for e in energy_groups]
    for v in np.atleast_2d(vel):
        split = []
        for g in groups:
            order = g[np.argsort(v[g], kind='stable')]
            cuts = np.flatnonzero(np.diff(v[order]) > 1.0e-5) + 1
            split += [s for s in np.split(order, cuts) if len(s) > 1]
        groups = split
    return [np.sort(g) for g in groups]


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
    k-point and calls :func:`~PAOFLOW.sparse.curvature.band_curvature_k`, which shares the work
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
