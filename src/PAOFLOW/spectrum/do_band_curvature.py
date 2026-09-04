def do_band_curvature(data_controller):
    """Compute the full band curvature (inverse effective mass) tensor :math:`d^2E/dk_i dk_j`.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``Hksp``, ``Rfft``, ``E_k``, ``dHksp``, ``v_k``, ``degen``.
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
    The curvature tensor is computed in two steps.

    First, the diagonal matrix elements of :math:`d^2H/dk_i dk_j` in the
    Bloch eigenstate basis are obtained by calling :func:`do_d2Hd2k_ij`.

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
    modified eigenvector set returned in ``dvec_list`` is reused here.
    Pairs with :math:`|E_n - E_m| < 10^{-5}` eV are excluded to avoid
    numerical divergences.
    """

    import numpy as np
    from numpy.linalg import eigh

    from ..hamiltonian.do_d2Hd2k import do_d2Hd2k_ij
    from ..utils.communication import scatter_full

    ary, attr = data_controller.data_dicts()
    bnd = attr['bnd']
    nawf = attr['nawf']
    E_k = ary['E_k']

    Dnm = scatter_full(
        np.reshape(ary['Dnm'], (attr['nawf'] * attr['nawf'], 3), order='C'), attr['npool']
    )

    # not really the inverse mass tensor..it's actually tksp
    # but we are calling it d2Ed2k for now to save memory.
    d2Ed2k, degen_idx, degen_d2Hdk, dvec_list = do_d2Hd2k_ij(
        ary['Hksp'],
        ary['dHksp'],
        Dnm,
        ary['Rfft'],
        attr['alat'],
        attr['npool'],
        ary['v_k'],
        bnd,
        ary['degen'],
    )

    # d2Ed2k is only the 6 unique components of the curvature
    # (inverse effective mass ) tensor. This is one to save memory.
    ij_ind = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]], dtype=int)
    E_temp = np.zeros((bnd, nawf), order='C')

    # ----------------------
    # for d2E/d2k_ij
    # ----------------------
    for ispin in range(d2Ed2k.shape[3]):
        for ik in range(d2Ed2k.shape[1]):
            # tksp_ij = <psi|d2Hd2k_ij|psi>

            # ij component of second derivative of the energy is:
            # tksp_ij + sum_i( (pksp_i*pksp_j.T + pksp_j*pksp_i.T)/(E_i-E_j) )
            E_temp = ((E_k[ik, :, ispin] - E_k[ik, :, ispin][:, None])[:, :]).T
            E_temp[np.where(np.abs(E_temp) < 1.0e-3)] = np.inf

            for ij in range(ij_ind.shape[0]):
                ipol = ij_ind[ij, 0]
                jpol = ij_ind[ij, 1]

                # to avoid a zero in the denominator when E_i=E_j
                if dvec_list[ij][ispin][ik].size:
                    v_k = dvec_list[ij][ispin][ik]
                else:
                    v_k = ary['v_k'][ik, :, :, ispin]

                pksp_i = np.conj(v_k.T).dot(ary['dHksp'][ik, ipol, :, :, ispin]).dot(v_k)
                pksp_j = np.conj(v_k.T).dot(ary['dHksp'][ik, jpol, :, :, ispin]).dot(v_k)

                # this is where d2Ed2k becomes the actual curvature tensor
                d2Ed2k[ij, ik, :, ispin] += np.sum(
                    (((pksp_i * pksp_j.T + pksp_j * pksp_i.T) / E_temp).real), axis=1
                )[:bnd]

                # second order perturbation for degeneracies of E and dEdk
                if degen_idx[ij][ispin][ik]:
                    for i in range(len(degen_idx[ij][ispin][ik])):
                        ll = degen_idx[ij][ispin][ik][i][0]
                        ul = degen_idx[ij][ispin][ik][i][-1] + 1

                        degen_d2Ed2k = (
                            degen_d2Hdk[ij][ispin][ik][i]
                            + pksp_i[ll:ul, ll:ul] @ (pksp_j[ll:ul, ll:ul] / E_temp[ll:ul, ll:ul])
                            + (pksp_i[ll:ul, ll:ul] @ (pksp_j[ll:ul, ll:ul] / E_temp[ll:ul, ll:ul]))
                            .conj()
                            .T
                        )

                        d2Ed2k[ij, ik, ll:ul, ispin], _ = eigh(degen_d2Ed2k)

    ary['d2Ed2k'] = d2Ed2k
