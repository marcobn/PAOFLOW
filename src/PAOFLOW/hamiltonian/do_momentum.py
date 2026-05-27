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
    import numpy as np

    from .perturb_split import perturb_split

    arry, attr = data_controller.data_dicts()

    nktot, _, nawf, nawf, nspin = arry['dHksp'].shape

    arry['pksp'] = np.zeros_like(arry['dHksp'])

    for ispin in range(nspin):
        for ik in range(nktot):
            for l in range(3):
                arry['pksp'][ik, l, :, :, ispin], _ = perturb_split(
                    arry['dHksp'][ik, l, :, :, ispin],
                    arry['dHksp'][ik, l, :, :, ispin],
                    arry['v_k'][ik, :, :, ispin],
                    arry['degen'][ispin][ik],
                )
                #  impose hermiticity
                arry['pksp'][ik, l, :, :, ispin] = (
                    arry['pksp'][ik, l, :, :, ispin] + np.conj(arry['pksp'][ik, l, :, :, ispin].T)
                ) / 2.0
            #  arry['pksp'][ik,l,attr['bnd']:,attr['bnd']:,ispin] = 0.0

    # for ispin in range(nspin):
    #   for ik in range(nktot):
    #     for l in range(3):
    #        arry['pksp'][ik,l,:,:,ispin] = arry['v_k'][ik,:,:,ispin]@arry['dHksp'][ik,l,:,:,ispin]@ \
    #                                       np.conj(arry['v_k'][ik,:,:,ispin]).T
