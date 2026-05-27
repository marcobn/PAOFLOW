def do_spin_texture(data_controller):
    """Compute the spin texture for Fermi-surface bands and write output files.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpnts, nawf, 1)``),
        ``v_k`` (shape ``(snktot, nawf, nawf, 1)``),
        ``Sj`` (shape ``(3, nawf, nawf)``).
        Required attributes: ``nawf``, ``nk1``, ``nk2``, ``nk3``, ``npool``,
        ``fermi_up``, ``fermi_dw``, ``opath``.

    Returns
    -------
    None
        Adds the following key to ``data_controller.data_arrays``:

        - ``ind_plot`` : list of int — indices of the bands that cross the
          Fermi window ``[fermi_dw, fermi_up]``.

        Writes one of the following output files on rank 0:

        - ``{opath}/spin-texture-bands.dat``: if ``kq`` is present (k-path
          mode), a text file with columns ``ik``, ``E``, ``Sx``, ``Sy``,
          ``Sz`` for each band in ``ind_plot``.
        - ``{opath}/spin_text_band_{ib}.npz``: one NPZ archive per Fermi band
          (key ``spinband``, shape ``(nk1, nk2, nk3, 3)``) for full k-grid mode.

    Notes
    -----
    The spin expectation value for band :math:`n` at k-point :math:`\\mathbf{k}`
    is computed as

    .. math::

        \\langle S_l \\rangle_{n\\mathbf{k}} =
            \\langle n\\mathbf{k} | S_l | n\\mathbf{k} \\rangle
            = \\mathbf{v}^\\dagger_{n\\mathbf{k}} S_l \\mathbf{v}_{n\\mathbf{k}}

    where :math:`S_l` (``Sj[l]``) is the :math:`l`-th Cartesian spin
    matrix and :math:`\\mathbf{v}_{n\\mathbf{k}}` are the Bloch eigenvectors.
    Only the diagonal elements (band index :math:`n`) of the full matrix product
    are retained.  The computation is valid only for non-collinear or spin–orbit
    coupled calculations (``nspin = 1``).
    """
    import os

    import numpy as np
    from mpi4py import MPI

    from .communication import gather_full

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    fermi_up, fermi_dw = attributes['fermi_up'], attributes['fermi_dw']
    nawf, nk1, nk2, nk3 = (
        attributes['nawf'],
        attributes['nk1'],
        attributes['nk2'],
        attributes['nk3'],
    )
    E_k_full = gather_full(arrays['E_k'], attributes['npool'])

    ind_plot = []
    icount = None
    if rank == 0:
        icount = 0
        for ib in range(nawf):
            E_k_min = np.amin(E_k_full[:, ib, 0])
            E_k_max = np.amax(E_k_full[:, ib, 0])
            btwUp = E_k_min < fermi_up and E_k_max > fermi_up
            btwDwn = E_k_min < fermi_dw and E_k_max > fermi_dw
            btwUaD = E_k_min > fermi_dw and E_k_max < fermi_up
            if btwUp or btwDwn or btwUaD:
                ind_plot.append(ib)
                icount += 1

    icount = comm.bcast(icount)
    ind_plot = comm.bcast(ind_plot)
    arrays['ind_plot'] = ind_plot

    Sj = arrays['Sj']
    snktot = arrays['v_k'].shape[0]
    sktxtaux = np.zeros((snktot, 3, nawf, nawf), dtype=complex)

    # Compute matrix elements of the spin operator
    for ik in range(snktot):
        for l in range(3):
            sktxtaux[ik, l, :, :] = (
                np.conj(arrays['v_k'][ik, :, :, 0].T)
                .dot(Sj[l, :, :])
                .dot(arrays['v_k'][ik, :, :, 0])
            )

    sktxtaux = np.take(np.diagonal(sktxtaux, axis1=2, axis2=3), ind_plot, axis=2)
    sktxt = gather_full(np.ascontiguousarray(sktxtaux), attributes['npool'])
    sktxtaux = None

    if rank == 0:
        if 'kq' in arrays and E_k_full.shape[0] == arrays['kq'].shape[1]:
            f = open(os.path.join(attributes['opath'], 'spin-texture-bands' + '.dat'), 'w')
            for ik in range(E_k_full.shape[0]):
                for ib in range(icount):
                    idx = ind_plot[ib]
                    f.write(
                        '\t'.join(
                            ['%d' % ik]
                            + ['% 5.8f' % E_k_full[ik, idx]]
                            + ['% 5.8f' % j for j in sktxt[ik, :, ib].real]
                        )
                        + '\n'
                    )
                f.write('\n')
            f.close()
        else:
            sktxt = np.reshape(sktxt, (nk1, nk2, nk3, 3, icount), order='C')
            for ib in range(icount):
                np.savez(
                    os.path.join(attributes['opath'], 'spin_text_band_' + str(ib)),
                    spinband=sktxt[:, :, :, :, ib],
                )

    arrays['sktxt'] = sktxt
    sktxt = None
    E_k_full = None
