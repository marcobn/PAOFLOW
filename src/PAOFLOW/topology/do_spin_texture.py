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
    import numpy as np

    from ..utils.communication import gather_full
    from .texture import fermi_window_bands, texture_k, write_texture

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    E_k_full = gather_full(arrays['E_k'], attributes['npool'])
    ind_plot = fermi_window_bands(data_controller, E_k_full)

    Sj = arrays['Sj']
    snktot = arrays['v_k'].shape[0]
    txtaux = np.zeros((snktot, 3, arrays['v_k'].shape[2]), dtype=complex)

    # Compute matrix elements of the spin operator
    for ik in range(snktot):
        txtaux[ik] = texture_k(arrays['v_k'][ik, :, :, 0], Sj)

    txtaux = np.take(txtaux, ind_plot, axis=2)
    arrays['sktxt'] = write_texture(data_controller, 'spin', E_k_full, txtaux, ind_plot)
    E_k_full = None
