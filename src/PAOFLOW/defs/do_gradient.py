def do_gradient(data_controller):
    """Compute the first-order k-space gradient :math:`dH/dk` of the Hamiltonian via FFT.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``Hksp`` (shape ``(snawf, nk1, nk2, nk3, nspin)``),
        ``Rfft`` (shape ``(nk1, nk2, nk3, 3)``), ``Dnm`` (shape
        ``(nawf*nawf, 3)``).
        Required attributes: ``nawf``, ``nk1``, ``nk2``, ``nk3``, ``nspin``,
        ``alat``, ``npool``, ``use_cuda``.

    Returns
    -------
    None
        Adds the following key to ``data_controller.data_arrays``:

        - ``dHksp`` : np.ndarray, shape ``(snawf, nk1, nk2, nk3, 3, nspin)``,
          complex — the Cartesian gradient of the k-space Hamiltonian
          :math:`dH(\\mathbf{k})/dk_l` for each orbital pair and Cartesian
          direction :math:`l = 0, 1, 2`.

        The in-place computation also overwrites ``Hksp`` with
        :math:`H(\\mathbf{R}) \\cdot i \\cdot a_{\\text{lat}}` (the intermediate
        real-space representation).

    Notes
    -----
    The gradient is computed in two stages for each orbital index ``n`` and
    spin channel:

    1. **Real-space transformation**: ``Hksp[n]`` is replaced by
       :math:`\\mathcal{F}^{-1}[H(\\mathbf{k})] \\cdot i \\cdot a_{\\text{lat}}`
       using either a CUDA or SciPy inverse FFT.

    2. **Gradient**: for each Cartesian direction :math:`l`,

       .. math::

           dH(\\mathbf{k})/dk_l
               = \\mathcal{F}\\left[ R_l \\cdot H(\\mathbf{R}) \\right]
               + i \\cdot H(\\mathbf{k}) \\cdot D^{nm}_l

       where :math:`R_l` is the :math:`l`-th component of the real-space grid
       ``Rfft`` and :math:`D^{nm}_l` is a diagonal tight-binding correction
       term from ``Dnm``.

    The FFT grid in real space is constructed by :func:`get_R_grid_fft`.
    The distributed array ``Dnm`` is scattered across pools by
    :func:`scatter_full`.
    """
    import numpy as np
    from scipy import fftpack as FFT

    from .communication import scatter_full
    from .get_R_grid_fft import get_R_grid_fft

    arry, attr = data_controller.data_dicts()

    if attr['use_cuda']:
        from .cuda_fft import cuda_ifftn

    # ----------------------
    # Compute the gradient of the k-space Hamiltonian
    # ----------------------

    snawf, nk1, nk2, nk3, nspin = arry['Hksp'].shape

    # fft grid in R shifted to have (0,0,0) in the center
    get_R_grid_fft(data_controller, nk1, nk2, nk3)

    #  dHaux = np.empty((nk1*nk2*nk3,3), dtype=complex, order='C')
    #  Haux = np.empty((nk1,nk2,nk3), dtype=complex, order='C')
    arry['dHksp'] = np.empty((snawf, nk1, nk2, nk3, 3, nspin), dtype=complex, order='C')
    Dnm = scatter_full(
        np.reshape(arry['Dnm'], (attr['nawf'] * attr['nawf'], 3), order='C'), attr['npool']
    )
    #  kdot = np.zeros((1,arry['R'].shape[0]), dtype=complex, order='C')
    #  kdot = np.tensordot(arry['R'], -2.0j*np.pi*arry['kgrid'], axes=([1],[0]))
    #  kdot = np.exp(kdot, kdot)
    #  kdoti = np.zeros((1,arry['R'].shape[0]), dtype=complex, order='C')
    #  kdoti = np.tensordot(arry['R'], 2.0j*np.pi*arry['kgrid'], axes=([1],[0]))
    #  kdoti = np.exp(kdoti, kdoti)
    for ispin in range(nspin):
        for n in range(snawf):
            Haux = arry['Hksp'][n, :, :, :, ispin].copy()
            ########################################
            ### real space grid replaces k space ###
            ########################################
            if attr['use_cuda']:
                arry['Hksp'][n, :, :, :, ispin] = (
                    cuda_ifftn(arry['Hksp'][n, :, :, :, ispin]) * 1.0j * attr['alat']
                )
            else:
                arry['Hksp'][n, :, :, :, ispin] = (
                    FFT.ifftn(arry['Hksp'][n, :, :, :, ispin]) * 1.0j * attr['alat']
                )
                # HRaux = arry['Hksp'][n,:,:,:,ispin].reshape(attr['nk1']*attr['nk2']*attr['nk3'], order='C')
                # HRaux = np.tensordot(HRaux, kdoti, axes=([0],[0]))/(attr['nk1']*attr['nk2']*attr['nk3'])
                # arry['Hksp'][n,:,:,:,ispin] =  HRaux.reshape((nk1,nk2,nk3), order='C')*1.0j*attr['alat']

            # Compute R*H(R) + diagonal TB correction
            for l in range(3):
                arry['dHksp'][n, :, :, :, l, ispin] = (
                    FFT.fftn(arry['Rfft'][:, :, :, l] * arry['Hksp'][n, :, :, :, ispin])
                    + 1j * Haux[:, :, :] * Dnm[n, l]
                )

            # HRaux = arry['Hksp'][n,:,:,:,ispin].reshape(attr['nk1']*attr['nk2']*attr['nk3'], order='C')
            # for l in range(3):
            #   dHaux = np.tensordot(HRaux, arry['R'][:,l]*kdot, axes=([0],[1]))
            #   arry['dHksp'][n,:,:,:,l,ispin] =  dHaux.reshape((nk1,nk2,nk3), order='C')
