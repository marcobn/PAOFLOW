def do_gradient(data_controller):
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
