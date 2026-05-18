def get_K_grid_fft(data_controller):
    """Build the Cartesian k-grid and uniform k-point weights from the FFT mesh.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``b_vectors`` (shape ``(3, 3)``).
        Required attributes: ``nk1``, ``nk2``, ``nk3``.

    Returns
    -------
    None
        Adds the following entries to ``data_controller.data_arrays``:

        - ``kgrid`` : np.ndarray, shape ``(3, nktot)`` — Cartesian k-coordinates
          (in units of :math:`2\\pi/a`) of all ``nktot = nk1*nk2*nk3`` grid
          points, centred around :math:`\\Gamma` (folded into :math:`[-0.5, 0.5)`).
        - ``kq_wght`` : np.ndarray, shape ``(nktot,)`` — uniform k-point weights,
          each equal to ``1 / nktot``.
    """
    import numpy as np

    arrays, attributes = data_controller.data_dicts()

    nk1 = attributes['nk1']
    nk2 = attributes['nk2']
    nk3 = attributes['nk3']
    b_vectors = arrays['b_vectors']

    nktot = nk1 * nk2 * nk3
    arrays['kq_wght'] = np.ones((nktot), dtype=float) / nktot
    #  return

    ### Not used
    Kint = np.zeros((3, nktot), dtype=float)
    idk = np.zeros((nk1, nk2, nk3), dtype=int)

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j * nk3 + i * nk2 * nk3
                Rx = float(i) / float(nk1)
                Ry = float(j) / float(nk2)
                Rz = float(k) / float(nk3)
                if Rx >= 0.5:
                    Rx = Rx - 1.0
                if Ry >= 0.5:
                    Ry = Ry - 1.0
                if Rz >= 0.5:
                    Rz = Rz - 1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)
                idk[i, j, k] = n
                Kint[:, n] = Rx * b_vectors[0, :] + Ry * b_vectors[1, :] + Rz * b_vectors[2, :]
    arrays['kgrid'] = Kint
    return


def get_K_grid_fft_crystal(nk1, nk2, nk3):
    """Build the k-grid in reduced crystal coordinates.

    Parameters
    ----------
    nk1 : int
        Number of k-points along the first reciprocal-lattice vector.
    nk2 : int
        Number of k-points along the second reciprocal-lattice vector.
    nk3 : int
        Number of k-points along the third reciprocal-lattice vector.

    Returns
    -------
    np.ndarray, shape ``(nk1*nk2*nk3, 3)``, float
        k-coordinates in reduced crystal units, centred around
        :math:`\\Gamma` (each component folded into :math:`[-0.5, 0.5)`).
    """
    import numpy as np

    nktot = nk1 * nk2 * nk3

    Kint = np.zeros((nktot, 3), dtype=float)

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j * nk3 + i * nk2 * nk3
                Rx = float(i) / float(nk1)
                Ry = float(j) / float(nk2)
                Rz = float(k) / float(nk3)
                if Rx >= 0.5:
                    Rx = Rx - 1.0
                if Ry >= 0.5:
                    Ry = Ry - 1.0
                if Rz >= 0.5:
                    Rz = Rz - 1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)

                Kint[n] = Rx, Ry, Rz

    return Kint
