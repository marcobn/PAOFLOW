def get_R_grid_fft(data_controller, nr1, nr2, nr3):
    """Build the real-space lattice grid for FFT-based Hamiltonian operations.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays``.
        Required array: ``a_vectors`` (shape ``(3, 3)``), the primitive
        lattice vectors in units of the lattice constant.
    nr1 : int
        Number of grid points along the first lattice vector.
    nr2 : int
        Number of grid points along the second lattice vector.
    nr3 : int
        Number of grid points along the third lattice vector.

    Returns
    -------
    None
        Adds the following keys to ``data_controller.data_arrays``:

        - ``R`` : np.ndarray, shape ``(nr1*nr2*nr3, 3)`` — Cartesian
          coordinates of each real-space grid point in units of the lattice
          constant, centred around the origin (i.e. components folded into
          :math:`[-0.5, 0.5)`).
        - ``Rfft`` : np.ndarray, shape ``(nr1, nr2, nr3, 3)`` — same vectors
          on the 3-D grid layout, used as multiplicative factors in the FFT
          gradient and curvature routines.
        - ``idx`` : np.ndarray, shape ``(nr1, nr2, nr3)``, int — linear
          index mapping the 3-D grid position to a row in ``R``.
        - ``R_wght`` : np.ndarray, shape ``(nr1*nr2*nr3,)`` — uniform
          weights, all set to ``1.0``.

    Notes
    -----
    Grid coordinates are generated in reduced (crystal) units as
    :math:`(i/nr_1, j/nr_2, k/nr_3)` and folded into :math:`[-0.5, 0.5)`
    by subtracting 1 when the component is :math:`\\geq 0.5`.  The Cartesian
    position is then

    .. math::

        \\mathbf{R}_{ijk} = \\tilde{R}_x \\, nr_1 \\, \\mathbf{a}_1
            + \\tilde{R}_y \\, nr_2 \\, \\mathbf{a}_2
            + \\tilde{R}_z \\, nr_3 \\, \\mathbf{a}_3

    where :math:`\\tilde{R}_{x,y,z}` are the folded reduced coordinates
    and :math:`\\mathbf{a}_{1,2,3}` are the rows of ``a_vectors``.
    """
    import numpy as np

    arrays = data_controller.data_arrays

    nrtot = nr1 * nr2 * nr3

    a_vectors = arrays['a_vectors']

    arrays['R'] = np.zeros((nrtot, 3), dtype=float)
    arrays['idx'] = np.zeros((nr1, nr2, nr3), dtype=int)
    arrays['Rfft'] = np.zeros((nr1, nr2, nr3, 3), dtype=float)
    arrays['R_wght'] = np.ones((nrtot), dtype=float)

    for i in range(nr1):
        for j in range(nr2):
            for k in range(nr3):
                n = k + j * nr3 + i * nr2 * nr3
                Rx = float(i) / float(nr1)
                Ry = float(j) / float(nr2)
                Rz = float(k) / float(nr3)
                if Rx >= 0.5:
                    Rx = Rx - 1.0
                if Ry >= 0.5:
                    Ry = Ry - 1.0
                if Rz >= 0.5:
                    Rz = Rz - 1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)

                # evec = np.array([[1,0,0],[0,1,0],[0,0,1]])*attributes['alat']
                # arrays['R'][n,:] = Rx*nr1*evec[0,:] + Ry*nr2*evec[1,:] + Rz*nr3*evec[2,:]
                arrays['R'][n, :] = (
                    Rx * nr1 * a_vectors[0, :]
                    + Ry * nr2 * a_vectors[1, :]
                    + Rz * nr3 * a_vectors[2, :]
                )
                arrays['Rfft'][i, j, k, :] = arrays['R'][n, :]
                arrays['idx'][i, j, k] = n
