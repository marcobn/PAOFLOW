def write2bxsf(data_controller, fname, bands, nbnd, indices, fermi_up, fermi_dw):
    """Write Fermi-surface band data to a BXSF file for visualisation.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``b_vectors`` (shape ``(3, 3)``).
        Required attributes: ``alat``, ``nk1``, ``nk2``, ``nk3``,
        ``Efermi``, ``workpath``, ``outputdir``.
    fname : str
        Output filename (written inside ``{workpath}/{outputdir}/``).
    bands : np.ndarray, shape ``(nk1, nk2, nk3+1, nbnd)``
        Band eigenvalues on the 3-D k-grid (in eV).
    nbnd : int
        Number of bands to write.
    indices : Optional[array_like of int]
        1-D array of included band indices (1-based in output).  If
        ``None``, indices default to zero.
    fermi_up : float
        Upper boundary of the Fermi window (eV), written to the file header.
    fermi_dw : float
        Lower boundary of the Fermi window (eV), written to the file header.

    Returns
    -------
    None
        Creates the BXSF file at ``{workpath}/{outputdir}/{fname}``.

    Notes
    -----
    The BXSF format (XCrySDen Band-grid XSF) stores band eigenvalues on a
    uniform 3-D k-grid together with the reciprocal lattice vectors and the
    Fermi energy.  It is read by XCrySDen and compatible Fermi-surface
    visualisation tools.  The grid is written with periodic wrap-around:
    one extra point is appended in each direction by repeating the first
    row/column/plane, yielding an ``(nk1+1, nk2+1, nk3+1)`` grid.
    Reciprocal-lattice vectors are output in units of :math:`2\\pi / a_{\\rm lat}`.
    """
    from os.path import join

    import numpy as np

    pi = np.pi
    arrays, attributes = data_controller.data_dicts()

    x0 = np.zeros(3, dtype=float)
    if indices is None:
        indices = np.zeros(nbnd, dtype=float)

    Efermi = attributes['Efermi']
    alat, b_vectors = attributes['alat'], arrays['b_vectors']
    nx, ny, nz = attributes['nk1'], attributes['nk2'], attributes['nk3']

    with open(join(attributes['workpath'], attributes['outputdir'], '{0}'.format(fname)), 'w') as f:
        f.write(
            '\nBEGIN_INFO\n  Fermi Energy: {:15.9f}\n  Shift Range: {:12.9f}eV to{:12.9f}eV\nEND_INFO\n'.format(
                Efermi, fermi_dw, fermi_up
            )
        )
        # BXSF scalar-field header
        f.write('\nBEGIN_BLOCK_BANDGRID_3D\nband_energies\nBEGIN_BANDGRID_3D_BANDS\n')
        # number of points in each direction
        f.write('{:12d}\n'.format(nbnd))
        f.write('{:12d}{:12d}{:12d}\n'.format(nx + 1, ny + 1, nz + 1))
        # origin (should be zero, if I understan correctly)
        f.write('  {}\n'.format(''.join('%10.6f' % F for F in x0)))
        # 1st spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f' % F for F in b_vectors[0] * 2 * pi / alat)))
        # 2nd spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f' % F for F in b_vectors[1] * 2 * pi / alat)))
        # 3rd spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f' % F for F in b_vectors[2] * 2 * pi / alat)))

        for ib in range(nbnd):
            f.write('  BAND: {:5d}\n'.format(int(indices[ib]) + 1))
            for ix in range(nx):
                for iy in range(ny):
                    f.write('    {}'.format(''.join('%15.9f' % F for F in bands[ix, iy, :, ib])))
                    f.write('{:15.9f}\n'.format(bands[ix, iy, 0, ib]))
                f.write('    {}'.format(''.join('%15.9f' % F for F in bands[ix, 0, :, ib])))
                f.write('{:15.9f}\n'.format(bands[ix, 0, 0, ib]))
            for iy in range(ny):
                f.write('    {}'.format(''.join('%15.9f' % F for F in bands[0, iy, :, ib])))
                f.write('{:15.9f}\n'.format(bands[0, iy, 0, ib]))
            f.write('    {}'.format(''.join('%15.9f' % F for F in bands[0, 0, :, ib])))
            f.write('{:15.9f}\n'.format(bands[0, 0, 0, ib]))
        f.write('END_BANDGRID_3D\nEND_BLOCK_BANDGRID_3d\n')
