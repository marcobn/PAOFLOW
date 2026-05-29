def write2bxsf4skeaf(data_controller, bands, nbnd, indices):
    """Write per-band BXSF files in SKEAF-compatible format.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``b_vectors`` (shape ``(3, 3)``).
        Required attributes: ``alat``, ``nk1``, ``nk2``, ``nk3``,
        ``workpath``, ``outputdir``.
    bands : np.ndarray, shape ``(nk1, nk2, nk3, nbnd)``
        Band eigenvalues on the 3-D k-grid (in eV).
    nbnd : int
        Number of bands to write; one BXSF file is created per band.
    indices : Optional[array_like of int]
        1-D array of included band indices (1-based in output).  If
        ``None``, indices default to zero.

    Returns
    -------
    None
        Creates ``nbnd`` files named ``Fermi_surf_band_{ib+1}.bxsf``
        inside ``{workpath}/{outputdir}/``.

    Notes
    -----
    Unlike :func:`write2bxsf`, this function writes one BXSF file per
    band to ensure compatibility with SKEAF (Supercell K-space Extremal
    Area Finder).  Band energies are converted from eV to Rydberg
    (``Ryd_conv = 0.0734986176``) and the Fermi energy is set to zero.
    Reciprocal-lattice vectors are written in units of :math:`1/a_{\\rm lat}`
    (without the :math:`2\\pi` factor required by SKEAF).
    """
    from os.path import join

    import numpy as np

    arrays, attributes = data_controller.data_dicts()

    x0 = np.zeros(3, dtype=float)
    if indices is None:
        indices = np.zeros(nbnd, dtype=float)

    Efermi = 0.0
    Ryd_conv = 0.0734986176
    Ryd_bands = np.zeros_like(bands)
    Ryd_bands[:, :, :, :] = Ryd_conv * bands[:, :, :, :]
    alat, b_vectors = attributes['alat'], arrays['b_vectors']
    nx, ny, nz = attributes['nk1'], attributes['nk2'], attributes['nk3']

    for ib in range(nbnd):
        with open(
            join(
                attributes['workpath'],
                attributes['outputdir'],
                'Fermi_surf_band_%d.bxsf' % (ib + 1),
            ),
            'w',
        ) as f:
            f.write('\nBEGIN_INFO\n  Fermi Energy: {:15.9f}\nEND_INFO\n'.format(Efermi))
            # BXSF scalar-field header
            f.write('\nBEGIN_BLOCK_BANDGRID_3D\nband_energies\nBANDGRID_3D_BANDS\n')
            # number of points in each direction
            f.write('{:12d}\n'.format(1))
            f.write('{:12d}{:12d}{:12d}\n'.format(nx + 1, ny + 1, nz + 1))
            # origin (should be zero, if I understan correctly)
            f.write('  {}\n'.format(''.join('%10.6f' % F for F in x0)))
            # 1st spanning (=lattice) vector
            f.write('  {}\n'.format(''.join('%10.6f' % F for F in b_vectors[0] / alat)))
            # 2nd spanning (=lattice) vector
            f.write('  {}\n'.format(''.join('%10.6f' % F for F in b_vectors[1] / alat)))
            # 3rd spanning (=lattice) vector
            f.write('  {}\n'.format(''.join('%10.6f' % F for F in b_vectors[2] / alat)))

            f.write('  BAND: {:5d}\n'.format(int(indices[ib]) + 1))
            combined_band = []
            for ix in range(nx):
                for iy in range(ny):
                    for F in Ryd_bands[ix, iy, :, ib]:
                        combined_band.append(F)
                    combined_band.append((Ryd_bands[ix, iy, 0, ib]))
                for F in Ryd_bands[ix, 0, :, ib]:
                    combined_band.append(F)
                combined_band.append(Ryd_bands[ix, 0, 0, ib])
            for iy in range(ny):
                for F in Ryd_bands[0, iy, :, ib]:
                    combined_band.append(F)
                combined_band.append((Ryd_bands[0, iy, 0, ib]))
            for F in Ryd_bands[0, 0, :, ib]:
                combined_band.append(F)
            combined_band.append((Ryd_bands[0, 0, 0, ib]))
            for i in range(len(combined_band)):
                if (i + 1) % 6 == 0:
                    f.write('{:15.9f}\n'.format(combined_band[i]))
                else:
                    f.write('{:15.9f}'.format(combined_band[i]))
            f.write('\nEND_BANDGRID_3D\nEND_BLOCK_BANDGRID_3d\n')
            f.close()
