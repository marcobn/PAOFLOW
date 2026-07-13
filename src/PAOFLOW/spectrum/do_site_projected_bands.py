def site_projeted_bands(data_controller, type):
    """Write site-projected band weights to a data file.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``v_k`` (shape ``(nkpi, nawf, bnd, nspin)``),
        ``E_k`` (shape ``(nkpi, bnd, nspin)``), ``site_proj``
        (1-D int array of site indices to project onto), ``naw``
        (number of atomic orbitals per atom).
        Required attributes: ``nawf``, ``do_spin_orbit``.
        Attribute ``opath`` is used for output.

    Returns
    -------
    None
        Writes one text file per spin channel to
        ``{opath}/site-projected-bands_{ispin}.dat``.
        Each line contains three whitespace-separated values:
        k-point index (int), band energy (float, eV), and the
        site-projected spectral weight (float).

    Notes
    -----
    For each selected site in ``site_proj``, the orbital indices
    :math:`[\\text{idx}, \\text{fdx})` belonging to that site are identified
    using the cumulative sum of ``naw``.  A complex mask is applied to the
    eigenvector array ``v_k`` to retain only those orbital components, and
    the spectral weight is computed as :math:`\\sum_\\mu |v^\\mu_{nk}|^2`.

    When ``do_spin_orbit`` is enabled, the spin-orbit-doubled basis is
    accounted for by also including the upper spin sector
    :math:`[\\text{idx}+s, \\text{fdx}+s)` where :math:`s = N_{\\text{wf}}/2`.
    """
    import numpy as np
    from os.path import join

    arry, attr = data_controller.data_dicts()

    nkpi = arry['kq'].shape[1]
    mask = np.zeros_like(arry['v_k'][:, :, :, 0])
    cs = np.zeros_like(arry['v_k'][:, :, :, 0])

    s = 0
    # Only used if ad-hoc SOC
    if attr['do_spin_orbit']:
        s = int(attr['nawf'] / 2)

    nspin = arry['v_k'].shape[3]
    if type == 'BZ_path':
        for ispin in range(nspin):
            f = open(join(attr['opath'], 'site-projected-bands_' + str(ispin) + '.dat'), 'w')

            for i in range(arry['site_proj'].shape[0]):
                idx = np.sum(arry['naw'][0 : arry['site_proj'][i]])
                fdx = idx + arry['naw'][arry['site_proj'][i]]

                mask[:, idx:fdx, :] = complex(1.0, 1.0)
                mask[:, idx + s : fdx + s, :] = complex(1.0, 1.0)  # Only used if ad-hoc SOC

                cs[:, :, :] = np.multiply(mask[:, :, :], arry['v_k'][:, :, :, ispin])

            for i in range(attr['nawf']):
                for k in range(nkpi):
                    f.write(
                        ''.join(
                            [
                                '%s %s %s\n'
                                % (
                                    k,
                                    float(arry['E_k'][k, i, ispin]),
                                    np.sum(np.absolute(np.square((cs[k, :, i])))),
                                )
                            ]
                        )
                    )
            f.close()
    elif type == 'BZ_mesh':
        arry['FS_orb'] = np.zeros_like(arry['E_k'], dtype=float)
        for ispin in range(nspin):
            mask.fill(0)

            for i in range(arry['site_proj'].shape[0]):
                idx = np.sum(arry['naw'][: arry['site_proj'][i]])
                fdx = idx + arry['naw'][arry['site_proj'][i]]

                mask[:, idx:fdx, :] = 1.0 + 0.0j
                mask[:, idx + s : fdx + s, :] = 1.0 + 0.0j

            cs = mask * arry['v_k'][:, :, :, ispin]

            arry['FS_orb'][:, :, ispin] = np.sum(np.abs(cs) ** 2, axis=1)
