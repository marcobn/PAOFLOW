def site_projeted_bands(data_controller):
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

    arry, attr = data_controller.data_dicts()

    mask = site_mask(arry['naw'], arry['site_proj'], attr['nawf'], attr['do_spin_orbit'])

    nk_local, _, nbands, nspin = arry['v_k'].shape
    weights = np.zeros((nk_local, nbands, nspin), dtype=float)
    for ispin in range(nspin):
        for k in range(nk_local):
            weights[k, :, ispin] = site_weights_k(arry['v_k'][k, :, :, ispin], mask)

    write_site_projected_bands(data_controller, arry['E_k'], weights)


def site_mask(naw, site_proj, nawf: int, do_spin_orbit: bool):
    """Orbital mask of the projected sites: ``1 + 1j`` on their orbitals, 0 elsewhere.

    Notes
    -----
    The complex unit factor is what :func:`site_projeted_bands` has always
    multiplied the eigenvectors by, so the weights it writes are twice the
    orbital populations; kept for the files' sake.  With ad-hoc spin-orbit
    (``do_spin_orbit``) the spin-down copy, ``nawf / 2`` further on, is
    included.
    """
    import numpy as np

    mask = np.zeros(nawf, dtype=complex)
    s = 0
    # Only used if ad-hoc SOC
    if do_spin_orbit:
        s = int(nawf / 2)
    for i in range(site_proj.shape[0]):
        idx = np.sum(naw[0 : site_proj[i]])
        fdx = idx + naw[site_proj[i]]
        mask[idx:fdx] = complex(1.0, 1.0)
        mask[idx + s : fdx + s] = complex(1.0, 1.0)  # Only used if ad-hoc SOC
    return mask


def site_weights_k(v_k, mask):
    """Weight of every state of one k-point on the masked orbitals.

    Parameters
    ----------
    v_k : np.ndarray, shape ``(nawf, nbands)``
        Eigenvectors (columns).
    mask : np.ndarray, shape ``(nawf,)``
        From :func:`site_mask`.

    Returns
    -------
    np.ndarray, shape ``(nbands,)``
    """
    import numpy as np

    cs = np.multiply(mask[:, None], v_k)
    return np.array([np.sum(np.absolute(np.square(cs[:, i]))) for i in range(cs.shape[1])])


def write_site_projected_bands(data_controller, E_k, weights) -> None:
    """Gather and write ``site-projected-bands_<ispin>.dat`` (rank 0).

    Parameters
    ----------
    E_k, weights : np.ndarray, shape ``(nk_local, nbands, nspin)``
        This rank's path energies and :func:`site_weights_k`; gathered here
        (the files used to be written from the local slice, so only a
        serial run was complete).
    """
    from os.path import join

    import numpy as np
    from mpi4py import MPI

    from ..utils.communication import gather_full

    arry, attr = data_controller.data_dicts()
    nbands = weights.shape[1]
    E_k = gather_full(np.ascontiguousarray(E_k[:, :nbands, :]), attr['npool'])
    weights = gather_full(weights, attr['npool'])
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    nkpi = arry['kq'].shape[1]
    for ispin in range(weights.shape[2]):
        f = open(join(attr['opath'], 'site-projected-bands_' + str(ispin) + '.dat'), 'w')
        for i in range(nbands):
            for k in range(nkpi):
                f.write(
                    ''.join(['%s %s %s\n' % (k, float(E_k[k, i, ispin]), weights[k, i, ispin])])
                )
        f.close()
