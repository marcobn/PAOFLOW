def do_fermisurf(data_controller):
    """Identify Fermi-surface bands and write BXSF and NPZ output files.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required array: ``E_k`` (shape ``(nkpnts, nawf, nspin)``), distributed
        over MPI pools.
        Required attributes: ``nawf``, ``nk1``, ``nk2``, ``nk3``, ``nspin``,
        ``fermi_up``, ``fermi_dw``, ``npool``, ``opath``, ``verbose``.

    Returns
    -------
    None
        On rank 0, writes the following output files for each spin channel
        ``ispin``:

        - ``FermiSurf_{ispin}.bxsf``: BXSF file containing all bands that
          intersect the Fermi window ``[fermi_dw, fermi_up]``.
        - ``Fermi_surf_band_{ib}_{ispin}.npz``: compressed NumPy archive with
          the band eigenvalues on the full 3-D k-grid (key ``nameband``,
          shape ``(nk1, nk2, nk3)``).

    Notes
    -----
    A band is included in the Fermi surface if at least one k-point lies
    below ``fermi_up`` and at least one lies above ``fermi_dw``, i.e. the
    band energy range overlaps the window ``[fermi_dw, fermi_up]``.  This
    covers three cases:

    - the band crosses ``fermi_up`` from below,
    - the band crosses ``fermi_dw`` from above, or
    - the band lies entirely within the window.

    The gathered eigenvalue array is reshaped to
    ``(nk1, nk2, nk3, nawf, nspin)`` before the BXSF writer is called.
    Only MPI rank 0 performs the band selection, the file I/O, and the NPZ
    saves; all ranks synchronise at the end via ``MPI.Barrier``.
    """
    import numpy as np
    from mpi4py import MPI

    from .communication import gather_full

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arry, attr = data_controller.data_dicts()

    # maximum number of bands crossing fermi surface
    ###### PARALLELIZATION
    E_kf = gather_full(arry['E_k'], attr['npool'])

    if rank == 0:
        if attr['verbose']:
            print('Writing bxsf file for Fermi Surface')

        nawf = attr['nawf']
        nk1, nk2, nk3 = attr['nk1'], attr['nk2'], attr['nk3']
        fermi_up, fermi_dw = attr['fermi_up'], attr['fermi_dw']

        E_ks = np.reshape(E_kf, (nk1, nk2, nk3, nawf, attr['nspin']))

        for ispin in range(attr['nspin']):
            ind_plot = []
            eigband = []

            for ib in range(nawf):
                E_k_min = np.amin(E_kf[:, ib, ispin])
                E_k_max = np.amax(E_kf[:, ib, ispin])
                btwUp = E_k_min < fermi_up and E_k_max > fermi_up
                btwDwn = E_k_min < fermi_dw and E_k_max > fermi_dw
                btwUaD = E_k_min > fermi_dw and E_k_max < fermi_up
                if btwUp or btwDwn or btwUaD:
                    ind_plot.append(ib)
                    eigband.append(E_ks[:, :, :, ib, ispin])

            feig = 'FermiSurf_%d.bxsf' % ispin
            eigband = np.array(eigband)
            data_controller.write_bxsf(
                feig, np.moveaxis(eigband, 0, 3), len(ind_plot), indices=ind_plot
            )

            for i, ib in enumerate(eigband):
                np.savez(
                    join(attr['opath'], 'Fermi_surf_band_%d_%d' % (ind_plot[i], ispin)), nameband=ib
                )

    comm.Barrier()
    E_kf = E_ks = None
