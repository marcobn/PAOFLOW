def do_fermisurf(data_controller, type, project):
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
    from os.path import join
    from ..utils.communication import gather_full

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arry, attr = data_controller.data_dicts()

    # maximum number of bands crossing fermi surface
    ###### PARALLELIZATION
    E_kf = gather_full(arry['E_k'], attr['npool'])
    if rank == 0:
        if attr['verbose']:
            print('Writing file for Fermi Surface :)')

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
            if type == 'bxsf':
                feig = 'FermiSurf_%d.bxsf' % ispin
                eigband = np.array(eigband)
                data_controller.write_bxsf(
                    feig, np.moveaxis(eigband, 0, 3), len(ind_plot), indices=ind_plot
                )
            elif type == 'fermisurfer':
                feig = f'FermiSurf_{ispin}.frmsf'

                eigband = np.array(eigband)

                bands4d = np.moveaxis(eigband, 0, 3)

                if project == 'velocity':
                    projection = Fermi_vel(data_controller, bands4d)
                elif project == 'orbital':
                    FS_orb = arry['FS_orb']
                    FS_orb = FS_orb.reshape(nk1, nk2, nk3, nawf, attr['nspin'])
                    projection = FS_orb[:, :, :, :, ispin]
                elif project == 'spin_Sx' or project == 'spin_Sy' or project == 'spin_Sz':
                    spintext = arry['sktxt']
                    spin_component = {
                        'spin_Sx': 0,
                        'spin_Sy': 1,
                        'spin_Sz': 2,
                    }
                    projection = spintext[:, :, :, spin_component[project], :].real
                elif project == 'omega':  ########TBI
                    berry = arry['Om_znk_fermi']
                    print(berry.shape)
                    pass  # TBI
                else:
                    projection = None
                write_frmsf(data_controller, feig, bands4d, projection)
            for i, ib in enumerate(eigband):
                np.savez(
                    join(attr['opath'], 'Fermi_surf_band_%d_%d' % (ind_plot[i], ispin)), nameband=ib
                )

    comm.Barrier()
    E_kf = E_ks = None


def write_frmsf(data_controller, fname, bands, projection):
    import numpy as np
    from os.path import join

    arrays, attributes = data_controller.data_dicts()

    alat = attributes['alat']
    nk1 = attributes['nk1']
    nk2 = attributes['nk2']
    nk3 = attributes['nk3']

    b_vectors = arrays['b_vectors']

    nbnd = bands.shape[3]

    b1 = b_vectors[0] * 2.0 * np.pi / alat
    b2 = b_vectors[1] * 2.0 * np.pi / alat
    b3 = b_vectors[2] * 2.0 * np.pi / alat

    with open(join(attributes['opath'], fname), 'w') as fo:
        fo.write(f'{nk1} {nk2} {nk3}\n')
        fo.write('1\n')
        fo.write(f'{nbnd}\n')

        fo.write(f'{b1[0]} {b1[1]} {b1[2]}\n')
        fo.write(f'{b2[0]} {b2[1]} {b2[2]}\n')
        fo.write(f'{b3[0]} {b3[1]} {b3[2]}\n')

        # energies already shifted by Ef
        for ib in range(nbnd):
            for ik1 in range(nk1):
                for ik2 in range(nk2):
                    for ik3 in range(nk3):
                        fo.write(f'{bands[ik1,ik2,ik3,ib]:.8f}\n')
        if projection is not None:
            for ib in range(nbnd):
                for ik1 in range(nk1):
                    for ik2 in range(nk2):
                        for ik3 in range(nk3):
                            fo.write(f'{projection[ik1,ik2,ik3,ib]:.8e}\n')


def Fermi_vel(data_controller, bands):
    import numpy as np

    arrays, attributes = data_controller.data_dicts()

    alat = attributes['alat']
    nk1 = attributes['nk1']
    nk2 = attributes['nk2']
    nk3 = attributes['nk3']

    b_vectors = arrays['b_vectors']

    nbnd = bands.shape[3]

    b1 = b_vectors[0] * 2.0 * np.pi / alat
    b2 = b_vectors[1] * 2.0 * np.pi / alat
    b3 = b_vectors[2] * 2.0 * np.pi / alat
    B = np.column_stack((b1, b2, b3))
    dk1 = np.linalg.norm(b1) / nk1
    dk2 = np.linalg.norm(b2) / nk2
    dk3 = np.linalg.norm(b3) / nk3
    vx = np.empty_like(bands)
    vy = np.empty_like(bands)
    vz = np.empty_like(bands)

    for ib in range(nbnd):
        vx[:, :, :, ib] = np.gradient(bands[:, :, :, ib], dk1, axis=0, edge_order=2)

        vy[:, :, :, ib] = np.gradient(bands[:, :, :, ib], dk2, axis=1, edge_order=2)

        vz[:, :, :, ib] = np.gradient(bands[:, :, :, ib], dk3, axis=2, edge_order=2)

    hbarAng = 6.582119569e-6

    grad_frac = np.stack((vx, vy, vz), axis=-1)
    grad_cart = grad_frac @ np.linalg.inv(B)

    return np.linalg.norm(grad_cart, axis=-1) / hbarAng
