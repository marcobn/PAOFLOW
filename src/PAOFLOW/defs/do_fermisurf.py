def do_fermisurf(data_controller):
    from os.path import join

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
