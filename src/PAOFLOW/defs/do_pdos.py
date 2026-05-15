import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_pdos(data_controller, emin, emax, ne, delta):
    arrays, attributes = data_controller.data_dicts()

    nawf = attributes['nawf']
    nspin = attributes['nspin']
    nktot = attributes['nkpnts']

    # PDOS calculation with gaussian smearing

    emax = np.amin(np.array([attributes['shift'], emax]))
    ene = np.linspace(emin, emax, ne)

    for ispin in range(nspin):
        pdosaux = np.zeros((nawf, ne), dtype=float)
        v_kaux = np.real(np.abs(arrays['v_k'][:, :, :, ispin]) ** 2)

        E_k = arrays['E_k'][:, :, ispin]

        for n in range(ne):
            taux = np.exp(-(((ene[n] - E_k) / delta) ** 2)) / np.sqrt(np.pi)
            for m in range(nawf):
                pdosaux[m, n] += np.sum(taux * v_kaux[:, m, :])

        pdos = np.zeros((nawf, ne), dtype=float) if rank == 0 else None

        comm.Reduce(pdosaux, pdos, op=MPI.SUM)
        pdosaux = None

        if rank == 0:
            pdos /= float(nktot) * np.sqrt(np.pi) * delta

        pdos_sum = np.zeros(ne, dtype=float) if rank == 0 else None

        for m in range(nawf):
            if rank == 0:
                pdos_sum += pdos[m]
            fpdos = '%d_pdos_%d.dat' % (m, ispin)
            data_controller.write_file_row_col(fpdos, ene, (pdos[m] if rank == 0 else None))

        fpdos = 'pdos_sum_%d.dat' % ispin
        data_controller.write_file_row_col(fpdos, ene, pdos_sum)


def do_pdos_adaptive(data_controller, emin, emax, ne):
    from .smearing import gaussian, metpax

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    # PDoS Calculation with Gaussian Smearing
    emax = np.amin(np.array([attributes['shift'], emax]))
    ene = np.linspace(emin, emax, ne)

    nawf = attributes['nawf']

    for ispin in range(attributes['nspin']):
        E_k = np.real(arrays['E_k'][:, :, ispin])

        pdosaux = np.zeros((nawf, ne), dtype=float)

        v_kaux = np.real(np.abs(arrays['v_k'][:, :, :, ispin]) ** 2)

        taux = np.zeros((arrays['deltakp'].shape[0], nawf), dtype=float)

        for n in range(ne):
            if attributes['smearing'] == 'gauss':
                taux = gaussian(ene[n], E_k, arrays['deltakp'][:, :, ispin])
            elif attributes['smearing'] == 'm-p':
                taux = metpax(ene[n], E_k, arrays['deltakp'][:, :, ispin])
            for i in range(nawf):
                # Adaptive Gaussian Smearing
                pdosaux[i, n] += np.sum(taux * v_kaux[:, i, :])

        pdos = np.zeros((nawf, ne), dtype=float) if rank == 0 else None

        comm.Reduce(pdosaux, pdos, op=MPI.SUM)
        pdosaux = None

        if rank == 0:
            pdos /= float(attributes['nkpnts'])

        pdos_sum = np.zeros(ne, dtype=float) if rank == 0 else None

        for m in range(nawf):
            if rank == 0:
                pdos_sum += pdos[m]
            fpdos = '%d_pdosdk_%d.dat' % (m, ispin)
            data_controller.write_file_row_col(fpdos, ene, (pdos[m] if rank == 0 else None))

        fpdos = 'pdosdk_sum_%d.dat' % ispin
        data_controller.write_file_row_col(fpdos, ene, pdos_sum)
