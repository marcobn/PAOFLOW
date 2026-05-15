import numpy as np
from mpi4py import MPI

from .communication import load_balancing
from .do_atwfc_proj import calc_atwfc_k, calc_gkspace, fft_allwfc_G2R, ortho_atwfc_k
from .write2xsf import write2xsf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_density(data_controller, nr1, nr2, nr3):
    arry, attr = data_controller.data_dicts()

    # Calculation of the electron density

    if rank == 0 and attr['verbose']:
        print('Writing density files')

    rhoaux = np.zeros((nr1, nr2, nr3, attr['nspin']), dtype=complex, order='C')

    ini_ik, end_ik = load_balancing(comm.Get_size(), rank, attr['nkpnts'])

    basis = arry['basis']
    eps = 1.0e-5
    for ispin in range(attr['nspin']):
        for ik in range(ini_ik, end_ik):
            gkspace = calc_gkspace(data_controller, ik, gamma_only=False)
            atwfcgk = calc_atwfc_k(basis, gkspace)
            oatwfcgk = ortho_atwfc_k(atwfcgk)
            atwfcr = fft_allwfc_G2R(oatwfcgk, gkspace, nr1, nr2, nr3, attr['omega'])
            for nb in range(attr['bnd']):
                if arry['E_k'][ik - ini_ik, nb, ispin] <= 0.0 + eps:
                    tmp = np.tensordot(
                        arry['v_k'][ik - ini_ik, :, nb, ispin], atwfcr[:, :, :, :], axes=(0, 0)
                    )
                    rhoaux[:, :, :, ispin] += (
                        2 * np.conj(tmp) * tmp / attr['nkpnts'] * attr['omega'] / (nr1 * nr2 * nr3)
                    )

        rho = (
            np.zeros((nr1, nr2, nr3, attr['nspin']), dtype=complex, order='C')
            if rank == 0
            else None
        )

        comm.Reduce(rhoaux, rho, op=MPI.SUM)
        rhoaux = None

        if rank == 0:
            fdensity = attr['outputdir'] + '/density_%s.xsf' % str(ispin)
            write2xsf(data_controller, filename=fdensity, data=np.real(rho[:, :, :, ispin]))
    if rank == 0:
        if attr['verbose']:
            print('Total charge = ', np.real(np.sum(rho)).round(3))
