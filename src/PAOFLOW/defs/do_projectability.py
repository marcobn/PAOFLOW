import numpy as np
from mpi4py import MPI


def build_Pn(nawf, nbnds, nkpnts, nspin, U):
    Pn = 0.0
    for ispin in range(nspin):
        for ik in range(nkpnts):
            UU = np.transpose(
                U[:, :, ik, ispin]
            )  # transpose of U. Now the columns of UU are the eigenvector of length nawf
            Pn += np.real(np.sum(np.conj(UU) * UU, axis=0)) / nkpnts / nspin
    return Pn


def do_projectability(data_controller):
    # ----------------------
    # Building the Projectability
    # ----------------------
    rank = MPI.COMM_WORLD.Get_rank()

    arry, attr = data_controller.data_dicts()

    shift = attr['shift']

    if rank != 0:
        attr['shift'] = None
    else:
        Pn = build_Pn(attr['nawf'], attr['nbnds'], attr['nkpnts'], attr['nspin'], arry['U'])

        if attr['verbose']:
            print('Projectability vector ', Pn)

        # Check projectability and decide bnd
        bnd = 0
        for n in range(attr['nbnds']):
            if Pn[n] > attr['pthr']:
                bnd += 1

        Pn = None
        attr['bnd'] = maxbnd = bnd
        warn_txt = 'WARNING: All bands meet the projectability threshold. Consider increasing number of bands.'
        if bnd == attr['nawf']:
            maxbnd = bnd - 1
            print(warn_txt)

        if 'shift' not in attr or attr['shift'] == 'auto':
            if maxbnd >= arry['my_eigsmat'].shape[0]:
                maxbnd = arry['my_eigsmat'].shape[0] - 1
                print(warn_txt)
            shift_v = np.amin(arry['my_eigsmat'][maxbnd, :, :])
            attr['shift'] = shift_v if shift == 'auto' else shift

        if attr['verbose']:
            print('# of bands with good projectability > {} = {}'.format(attr['pthr'], bnd))
        if attr['verbose'] and bnd < attr['nbnds']:
            print(
                'Range of suggested shift ',
                np.amin(arry['my_eigsmat'][maxbnd, :, :]),
                ' , ',
                np.amax(arry['my_eigsmat'][maxbnd, :, :]),
            )

    # Broadcast
    data_controller.broadcast_attribute('bnd')
    data_controller.broadcast_attribute('shift')
