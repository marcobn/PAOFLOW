import numpy as np
from mpi4py import MPI


def build_Pn(nawf, nbnds, nkpnts, nspin, U):
    """Compute the k-averaged projectability for each DFT band.

    Parameters
    ----------
    nawf : int
        Number of atomic wave-function projectors.
    nbnds : int
        Total number of DFT bands.
    nkpnts : int
        Number of k-points.
    nspin : int
        Number of spin channels.
    U : np.ndarray, shape ``(nawf, nbnds, nkpnts, nspin)``
        Overlap matrix between PAO projectors and DFT eigenstates.

    Returns
    -------
    np.ndarray, shape ``(nbnds,)``, float
        Projectability vector :math:`P_n`; values close to 1 indicate
        that band ``n`` is well represented in the PAO subspace.

    Notes
    -----
    The projectability is defined as the average squared norm of the
    projection coefficients:

    .. math::

        P_n = \\frac{1}{N_k N_\\sigma}
            \\sum_{\\mathbf{k}, \\sigma}
            \\sum_i |U_{i n \\mathbf{k} \\sigma}|^2
    """
    Pn = 0.0
    for ispin in range(nspin):
        for ik in range(nkpnts):
            UU = np.transpose(
                U[:, :, ik, ispin]
            )  # transpose of U. Now the columns of UU are the eigenvector of length nawf
            Pn += np.real(np.sum(np.conj(UU) * UU, axis=0)) / nkpnts / nspin
    return Pn


def do_projectability(data_controller):
    """Determine the number of well-projected bands and the optimal energy shift.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``U`` (shape ``(nawf, nbnds, nkpnts, nspin)``),
        ``my_eigsmat`` (shape ``(nbnds, nkpnts, nspin)``).
        Required attributes: ``nawf``, ``nbnds``, ``nkpnts``, ``nspin``,
        ``pthr``, ``shift``, ``verbose``.

    Returns
    -------
    None
        Updates the following attributes in ``data_controller.data_attributes``
        and broadcasts them to all MPI ranks:

        - ``bnd`` : int — number of bands whose projectability exceeds ``pthr``.
        - ``shift`` : float — the energy shift :math:`\\eta` to apply to states
          outside the PAO subspace (set from the minimum eigenvalue at the
          band edge when ``shift == 'auto'``).

    Notes
    -----
    Only rank 0 calls :func:`build_Pn` and determines ``bnd`` and ``shift``.
    Both values are then broadcast via
    :meth:`DataController.broadcast_attribute`.  A warning is printed if all
    bands meet the threshold, indicating that the number of DFT bands may be
    too small.
    """
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
            print('Projectability vector: \n', Pn)

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
