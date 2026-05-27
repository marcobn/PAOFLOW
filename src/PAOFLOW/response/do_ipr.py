import numpy as np


def inverse_participation_ratio(data_controller):
    """Compute the inverse participation ratio (IPR) for each Bloch eigenstate.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``v_k`` (shape ``(nkpnts, nawf, bnd, nspin)``),
        ``E_k`` (shape ``(nkpnts, bnd, nspin)``), and either ``kpnts``
        (shape ``(nkpnts, 3)``) or ``kq`` (shape ``(3, nkpnts)``) for the
        k-point coordinates.
        Required attribute: ``bnd``.

    Returns
    -------
    np.ndarray, shape ``(nspin, nkpnts, nbands, 3)``
        Array of object dtype.  For each spin, k-point, and band the three
        elements along the last axis are:

        - index 0 : np.ndarray, shape ``(3,)`` — crystal k-point coordinates.
        - index 1 : float — band eigenvalue :math:`E_{nk}` in eV.
        - index 2 : float — inverse participation ratio value.

    Notes
    -----
    The IPR quantifies the spatial localisation of a Bloch eigenstate.  For
    eigenstate :math:`|\\psi_{nk}\\rangle` expressed in a localised basis
    :math:`\\{|w_m\\rangle\\}` with coefficients :math:`v_{nk,m}`, the IPR is

    .. math::

        \\text{IPR}_{nk} =
            \\frac{\\sum_m |v_{nk,m}|^4}{\\left(\\sum_m |v_{nk,m}|^2\\right)^2}

    A value of 1 indicates a state fully localised on a single basis function;
    values near :math:`1/N_{\\text{orb}}` indicate delocalised (Bloch-like) states.
    The function works for both k-path (bands) and full k-grid computations,
    selecting ``kpnts`` or the transpose of ``kq`` accordingly.
    """
    arry, attr = data_controller.data_dicts()

    nbands = attr['bnd']

    if 'Hksp' in arry:
        kpts = arry['kpnts']
    else:
        kpts = arry['kq'].T

    nkpts = kpts.shape[0]

    nspin = attr['nspin']

    ipr = np.zeros((nspin, nkpts, nbands, 3), dtype=object)

    for ispin in range(nspin):
        for ikpt in range(nkpts):
            for iband in range(nbands):
                vk = arry['v_k'][ikpt, :, iband, ispin]
                ek = arry['E_k'][ikpt, iband, ispin]

                vk_abs = np.abs(vk)

                ipr[ispin, ikpt, iband, 0] = kpts[ikpt]
                ipr[ispin, ikpt, iband, 1] = ek
                ipr[ispin, ikpt, iband, 2] = np.sum(vk_abs**4) / (np.sum(vk_abs**2) ** 2)

    return ipr
