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
    np.ndarray or None, shape ``(nspin, nkpnts, nbands, 3)``
        Array of object dtype.  For each spin, k-point, and band the three
        elements along the last axis are:

        - index 0 : np.ndarray, shape ``(3,)`` — crystal k-point coordinates.
        - index 1 : float — band eigenvalue :math:`E_{nk}` in eV.
        - index 2 : float — inverse participation ratio value.

        On rank 0; ``None`` on the other ranks (the values are gathered).

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
    from ..utils.communication import gather_full

    arry, attr = data_controller.data_dicts()

    nbands = attr['bnd']

    if 'Hksp' in arry:
        kpts = arry['kpnts']
    else:
        kpts = arry['kq'].T

    nspin = attr['nspin']
    nk_local = arry['v_k'].shape[0]

    # per-state values on this rank's k-points, gathered below (indexing the
    # global k list with the local eigenvectors only worked on one rank)
    values = np.zeros((nk_local, nbands, nspin), dtype=float)
    for ispin in range(nspin):
        for ikpt in range(nk_local):
            values[ikpt, :, ispin] = ipr_k(arry['v_k'][ikpt, :, :nbands, ispin])

    values = gather_full(values, attr['npool'])
    energies = gather_full(np.ascontiguousarray(arry['E_k'][:, :nbands, :]), attr['npool'])
    if values is None:
        return None
    return ipr_table(kpts, energies, values)


def ipr_k(v_k: np.ndarray) -> np.ndarray:
    """Inverse participation ratio of each state at one k-point.

    Parameters
    ----------
    v_k : np.ndarray, shape ``(nawf, nbands)``
        Eigenvectors (columns).

    Returns
    -------
    np.ndarray, shape ``(nbands,)``
        :math:`\\sum_m |v_m|^4 / (\\sum_m |v_m|^2)^2` per column.
    """
    out = np.empty(v_k.shape[1], dtype=float)
    for iband in range(v_k.shape[1]):
        vk_abs = np.abs(v_k[:, iband])
        out[iband] = np.sum(vk_abs**4) / (np.sum(vk_abs**2) ** 2)
    return out


def ipr_table(kpts: np.ndarray, energies: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Assemble the ``(nspin, nkpts, nbands, 3)`` object array of :func:`inverse_participation_ratio`.

    Parameters
    ----------
    kpts : np.ndarray, shape ``(nkpts, 3)``
        Coordinates stored in field 0.
    energies, values : np.ndarray, shape ``(nkpts, nbands, nspin)``
        Band energies (field 1) and IPR values (field 2).
    """
    nkpts, nbands, nspin = values.shape
    ipr = np.zeros((nspin, nkpts, nbands, 3), dtype=object)
    for ispin in range(nspin):
        for ikpt in range(nkpts):
            for iband in range(nbands):
                ipr[ispin, ikpt, iband, 0] = kpts[ikpt]
                ipr[ispin, ikpt, iband, 1] = energies[ikpt, iband, ispin]
                ipr[ispin, ikpt, iband, 2] = values[ikpt, iband, ispin]
    return ipr
