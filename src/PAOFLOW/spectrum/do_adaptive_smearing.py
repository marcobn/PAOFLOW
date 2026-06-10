def do_adaptive_smearing(data_controller, smearing, afac):
    """Compute adaptive smearing widths for each k-point and band.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required array: ``pksp`` (shape ``(npks, 3, nawf, nawf, nspin)``) —
        the momentum matrix elements in the Bloch eigenstate basis.
        Required attributes: ``nawf``, ``nspin``, ``nkpnts``, ``omega``.
    smearing : str
        Smearing method identifier.  Pass ``'m-p'`` for Methfessel–Paxton;
        any other value selects the default prefactor.
    afac : Optional[float]
        Adaptive smearing prefactor :math:`\\alpha`.  If ``None``, defaults
        to ``1.0`` for ``'m-p'`` smearing and ``0.7`` otherwise.

    Returns
    -------
    None
        Adds the following keys to ``data_controller.data_arrays``:

        - ``deltakp`` : np.ndarray, shape ``(npks, nawf, nspin)`` —
          band-resolved adaptive smearing widths
          :math:`\\sigma_{nk} = \\alpha \\, |\\nabla_k E_n| \\, \\delta k`.
        - ``deltakp2`` : np.ndarray, shape ``(npks, nawf, nawf, nspin)`` —
          interband adaptive smearing widths proportional to
          :math:`|\\nabla_k E_n - \\nabla_k E_m|`.

    Notes
    -----
    The mean k-point spacing is estimated as

    .. math::

        \\delta k = \\left(\\frac{8\\pi^3}{\\Omega \\, N_k}\\right)^{1/3}

    where :math:`\\Omega` is the unit-cell volume and :math:`N_k` is the total
    number of k-points.  The diagonal elements of ``pksp`` (proportional to
    the band velocities) are used as a proxy for :math:`\\nabla_k E_n`.

    Reference: J. R. Yates, X. Wang, D. Vanderbilt, I. Souza,
    Phys. Rev. B **75**, 195121 (2007).
    """
    from numpy.linalg import norm
    import numpy as np

    arrays, attributes = data_controller.data_dicts()

    # ----------------------
    # adaptive smearing as in Yates et al. Phys. Rev. B 75, 195121 (2007).
    # ----------------------

    nawf = attributes['nawf']
    nspin = attributes['nspin']
    nkpnts = attributes['nkpnts']
    npks = arrays['pksp'].shape[0]

    diag = np.diag_indices(nawf)

    dk = (8.0 * np.pi**3 / attributes['omega'] / (nkpnts)) ** (1.0 / 3.0)

    if afac == None:
        afac = 1.0 if smearing == 'm-p' else 0.7

    ## DEV: Try to make contiguinuity conditional. Requires benchmark testing
    # Source of the band group velocity nabla_k E_n used for the Yates width.
    # When the non-local velocity (NLV) correction is active, gradient_and_momenta
    # stores the BARE eigenbasis velocity diagonal under 'velkp_bare'. The
    # diagonal of the (NLV-corrected) 'pksp' is no longer the true group
    # velocity -- the interband position-commutator correction contaminates it
    # and inflates the adaptive widths, producing a spurious epsilon spike near
    # omega -> 0. Prefer the bare diagonal whenever it is present (opt out via
    # attr['adaptive_smearing_bare_velocity'] = False).
    use_bare = attributes.get('adaptive_smearing_bare_velocity', True)
    if use_bare and 'velkp_bare' in arrays:
        pksaux = np.ascontiguousarray(arrays['velkp_bare'])
    else:
        pksaux = np.ascontiguousarray(arrays['pksp'][:, :, diag[0], diag[1]])

    deltakp = np.zeros((npks, nawf, nspin), dtype=float)
    deltakp2 = np.zeros((npks, nawf, nawf, nspin), dtype=float)
    for n in range(nawf):
        deltakp[:, n] = norm(np.real(pksaux[:, :, n]), axis=1)
        for m in range(nawf):
            deltakp2[:, n, m, :] = norm(pksaux[:, :, n, :] - pksaux[:, :, m, :], axis=1)

    pksaux = None
    deltakp *= afac * dk
    deltakp2 *= afac * dk

    arrays['deltakp'] = deltakp
    arrays['deltakp2'] = deltakp2
