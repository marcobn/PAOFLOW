def do_adaptive_smearing(data_controller, smearing, afac):
    import numpy as np
    from numpy.linalg import norm

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
