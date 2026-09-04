def do_transport(
    data_controller, temps, ene, velkp, channels, weights, do_hall, write_to_file, save_tensors
):
    """Compute electronic transport tensors from Boltzmann transport theory.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required attributes: ``smearing``, ``nspin``, ``dftSO``, ``omega``,
        ``opath``.
    temps : array_like of float
        Sequence of temperatures in Kelvin at which transport properties
        are evaluated.
    ene : np.ndarray, shape ``(esize,)``
        Energy grid (eV) at which the transport tensors are sampled.
    velkp : np.ndarray
        Band velocities at each k-point, passed directly to
        :func:`do_Boltz_tensors`.
    channels : array_like
        Channel specification passed to :func:`do_Boltz_tensors`.
    weights : array_like
        k-point weights passed to :func:`do_Boltz_tensors`.
    do_hall : bool
        If ``True``, also compute the antisymmetric Hall coefficient tensor.
    write_to_file : bool
        If ``True``, write all transport tensors to formatted ``.dat`` files
        in ``opath``.
    save_tensors : bool
        If ``True``, store computed tensors in ``data_controller.data_arrays``
        and broadcast them across MPI ranks.

    Returns
    -------
    None
        On rank 0, writes (when ``write_to_file`` is ``True``) one file per
        spin channel and transport quantity:

        - ``sigma[smearing]_{ispin}.dat``: electrical conductivity
          :math:`\\sigma` in S m⁻¹.
        - ``Seebeck[smearing]_{ispin}.dat``: Seebeck coefficient
          :math:`S` in V K⁻¹.
        - ``kappa[smearing]_{ispin}.dat``: electron thermal conductivity
          :math:`\\kappa` in W m⁻¹ K⁻¹.
        - ``PF[smearing]_{ispin}.dat``: power factor
          :math:`S^2 \\sigma` in W m⁻¹ K⁻².
        - ``hall_trace_{ispin}.dat`` (if ``do_hall``): trace of the Hall
          coefficient tensor :math:`R_H` in m³ C⁻¹.

        When ``save_tensors`` is ``True``, adds the following keys to
        ``data_controller.data_arrays``:

        - ``sigma`` : np.ndarray, shape ``(3, 3, esize)`` — conductivity
          tensor in S m⁻¹.
        - ``S`` : np.ndarray, shape ``(3, 3, esize)`` — Seebeck tensor
          in V K⁻¹.
        - ``kappa`` : np.ndarray, shape ``(3, 3, esize)`` — thermal
          conductivity tensor in W m⁻¹ K⁻¹.
        - ``R_hall_trace`` : np.ndarray, shape ``(esize,)`` — averaged
          Hall coefficient in m³ C⁻¹ (only if ``do_hall``).

    Notes
    -----
    The transport tensors are derived from the generalised transport
    integrals :math:`\\mathcal{L}^{(\\alpha)}` via the Boltzmann transport
    equation in the relaxation-time approximation:

    .. math::

        \\sigma = e^2 \\mathcal{L}^{(0)}, \\quad
        S = -\\frac{1}{eT} (\\mathcal{L}^{(0)})^{-1} \\mathcal{L}^{(1)}, \\quad
        \\kappa = \\frac{1}{T}\\left[
            \\mathcal{L}^{(2)} - T\\, \\mathcal{L}^{(1)}
            (\\mathcal{L}^{(0)})^{-1} \\mathcal{L}^{(1)}
        \\right]

    The Hall coefficient is obtained from the antisymmetric part of the
    Hall transport tensor :math:`\\mathcal{L}^{(0,\\text{Hall})}` as

    .. math::

        R_H = (\\sigma^{-1})\\, \\mathcal{L}^{(0,\\text{Hall})}\\, (\\sigma^{-1})

    and reported as the average of the three even-permutation components
    :math:`(R_{012} + R_{201} + R_{120}) / 3`.

    Conversion factors: ``siemen_conv = 6.9884`` converts internal units to
    S m⁻¹ × 10⁻²¹; ``temp_conv = 11604.525`` converts Kelvin to eV⁻¹;
    ``hall_SI = 9.249\\times10^{-13}`` converts to m³ C⁻¹.
    """
    from os.path import join

    import numpy as np
    from numpy import linalg as npl

    from .do_Boltz_tensors import do_Boltz_tensors, do_Boltz_tensors_hall

    comm, rank = data_controller.comm, data_controller.rank
    arrays, attr = data_controller.data_dicts()

    esize = ene.size
    siemen_conv, temp_conv, hall_SI = 6.9884, 11604.52500617, 9.248931724005307e-13
    nspin = attr['nspin']
    spin_mult = 1.0 if nspin == 2 or attr['dftSO'] else 2.0

    for ispin in range(nspin):
        # Quick function opens file in output folder with name 's'
        if write_to_file:
            ojf = lambda st, sp: open(join(attr['opath'], '%s_%d.dat' % (st, sp)), 'w')

            if attr['smearing'] is None:
                fsigma = ojf('sigma', ispin)
                fPF = ojf('PF', ispin)
                fkappa = ojf('kappa', ispin)
                fSeebeck = ojf('Seebeck', ispin)
                if do_hall:
                    fhall_trace = ojf('hall_trace', ispin)
                    fhall = ojf('hall', ispin)
                    fnernst = ojf('nernst', ispin)
            else:
                fsigma = ojf('sigma_' + attr['smearing'], ispin)
                fPF = ojf('PF_' + attr['smearing'], ispin)
                fkappa = ojf('kappa_' + attr['smearing'], ispin)
                fSeebeck = ojf('Seebeck_' + attr['smearing'], ispin)
                if do_hall:
                    fhall_trace = ojf('hall_trace_' + attr['smearing'], ispin)
                    fhall = ojf('hall_' + attr['smearing'], ispin)
                    fnernst = ojf('nernst_' + attr['smearing'], ispin)

        for temp in temps:
            itemp = temp / temp_conv

            # Quick function to write Transport Formatted line to file
            wtup = lambda fn, tu: fn.write(
                '%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n' % tu
            )

            # Quick function to get tuple elements to write
            gtup = lambda tu, i: (
                temp,
                ene[i],
                tu[0, 0, i],
                tu[1, 1, i],
                tu[2, 2, i],
                tu[0, 1, i],
                tu[0, 2, i],
                tu[1, 2, i],
            )

            if do_hall:
                wtup_hall_trace = lambda fn, tu: fn.write('%8.2f % .5f % 9.5e \n' % tu)
                gtup_hall_trace = lambda tu, i: (temp, ene[i], tu[i])

                wtup_full_hall = lambda fn, tu: fn.write(
                    '%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e \n'
                    % tu
                )
                gtup_full_hall = lambda tu, i: (
                    temp,
                    ene[i],
                    (tu[0, 1, 2, i] - tu[1, 0, 2, i]) / 2,
                    (tu[0, 2, 1, i] - tu[2, 0, 1, i]) / 2,
                    (tu[1, 2, 0, i] - tu[2, 1, 0, i]) / 2,
                    (tu[0, 1, 0, i] - tu[1, 0, 0, i]) / 2,
                    (tu[0, 1, 1, i] - tu[1, 0, 1, i]) / 2,
                    (tu[0, 2, 0, i] - tu[2, 0, 0, i]) / 2,
                    (tu[0, 2, 2, i] - tu[2, 0, 2, i]) / 2,
                    (tu[1, 2, 1, i] - tu[2, 1, 1, i]) / 2,
                    (tu[1, 2, 2, i] - tu[2, 1, 2, i]) / 2,
                )

                wtup_nernst = lambda fn, tu: fn.write(
                    '%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e \n'
                    % tu
                )
                gtup_nernst = lambda tu, i: (
                    temp,
                    ene[i],
                    tu[0, 1, 2, i],
                    tu[1, 0, 2, i],
                    tu[0, 2, 1, i],
                    tu[2, 0, 1, i],
                    tu[1, 2, 0, i],
                    tu[2, 1, 0, i],
                    tu[0, 1, 0, i],
                    tu[1, 0, 0, i],
                    tu[0, 1, 1, i],
                    tu[1, 0, 1, i],
                    tu[0, 2, 0, i],
                    tu[2, 0, 0, i],
                    tu[0, 2, 2, i],
                    tu[2, 0, 2, i],
                    tu[1, 2, 1, i],
                    tu[2, 1, 1, i],
                    tu[1, 2, 2, i],
                    tu[2, 1, 2, i],
                )

            L0, L1, L2 = do_Boltz_tensors(
                data_controller, attr['smearing'], itemp, ene, velkp, ispin, channels, weights
            )

            if do_hall:
                L0_hall, L1_hall = do_Boltz_tensors_hall(
                    data_controller, attr['smearing'], itemp, ene, velkp, ispin, channels, weights
                )

            if rank == 0:
                # ----------------------
                # Conductivity (in units of /Ohm/m/s)
                # convert in units of 10*21 siemens m^-1 s^-1
                # ----------------------
                L0_unconverted = L0 * spin_mult / attr['omega']
                L0 *= spin_mult * siemen_conv / attr['omega']
                sigma = L0 * 1.0e21  # convert in units of siemens m^-1 s^-1
                if write_to_file:
                    for i in range(esize):
                        wtup(fsigma, gtup(sigma, i))
                    sigma = None
                if save_tensors:
                    arrays['sigma'] = sigma

                # ----------------------
                # Seebeck (in units of V/K)
                # convert in units of 10^21 Amperes m^-1 s^-1
                # ----------------------
                L1 *= spin_mult * siemen_conv / (temp * attr['omega'])

                S = np.zeros((3, 3, esize), dtype=float)

                for n in range(esize):
                    try:
                        S[:, :, n] = -1.0 * npl.inv(L0[:, :, n]) @ L1[:, :, n]
                    except Exception:
                        from ..utils.report_exception import report_exception

                        print('check t_tensor components - matrix cannot be singular')
                        report_exception()
                        raise
                if write_to_file:
                    for i in range(esize):
                        wtup(fSeebeck, gtup(S, i))
                if save_tensors:
                    arrays['S'] = S

                # ----------------------
                # Electron thermal conductivity ((in units of W/m/K/s)
                # convert in units of kg m s^-4
                # ----------------------
                L2 *= spin_mult * siemen_conv * 1.0e15 / (temp * attr['omega'])

                kappa = np.zeros((3, 3, esize), dtype=float)
                for n in range(esize):
                    kappa[:, :, n] = (
                        L2[:, :, n] - temp * L1[:, :, n] @ npl.inv(L0[:, :, n]) @ L1[:, :, n]
                    ) * 1.0e6
                L2 = None
                if write_to_file:
                    for i in range(esize):
                        wtup(fkappa, gtup(kappa, i))
                    kappa = None
                if save_tensors:
                    arrays['kappa'] = kappa

                PF = np.zeros((3, 3, esize), dtype=float)
                for n in range(esize):
                    PF[:, :, n] = np.dot(np.dot(S[:, :, n], L0[:, :, n]), S[:, :, n]) * 1.0e21
                S = L0 = None
                if write_to_file:
                    for i in range(esize):
                        wtup(fPF, gtup(PF, i))
                    PF = None

                if do_hall:
                    L0_hall *= spin_mult / (attr['omega'])

                    R_hall = np.zeros((3, 3, 3, esize), dtype=float)
                    R_hall_trace = np.zeros((esize), dtype=float)
                    for n in range(esize):
                        try:
                            for r in range(3):
                                R_hall[:, :, r, n] = (
                                    -npl.inv(L0_unconverted[:, :, n])
                                    @ L0_hall[:, :, r, n]
                                    @ npl.inv(L0_unconverted[:, :, n])
                                )
                                # ----------------------
                                # The equivalent to the trace of the Hall tensor is an average
                                # over the even permutations of [0, 1, 2].
                                # ----------------------
                            R_hall_trace[n] = (
                                (R_hall[0, 1, 2, n] + R_hall[2, 0, 1, n] + R_hall[1, 2, 0, n])
                                * hall_SI
                                / 3
                            )

                        except Exception:
                            from ..utils.report_exception import report_exception

                            print('check t_tensor components - matrix cannot be singular')
                            report_exception()
                            raise

                    if write_to_file:
                        for i in range(esize):
                            wtup_hall_trace(fhall_trace, gtup_hall_trace(R_hall_trace, i))
                    if save_tensors:
                        arrays['R_hall_trace'] = R_hall_trace

                    R_hall *= hall_SI

                    if write_to_file:
                        for i in range(esize):
                            wtup_full_hall(fhall, gtup_full_hall(R_hall, i))
                    if save_tensors:
                        arrays['R_hall'] = R_hall

                    L1_hall *= spin_mult / (temp * attr['omega'])

                    nernst = np.zeros((3, 3, 3, esize), dtype=float)
                    for n in range(esize):
                        try:
                            for r in range(3):
                                nernst[:, :, r, n] = (
                                    R_hall[:, :, r, n] @ L1[:, :, n]
                                    + npl.inv(L0_unconverted[:, :, n])
                                    @ L1_hall[:, :, r, n]
                                    * siemen_conv
                                    * hall_SI
                                )

                        except Exception:
                            from ..utils.report_exception import report_exception

                            print('check t_tensor components - matrix cannot be singular')
                            report_exception()
                            raise

                    nernst *= 1e21

                    if write_to_file:
                        for i in range(esize):
                            wtup_nernst(fnernst, gtup_nernst(nernst, i))
                    if save_tensors:
                        arrays['nernst'] = nernst

            comm.Barrier()

        if write_to_file:
            fsigma.close()
            fPF.close()
            fkappa.close()
            fSeebeck.close()
            if do_hall:
                fhall_trace.close()
                fhall.close()
                fnernst.close()

        if save_tensors:
            data_controller.broadcast_single_array('sigma', dtype=float)
            data_controller.broadcast_single_array('S', dtype=float)
            data_controller.broadcast_single_array('kappa', dtype=float)
            if do_hall:
                data_controller.broadcast_single_array('R_hall', dtype=float)
                data_controller.broadcast_single_array('nernst', dtype=float)
                data_controller.broadcast_single_array('R_hall_trace', dtype=float)
