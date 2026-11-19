import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_spin_Hall(data_controller, twoD, do_ac, P):
    """Compute the spin Hall conductivity tensor and optionally the AC spin Hall conductivity.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``s_tensor`` (shape ``(n, 3)`` specifying
        :math:`(i_{\\rm pol}, j_{\\rm pol}, s_{\\rm pol})` triplets),
        ``dHksp``, ``v_k``, ``degen``, ``Sj``, ``E_k``, ``deltakp``.
        Required attributes: ``dftSO``, ``nk1``, ``nk2``, ``nk3``,
        ``omega``, ``alat``, ``verbose``, ``opath``.
    twoD : bool
        If ``True``, normalise by the 2-D cross-sectional area instead of
        the 3-D volume.
    do_ac : bool
        If ``True``, also compute the AC spin Hall conductivity.
    P : np.ndarray, shape ``(nawf, nawf)``
        Projection operator used to symmetrize the spin current operator.

    Returns
    -------
    None
        Writes per-component output files to ``opath``:

        - ``Spin_Berry_{spol}_{ipol}{jpol}.bxsf`` — spin Berry curvature on
          the k-grid.
        - ``shcEf_{spol}_{ipol}{jpol}.dat`` — spin Hall conductivity vs.
          Fermi energy.
        - When ``do_ac``: ``ac_shcr_{...}.dat`` and ``ac_shci_{...}.dat``.
    """
    from ..utils.constants import LL
    from .kubo_units import conductivity_conversion
    from ..utils.perturb_split import perturb_split

    arry, attr = data_controller.data_dicts()

    s_tensor = arry['s_tensor']

    # -----------------------
    # Spin Hall calculation
    # -----------------------
    if attr['dftSO'] == False:
        if rank == 0:
            print('Relativistic calculation with SO required')
            comm.Abort()
        comm.Barrier()

    if rank == 0 and attr['verbose']:
        print('Writing bxsf files for Spin Berry Curvature')

    snktot, _, nawf, _, nspin = arry['dHksp'].shape

    for n in range(s_tensor.shape[0]):
        ipol = s_tensor[n][0]
        jpol = s_tensor[n][1]
        spol = s_tensor[n][2]
        Sj = arry['Sj'][spol]

        # ----------------------------------------------
        # Compute the spin current operator j^l_n,m(k)
        # ----------------------------------------------
        # The spin current is built one k-point at a time to avoid holding a
        # full (snktot,nawf,nawf,nspin) temporary alongside the two outputs.
        jksp_is = np.empty((snktot, nawf, nawf, nspin), dtype=complex)
        pksp_j = np.empty((snktot, nawf, nawf, nspin), dtype=complex)

        for ik in range(snktot):
            for ispin in range(nspin):
                dHk = arry['dHksp'][ik, ipol, :, :, ispin]
                jdHk = 0.5 * (Sj @ dHk + dHk @ Sj)
                jksp_is[ik, :, :, ispin], pksp_j[ik, :, :, ispin] = perturb_split(
                    0.5 * (P @ jdHk + jdHk @ P),
                    arry['dHksp'][ik, jpol, :, :, ispin],
                    arry['v_k'][ik, :, :, ispin],
                    arry['degen'][ispin][ik],
                )
        dHk = jdHk = None

        # ---------------------------------
        # Compute spin Berry curvature...
        # ---------------------------------
        ene, shc, Om_k = do_Berry_curvature(data_controller, jksp_is, pksp_j)

        jksp_is = pksp_j = None

        cgs_conv = conductivity_conversion(data_controller, twoD)
        if rank == 0:
            shc *= cgs_conv

        cart_indices = (str(LL[spol]), str(LL[ipol]), str(LL[jpol]))

        fBerry = 'Spin_Berry_%s_%s%s.bxsf' % cart_indices
        nk1, nk2, nk3 = attr['nk1'], attr['nk2'], attr['nk3']
        Om_kps = np.empty((nk1, nk2, nk3, 2), dtype=float) if rank == 0 else None
        if rank == 0:
            Om_kps[:, :, :, 0] = Om_kps[:, :, :, 1] = Om_k[:, :, :]
        data_controller.write_bxsf(fBerry, Om_kps, 2, skeaf_sheets=False)

        Om_k = Om_kps = None

        fshc = 'shcEf_%s_%s%s.dat' % cart_indices
        data_controller.write_file_row_col(fshc, ene, shc)

        ene = shc = None

        if do_ac:
            jksp_js = np.empty((snktot, nawf, nawf, nspin), dtype=complex)
            pksp_i = np.empty((snktot, nawf, nawf, nspin), dtype=complex)

            for ik in range(snktot):
                for ispin in range(nspin):
                    dHk = arry['dHksp'][ik, ipol, :, :, ispin]
                    jksp_js[ik, :, :, ispin], pksp_i[ik, :, :, ispin] = perturb_split(
                        0.5 * (Sj @ dHk + dHk @ Sj),
                        arry['dHksp'][ik, jpol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            dHk = None

            ene, sigxy = do_ac_conductivity(data_controller, jksp_js, pksp_i, ipol, jpol)

            jksp_js = pksp_i = None

            if rank == 0:
                sigxy *= cgs_conv

            sigxyi = np.imag(ene * sigxy / 105.4571) if rank == 0 else None
            sigxyr = np.real(sigxy) if rank == 0 else None
            sigxy = None

            fsigI = 'SCDi_%s_%s%s.dat' % cart_indices
            data_controller.write_file_row_col(fsigI, ene, sigxyi)

            fsigR = 'SCDr_%s_%s%s.dat' % cart_indices
            data_controller.write_file_row_col(fsigR, ene, sigxyr)


def do_orbital_Hall(data_controller, twoD, do_ac, P):
    """Compute the orbital Hall conductivity tensor and optionally the AC spin Hall conductivity.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``o_tensor`` (shape ``(n, 3)`` specifying
        :math:`(i_{\\rm pol}, j_{\\rm pol}, s_{\\rm pol})` triplets),
        ``dHksp``, ``v_k``, ``degen``, ``Sj``, ``E_k``, ``deltakp``.
        Required attributes: ``dftSO``, ``nk1``, ``nk2``, ``nk3``,
        ``omega``, ``alat``, ``verbose``, ``opath``.
    twoD : bool
        If ``True``, normalise by the 2-D cross-sectional area instead of
        the 3-D volume.
    do_ac : bool
        If ``True``, also compute the AC orbital Hall conductivity.
    P : np.ndarray, shape ``(nawf, nawf)``
        Projection operator used to symmetrize the orbital current operator.

    Returns
    -------
    None
        Writes per-component output files to ``opath``:

        - ``Orbital_Berry_{spol}_{ipol}{jpol}.bxsf`` — orbital Berry curvature on
          the k-grid.
        - ``ohcEf_{spol}_{ipol}{jpol}.dat`` — orbital Hall conductivity vs.
          Fermi energy.
        - When ``do_ac``: ``ac_ohcr_{...}.dat`` and ``ac_ohci_{...}.dat``.
    """
    from ..utils.constants import LL
    from .kubo_units import conductivity_conversion
    from ..utils.perturb_split import perturb_split

    arry, attr = data_controller.data_dicts()

    o_tensor = arry['o_tensor']

    # -----------------------
    # Orbital Hall calculation
    # -----------------------
    # No spin-orbit requirement: the orbital Hall effect exists without it. ``Lj`` is built by
    # PAOFLOW.orbital_operator in the spinor basis (dftSO or adhoc_SO) or, since 14 Nov 2026, in
    # the scalar real-harmonic basis.
    if 'Lj' not in arry:
        raise KeyError('do_orbital_Hall needs the orbital operator Lj (PAOFLOW.orbital_operator)')

    if rank == 0 and attr['verbose']:
        print('Writing bxsf files for Orbital Berry Curvature')

    for n in range(o_tensor.shape[0]):
        ipol = o_tensor[n][0]
        jpol = o_tensor[n][1]
        spol = o_tensor[n][2]
        # ----------------------------------------------
        # Compute the orbital current operator j^l_n,m(k)
        # ----------------------------------------------
        jdHksp = do_orbital_current(data_controller, spol, ipol)

        jksp_is = np.empty_like(jdHksp)
        pksp_j = np.empty_like(jdHksp)

        for ik in range(jdHksp.shape[0]):
            for ispin in range(jdHksp.shape[3]):
                jksp_is[ik, :, :, ispin], pksp_j[ik, :, :, ispin] = perturb_split(
                    0.5 * (P @ jdHksp[ik, :, :, ispin] + jdHksp[ik, :, :, ispin] @ P),
                    arry['dHksp'][ik, jpol, :, :, ispin],
                    arry['v_k'][ik, :, :, ispin],
                    arry['degen'][ispin][ik],
                )
        jdHksp = None

        # ---------------------------------
        # Compute orbital Berry curvature...
        # ---------------------------------
        ene, ohc, Om_k = do_Berry_curvature(data_controller, jksp_is, pksp_j)

        cgs_conv = conductivity_conversion(data_controller, twoD)
        if not attr['dftSO'] and attr['nspin'] == 1:
            # scalar basis: every band holds both spins (as in the Boltzmann routes)
            cgs_conv *= 2.0
        if rank == 0:
            ohc *= cgs_conv

        cart_indices = (str(LL[spol]), str(LL[ipol]), str(LL[jpol]))

        fBerry = 'Orbital_Berry_%s_%s%s.bxsf' % cart_indices
        nk1, nk2, nk3 = attr['nk1'], attr['nk2'], attr['nk3']
        Om_kps = np.empty((nk1, nk2, nk3, 2), dtype=float) if rank == 0 else None
        if rank == 0:
            Om_kps[:, :, :, 0] = Om_kps[:, :, :, 1] = Om_k[:, :, :]
        data_controller.write_bxsf(fBerry, Om_kps, 2, skeaf_sheets=False)

        Om_k = Om_kps = None

        fshc = 'ohcEf_%s_%s%s.dat' % cart_indices
        data_controller.write_file_row_col(fshc, ene, ohc)

        ene = ohc = None

        if do_ac:
            jdHksp = do_orbital_current(data_controller, spol, ipol)

            jksp_js = np.empty_like(jdHksp)
            pksp_i = np.empty_like(jdHksp)

            for ik in range(jdHksp.shape[0]):
                for ispin in range(jdHksp.shape[3]):
                    jksp_js[ik, :, :, ispin], pksp_i[ik, :, :, ispin] = perturb_split(
                        jdHksp[ik, :, :, ispin],
                        arry['dHksp'][ik, jpol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            jdHksp = None

            ene, sigxy = do_ac_conductivity(data_controller, jksp_js, pksp_i, ipol, jpol)
            if rank == 0:
                sigxy *= cgs_conv

            sigxyi = np.imag(ene * sigxy / 105.4571) if rank == 0 else None
            sigxyr = np.real(sigxy) if rank == 0 else None
            sigxy = None

            fsigI = 'OCDi_%s_%s%s.dat' % cart_indices
            data_controller.write_file_row_col(fsigI, ene, sigxyi)

            fsigR = 'OCDr_%s_%s%s.dat' % cart_indices
            data_controller.write_file_row_col(fsigR, ene, sigxyr)


def do_anomalous_Hall(data_controller, do_ac: bool, twoD: bool = False) -> None:
    """Compute the anomalous Hall conductivity and optionally the magneto-optical conductivity.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``a_tensor`` (shape ``(n, 2)`` specifying
        :math:`(i_{\\rm pol}, j_{\\rm pol})` pairs), ``dHksp``, ``v_k``,
        ``degen``, ``E_k``, ``deltakp``.
        Required attributes: ``dftSO``, ``nk1``, ``nk2``, ``nk3``,
        ``omega``, ``alat``, ``verbose``, ``opath``.
    do_ac : bool
        If ``True``, also compute the MCD (magneto-circular dichroism)
        optical conductivity.
    twoD : bool
        Sheet conductivity in units of :math:`e^2/\\hbar` (area of the first two lattice
        vectors) instead of S/cm. Default ``False``.

    Returns
    -------
    None
        Writes per-component output files to ``opath``:

        - ``Berry_{ipol}{jpol}.bxsf`` — Berry curvature on the k-grid.
        - ``ahcEf_{ipol}{jpol}.dat`` — anomalous Hall conductivity vs.
          Fermi energy.
        - When ``do_ac``: ``MCDr_{...}.dat`` and ``MCDi_{...}.dat``.
    """
    from ..utils.constants import LL
    from .kubo_units import conductivity_conversion
    from ..utils.perturb_split import perturb_split

    arry, attr = data_controller.data_dicts()

    a_tensor = arry['a_tensor']

    # ----------------------------
    # Anomalous Hall calculation
    # ----------------------------
    if attr['dftSO'] == False:
        if rank == 0:
            print('Relativistic calculation with SO required')
            comm.Abort()
        comm.Barrier()

    if rank == 0 and attr['verbose']:
        print('Writing bxsf files for Berry Curvature')

    for n in range(a_tensor.shape[0]):
        ipol = a_tensor[n][0]
        jpol = a_tensor[n][1]

        dks = arry['dHksp'].shape

        pksp_i = np.zeros((dks[0], dks[2], dks[3], dks[4]), order='C', dtype=complex)
        pksp_j = np.zeros_like(pksp_i)

        for ik in range(dks[0]):
            for ispin in range(dks[4]):
                pksp_i[ik, :, :, ispin], pksp_j[ik, :, :, ispin] = perturb_split(
                    arry['dHksp'][ik, ipol, :, :, ispin],
                    arry['dHksp'][ik, jpol, :, :, ispin],
                    arry['v_k'][ik, :, :, ispin],
                    arry['degen'][ispin][ik],
                )

        ene, ahc, Om_k = do_Berry_curvature(data_controller, pksp_i, pksp_j)

        cgs_conv = conductivity_conversion(data_controller, twoD)

        cart_indices = (str(LL[ipol]), str(LL[jpol]))

        fBerry = 'Berry_%s%s.bxsf' % cart_indices
        nk1, nk2, nk3 = attr['nk1'], attr['nk2'], attr['nk3']
        Om_kps = np.empty((nk1, nk2, nk3, 2), dtype=float) if rank == 0 else None
        if rank == 0:
            Om_kps[:, :, :, 0] = Om_kps[:, :, :, 1] = Om_k[:, :, :]
        data_controller.write_bxsf(fBerry, Om_kps, 2, skeaf_sheets=False)

        Om_k = Om_kps = None

        if rank == 0:
            ahc *= cgs_conv
        fahc = 'ahcEf_%s%s.dat' % cart_indices
        data_controller.write_file_row_col(fahc, ene, ahc)

        ene = ahc = None

        if do_ac:
            ene, sigxy = do_ac_conductivity(data_controller, pksp_i, pksp_j, ipol, jpol)
            if rank == 0:
                sigxy *= cgs_conv

            sigxyi = np.imag(ene * sigxy / 105.4571) if rank == 0 else None
            sigxyr = np.real(sigxy) if rank == 0 else None
            sigxy = None

            fsigI = 'MCDi_%s%s.dat' % cart_indices
            data_controller.write_file_row_col(fsigI, ene, sigxyi)

            fsigR = 'MCDr_%s%s.dat' % cart_indices
            data_controller.write_file_row_col(fsigR, ene, sigxyr)

        pksp_i = pksp_j = None


def do_Berry_curvature(data_controller, jksp, pksp):
    """Compute the Berry curvature and its Fermi-energy integral via the Kubo formula.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(snktot, nawf, nspin)``),
        ``deltakp``, ``deltakp2``.
        Required attributes: ``npool``, ``nkpnts``, ``nk1``, ``nk2``,
        ``nk3``, ``fermi_up``, ``fermi_dw``, ``deltaH``, ``smearing``,
        ``eminH``, ``emaxH``, ``esizeH``, ``shift``.
    jksp : np.ndarray, shape ``(snktot, nawf, nawf, nspin)``, complex
        Left matrix element (spin current or momentum).
    pksp : np.ndarray, shape ``(snktot, nawf, nawf, nspin)``, complex
        Right matrix element (momentum).

    Returns
    -------
    ene : np.ndarray, shape ``(esizeH,)``
        Energy grid (eV).
    shc : np.ndarray or None
        Berry-curvature integral as a function of Fermi energy (rank 0 only).
    Om_zk : np.ndarray or None
        Berry curvature on the k-grid reshaped to ``(nk1, nk2, nk3)``
        evaluated at ``fermi_up`` minus the value at ``fermi_dw``
        (rank 0 only).

    Notes
    -----
    Uses the Kubo formula

    .. math::

        \\Omega_n(\\mathbf{k}) = -2 \\sum_{m \\ne n}
            \\frac{\\mathrm{Im}[j_{nm} p_{mn}]}
            {(\\varepsilon_n - \\varepsilon_m)^2 + \\delta^2}

    summed over occupied bands with occupation determined by the
    selected smearing scheme.
    """
    # ----------------------
    # Compute spin Berry curvature
    # ----------------------
    from ..utils.communication import gather_full
    from ..utils.smearing import intgaussian, intmetpax
    from .kubo_kernels import berry_curvature

    arrays, attributes = data_controller.data_dicts()

    snktot, nawf, _, nspin = pksp.shape
    fermi_up, fermi_dw = attributes['fermi_up'], attributes['fermi_dw']
    nk1, nk2, nk3 = attributes['nk1'], attributes['nk2'], attributes['nk3']

    # Compute only Omega_z(k)
    Om_znkaux = np.zeros((snktot, nawf), dtype=float)

    deltap = attributes['deltaH']

    for ik in range(snktot):
        Om_znkaux[ik] = berry_curvature(
            arrays['E_k'][ik, :, 0], jksp[ik, :, :, 0], pksp[ik, :, :, 0], deltap
        )

    emax = attributes['emaxH']
    if attributes['shift'] != 0.0:
        emax = min(attributes['shift'], emax)
    esize = attributes['esizeH']
    ene = np.linspace(attributes['eminH'], emax, esize)

    Om_zkaux = np.zeros((snktot, esize), dtype=float)

    for i in range(esize):
        if attributes['smearing'] == 'gauss':
            Om_zkaux[:, i] = np.sum(
                (
                    Om_znkaux[:, :]
                    * intgaussian(arrays['E_k'][:, :, 0], ene[i], arrays['deltakp'][:, :, 0])
                ),
                axis=1,
            )
        elif attributes['smearing'] == 'm-p':
            Om_zkaux[:, i] = np.sum(
                Om_znkaux[:, :]
                * intmetpax(arrays['E_k'][:, :, 0], ene[i], arrays['deltakp'][:, :, 0]),
                axis=1,
            )
        else:
            Om_zkaux[:, i] = np.sum(
                Om_znkaux[:, :] * (0.5 * (-np.sign(arrays['E_k'][:, :, 0] - ene[i]) + 1)),
                axis=1,
            )

    Om_zk = gather_full(Om_zkaux, attributes['npool'])
    Om_zkaux = None

    shc = None
    if rank == 0:
        shc = np.sum(Om_zk, axis=0) / float(attributes['nkpnts'])

    n0 = 0
    n = esize - 1
    if rank == 0:
        for i in range(esize - 1):
            if ene[i] <= fermi_dw and ene[i + 1] >= fermi_dw:
                n0 = i
            if ene[i] <= fermi_up and ene[i + 1] >= fermi_up:
                n = i
        Om_zk = np.reshape(Om_zk, (nk1, nk2, nk3, esize), order='C')
        Om_zk = Om_zk[:, :, :, n] - Om_zk[:, :, :, n0]

    return (ene, shc, Om_zk)


def do_ac_conductivity(data_controller, jksp, pksp, ipol, jpol):
    """Compute the frequency-dependent (optical) conductivity via the Kubo-Greenwood formula.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k``, ``deltakp``, ``deltakp2``.
        Required attributes: ``shift``, ``esizeH``, ``nkpnts``, ``smearing``,
        ``temp``, ``delta``.
    jksp : np.ndarray, shape ``(snktot, nawf, nawf, nspin)``, complex
        Left matrix element.
    pksp : np.ndarray, shape ``(snktot, nawf, nawf, nspin)``, complex
        Right matrix element.
    ipol : int
        Polarisation index of the left operator (0=x, 1=y, 2=z).
    jpol : int
        Polarisation index of the right operator.

    Returns
    -------
    ene : np.ndarray or None
        Frequency grid (eV) on rank 0; ``None`` on other ranks.
    sigxy : np.ndarray or None
        Complex optical conductivity :math:`\\sigma_{ij}(\\omega)` on
        rank 0; ``None`` on other ranks.
    """
    arry, attr = data_controller.data_dicts()

    # Compute the optical conductivity tensor sigma_xy(ene)

    ispin = 0

    emin = 0.0
    emax = attr['shift']
    width_floor = 0.0
    if not np.isfinite(emax):
        # Tight-binding models have no projectability shift (shift = inf): the photon energies
        # then span the band width, and the adaptive pair widths (zero for pairs with equal
        # velocities) are floored at ``delta``. Until 14 Nov 2026 the grid was linspace(0, inf),
        # i.e. NaN. DFT runs (finite shift) are unchanged.
        e_top = comm.allreduce(float(np.max(arry['E_k'])), op=MPI.MAX)
        e_bottom = comm.allreduce(float(np.min(arry['E_k'])), op=MPI.MIN)
        emax = e_top - e_bottom
        width_floor = attr['deltaH']
    esize = attr['esizeH']
    ene = np.linspace(emin, emax, esize)

    sigxy_aux = smear_sigma_loop(
        data_controller, ene, jksp, pksp, ispin, ipol, jpol, width_floor=width_floor
    )

    sigxy = np.zeros((esize), dtype=complex) if rank == 0 else None
    sigxyR = np.zeros((esize), dtype=float) if rank == 0 else None
    sigxyI = np.zeros((esize), dtype=float) if rank == 0 else None

    sigxy_auxR = np.ascontiguousarray(np.real(sigxy_aux))
    sigxy_auxI = np.ascontiguousarray(np.imag(sigxy_aux))

    comm.Reduce(sigxy_auxR, sigxyR, op=MPI.SUM)
    comm.Reduce(sigxy_auxI, sigxyI, op=MPI.SUM)

    sigxy_aux = sigxy_auxR = sigxy_auxI = None

    if rank == 0:
        sigxy = (sigxyR + 1j * sigxyI) / float(attr['nkpnts'])
        return (ene, sigxy)
    else:
        return (None, None)


def smear_sigma_loop(
    data_controller, ene, pksp_i, pksp_j, ispin, ipol, jpol, width_floor: float = 0.0
) -> np.ndarray:
    """Evaluate the smeared off-diagonal conductivity sum for a single spin channel.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k``, ``deltakp``, ``deltakp2``, ``delta``.
        Required attributes: ``smearing``, ``temp``.
    ene : np.ndarray, shape ``(esize,)``
        Frequency grid (eV).
    pksp_i : np.ndarray, shape ``(snktot, nawf, nawf, nspin)``, complex
        Left matrix elements.
    pksp_j : np.ndarray, shape ``(snktot, nawf, nawf, nspin)``, complex
        Right matrix elements.
    ispin : int
        Spin channel index.
    ipol : int
        Left polarisation index.
    jpol : int
        Right polarisation index.
    width_floor : float
        Lower bound of the adaptive pair widths ``deltakp2`` (eV); 0 keeps them as they are.

    Returns
    -------
    np.ndarray, shape ``(esize,)``, complex
        Per-rank partial sum of the conductivity; call ``MPI.Reduce`` to
        accumulate the global result.
    """
    from ..utils.smearing import intgaussian, intmetpax

    arry, attr = data_controller.data_dicts()

    esize = ene.size
    sigxy = np.zeros((esize), dtype=complex)

    snktot, nawf, _, nspin = pksp_j.shape
    f_nm = np.zeros((snktot, nawf, nawf), dtype=float)
    E_diff_nm = np.zeros((snktot, nawf, nawf), dtype=float)

    Ef = 0.0
    eps = 1.0e-16

    if attr['smearing'] == None:
        fn = 1.0 / (np.exp(arry['E_k'][:, :, ispin] / attr['temp']) + 1)
    elif attr['smearing'] == 'gauss':
        fn = intgaussian(arry['E_k'][:, :, ispin], Ef, arry['deltakp'][:, :, ispin])
    elif attr['smearing'] == 'm-p':
        fn = intmetpax(arry['E_k'][:, :, ispin], Ef, arry['deltakp'][:, :, ispin])

    # Collapsing the sum over k points
    for n in range(nawf):
        for m in range(nawf):
            if m != n:
                E_diff_nm[:, n, m] = (arry['E_k'][:, n, ispin] - arry['E_k'][:, m, ispin]) ** 2
                f_nm[:, n, m] = (fn[:, n] - fn[:, m]) * np.imag(
                    pksp_j[:, n, m, ispin] * pksp_i[:, m, n, ispin]
                )

    fn = None

    widths = arry['deltakp2'][:, :nawf, :nawf, ispin]
    if width_floor > 0.0:
        widths = np.maximum(widths, width_floor)
    for e in range(esize):
        if attr['smearing'] != None:
            sigxy[e] = np.sum(
                f_nm[:, :, :] / (E_diff_nm[:, :, :] - (ene[e] + 1.0j * widths) ** 2 + eps)
            )
        else:
            sigxy[e] = np.sum(
                f_nm[:, :, :] / (E_diff_nm[:, :, :] - (ene[e] + 1.0j * arry['delta']) ** 2 + eps)
            )

    E_diff_nm = None

    return np.nan_to_num(sigxy)


def do_spin_current(data_controller, spol, ipol):
    """Compute the spin current operator :math:`j^{s}_{\\rm pol}` in the band basis.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays``.
        Required arrays: ``Sj`` (shape ``(3, nawf, nawf)``),
        ``dHksp`` (shape ``(snktot, 3, nawf, nawf, nspin)``).
    spol : int
        Spin polarisation component index (0=x, 1=y, 2=z).
    ipol : int
        Momentum/velocity component index.

    Returns
    -------
    np.ndarray, shape ``(snktot, nawf, nawf, nspin)``, complex
        Symmetrised spin current matrix elements
        :math:`j^{s_\\rm pol}_{nm}(\\mathbf{k}) =
        \\tfrac{1}{2}\\{S^{s_\\rm pol}, \\partial_{i_\\rm pol} H\\}_{nm}`.
    """
    arry, attr = data_controller.data_dicts()

    Sj = arry['Sj'][spol]
    snktot, _, nawf, nawf, nspin = arry['dHksp'].shape

    jdHksp = np.empty((snktot, nawf, nawf, nspin), dtype=complex)

    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik, :, :, ispin] = 0.5 * (
                np.dot(Sj, arry['dHksp'][ik, ipol, :, :, ispin])
                + np.dot(arry['dHksp'][ik, ipol, :, :, ispin], Sj)
            )

    return jdHksp


def do_orbital_current(data_controller, spol, ipol):
    """Compute the orbital current operator :math:`j^{s}_{\\rm pol}` in the band basis.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays``.
        Required arrays: ``Lj`` (shape ``(3, nawf, nawf)``),
        ``dHksp`` (shape ``(snktot, 3, nawf, nawf, norbital)``).
    spol : int
        Orbital polarisation component index (0=x, 1=y, 2=z).
    ipol : int
        Momentum/velocity component index.

    Returns
    -------
    np.ndarray, shape ``(snktot, nawf, nawf, norbital)``, complex
        Symmetrised orbital current matrix elements
        :math:`j^{s_\\rm pol}_{nm}(\\mathbf{k}) =
        \\tfrac{1}{2}\\{L^{s_\\rm pol}, \\partial_{i_\\rm pol} H\\}_{nm}`.
    """
    arry, attr = data_controller.data_dicts()

    Lj = arry['Lj'][spol]
    snktot, _, nawf, nawf, nspin = arry['dHksp'].shape

    jdHksp = np.empty((snktot, nawf, nawf, nspin), dtype=complex)

    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik, :, :, ispin] = 0.5 * (
                np.dot(Lj, arry['dHksp'][ik, ipol, :, :, ispin])
                + np.dot(arry['dHksp'][ik, ipol, :, :, ispin], Lj)
            )

    return jdHksp
