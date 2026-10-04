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
                jdHk = current_operator(Sj, dHk)
                jksp_is[ik, :, :, ispin], pksp_j[ik, :, :, ispin] = perturb_split(
                    project_current(P, jdHk),
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

        cgs_conv = hall_conversion(data_controller, twoD)
        names = hall_file_names('spin', ipol, jpol, spol)
        write_berry_outputs(data_controller, names, ene, shc, Om_k, cgs_conv)
        ene = shc = Om_k = None

        if do_ac:
            jksp_js = np.empty((snktot, nawf, nawf, nspin), dtype=complex)
            pksp_i = np.empty((snktot, nawf, nawf, nspin), dtype=complex)
            for ik in range(snktot):
                for ispin in range(nspin):
                    dHk = arry['dHksp'][ik, ipol, :, :, ispin]
                    jksp_js[ik, :, :, ispin], pksp_i[ik, :, :, ispin] = perturb_split(
                        current_operator(Sj, dHk),
                        arry['dHksp'][ik, jpol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            dHk = None

            ene, sigxy = do_ac_conductivity(data_controller, jksp_js, pksp_i, ipol, jpol)
            jksp_js = pksp_i = None
            write_ac_outputs(data_controller, names, ene, sigxy, cgs_conv)


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
    from ..utils.perturb_split import perturb_split

    arry, attr = data_controller.data_dicts()

    o_tensor = arry['o_tensor']

    # -----------------------
    # Orbital Hall calculation
    # -----------------------
    if attr['dftSO'] == False:
        if rank == 0:
            print('Relativistic calculation with SO required')
            comm.Abort()
        comm.Barrier()

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
                    project_current(P, jdHksp[ik, :, :, ispin]),
                    arry['dHksp'][ik, jpol, :, :, ispin],
                    arry['v_k'][ik, :, :, ispin],
                    arry['degen'][ispin][ik],
                )
        jdHksp = None

        # ---------------------------------
        # Compute orbital Berry curvature...
        # ---------------------------------
        ene, ohc, Om_k = do_Berry_curvature(data_controller, jksp_is, pksp_j)

        cgs_conv = hall_conversion(data_controller, twoD)
        names = hall_file_names('orbital', ipol, jpol, spol)
        write_berry_outputs(data_controller, names, ene, ohc, Om_k, cgs_conv)
        ene = ohc = Om_k = None

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
            write_ac_outputs(data_controller, names, ene, sigxy, cgs_conv)


def do_anomalous_Hall(data_controller, do_ac):
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

    Returns
    -------
    None
        Writes per-component output files to ``opath``:

        - ``Berry_{ipol}{jpol}.bxsf`` — Berry curvature on the k-grid.
        - ``ahcEf_{ipol}{jpol}.dat`` — anomalous Hall conductivity vs.
          Fermi energy.
        - When ``do_ac``: ``MCDr_{...}.dat`` and ``MCDi_{...}.dat``.
    """
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

        cgs_conv = hall_conversion(data_controller, False)
        names = hall_file_names('anomalous', ipol, jpol)
        write_berry_outputs(data_controller, names, ene, ahc, Om_k, cgs_conv)
        ene = ahc = Om_k = None

        if do_ac:
            ene, sigxy = do_ac_conductivity(data_controller, pksp_i, pksp_j, ipol, jpol)
            write_ac_outputs(data_controller, names, ene, sigxy, cgs_conv)

        pksp_i = pksp_j = None


def current_operator(O, dH):
    """Symmetrized current :math:`\\tfrac12\\{O, \\partial_i H\\}` of an operator ``O``.

    Notes
    -----
    Written with ``@`` so it serves ndarrays (``do_spin_Hall``) and the
    sparse backend's CSR matrices alike.
    """
    return 0.5 * (O @ dH + dH @ O)


def project_current(P, J):
    """Site projection :math:`\\tfrac12 (P J + J P)` of a current operator."""
    return 0.5 * (P @ J + J @ P)


_HALL_FILES = {
    # Berry curvature bxsf, Fermi-energy scan, AC imaginary / real parts
    'anomalous': ('Berry_%s%s.bxsf', 'ahcEf_%s%s.dat', 'MCDi_%s%s.dat', 'MCDr_%s%s.dat'),
    'spin': (
        'Spin_Berry_%s_%s%s.bxsf',
        'shcEf_%s_%s%s.dat',
        'SCDi_%s_%s%s.dat',
        'SCDr_%s_%s%s.dat',
    ),
    'orbital': (
        'Orbital_Berry_%s_%s%s.bxsf',
        'ohcEf_%s_%s%s.dat',
        'OCDi_%s_%s%s.dat',
        'OCDr_%s_%s%s.dat',
    ),
}


def hall_file_names(kind: str, ipol: int, jpol: int, spol: int | None = None) -> tuple[str, ...]:
    """Output names of one Hall tensor component: bxsf, Fermi scan, AC imaginary, AC real."""
    from ..utils.constants import LL

    if spol is None:
        cart_indices = (str(LL[ipol]), str(LL[jpol]))
    else:
        cart_indices = (str(LL[spol]), str(LL[ipol]), str(LL[jpol]))
    return tuple(name % cart_indices for name in _HALL_FILES[kind])


def hall_conversion(data_controller, twoD: bool):
    """Conversion of the Berry-curvature integral to conductivity units (rank 0, else None).

    ``twoD`` uses the in-plane cell area (Ohm^-1) instead of the volume.
    """
    from ..utils.constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI

    arry, attr = data_controller.data_dicts()
    if rank != 0:
        return None
    if twoD:
        av0, av1 = arry['a_vectors'][0, :], arry['a_vectors'][1, :]
        return 1.0 / (np.linalg.norm(np.cross(av0, av1)) * attr['alat'] ** 2)
    return 1.0e8 * ANGSTROM_AU * ELECTRONVOLT_SI**2 / (H_OVER_TPI * attr['omega'])


def write_berry_outputs(data_controller, names, ene, value, Om_k, cgs_conv) -> None:
    """Write the Berry-curvature bxsf and the conductivity scan of one component.

    ``value`` is scaled by ``cgs_conv`` in place on rank 0 (collective call).
    """
    attr = data_controller.data_attributes
    nk1, nk2, nk3 = attr['nk1'], attr['nk2'], attr['nk3']
    Om_kps = np.empty((nk1, nk2, nk3, 2), dtype=float) if rank == 0 else None
    if rank == 0:
        Om_kps[:, :, :, 0] = Om_kps[:, :, :, 1] = Om_k[:, :, :]
    data_controller.write_bxsf(names[0], Om_kps, 2)
    Om_kps = None

    if rank == 0:
        value *= cgs_conv
    data_controller.write_file_row_col(names[1], ene, value)


def write_ac_outputs(data_controller, names, ene, sigxy, cgs_conv) -> None:
    """Write the imaginary and real AC conductivity of one component (collective call)."""
    if rank == 0:
        sigxy *= cgs_conv
    sigxyi = np.imag(ene * sigxy / 105.4571) if rank == 0 else None
    sigxyr = np.real(sigxy) if rank == 0 else None
    sigxy = None

    data_controller.write_file_row_col(names[2], ene, sigxyi)
    data_controller.write_file_row_col(names[3], ene, sigxyr)


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
    arrays, attributes = data_controller.data_dicts()

    snktot, nawf, _, nspin = pksp.shape

    # Compute only Omega_z(k)
    Om_znkaux = np.zeros((snktot, nawf), dtype=float)

    deltap = attributes['deltaH']
    for ik in range(snktot):
        Om_znkaux[ik] = berry_curvature_k(
            arrays['E_k'][ik, :, 0], jksp[ik, :, :, 0], pksp[ik, :, :, 0], deltap
        )

    ene = berry_energy_grid(attributes)
    Om_zkaux = berry_occupation_sum(
        arrays['E_k'][:, :, 0], arrays['deltakp'][:, :, 0], Om_znkaux, ene, attributes['smearing']
    )
    return berry_reduce(data_controller, Om_zkaux, ene)


def berry_curvature_k(E: np.ndarray, jk: np.ndarray, pk: np.ndarray, deltap: float) -> np.ndarray:
    """Band-resolved Berry curvature :math:`\\Omega_n` at one k-point.

    Parameters
    ----------
    E : np.ndarray, shape ``(nawf,)``
        Every eigenvalue at this k-point (eV); the sum over ``m`` runs over
        all of them.
    jk, pk : np.ndarray, shape ``(nawf, nawf)``, complex
        Left (current) and right (momentum) matrix elements in the band basis.
    deltap : float
        Broadening :math:`\\delta` of the energy denominator (eV).

    Returns
    -------
    np.ndarray, shape ``(nawf,)``
        :math:`-2 \\sum_m \\mathrm{Im}[j_{nm} p_{mn}] / ((E_n - E_m)^2 + \\delta^2)`,
        with near-zero denominators (below 1e-4) excluded.
    """
    E_nm = (E - E[:, None]) ** 2 + deltap**2
    E_nm[np.where(E_nm < 1.0e-4)] = np.inf
    return -2.0 * np.sum(np.imag(jk * pk.T) / E_nm, axis=1)


def berry_energy_grid(attributes: dict) -> np.ndarray:
    """Fermi-energy grid of the Berry-curvature integral.

    Notes
    -----
    Clips ``attributes['emaxH']`` to the PAO ``shift`` in place, as
    :func:`do_Berry_curvature` always has.
    """
    if attributes['shift'] != 0.0:
        attributes['emaxH'] = np.amin(np.array([attributes['shift'], attributes['emaxH']]))

    ### Hardcoded 'de'
    esize = attributes['esizeH']
    return np.linspace(attributes['eminH'], attributes['emaxH'], esize)


def berry_occupation_sum(
    E_k: np.ndarray,
    deltakp: np.ndarray,
    Om_znkaux: np.ndarray,
    ene: np.ndarray,
    smearing: str | None,
) -> np.ndarray:
    """Occupation-weighted band sum of the Berry curvature, per k-point and Fermi energy.

    Parameters
    ----------
    E_k, deltakp : np.ndarray, shape ``(nk, nbands)``
        Eigenvalues and adaptive widths of a block of k-points.
    Om_znkaux : np.ndarray, shape ``(nk, nbands)``
        :func:`berry_curvature_k` of the same k-points.
    ene : np.ndarray, shape ``(esize,)``
        Fermi energies.
    smearing : {'gauss', 'm-p', None}
        Occupation function.

    Returns
    -------
    np.ndarray, shape ``(nk, esize)``
        Each row depends only on its own k-point, so a one-point block gives
        the row of a whole-slice call bit for bit.
    """
    from ..utils.smearing import intgaussian, intmetpax

    esize = ene.size
    Om_zkaux = np.zeros((E_k.shape[0], esize), dtype=float)
    for i in range(esize):
        if smearing == 'gauss':
            Om_zkaux[:, i] = np.sum((Om_znkaux[:, :] * intgaussian(E_k, ene[i], deltakp)), axis=1)
        elif smearing == 'm-p':
            Om_zkaux[:, i] = np.sum(Om_znkaux[:, :] * intmetpax(E_k, ene[i], deltakp), axis=1)
        else:
            Om_zkaux[:, i] = np.sum(Om_znkaux[:, :] * (0.5 * (-np.sign(E_k - ene[i]) + 1)), axis=1)
    return Om_zkaux


def berry_reduce(data_controller, Om_zkaux: np.ndarray, ene: np.ndarray):
    """Gather the per-k Berry sums and integrate over the BZ.

    Parameters
    ----------
    data_controller : DataController
        Supplies ``npool``, ``nkpnts``, the mesh and ``fermi_up``/``fermi_dw``.
    Om_zkaux : np.ndarray, shape ``(nk_local, esize)``
        :func:`berry_occupation_sum` of this rank's k-points.
    ene : np.ndarray, shape ``(esize,)``

    Returns
    -------
    (ene, shc, Om_zk) : tuple
        As :func:`do_Berry_curvature` (``shc`` and ``Om_zk`` on rank 0 only).
    """
    from ..utils.communication import gather_full

    arrays, attributes = data_controller.data_dicts()
    fermi_up, fermi_dw = attributes['fermi_up'], attributes['fermi_dw']
    nk1, nk2, nk3 = attributes['nk1'], attributes['nk2'], attributes['nk3']
    esize = ene.size

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
    ### Hardcode 'de'
    esize = attr['esizeH']
    ene = np.linspace(emin, emax, esize)

    sigxy_aux = smear_sigma_loop(data_controller, ene, jksp, pksp, ispin, ipol, jpol)
    return ac_conductivity_reduce(data_controller, sigxy_aux, ene)


def ac_frequency_grid(attributes: dict) -> np.ndarray:
    """Frequency grid of :func:`do_ac_conductivity`, ``[0, shift]``."""
    ### Hardcode 'de'
    return np.linspace(0.0, attributes['shift'], attributes['esizeH'])


def ac_conductivity_reduce(data_controller, sigxy_aux: np.ndarray, ene: np.ndarray):
    """Sum the per-rank optical conductivity and normalize by the k count.

    Returns
    -------
    (ene, sigxy) : tuple
        As :func:`do_ac_conductivity`; ``(None, None)`` off rank 0.
    """
    arry, attr = data_controller.data_dicts()
    esize = ene.size

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


def smear_sigma_loop(data_controller, ene, pksp_i, pksp_j, ispin, ipol, jpol):
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

    Returns
    -------
    np.ndarray, shape ``(esize,)``, complex
        Per-rank partial sum of the conductivity; call ``MPI.Reduce`` to
        accumulate the global result.
    """
    arry, attr = data_controller.data_dicts()

    fn = occupations(arry['E_k'][:, :, ispin], arry['deltakp'][:, :, ispin], attr)
    nawf = pksp_j.shape[1]
    if attr['smearing'] != None:
        broadening = arry['deltakp2'][:, :nawf, :nawf, ispin]
    else:
        broadening = arry['delta']
    return smear_sigma_block(
        arry['E_k'][:, :, ispin],
        fn,
        pksp_i[:, :, :, ispin],
        pksp_j[:, :, :, ispin],
        ene,
        broadening,
    )


def occupations(E_k: np.ndarray, deltakp: np.ndarray, attr: dict) -> np.ndarray:
    """Occupations at :math:`E_F = 0` of :func:`smear_sigma_loop`.

    Parameters
    ----------
    E_k, deltakp : np.ndarray, shape ``(nk, nbands)``
        Eigenvalues and adaptive widths of a block of k-points.
    attr : dict
        Supplies ``smearing`` and, without smearing, ``temp``.
    """
    from ..utils.smearing import intgaussian, intmetpax

    Ef = 0.0
    if attr['smearing'] == None:
        fn = 1.0 / (np.exp(E_k / attr['temp']) + 1)
    elif attr['smearing'] == 'gauss':
        fn = intgaussian(E_k, Ef, deltakp)
    elif attr['smearing'] == 'm-p':
        fn = intmetpax(E_k, Ef, deltakp)
    return fn


def smear_sigma_block(
    E_k: np.ndarray,
    fn: np.ndarray,
    pksp_i: np.ndarray,
    pksp_j: np.ndarray,
    ene: np.ndarray,
    broadening,
) -> np.ndarray:
    """Smeared off-diagonal conductivity sum over a block of k-points.

    Parameters
    ----------
    E_k, fn : np.ndarray, shape ``(nk, nawf)``
        Eigenvalues and occupations; every state of each k-point.
    pksp_i, pksp_j : np.ndarray, shape ``(nk, nawf, nawf)``, complex
        Left and right matrix elements in the band basis.
    ene : np.ndarray, shape ``(esize,)``
        Frequency grid (eV).
    broadening : np.ndarray, shape ``(nk, nawf, nawf)``, or float
        Interband adaptive widths ``deltakp2``, or a fixed width.

    Returns
    -------
    np.ndarray, shape ``(esize,)``, complex
        The block's partial sum; the dense kernel passes its whole k-slice,
        the sparse backend one k-point at a time.
    """
    esize = ene.size
    sigxy = np.zeros((esize), dtype=complex)
    snktot, nawf, _ = pksp_j.shape
    f_nm = np.zeros((snktot, nawf, nawf), dtype=float)
    E_diff_nm = np.zeros((snktot, nawf, nawf), dtype=float)
    eps = 1.0e-16

    # Collapsing the sum over k points
    for n in range(nawf):
        for m in range(nawf):
            if m != n:
                E_diff_nm[:, n, m] = (E_k[:, n] - E_k[:, m]) ** 2
                f_nm[:, n, m] = (fn[:, n] - fn[:, m]) * np.imag(pksp_j[:, n, m] * pksp_i[:, m, n])
    fn = None

    for e in range(esize):
        sigxy[e] = np.sum(
            f_nm[:, :, :] / (E_diff_nm[:, :, :] - (ene[e] + 1.0j * broadening) ** 2 + eps)
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
