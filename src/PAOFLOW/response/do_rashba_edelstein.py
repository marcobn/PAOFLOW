from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_rashba_edelstein(
    data_controller,
    ene,
    temperature,
    regularization,
    twoD_structure,
    lattice_height,
    structure_thickness,
    write_to_file,
    Op_text,
    filename,
):
    """Compute the Rashba-Edelstein effect tensor as a function of energy.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``v_k`` (shape ``(nkpnts, nawf, nawf, nspin)``),
        ``pksp`` (shape ``(nkpnts, 3, nawf, nawf, nspin)``),
        ``deltakp`` (adaptive smearing widths), ``E_k``,
        ``sktxt`` (spin texture), ``ind_plot``.
        Required attributes: ``smearing``, ``opath``.
    ene : np.ndarray, shape ``(ne,)``
        Energy grid (eV) at which the tensors are evaluated.
    temperature : float
        Electronic temperature (eV).  Use ``0`` to apply the zero-temperature
        (delta-function) Gaussian smearing.
    regularization : float
        Small positive constant (SI units) added to the current denominator
        to avoid divergences.
    twoD_structure : bool
        If ``True``, the tensor is rescaled by ``lattice_height / structure_thickness``
        to convert from 3-D to 2-D units.
    lattice_height : float
        Out-of-plane lattice constant used for 2-D rescaling.
    structure_thickness : float
        Physical thickness of the 2-D slab used for 2-D rescaling.
    write_to_file : bool
        If ``True``, write ``kai.dat``, ``current.dat``, and
        ``Ekai_{si}{sj}.dat`` files to ``opath``.

    Returns
    -------
    None
        All output is written to disk when ``write_to_file`` is ``True``.

    Notes
    -----
    The Rashba-Edelstein (inverse spin galvanic) tensor component
    :math:`\\chi_{ij}` and the longitudinal current tensor :math:`j_{ii}`
    are computed via a Boltzmann-like sum over k-points and bands within
    the energy window set by ``ind_plot``.  The effective field response is

    .. math::

        E^{\\rm kai}_{ij}(\\varepsilon) =
            -\\frac{\\hbar\\,\\chi_{ij}(\\varepsilon)}
            {j_{jj}(\\varepsilon)\\, e a_0}

    Smearing is applied through either the zero-temperature Gaussian or
    the finite-temperature derivative of the Fermi-Dirac function
    :math:`-\\partial f / \\partial E`.
    """
    import numpy as np

    arrays, attr = data_controller.data_dicts()

    snktot = arrays['v_k'].shape[0]
    ind_plot = arrays['ind_plot']
    nstates = len(ind_plot)

    pksp = np.take(
        np.diagonal(np.real(arrays['pksp'][:, :, :, :, 0]), axis1=2, axis2=3), ind_plot, axis=2
    )

    deltakp = np.take(arrays['deltakp'], ind_plot, axis=1)[:, :, 0]
    E_k = np.take(arrays['E_k'], ind_plot, axis=1)[:, :, 0]
    St = _local_texture(data_controller, Op_text, snktot, nstates)
    ree_tensor(
        data_controller,
        ene,
        St,
        pksp,
        deltakp,
        E_k,
        temperature,
        regularization,
        twoD_structure,
        lattice_height,
        structure_thickness,
        write_to_file,
        filename,
    )


def _local_texture(data_controller, Op_text, snktot: int, nstates: int):
    """This rank's k-slice of a texture that ``do_spin_texture`` gathered on rank 0.

    Notes
    -----
    In a serial run the gathered array is the local one; under MPI it is
    scattered back to the k-distribution of ``pksp`` (previously the
    gathered array was reshaped to the local k count, which only works on
    one rank).
    """
    import numpy as np

    from ..utils.communication import scatter_full

    if data_controller.comm.Get_size() == 1:
        return np.real(Op_text).reshape(snktot, 3, nstates)
    full = None
    if data_controller.rank == 0:
        full = np.ascontiguousarray(np.real(Op_text).reshape(-1, 3, nstates))
    return scatter_full(full, data_controller.data_attributes['npool'])


def ree_tensor(
    data_controller,
    ene,
    St,
    pksp,
    deltakp,
    E_k,
    temperature,
    regularization,
    twoD_structure,
    lattice_height,
    structure_thickness,
    write_to_file,
    filename,
):
    """Rashba-Edelstein sums over this rank's k-points, reduced and written.

    Parameters
    ----------
    data_controller : DataController
        Supplies ``smearing`` and ``opath``; performs the reduction.
    ene : np.ndarray, shape ``(ne,)``
    St, pksp : np.ndarray, shape ``(nk_local, 3, nstates)``
        Band-diagonal texture and velocities of the selected bands.
    deltakp, E_k : np.ndarray, shape ``(nk_local, nstates)``
        Their adaptive widths and energies.
    temperature, regularization, twoD_structure, lattice_height,
    structure_thickness, write_to_file, filename
        As :func:`do_rashba_edelstein`.

    Returns
    -------
    None
        Rank 0 writes the files when ``write_to_file``.
    """
    from os.path import join

    import numpy as np

    from ..utils.constants import BOHR_RADIUS_CM, ELECTRONVOLT_SI, HBAR
    from ..utils.smearing import gaussian

    comm, rank = data_controller.comm, data_controller.rank
    arrays, attr = data_controller.data_dicts()

    snktot = E_k.shape[0]
    nstates = E_k.shape[1]
    tau_const = 1.0
    esize = ene.size

    kai_aux = np.zeros((snktot, 3, 3, nstates), dtype=float)
    j_aux = np.zeros((snktot, 3, 3, nstates), dtype=float)
    for l in range(3):
        for m in range(3):
            kai_aux[:, l, m, :] = tau_const * St[:, l, :] * pksp[:, m, :]
            j_aux[:, l, l, :] = tau_const * pksp[:, l, :] * pksp[:, l, :]

    kai_eaux = np.zeros((snktot, 3, 3, esize), dtype=float)
    j_eaux = np.zeros((snktot, 3, 3, esize), dtype=float)

    def dfermi(E, ene, temp):
        return -1 / (4 * temp * (np.cosh((E - ene) / (2 * temp)) ** 2))

    for i in range(esize):
        gaussian_smear = None
        if attr['smearing'] == 'gauss':
            if temperature == 0:
                gaussian_smear = gaussian(E_k, ene[i], deltakp)
            else:
                gaussian_smear = dfermi(E_k, ene[i], temperature)
        else:
            raise ValueError("Routine requires 'gauss' smearing")
        for l in range(3):
            for m in range(3):
                kai_eaux[:, l, m, i] = np.sum(kai_aux[:, l, m, :] * gaussian_smear, axis=1)
                j_eaux[:, l, l, i] = np.sum(j_aux[:, l, l, :] * gaussian_smear, axis=1)
    kai_aux = None
    j_aux = None

    kai = np.zeros((3, 3, esize), dtype=float) if rank == 0 else None
    jc = np.zeros((3, 3, esize), dtype=float) if rank == 0 else None

    kai_eaux = np.ascontiguousarray(np.sum(kai_eaux, axis=0))
    j_eaux = np.ascontiguousarray(np.sum(j_eaux, axis=0))

    comm.Reduce(kai_eaux, kai, op=MPI.SUM)
    comm.Reduce(j_eaux, jc, op=MPI.SUM)

    if rank == 0:
        Ekai = np.empty((3, 3, esize), dtype=float)
        for i in range(3):
            for j in range(3):
                Ekai[i, j] = (
                    -HBAR
                    * kai[i, j]
                    / (jc[j, j] * ELECTRONVOLT_SI * BOHR_RADIUS_CM + regularization)
                )

        if twoD_structure:
            Ekai *= lattice_height / structure_thickness

        sEkai = {0: 'x', 1: 'y', 2: 'z'}
        wEkai = lambda fn, e, t: fn.write('% .5f % 9.5e\n' % (e, t))
        wtup = lambda fn, tu: fn.write(
            '% .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n' % tu
        )
        gtup = lambda tu, i: (
            ene[i],
            tu[0, 0, i],
            tu[0, 1, i],
            tu[0, 2, i],
            tu[1, 0, i],
            tu[1, 1, i],
            tu[1, 2, i],
            tu[2, 0, i],
            tu[2, 1, i],
            tu[2, 2, i],
        )

        if write_to_file:
            fkai = open(join(attr['opath'], filename + 'kai.dat'), 'w')
            fcurrent = open(join(attr['opath'], filename + 'current.dat'), 'w')

            ofE = lambda si, sj: open(join(attr['opath'], filename + f'Ekai_{si}{sj}.dat'), 'w')
            fEkai = [[ofE(sEkai[i], sEkai[j]) for j in range(3)] for i in range(3)]

            for ie in range(esize):
                wtup(fkai, gtup(kai, ie))
                wtup(fcurrent, gtup(jc, ie))
                for i in range(3):
                    for j in range(3):
                        wEkai(fEkai[i][j], ene[ie], Ekai[i, j, ie])

            fkai.close()
            fcurrent.close()
            for i in range(3):
                for j in range(3):
                    fEkai[i][j].close()


def ree_intra_products_k(operator, dH_i, v_k, degen):
    """Band-diagonal :math:`\\langle O\\rangle\\langle v_i\\rangle` and :math:`\\langle v_i\\rangle^2` at one k-point.

    Parameters
    ----------
    operator : np.ndarray or scipy.sparse matrix, shape ``(nawf, nawf)``
        The (site-projected) spin or orbital component.
    dH_i : np.ndarray or scipy.sparse matrix, shape ``(nawf, nawf)``
        ``dH/dk`` along the field direction.
    v_k : np.ndarray, shape ``(nawf, m)``
    degen : list of np.ndarray

    Returns
    -------
    (spin_velocity, velocity_squared) : tuple of np.ndarray, shape ``(m,)``
        Diagonals of ``O1 * O2`` and ``O2 * O2``, both projected in the basis
        that diagonalizes ``operator`` inside each degenerate group.
    """
    import numpy as np

    from ..utils.perturb_split import perturb_split

    O1, O2 = perturb_split(operator, dH_i, v_k, degen)
    return np.diagonal(O1 * O2), np.diagonal(O2 * O2)


def do_rashba_edelstein_intra(data_controller, prefix_file, ene, delta, ipol, spol, Op1, P):
    """Intra-band Rashba-Edelstein response :math:`\\chi_{ij}` of one component.

    Notes
    -----
    Per k-point the work is :func:`ree_intra_products_k`; the smearing,
    reduction and write are :func:`ree_intra_from_products`, which the
    sparse backend calls with products streamed from its mesh pass.
    """
    import numpy as np

    arrays, attributes = data_controller.data_dicts()

    nspin = attributes['nspin']
    nktot = arrays['pksp'].shape[0]
    nawf = attributes['nawf']

    spin_velocity = np.zeros((nktot, nawf, nspin), dtype=complex)
    velocity_squared = np.zeros((nktot, nawf, nspin), dtype=complex)
    for ik in range(nktot):
        for ispin in range(nspin):
            spin_velocity[ik, :, ispin], velocity_squared[ik, :, ispin] = ree_intra_products_k(
                0.5 * (P @ Op1[spol, :, :] + Op1[spol, :, :] @ P),
                arrays['dHksp'][ik, ipol, :, :, ispin],
                arrays['v_k'][ik, :, :, ispin],
                arrays['degen'][ispin][ik],
            )

    for ispin in range(nspin):
        ree_intra_from_products(
            data_controller,
            prefix_file,
            ene,
            delta,
            ipol,
            spol,
            spin_velocity[:, :, ispin],
            velocity_squared[:, :, ispin],
            ispin,
        )


def ree_intra_from_products(
    data_controller, prefix_file, ene, delta, ipol, spol, spin_velocity, velocity_squared, ispin
):
    """Smear, reduce and write the intra-band Rashba-Edelstein response of one spin.

    Parameters
    ----------
    data_controller : DataController
        Supplies ``E_k``/``deltakp`` (bands of the products) and the
        ``ree_*`` unit factors; performs the write.
    spin_velocity, velocity_squared : np.ndarray, shape ``(nk_local, nbands)``
        :func:`ree_intra_products_k` of every local k-point.
    """
    import numpy as np

    from ..utils.constants import BOHR_RADIUS_CM, ELECTRONVOLT_SI, HBAR, LL
    from ..utils.smearing import gaussian, metpax

    arrays, attributes = data_controller.data_dicts()

    # Unit-defining factors, shared with do_rashba_edelstein via the wrapper, so
    # this intra-band routine reports chi_{ij} = -hbar * kai_{ij} / (jc_{jj} e a0)
    # in the SAME units (kai = Sum <S_i><v_j> delta ; jc = Sum <v_j><v_j> delta).
    reg = attributes.get('ree_reg', 1e-30)
    twoD = attributes.get('ree_twoD', False)
    lt = attributes.get('ree_lt', 1.0)
    st = attributes.get('ree_st', 1.0)

    ne = ene.size
    nbands = spin_velocity.shape[1]

    E_k = np.real(arrays['E_k'][:, :nbands, ispin])
    deltakp = arrays['deltakp'][:, :nbands, ispin]

    accaux = np.zeros((ne), dtype=float)  # kai numerator:  Sum <S_spol><v_ipol>
    jcaux = np.zeros((ne), dtype=float)  # current (field dir.): Sum <v_ipol><v_ipol>

    for n in range(ne):
        # Adaptive Gaussian Smearing
        if attributes['smearing'] == 'gauss':
            taux = gaussian(ene[n], E_k, deltakp)
        # Adaptive M-P smearing
        elif attributes['smearing'] == 'm-p':
            taux = metpax(ene[n], E_k, deltakp)
        elif attributes['smearing'] == None:
            taux = np.exp(-(((ene[n] - E_k[:, :]) / delta) ** 2)) / np.sqrt(np.pi)

        accaux[n] += np.sum(np.real(taux * spin_velocity))
        jcaux[n] += np.sum(np.real(taux * velocity_squared))

    acc = np.zeros((ne), dtype=float) if rank == 0 else None
    jc = np.zeros((ne), dtype=float) if rank == 0 else None

    comm.Reduce(accaux, acc, op=MPI.SUM)
    comm.Reduce(jcaux, jc, op=MPI.SUM)
    accaux = jcaux = None

    # chi_{ij} = -hbar * kai / (jc * e * a0), as in do_rashba_edelstein.
    # tau and E_x cancel in the kai/jc ratio, and so do the (1/nkpnts) and
    # smearing normalizations (both numerator and denominator carry them).
    chi = None
    if rank == 0:
        chi = -HBAR * acc / (jc * ELECTRONVOLT_SI * BOHR_RADIUS_CM + reg)
        if twoD:
            chi *= lt / st

    facc = '%s_reeEf_%s%s.dat' % (prefix_file, str(LL[ipol]), str(LL[spol]))
    data_controller.write_file_row_col(facc, ene, chi)
