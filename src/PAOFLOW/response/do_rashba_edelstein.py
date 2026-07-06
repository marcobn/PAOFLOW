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
    from os.path import join
    from ..utils.smearing import gaussian
    from ..utils.constants import ELECTRONVOLT_SI, BOHR_RADIUS_CM, HBAR

    comm, rank = data_controller.comm, data_controller.rank
    arrays, attr = data_controller.data_dicts()

    snktot = arrays['v_k'].shape[0]
    ind_plot = arrays['ind_plot']
    nstates = len(ind_plot)
    tau_const = 1.0
    esize = ene.size

    pksp = np.take(
        np.diagonal(np.real(arrays['pksp'][:, :, :, :, 0]), axis1=2, axis2=3), ind_plot, axis=2
    )

    deltakp = np.take(arrays['deltakp'], ind_plot, axis=1)[:, :, 0]
    E_k = np.take(arrays['E_k'], ind_plot, axis=1)[:, :, 0]
    St = np.real(Op_text).reshape(snktot, 3, nstates)

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
            fkai = open(join(attr['opath'], filename + '_kai.dat'), 'w')
            fcurrent = open(join(attr['opath'], filename + '_current.dat'), 'w')

            ofE = lambda si, sj: open(join(attr['opath'], filename + f'_Ekai_{si}{sj}.dat'), 'w')
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
def do_rashba_edelstein_intra(data_controller, prefix_file, ene, delta, ipol, spol, Op1, P):
    import numpy as np
    from ..utils.perturb_split import perturb_split
    from ..utils.smearing import metpax, gaussian
    from ..utils.constants import LL, HBAR, ELECTRONVOLT_SI, BOHR_RADIUS_CM

    arrays, attributes = data_controller.data_dicts()

    # Unit-defining factors, shared with do_rashba_edelstein via the wrapper, so
    # this intra-band routine reports chi_{ij} = -hbar * kai_{ij} / (jc_{jj} e a0)
    # in the SAME units (kai = Sum <S_i><v_j> delta ; jc = Sum <v_j><v_j> delta).
    reg = attributes.get('ree_reg', 1e-30)
    twoD = attributes.get('ree_twoD', False)
    lt = attributes.get('ree_lt', 1.0)
    st = attributes.get('ree_st', 1.0)

    nawf = attributes['nawf']
    nspin = attributes['nspin']
    nktot = arrays['pksp'].shape[0]
    ne = ene.size

    # Hall Magnetization Calculation with Gaussian Smearing

    nawf = attributes['nawf']

    v_kaux = np.zeros_like(arrays['v_k'])

    O1 = np.zeros_like(arrays['pksp'])
    O2 = np.zeros_like(arrays['pksp'])

    for ik in range(nktot):
        for ispin in range(nspin):
            O1[ik, spol, :, :, ispin], O2[ik, ipol, :, :, ispin] = perturb_split(
                0.5 * (P @ Op1[spol, :, :] + Op1[spol, :, :] @ P),
                arrays['dHksp'][ik, ipol, :, :, ispin],
                arrays['v_k'][ik, :, :, ispin],
                arrays['degen'][ispin][ik],
            )

            # O1[ik,spol,:,:,ispin],O2[ik,ipol,:,:,ispin]= perturb_split(
            #    arrays['dHksp'][ik,ipol,:,:,ispin],
            #    arrays['dHksp'][ik,spol,:,:,ispin],
            #    arrays['v_k'][ik,:,:,ispin],
            #    arrays['degen'][ispin][ik]
            # )

    j_kaux = np.zeros_like(arrays['v_k'])

    for ispin in range(attributes['nspin']):
        E_k = np.real(arrays['E_k'][:, :, ispin])

        accaux = np.zeros((ne), dtype=float)   # kai numerator:  Sum <S_spol><v_ipol>
        jcaux = np.zeros((ne), dtype=float)     # current (field dir.): Sum <v_ipol><v_ipol>

        for ik in range(nktot):
            v_kaux[ik, :, :, ispin] = O1[ik, spol, :, :, ispin] * O2[ik, ipol, :, :, ispin]
            j_kaux[ik, :, :, ispin] = O2[ik, ipol, :, :, ispin] * O2[ik, ipol, :, :, ispin]

        if attributes['smearing'] != None:
            taux = np.zeros((arrays['deltakp'].shape[0], nawf), dtype=float)

        for n in range(ne):
            # Adaptive Gaussian Smearing
            if attributes['smearing'] == 'gauss':
                taux = gaussian(ene[n], E_k, arrays['deltakp'][:, :, ispin])
            # Adaptive M-P smearing
            elif attributes['smearing'] == 'm-p':
                taux = metpax(ene[n], E_k, arrays['deltakp'][:, :, ispin])
            elif attributes['smearing'] == None:
                taux = np.exp(-(((ene[n] - E_k[:, :]) / delta) ** 2)) / np.sqrt(np.pi)

            accaux[n] += np.sum(
                np.real(taux * np.diagonal(v_kaux[:, :, :, ispin], axis1=1, axis2=2))
            )
            jcaux[n] += np.sum(
                np.real(taux * np.diagonal(j_kaux[:, :, :, ispin], axis1=1, axis2=2))
            )

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
