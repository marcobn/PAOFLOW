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
    from .smearing import gaussian
    from .constants import ELECTRONVOLT_SI, BOHR_RADIUS_CM, HBAR

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
    St = np.real(arrays['sktxt']).reshape(snktot, 3, nstates)

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
            fkai = open(join(attr['opath'], 'kai.dat'), 'w')
            fcurrent = open(join(attr['opath'], 'current.dat'), 'w')

            ofE = lambda si, sj: open(join(attr['opath'], f'Ekai_{si}{sj}.dat'), 'w')
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
