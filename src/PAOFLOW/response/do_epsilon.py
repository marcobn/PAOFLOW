import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from ..utils.smearing import gaussian, intgaussian, intmetpax, metpax


def do_dielectric_tensor(data_controller, ene):
    """Compute and write the frequency-dependent dielectric tensor.

    Iterates over every tensor component pair ``(i, j)`` listed in
    ``arrays['d_tensor']`` and for each one calls :func:`do_epsilon` to
    obtain the imaginary part ``\u03b5\u2082`` (``epsi``), real part ``\u03b5\u2081`` (``epsr``),
    electron energy-loss spectrum ``\u03b5\u2082/(\u03b5\u2081\u00b2+\u03b5\u2082\u00b2)`` (``eels``), and the
    Kramers\u2013Kronig-derived real part (``ieps``).  Each quantity is written
    to a two-column ``.dat`` file.

    For diagonal components the plasmon frequency is estimated from the
    f-sum rule:

    .. math::

        \\omega_p = \\sqrt{\\frac{2}{\\pi} \\int_0^\\infty \\omega\\,\\varepsilon_2(\\omega)\,d\\omega}

    and printed to stdout.

    Spin-polarised runs (``nspin = 2``) loop over both spin channels and
    append ``_0`` / ``_1`` suffixes to each output file.

    Parameters
    ----------
    data_controller : DataController
        Provides ``d_tensor``, ``nspin``, ``smearing``, ``degauss``, ``opath``.
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV.  Must not start exactly at zero (shifted
        internally to ``1e-5 eV`` by :func:`do_epsilon`).

    Output files (written to ``opath``)
    ------------------------------------
    ``epsi_XY.dat``, ``epsr_XY.dat``, ``eels_XY.dat``, ``ieps_XY.dat``
        Two-column (energy, value) files for each tensor component ``XY``;
        spin-polarised runs also produce ``_0`` and ``_1`` variants.
    """
    from ..utils.constants import LL

    arrays, attributes = data_controller.data_dicts()
    d_tensor = arrays['d_tensor']
    nspin = attributes['nspin']

    # if from_wfc:
    #     nbnds = attributes['nbnds']
    #     nkpnts = attributes['nkpnts']
    #     nspin = attributes['nspin']
    #     arrays['pksp'] = np.empty((nkpnts, 3, nbnds, nbnds, nspin), dtype=np.complex128)
    #     for ispin in range(nspin):
    #         for ik in range(nkpnts):
    #             arrays['pksp'][ik, :, :, :, ispin] = calc_dipole(
    #                 arrays, attributes, ik, ispin, arrays['b_vectors']
    #             )

    smearing = attributes['smearing']
    if smearing == None:
        if rank == 0:
            print('No smearing, fixed occupation')
    else:
        if rank == 0:
            print('Using fixed smearing = %.3f eV' % attributes['degauss'])

    if nspin == 1:
        for n in range(d_tensor.shape[0]):
            ipol = d_tensor[n][0]
            jpol = d_tensor[n][1]

            epsi, epsr, eels, ieps = do_epsilon(data_controller, ene, 0, ipol, jpol)
            # Write files
            indices = (LL[ipol], LL[jpol])
            for ep, es in [(epsi, 'epsi'), (epsr, 'epsr'), (eels, 'eels'), (ieps, 'ieps')]:
                fn = '%s_%s%s.dat' % ((es,) + indices)
                data_controller.write_file_row_col(fn, ene, ep)

            if rank == 0 and ipol == jpol:
                renorm = np.sqrt((2.0 / np.pi) * np.trapezoid(epsi * ene, x=ene))
                component = LL[ipol] + LL[jpol]
                print('Component', component, ', plasmon frequency = ', renorm, 'eV')

    else:
        for n in range(d_tensor.shape[0]):
            ipol = d_tensor[n][0]
            jpol = d_tensor[n][1]

            epsi_0, epsr_0, eels_0, ieps_0 = do_epsilon(data_controller, ene, 0, ipol, jpol)
            epsi_1, epsr_1, eels_1, ieps_1 = do_epsilon(data_controller, ene, 1, ipol, jpol)
            # Write files
            indices = (LL[ipol], LL[jpol], 0)
            for ep, es in [(epsi_0, 'epsi'), (epsr_0, 'epsr'), (eels_0, 'eels'), (ieps_0, 'ieps')]:
                fn = '%s_%s%s_%d.dat' % ((es,) + indices)
                data_controller.write_file_row_col(fn, ene, ep)
            indices = (LL[ipol], LL[jpol], 1)
            for ep, es in [(epsi_1, 'epsi'), (epsr_1, 'epsr'), (eels_1, 'eels'), (ieps_1, 'ieps')]:
                fn = '%s_%s%s_%d.dat' % ((es,) + indices)
                data_controller.write_file_row_col(fn, ene, ep)

            if rank == 0 and ipol == jpol:
                epsi = epsi_0 + epsi_1
                renorm = np.sqrt((2.0 / np.pi) * np.trapezoid(epsi * ene, x=ene))
                component = LL[ipol] + LL[jpol]
                print('Component', component, ', plasmon frequency = ', renorm, 'eV')


def do_jdos(data_controller, ene, jdos_smeartype):
    """Compute and write the joint density of states (JDOS).

    The JDOS counts the number of interband transitions at each photon
    energy

    .. math::

        J(\\omega) = \\sum_{nm\\mathbf{k}} f_{n\\mathbf{k}}
                     (1 - f_{m\\mathbf{k}})
                     \\delta(E_{m\\mathbf{k}} - E_{n\\mathbf{k}} - \\omega)

    broadened with either a Gaussian or Lorentzian of width ``delta``.
    Results from all MPI ranks are summed with ``MPI.Allreduce`` and written
    to ``jdos.dat`` (spin-unpolarised) or ``jdos_0.dat`` / ``jdos_1.dat``
    (spin-polarised).

    Parameters
    ----------
    data_controller : DataController
        Provides ``my_eigsmat``, ``kpnts_wght``, ``nbnds``, ``nspin``,
        ``dftSO``, ``smearing``, ``degauss``, ``delta``, ``insulator``,
        ``opath``.
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV.
    jdos_smeartype : str
        Broadening kernel: ``'gauss'`` (Gaussian) or ``'lorentz'``
        (Lorentzian).

    Output files (written to ``opath``)
    ------------------------------------
    ``jdos.dat`` or ``jdos_0.dat`` / ``jdos_1.dat``
        Two-column (energy, JDOS) file(s).
    """
    _, attributes = data_controller.data_dicts()
    esize = ene.size

    nspin = attributes['nspin']
    if nspin == 1:
        jdos_aux = jdos_loop(data_controller, ene, 0, jdos_smeartype)
        jdos = np.zeros(esize, dtype=float)
        comm.Allreduce(jdos_aux, jdos, op=MPI.SUM)
        jdos_aux = None

        fn = 'jdos.dat'
        data_controller.write_file_row_col(fn, ene, jdos)

        if rank == 0:
            print('Integration over JDOS = ', (np.trapezoid(jdos, x=ene)))
    else:
        jdos_aux0 = jdos_loop(data_controller, ene, 0, jdos_smeartype)
        jdos_aux1 = jdos_loop(data_controller, ene, 1, jdos_smeartype)
        jdos0 = np.zeros(esize, dtype=float)
        jdos1 = np.zeros(esize, dtype=float)
        comm.Allreduce(jdos_aux0, jdos0, op=MPI.SUM)
        comm.Allreduce(jdos_aux1, jdos1, op=MPI.SUM)
        jdos_aux0 = None
        jdos_aux1 = None

        fn0 = 'jdos_0.dat'
        data_controller.write_file_row_col(fn0, ene, jdos0)
        fn1 = 'jdos_1.dat'
        data_controller.write_file_row_col(fn1, ene, jdos1)

        if rank == 0:
            print('Integration over JDOS = ', (np.trapezoid(jdos0 + jdos1, x=ene)))


def do_epsilon(data_controller, ene, ispin, ipol, jpol):
    """Compute the dielectric-function components for one spin channel and polarization pair.

    Calls :func:`eps_loop` to accumulate the imaginary and real parts of the
    interband dielectric function across all k-points (reducing partial sums
    via ``MPI.Allreduce``), then applies the physical prefactor and constructs
    the remaining spectra:

    * ``epsi`` \u2014 imaginary part ``\u03b5\u2082(\u03c9)`` (interband absorption)
    * ``epsr`` \u2014 real part ``\u03b5\u2081(\u03c9)`` (Sellmeier dispersion)
    * ``eels`` \u2014 electron energy-loss spectrum
      ``\u03b5\u2082 / (\u03b5\u2081\u00b2 + \u03b5\u2082\u00b2)``
    * ``ieps`` \u2014 Kramers\u2013Kronig real part reconstructed by a discrete
      numerical integration

    The SI prefactor is

    .. math::

        A = \\frac{2\\,e\\,(10^{10})
                   a_0^{\\,2}}{\\varepsilon_0\,N_k\,\\Omega}

    with ``a_0`` the Bohr radius, ``\u03b50`` the vacuum permittivity, ``N_k`` the
    number of k-points, and ``\u03a9`` the unit-cell volume in Bohr\u00b3.

    Parameters
    ----------
    data_controller : DataController
        Provides ``E_k``, ``pksp``, ``nkpnts``, ``omega``, ``bnd``, ``nspin``,
        ``dftSO``, ``insulator``, ``smearing``, ``degauss``, ``delta``,
        ``intrasmear``.
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV.  A value of exactly zero is shifted to
        ``1e-5 eV`` to avoid division by zero.
    ispin : int
        Spin channel index (0 or 1).
    ipol : int
        Polarization index of the electric field (0\u20132 = x, y, z).
    jpol : int
        Second polarization index (0\u20132 = x, y, z).

    Returns
    -------
    epsi : ndarray, shape (ne,)
        Imaginary part of the dielectric function.
    epsr : ndarray, shape (ne,)
        Real part of the dielectric function.
    eels : ndarray, shape (ne,)
        Electron energy-loss spectrum.
    ieps : ndarray, shape (ne,)
        Kramers\u2013Kronig-derived real part.
    """
    from ..utils.constants import BOHR_RADIUS_ANGS, ELECTRONVOLT_SI

    # Compute the dielectric tensor

    _, attributes = data_controller.data_dicts()

    esize = ene.size
    if ene[0] == 0.0:
        ene[0] = 0.00001

    # if from_wfc:
    #     factor = (
    #         (2 * np.pi / attributes['alat']) ** 2
    #         * RYTOEV**3
    #         * 64.0
    #         * np.pi
    #         / (attributes['omega'] * attributes['nkpnts'])
    #     )
    # else:
    # 8.8541878188e-12 = \epsilon_0
    factor = (
        2
        * ELECTRONVOLT_SI
        * (1e10)
        / (8.8541878188e-12)
        * BOHR_RADIUS_ANGS**2
        / attributes['nkpnts']
        / (attributes['omega'] * BOHR_RADIUS_ANGS**3)
    )

    # =======================
    # EPS
    # =======================

    epsi_aux, epsr_aux = eps_loop(data_controller, ene, ispin, ipol, jpol)
    epsi = np.zeros(esize, dtype=float)
    comm.Allreduce(epsi_aux, epsi, op=MPI.SUM)
    epsi_aux = None

    epsr = np.zeros(esize, dtype=float)
    comm.Allreduce(epsr_aux, epsr, op=MPI.SUM)
    epsr_aux = None

    ### TNeeds revision. Each processor is allocating zeros here, when only rank 0 needs it.
    ### Can be condensed

    ieps = np.zeros(esize, dtype=float)

    epsi *= factor
    epsr = 1.0 * (ipol == jpol) + epsr * factor
    eels = epsi / (epsi**2 + epsr**2)
    for i in range(esize):
        for j in range(1, esize):
            ieps[i] += ene[j] * epsi[j] / (ene[i] ** 2 + ene[j] ** 2)
    ieps = 1.0 + (2.0 / np.pi) * ieps * (ene[3] - ene[2])

    return (epsi, epsr, eels, ieps)


def eps_loop(data_controller, ene, ispin, ipol, jpol):
    """Inner BZ loop: accumulate the interband dielectric-function integrands.

    Evaluates the Kubo\u2013Greenwood sum

    .. math::

        \\varepsilon_2(\\omega) \\propto \\sum_{nm\\mathbf{k}}
            \\frac{f_n - f_m}{E_{mn}}
            |\\langle m | p_i | n \\rangle|^2
            \\frac{\\eta\\omega}
                 {(E_{mn}^2 - \\omega^2)^2 + \\eta^2\\omega^2}

    and its real-part companion simultaneously, using the PAO momentum
    matrix ``pksp``.  A separate Drude-like intraband term is added for
    metals when ``attributes['insulator']`` is ``False``.

    Occupations are computed from the integrated Fermi function (step,
    Gaussian, or Methfessel\u2013Paxton) according to ``attributes['smearing']``.
    Overflow errors from large exponentials in the Fermi function are
    silenced locally and restored on return.

    This function operates on the local k-point slice held by the calling
    MPI rank; the caller is responsible for reducing partial sums across
    ranks.

    Parameters
    ----------
    data_controller : DataController
        Provides ``E_k``, ``pksp``, ``bnd``, ``nspin``, ``dftSO``,
        ``insulator``, ``smearing``, ``degauss``, ``delta``, ``intrasmear``.
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV.
    ispin : int
        Spin channel index.
    ipol : int
        First polarization index (row of momentum matrix).
    jpol : int
        Second polarization index (column of momentum matrix).

    Returns
    -------
    epsi : ndarray, shape (ne,)
        Local partial sum for the imaginary part (unscaled).
    epsr : ndarray, shape (ne,)
        Local partial sum for the real part (unscaled).
    """
    orig_over_err = np.geterr()['over']
    np.seterr(over='raise')

    arrays, attributes = data_controller.data_dicts()

    esize = ene.size
    # if from_wfc:
    #     bndmax = attributes['nbnds']
    #     Ek = np.swapaxes(arrays['my_eigsmat'][:, :, ispin], 0, 1)
    # else:
    bndmax = attributes['bnd']
    Ek = arrays['E_k'][:, :bndmax, ispin]

    intersmear = attributes['delta']
    smearing = attributes['smearing']

    spin_factor = 2 if (attributes['nspin'] == 1 and not attributes['dftSO']) else 1
    Ef = 1.0e-9

    epsi = np.zeros(esize, dtype=float)
    epsr = np.zeros(esize, dtype=float)

    if smearing != None:
        degauss = attributes['degauss']
        # Adaptive smearing not implemented
        # if 'deltakp' in arrays:  # check whether adaptive smearing is used
        #     degauss = arrays['deltakp'][:, :bndmax, ispin]
        #     if rank == 0:
        #         print('Using adaptive smearing')

    if smearing == None:
        fn = spin_factor * (Ek <= Ef)  # fixed occupation for insulator, no smearing
    elif smearing == 'gauss':
        fn = spin_factor * intgaussian(Ek, Ef, degauss)
    else:  # smearing == 'm-p':
        fn = spin_factor * intmetpax(Ek, Ef, degauss)

    th0 = 1.0e-3 * spin_factor
    th1 = 0.5e-4 * spin_factor

    if not attributes['insulator']:
        intrasmear = attributes['intrasmear']
        epsi_metal = np.zeros_like(epsi)
        epsr_metal = np.zeros_like(epsr)

        if smearing == 'gauss':
            fnF = spin_factor * gaussian(Ek, Ef, degauss)
        elif smearing == 'm-p':
            fnF = spin_factor * metpax(Ek, Ef, degauss)
        else:
            print('Smearing is None for a metal, switching to gaussian smearing')
            fnF = spin_factor * gaussian(Ek, Ef, degauss)

    for ik in range(fn.shape[0]):
        for iband2 in range(bndmax):
            for iband1 in range(bndmax):
                if iband1 != iband2:
                    E_diff_nm = Ek[ik, iband2] - Ek[ik, iband1]
                    f_nm = fn[ik, iband2] - fn[ik, iband1]
                    if np.abs(f_nm) > th0 and fn[ik, iband1] > th1 and fn[ik, iband2] < spin_factor:
                        pksp2 = np.real(
                            arrays['pksp'][ik, ipol, iband1, iband2, ispin]
                            * arrays['pksp'][ik, jpol, iband2, iband1, ispin]
                        )
                        # pksp2 in unit of (AU*eV)^2
                        epsi[:] += (
                            pksp2
                            * intersmear
                            * ene[:]
                            * fn[ik, iband1]
                            / (
                                ((E_diff_nm**2 - ene[:] ** 2) ** 2 + intersmear**2 * ene[:] ** 2)
                                * (E_diff_nm)
                            )
                        )
                        epsr[:] += (
                            pksp2
                            * (E_diff_nm**2 - ene[:] ** 2)
                            * fn[ik, iband1]
                            / (
                                ((E_diff_nm**2 - ene[:] ** 2) ** 2 + intersmear**2 * ene[:] ** 2)
                                * (E_diff_nm)
                            )
                        )

                elif not attributes['insulator']:
                    pksp2 = np.real(
                        arrays['pksp'][ik, ipol, iband1, iband1, ispin]
                        * arrays['pksp'][ik, jpol, iband1, iband1, ispin]
                    )
                    epsi_metal[:] += (
                        pksp2
                        * intrasmear
                        * ene[:]
                        * fnF[ik, iband1]
                        / (ene[:] ** 4 + intrasmear**2 * ene[:] ** 2)
                    )
                    epsr_metal[:] -= (
                        pksp2
                        * fnF[ik, iband1]
                        * ene[:] ** 2
                        / (ene[:] ** 4 + intrasmear**2 * ene[:] ** 2)
                    )
                else:
                    pass

    if not attributes['insulator']:
        # if from_wfc:
        #     from ..utils.constants import RYTOEV
        #     epsi_metal *= 0.5*spin_factor/RYTOEV
        #     epsr_metal *= 0.5*spin_factor/RYTOEV

        epsi += epsi_metal
        epsr += epsr_metal

    np.seterr(over=orig_over_err)
    return (epsi, epsr)


def jdos_loop(data_controller, ene, ispin, jdos_smeartype):
    """Inner BZ loop: accumulate the joint-density-of-states integrand.

    Iterates over all k-points and interband pairs ``(n, m)`` with
    ``E_{mn} > 0`` and computes

    .. math::

        J(\\omega) = \\sum_{nm\\mathbf{k}} w_k\,f_{n\\mathbf{k}}
                     (1 - f_{m\\mathbf{k}})
                     g(E_{mn}, \\omega, \\eta)

    where ``g`` is either a Gaussian (``'gauss'``) or Lorentzian
    (``'lorentz'``) of width ``delta``, and ``w_k`` are the k-point
    weights.  The result is normalized by the total oscillator-strength
    sum and the spin degeneracy factor.

    Occupations are computed from the integrated Fermi function (step,
    Gaussian, or Methfessel\u2013Paxton) selected by ``attributes['smearing']``.

    This function operates on the local k-point slice; the caller reduces
    partial sums with ``MPI.Allreduce``.

    Parameters
    ----------
    data_controller : DataController
        Provides ``my_eigsmat``, ``kpnts_wght``, ``nbnds``, ``nspin``,
        ``dftSO``, ``smearing``, ``degauss``, ``delta``, ``insulator``.
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV.
    ispin : int
        Spin channel index.
    jdos_smeartype : str
        Broadening kernel: ``'gauss'`` or ``'lorentz'``.

    Returns
    -------
    jdos : ndarray, shape (ne,)
        Local partial JDOS contribution from this MPI rank's k-points
        (normalized but not yet globally reduced).

    Raises
    ------
    ValueError
        If ``jdos_smeartype`` is not ``'gauss'`` or ``'lorentz'``.
    """
    arrays, attributes = data_controller.data_dicts()
    intersmear = attributes['delta']
    smearing = attributes['smearing']
    esize = ene.size
    # bndmax = attributes['bnd']
    # Ek = arrays['E_k'][:, :bndmax, ispin]
    bndmax = attributes['nbnds']
    Ek = np.swapaxes(arrays['my_eigsmat'][:, :, ispin], 0, 1)
    kweights = arrays['kpnts_wght']
    nkpnts = Ek.shape[0]
    jdos = np.zeros(esize, dtype=float)
    Ef = 1.0e-9

    if smearing == None or attributes['insulator']:
        fn = 1.0 * (Ek <= Ef)  # fixed occupation for insulator, no smearing
    elif smearing == 'gauss':
        degauss = attributes['degauss']
        fn = intgaussian(Ek, Ef, degauss)
    else:  # smearing == 'm-p':
        degauss = attributes['degauss']
        fn = intmetpax(Ek, Ef, degauss)

    count = 0.0
    if jdos_smeartype == 'gauss':
        for ik in range(nkpnts):
            for iband2 in range(bndmax):
                for iband1 in range(bndmax):
                    E_diff_nm = Ek[ik, iband2] - Ek[ik, iband1]
                    if fn[ik, iband1] > 1.0e-4 and fn[ik, iband2] < 2.0 and E_diff_nm > 1e-10:
                        f_nm = fn[ik, iband1] - fn[ik, iband2]
                        jdos += f_nm * gaussian(E_diff_nm, ene, intersmear) * kweights[ik]
                        count += f_nm

    elif jdos_smeartype == 'lorentz':
        for ik in range(nkpnts):
            for iband2 in range(bndmax):
                for iband1 in range(bndmax):
                    E_diff_nm = Ek[ik, iband2] - Ek[ik, iband1]
                    if fn[ik, iband1] > 1.0e-4 and fn[ik, iband2] < 2.0 and E_diff_nm > 1e-10:
                        f_nm = fn[ik, iband1] - fn[ik, iband2]
                        jdos += (
                            f_nm
                            * intersmear
                            / (np.pi * ((E_diff_nm - ene) ** 2 + intersmear**2))
                            * kweights[ik]
                        )
                        count += f_nm

    else:
        raise ValueError("jdos_smeartype must be 'gauss' or 'lorentz' ")

    spin_factor = 2 if (attributes['nspin'] == 1 and not attributes['dftSO']) else 1
    jdos *= nkpnts / count / spin_factor

    return jdos


"""
# Function to calculate dipole matrix element from coefficients of wavefunction,
# following the routine of epsilon.x
def calc_dipole(arry, attr, ik, ispin, b_vector):
    from ..projection.do_atwfc_proj import calc_atwfc_k, ortho_atwfc_k, calc_gkspace
    from scipy.io import FortranFile
    import os

    if attr['nspin'] == 1 or attr['nspin'] == 4:
        wfcfile = 'wfc{0}.dat'.format(ik + 1)
    elif attr['nspin'] == 2 and ispin == 0:
        wfcfile = 'wfcdw{0}.dat'.format(ik + 1)
    elif attr['nspin'] == 2 and ispin == 1:
        wfcfile = 'wfcup{0}.dat'.format(ik + 1)
    else:
        print('no wfc file found')

    with FortranFile(os.path.join(attr['fpath'], wfcfile), 'r') as f:
        record = f.read_ints(np.int32)
        assert len(record) == 11, 'something wrong reading fortran binary file'

        ik_ = record[0]
        assert ik + 1 == ik_, 'wrong k-point in wfc file???'

        # xk = np.frombuffer(record[1:7], np.float64)
        # ispin = record[7]
        # gamma_only = (record[8] != 0)
        _, igwx, _, nbnds = f.read_ints(np.int32)
        f.read_reals(np.float64).reshape(3, 3, order='F')
        mill = f.read_ints(np.int32).reshape(3, igwx, order='F')
        mill = b_vector.T @ mill + np.full((igwx, 3), arry['kpnts'][ik]).T

        wfc = []
        for i in range(nbnds):
            wfc.append(f.read_reals(np.complex128))

    dipole_aux = np.zeros((3, nbnds, nbnds), dtype=np.complex128)
    for iband2 in range(nbnds):
        for iband1 in range(nbnds):
            if attr['dftSO']:
                dipole_aux[:, iband1, iband2] = (wfc[iband2][:igwx] * mill) @ np.conjugate(
                    wfc[iband1][:igwx]
                )
                +(wfc[iband2][igwx:] * mill) @ np.conjugate(wfc[iband1][igwx:])
            else:
                dipole_aux[:, iband1, iband2] = (wfc[iband2] * mill) @ np.conjugate(wfc[iband1])
    return dipole_aux


# Function to calculate dipole matrix element from the eigenvector of the PAO Hamiltonian
# expanded in the real space of the atomic basis functions
def calc_dipole_internal(data_controller, ik, ispin):
    arry, attr = data_controller.data_dicts()
    basis = arry['basis']
    gkspace = calc_gkspace(data_controller, ik, gamma_only=False)
    _, igwx, mill, _, _ = [gkspace[s] for s in ('xk', 'igwx', 'mill', 'bg', 'gamma_only')]
    atwfcgk = calc_atwfc_k(basis, gkspace, attr['dftSO'])
    oatwfcgk = ortho_atwfc_k(atwfcgk)  # these are the atomic orbitals on the G vector grid

    # build the full wavefunction with the coefficients v_k
    bnd = attr['bnd']
    wfc = []
    # for nb in range(attr['bnd']):
    for nb in range(bnd):
        wfc.append(np.tensordot(arry['v_k'][ik, :, nb, ispin], oatwfcgk, axes=(0, 0)))

    # build k+G
    mill = arry['b_vectors'].T @ mill + np.full((igwx, 3), arry['kgrid'][:, ik]).T

    nbnds = attr['nawf']
    dipole_aux = np.zeros((3, nbnds, nbnds), dtype=np.complex128)
    for iband2 in range(bnd):
        for iband1 in range(bnd):
            if attr['dftSO']:
                # check indexing with nbnds and bnd!!!!!
                dipole_aux[:, iband1, iband2] = (wfc[iband2][:igwx] * mill) @ np.conjugate(
                    wfc[iband1][:igwx]
                )
                +(wfc[iband2][igwx:] * mill) @ np.conjugate(wfc[iband1][igwx:])
            else:
                dipole_aux[:, iband1, iband2] = (wfc[iband2] * mill) @ np.conjugate(wfc[iband1])
    return dipole_aux

def epsr_kramerskronig ( data_controller, ene, epsi ):
  from ..utils.smearing import intmetpax
  from scipy.integrate import simpson
  from ..utils.communication import load_balancing

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size
  de = ene[1] - ene[0]

  epsr = np.zeros(esize, dtype=float)

  ini_ie,end_ie = load_balancing(comm.Get_size(), rank, esize)

  # Range checks for Simpson Integrals
  if end_ie == ini_ie:
    return
  if ini_ie < 3:
    ini_ie = 3
  if end_ie == esize:
    end_ie = esize-1

  f_ene = intmetpax(ene, attributes['shift'], 1.)
  for ie in range(ini_ie, end_ie):
    I1 = simpson(ene[1:(ie-1)]*de*epsi[1:(ie-1)]*f_ene[1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2))
    I2 = simpson(ene[(ie+1):esize]*de*epsi[(ie+1):esize]*f_ene[(ie+1):esize]/(ene[(ie+1):esize]**2-ene[ie]**2))
    epsr[ie] = 2.*(I1+I2)/np.pi

  return epsr
"""
