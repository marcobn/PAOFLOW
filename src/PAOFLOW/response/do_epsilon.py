import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from ..utils.smearing import gaussian, intgaussian, intmetpax, metpax


def do_dielectric_tensor(data_controller, ene):
    r"""Compute and write the frequency-dependent dielectric tensor.

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
    ``epsi_XY.dat``, ``epsr_XY.dat``, ``eels_XY.dat``, ``ieps_XY.dat``,
    ``sigmar_XY.dat``, ``sigmai_XY.dat``
        Two-column (energy, value) files for each tensor component ``XY``;
        spin-polarised runs also produce ``_0`` and ``_1`` variants.
        ``sigmar``/``sigmai`` are the real (absorptive) and imaginary
        (dispersive) parts of the optical conductivity in SI units (S/m).
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

    # if 'deltakp2' in arrays and rank == 0:
    #     print('Adaptive (Yates) interband broadening enabled for the dielectric tensor')

    if nspin == 1:
        for n in range(d_tensor.shape[0]):
            ipol = d_tensor[n][0]
            jpol = d_tensor[n][1]

            epsi, epsr, eels, ieps = do_epsilon(data_controller, ene, 0, ipol, jpol)
            sigmar, sigmai = optical_conductivity(ene, epsi, epsr, ipol, jpol)
            # Write files. EELS = -Im(1/eps) is only physically meaningful for
            # diagonal tensor components, so we skip it for off-diagonal pairs.
            # The same applies to the refractive index n + iκ and the derived
            # absorption coefficient and reflectivity (Option A: per principal
            # axis; off-diagonal complex permittivity requires tensor
            # diagonalisation, out of scope here).
            indices = (LL[ipol], LL[jpol])
            spectra = [
                (epsi, 'epsi'),
                (epsr, 'epsr'),
                (ieps, 'ieps'),
                (sigmar, 'sigmar'),
                (sigmai, 'sigmai'),
            ]
            if ipol == jpol:
                nref, kref, alpha, refl = refractive_index(ene, epsi, epsr)
                spectra.extend(
                    [
                        (eels, 'eels'),
                        (nref, 'nref'),
                        (kref, 'kref'),
                        (alpha, 'alpha'),
                        (refl, 'refl'),
                    ]
                )
            for ep, es in spectra:
                fn = '%s_%s%s.dat' % ((es,) + indices)
                data_controller.write_file_row_col(fn, ene, ep)

            if ipol == jpol and attributes.get('emissivity', False):
                write_emissivity(data_controller, ene, epsr, epsi, LL[ipol] + LL[jpol], '')

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
            sigmar_0, sigmai_0 = optical_conductivity(ene, epsi_0, epsr_0, ipol, jpol)
            sigmar_1, sigmai_1 = optical_conductivity(ene, epsi_1, epsr_1, ipol, jpol)
            # Write files. EELS = -Im(1/eps) is only physically meaningful for
            # diagonal tensor components, so we skip it for off-diagonal pairs.
            indices = (LL[ipol], LL[jpol], 0)
            spectra0 = [
                (epsi_0, 'epsi'),
                (epsr_0, 'epsr'),
                (ieps_0, 'ieps'),
                (sigmar_0, 'sigmar'),
                (sigmai_0, 'sigmai'),
            ]
            if ipol == jpol:
                nref_0, kref_0, alpha_0, refl_0 = refractive_index(ene, epsi_0, epsr_0)
                spectra0.extend(
                    [
                        (eels_0, 'eels'),
                        (nref_0, 'nref'),
                        (kref_0, 'kref'),
                        (alpha_0, 'alpha'),
                        (refl_0, 'refl'),
                    ]
                )
            for ep, es in spectra0:
                fn = '%s_%s%s_%d.dat' % ((es,) + indices)
                data_controller.write_file_row_col(fn, ene, ep)
            if ipol == jpol and attributes.get('emissivity', False):
                write_emissivity(data_controller, ene, epsr_0, epsi_0, LL[ipol] + LL[jpol], '_0')
            indices = (LL[ipol], LL[jpol], 1)
            spectra1 = [
                (epsi_1, 'epsi'),
                (epsr_1, 'epsr'),
                (ieps_1, 'ieps'),
                (sigmar_1, 'sigmar'),
                (sigmai_1, 'sigmai'),
            ]
            if ipol == jpol:
                nref_1, kref_1, alpha_1, refl_1 = refractive_index(ene, epsi_1, epsr_1)
                spectra1.extend(
                    [
                        (eels_1, 'eels'),
                        (nref_1, 'nref'),
                        (kref_1, 'kref'),
                        (alpha_1, 'alpha'),
                        (refl_1, 'refl'),
                    ]
                )
            for ep, es in spectra1:
                fn = '%s_%s%s_%d.dat' % ((es,) + indices)
                data_controller.write_file_row_col(fn, ene, ep)
            if ipol == jpol and attributes.get('emissivity', False):
                write_emissivity(data_controller, ene, epsr_1, epsi_1, LL[ipol] + LL[jpol], '_1')

            if rank == 0 and ipol == jpol:
                epsi = epsi_0 + epsi_1
                renorm = np.sqrt((2.0 / np.pi) * np.trapezoid(epsi * ene, x=ene))
                component = LL[ipol] + LL[jpol]
                print('Component', component, ', plasmon frequency = ', renorm, 'eV')


def do_jdos(data_controller, ene, jdos_smeartype):
    r"""Compute and write the joint density of states (JDOS).

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
    r"""Compute the dielectric-function components for one spin channel and polarization pair.

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

    epsi *= factor
    epsr = 1.0 * (ipol == jpol) + epsr * factor

    # EELS = -Im(1/eps) = eps2/(eps1^2+eps2^2) is only physically meaningful
    # for diagonal tensor components. Emit zeros for off-diagonals and let the
    # caller decide whether to write the file (see ``do_dielectric_tensor``).
    if ipol == jpol:
        eels = epsi / (epsi**2 + epsr**2)
    else:
        eels = np.zeros(esize, dtype=float)

    # London transform: dielectric on the imaginary frequency axis
    #   eps(iω) = 1 + (2/π) ∫₀^∞ ω' eps₂(ω') / (ω'² + ω²) dω'
    # Integrand evaluated on the existing energy grid (any spacing).
    # The j=0 sample is skipped because ene[0] is shifted to ~1e-5 eV to avoid
    # a 1/0 divergence elsewhere; its contribution is negligible.
    ieps = np.empty(esize, dtype=float)
    integrand_num = ene[1:] * epsi[1:]
    for i in range(esize):
        ieps[i] = np.trapezoid(integrand_num / (ene[i] ** 2 + ene[1:] ** 2), x=ene[1:])
    ieps = 1.0 + (2.0 / np.pi) * ieps

    return (epsi, epsr, eels, ieps)


def optical_conductivity(ene, epsi, epsr, ipol, jpol):
    r"""Optical (AC) conductivity from the complex dielectric function.

    The optical conductivity is related to the complex relative permittivity
    :math:`\varepsilon(\omega) = \varepsilon_1 + i\varepsilon_2` by

    .. math::

        \sigma(\omega) = -i\,\varepsilon_0\,\omega\,
                          \bigl(\varepsilon(\omega) - 1\bigr)

    so that its real (absorptive) and imaginary (dispersive) parts are

    .. math::

        \sigma_1(\omega) = \varepsilon_0\,\omega\,\varepsilon_2(\omega), \qquad
        \sigma_2(\omega) = -\varepsilon_0\,\omega\,
                            \bigl(\varepsilon_1(\omega) - \delta_{ij}\bigr).

    The ``-1`` (``\delta_{ij}``) vacuum subtraction is applied only to
    diagonal tensor components, consistent with the ``+1`` added to
    ``epsr`` in :func:`do_epsilon`.

    Parameters
    ----------
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV.  The angular frequency is
        :math:`\omega = E / \hbar` with ``\hbar`` in eV s.
    epsi : ndarray, shape (ne,)
        Imaginary part of the dielectric function :math:`\varepsilon_2`.
    epsr : ndarray, shape (ne,)
        Real part of the dielectric function :math:`\varepsilon_1`
        (already including the ``+1`` for diagonal components).
    ipol, jpol : int
        Tensor component indices (0-2 = x, y, z).

    Returns
    -------
    sigma_r : ndarray, shape (ne,)
        Real (absorptive) part of the optical conductivity in S/m.
    sigma_i : ndarray, shape (ne,)
        Imaginary (dispersive) part of the optical conductivity in S/m.
    """
    from ..utils.constants import HBAR

    # SI vacuum permittivity (F/m), matching the prefactor used in do_epsilon.
    eps0_si = 8.8541878188e-12
    # Angular frequency in rad/s (HBAR is in eV s, ene in eV).
    omega = ene / HBAR

    sigma_r = eps0_si * omega * epsi
    sigma_i = -eps0_si * omega * (epsr - 1.0 * (ipol == jpol))

    return (sigma_r, sigma_i)


def refractive_index(ene, epsi, epsr):
    r"""Complex refractive index and derived spectra from $\varepsilon(\omega)$.

    For a diagonal component of the complex relative permittivity
    $\varepsilon = \varepsilon_1 + i\varepsilon_2$ the complex refractive
    index $\tilde n = n + i\kappa$ satisfies $\tilde n^2 = \varepsilon$, so

    .. math::

        n      &= \sqrt{(|\varepsilon| + \varepsilon_1)/2}, \\
        \kappa &= \sqrt{(|\varepsilon| - \varepsilon_1)/2}.

    From these the absorption coefficient and normal-incidence
    reflectivity follow:

    .. math::

        \alpha(\omega) &= 2\,\omega\,\kappa(\omega)/c, \\
        R(\omega)      &= \frac{(n-1)^2 + \kappa^2}{(n+1)^2 + \kappa^2}.

    Parameters
    ----------
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV.
    epsi, epsr : ndarray, shape (ne,)
        Imaginary and real parts of the (diagonal) dielectric function,
        with the ``+1`` already included in ``epsr``.

    Returns
    -------
    n : ndarray, shape (ne,)
        Real part of the refractive index (unitless).
    kappa : ndarray, shape (ne,)
        Extinction coefficient (unitless).
    alpha : ndarray, shape (ne,)
        Absorption coefficient in 1/m.
    refl : ndarray, shape (ne,)
        Normal-incidence reflectivity (unitless, in [0, 1]).
    """
    from ..utils.constants import HBAR, SPEED_OF_LIGHT

    mod_eps = np.sqrt(epsr * epsr + epsi * epsi)
    n = np.sqrt(0.5 * (mod_eps + epsr))
    kappa = np.sqrt(np.maximum(0.5 * (mod_eps - epsr), 0.0))

    omega = ene / HBAR
    alpha = 2.0 * omega * kappa / SPEED_OF_LIGHT

    refl = ((n - 1.0) ** 2 + kappa**2) / ((n + 1.0) ** 2 + kappa**2)

    return (n, kappa, alpha, refl)


def directional_reflectivity(epsr, epsi, theta):
    r"""Fresnel directional reflectivity for an opaque (optically thick) medium.

    Treats the diagonal complex relative permittivity
    :math:`\tilde\varepsilon = \varepsilon_1 + i\varepsilon_2` as the squared
    complex refractive index :math:`\tilde n^2 = \tilde\varepsilon`.  For an
    opaque bulk material light does not transmit, so the spectral directional
    reflectivity follows from the Fresnel equations.  The complex refraction
    term is

    .. math::

        q(\theta, \omega) = \sqrt{\tilde n^2 - \sin^2\theta},

    and the polarization-resolved reflectivities for the transverse-electric
    (s) and transverse-magnetic (p) components are

    .. math::

        R_s(\theta, \omega) &= \left\lvert
            \frac{\cos\theta - q}{\cos\theta + q}\right\rvert^2, \\
        R_p(\theta, \omega) &= \left\lvert
            \frac{\tilde n^2\cos\theta - q}{\tilde n^2\cos\theta + q}
            \right\rvert^2.

    The unpolarized directional reflectivity is the average

    .. math::

        R(\theta, \omega) = \tfrac{1}{2}\bigl(R_s + R_p\bigr).

    At normal incidence (:math:`\theta = 0`) this reduces to the standard
    expression :math:`R = [(n-1)^2 + \kappa^2]/[(n+1)^2 + \kappa^2]` returned
    by :func:`refractive_index`.

    Parameters
    ----------
    epsr, epsi : ndarray, shape (ne,)
        Real and imaginary parts of the (diagonal) dielectric function, with
        the ``+1`` already included in ``epsr``.
    theta : float
        Incidence angle relative to the surface normal, in radians
        (``0 <= theta < pi/2``).

    Returns
    -------
    refl : ndarray, shape (ne,)
        Unpolarized directional reflectivity (unitless, in [0, 1]).
    """
    n2 = epsr + 1j * epsi
    cos_t = np.cos(theta)
    sin2_t = np.sin(theta) ** 2
    q = np.sqrt(n2 - sin2_t)

    r_s = np.abs((cos_t - q) / (cos_t + q)) ** 2
    r_p = np.abs((n2 * cos_t - q) / (n2 * cos_t + q)) ** 2

    return 0.5 * (r_s + r_p)


def spectral_hemispherical_emissivity(epsr, epsi, ntheta):
    r"""Spectral hemispherical emissivity from the directional reflectivity.

    By Kirchhoff's law the directional spectral emissivity of an opaque
    material equals its directional spectral absorptivity,
    :math:`\varepsilon(\theta, \omega) = 1 - R(\theta, \omega)`.  The
    hemispherical emissivity removes the directional dependence by averaging
    over all solid angles in the upper hemisphere, weighted by the projected
    area :math:`\cos\theta`:

    .. math::

        \varepsilon(\omega) = 2 \int_0^{\pi/2}
            \varepsilon(\theta, \omega)\,\cos\theta\,\sin\theta\,d\theta.

    The polar-angle integral is evaluated numerically on a uniform grid of
    ``ntheta`` points in :math:`[0, \pi/2]` via the trapezoidal rule, with
    :func:`directional_reflectivity` supplying :math:`R(\theta, \omega)` at
    each angle.

    Parameters
    ----------
    epsr, epsi : ndarray, shape (ne,)
        Real and imaginary parts of the (diagonal) dielectric function.
    ntheta : int
        Number of polar-angle samples in ``[0, pi/2]``.

    Returns
    -------
    emis : ndarray, shape (ne,)
        Spectral hemispherical emissivity (unitless, in [0, 1]).
    """
    thetas = np.linspace(0.0, 0.5 * np.pi, ntheta)
    # integrand[k, :] = epsilon(theta_k, omega) * cos(theta_k) * sin(theta_k)
    integrand = np.empty((ntheta, epsr.size), dtype=float)
    for k, theta in enumerate(thetas):
        emis_dir = 1.0 - directional_reflectivity(epsr, epsi, theta)
        integrand[k] = emis_dir * np.cos(theta) * np.sin(theta)

    return 2.0 * np.trapezoid(integrand, x=thetas, axis=0)


def total_hemispherical_emissivity(ene, emis_w, temperature):
    r"""Total (Planck-weighted) hemispherical emissivity at a temperature.

    The total hemispherical emissivity weights the spectral hemispherical
    emissivity :math:`\varepsilon(\omega)` against the Planck blackbody
    spectral intensity and integrates over frequency:

    .. math::

        \varepsilon(T) = \frac{\int_0^\infty \varepsilon(\omega)\,
                               I_{bb}(\omega, T)\,d\omega}
                              {\int_0^\infty I_{bb}(\omega, T)\,d\omega},
        \qquad
        I_{bb}(\omega, T) = \frac{\hbar\omega^3}{4\pi^3 c^2}
            \frac{1}{\exp(\hbar\omega / k_B T) - 1}.

    Because the result is a ratio, the constant prefactor
    :math:`\hbar/(4\pi^3 c^2)` cancels and the angular frequency may be
    replaced by the photon energy :math:`E = \hbar\omega` (in eV), so the
    weight reduces to :math:`w(E) = E^3 / [\exp(E / k_B T) - 1]` with
    :math:`k_B T` expressed in eV.

    .. note::

        The integral is evaluated over the supplied energy grid ``ene`` only.
        At moderate temperatures the Planck weight peaks at low photon energy
        (:math:`\sim k_B T` to :math:`\sim 10\,k_B T`), so for a meaningful
        :math:`\varepsilon(T)` the grid should start near zero and resolve the
        thermally relevant range; otherwise the truncated integral
        underestimates the contribution from unsampled frequencies.

    Parameters
    ----------
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV (assumed strictly positive).
    emis_w : ndarray, shape (ne,)
        Spectral hemispherical emissivity :math:`\varepsilon(\omega)`.
    temperature : float
        Absolute temperature in kelvin.

    Returns
    -------
    emis_total : float
        Total hemispherical emissivity (unitless, in [0, 1]).
    """
    from ..utils.constants import ELECTRONVOLT_SI, K_BOLTZMAN_SI

    # Boltzmann constant in eV/K so that E/(kT) is dimensionless with E in eV.
    kb_ev = K_BOLTZMAN_SI / ELECTRONVOLT_SI
    kt = kb_ev * temperature

    # Planck weight w(E) = E^3 / (exp(E/kT) - 1); the constant prefactor
    # hbar/(4 pi^3 c^2) cancels in the ratio. Guard the exponential against
    # overflow for E >> kT (where the weight is negligibly small anyway).
    x = ene / kt
    with np.errstate(over='ignore'):
        weight = np.where(x < 700.0, ene**3 / np.expm1(x), 0.0)

    denom = np.trapezoid(weight, x=ene)
    if denom == 0.0:
        return 0.0
    return np.trapezoid(emis_w * weight, x=ene) / denom


def write_emissivity(data_controller, ene, epsr, epsi, comp, spin_tag):
    r"""Compute and write all emissivity spectra for one diagonal component.

    Driven by the configuration stored on ``data_controller`` by
    :meth:`PAOFLOW.dielectric_tensor` when ``emissivity=True``:

    * ``emis_angles`` \u2014 incidence angles (degrees) at which the Fresnel
      directional reflectivity :math:`R(\theta, \omega)` and emissivity
      :math:`\varepsilon(\theta, \omega) = 1 - R` are tabulated.
    * ``emis_ntheta`` \u2014 number of polar-angle samples used for the
      hemispherical integral.
    * ``emis_temperature`` \u2014 temperature(s) (K) at which the Planck-weighted
      total hemispherical emissivity is evaluated.

    Parameters
    ----------
    data_controller : DataController
        Provides ``emis_angles``, ``emis_ntheta``, ``emis_temperature`` and
        the ``write_file_row_col`` output method.
    ene : ndarray, shape (ne,)
        Photon-energy grid in eV.
    epsr, epsi : ndarray, shape (ne,)
        Real and imaginary parts of the diagonal dielectric function.
    comp : str
        Two-letter tensor-component tag (e.g. ``'xx'``) for output filenames.
    spin_tag : str
        Spin suffix appended to filenames (``''``, ``'_0'`` or ``'_1'``).

    Output files (written to ``opath``)
    ------------------------------------
    ``refl_th{deg}_{comp}{spin}.dat``, ``emis_th{deg}_{comp}{spin}.dat``
        Directional reflectivity and emissivity at each selected angle.
    ``emish_{comp}{spin}.dat``
        Spectral hemispherical emissivity :math:`\varepsilon(\omega)`.
    ``emist_{comp}{spin}.dat``
        Two-column (temperature, total hemispherical emissivity) file.
    """
    from ..utils.constants import DEGTORAD

    _, attributes = data_controller.data_dicts()
    angles_deg = np.atleast_1d(attributes['emis_angles']).astype(float)
    ntheta = int(attributes['emis_ntheta'])
    temps = np.atleast_1d(attributes['emis_temperature']).astype(float)

    # Directional reflectivity and emissivity at each requested incidence angle.
    for ang in angles_deg:
        refl_dir = directional_reflectivity(epsr, epsi, ang * DEGTORAD)
        deg_tag = int(round(ang))
        data_controller.write_file_row_col(
            'refl_th%d_%s%s.dat' % (deg_tag, comp, spin_tag), ene, refl_dir
        )
        data_controller.write_file_row_col(
            'emis_th%d_%s%s.dat' % (deg_tag, comp, spin_tag), ene, 1.0 - refl_dir
        )

    # Spectral hemispherical emissivity.
    emis_hemi = spectral_hemispherical_emissivity(epsr, epsi, ntheta)
    data_controller.write_file_row_col('emish_%s%s.dat' % (comp, spin_tag), ene, emis_hemi)

    # Total hemispherical emissivity at each requested temperature.
    emis_tot = np.array(
        [total_hemispherical_emissivity(ene, emis_hemi, T) for T in temps]
    )
    data_controller.write_file_row_col('emist_%s%s.dat' % (comp, spin_tag), temps, emis_tot)

    if rank == 0:
        for T, e in zip(temps, emis_tot):
            print(
                'Component %s%s, total hemispherical emissivity at %.1f K = %.6f'
                % (comp, spin_tag, T, e)
            )


def eps_loop(data_controller, ene, ispin, ipol, jpol):
    r"""Inner BZ loop: accumulate the interband dielectric-function integrands.

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

    # Optional adaptive (Yates et al., PRB 75, 195121 (2007)) interband
    # broadening. When ``pf.adaptive_smearing()`` has been run it stores the
    # per-(k, n, m) widths ``deltakp2`` = alpha |grad_k(E_n - E_m)| dk; use them
    # in place of the fixed scalar ``delta`` for the Lorentzian denominator.
    # Where two bands are locally parallel (|grad_k(E_n - E_m)| -> 0) the
    # adaptive width collapses and the Lorentzian becomes a near-delta spike on
    # a single frequency-grid point. To suppress those divergences without
    # washing out the physics the width is floored at the frequency-grid
    # spacing by default: every Lorentzian is then at least one bin wide (no
    # single-point spikes) while sharp van Hove singularities are preserved.
    # Set attr['adaptive_smearing_floor'] to override (e.g. to the fixed
    # ``delta`` for a smoother, purely additive broadening).
    adaptive = 'deltakp2' in arrays
    if adaptive:
        grid_spacing = (ene[1] - ene[0]) if ene.size > 1 else intersmear
        eta_floor = attributes.get('adaptive_smearing_floor', grid_spacing)
        deltakp2 = arrays['deltakp2'][:, :bndmax, :bndmax, ispin]
        # if rank == 0:
        #     print('Using adaptive (Yates) interband smearing for the dielectric tensor')

    spin_factor = 2 if (attributes['nspin'] == 1 and not attributes['dftSO']) else 1
    Ef = 1.0e-9

    epsi = np.zeros(esize, dtype=float)
    epsr = np.zeros(esize, dtype=float)

    # ``degauss`` is needed both for the integrated occupations (insulators)
    # and for the Drude term (metals); pull it unconditionally so the metal
    # fallback below cannot NameError when ``smearing is None``.
    degauss = attributes.get('degauss', None)

    if smearing == None or attributes['insulator']:
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
            if degauss is None:
                raise ValueError(
                    'Metal dielectric requires a smearing width: pass'
                    " `smearing='gauss'` (or 'm-p') and a `degauss` value."
                )
            if rank == 0:
                print(
                    'Smearing is None for a metal, switching to gaussian'
                    ' smearing with degauss = %.4f eV' % degauss
                )
            fnF = spin_factor * gaussian(Ek, Ef, degauss)

    for ik in range(fn.shape[0]):
        for iband2 in range(bndmax):
            for iband1 in range(bndmax):
                if iband1 != iband2:
                    E_diff_nm = Ek[ik, iband2] - Ek[ik, iband1]
                    f_nm = fn[ik, iband2] - fn[ik, iband1]
                    if np.abs(f_nm) > th0 and fn[ik, iband1] > th1 and fn[ik, iband2] < spin_factor:
                        # Interband broadening: adaptive per-(k, n, m) width
                        # when available, otherwise the fixed scalar value.
                        eta = (
                            max(deltakp2[ik, iband1, iband2], eta_floor) if adaptive else intersmear
                        )
                        pksp2 = np.real(
                            arrays['pksp'][ik, ipol, iband1, iband2, ispin]
                            * arrays['pksp'][ik, jpol, iband2, iband1, ispin]
                        )
                        # pksp2 in unit of (AU*eV)^2
                        epsi[:] += (
                            pksp2
                            * eta
                            * ene[:]
                            * fn[ik, iband1]
                            / (
                                ((E_diff_nm**2 - ene[:] ** 2) ** 2 + eta**2 * ene[:] ** 2)
                                * (E_diff_nm)
                            )
                        )
                        epsr[:] += (
                            pksp2
                            * (E_diff_nm**2 - ene[:] ** 2)
                            * fn[ik, iband1]
                            / (
                                ((E_diff_nm**2 - ene[:] ** 2) ** 2 + eta**2 * ene[:] ** 2)
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
        # The intraband (Drude) contribution carries a 4\u03c0 prefactor in QE
        # eq. (8) line 1, while the interband contribution carries 8\u03c0
        # (eq. 8 line 2). Both branches are multiplied by the common
        # ``factor`` in ``do_epsilon`` (built for the 8\u03c0 case), so the
        # Drude part has to be rescaled by 1/2 to recover the correct
        # absolute normalisation.
        epsi += 0.5 * epsi_metal
        epsr += 0.5 * epsr_metal

    np.seterr(over=orig_over_err)
    return (epsi, epsr)


def jdos_loop(data_controller, ene, ispin, jdos_smeartype):
    r"""Inner BZ loop: accumulate the joint-density-of-states integrand.

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
    from ..utils.communication import load_balancing

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

    # ``my_eigsmat`` holds the full BZ on every rank, so partition the k-loop
    # here: each rank accumulates only its slice and the caller reduces the
    # partial sums with MPI.Allreduce.  Without this every rank would compute
    # the complete JDOS and the Allreduce would scale the result by the number
    # of ranks (spurious dependence on the core count).
    ini_ik, end_ik = load_balancing(comm.Get_size(), rank, nkpnts)

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
        for ik in range(ini_ik, end_ik):
            for iband2 in range(bndmax):
                for iband1 in range(bndmax):
                    E_diff_nm = Ek[ik, iband2] - Ek[ik, iband1]
                    if fn[ik, iband1] > 1.0e-4 and fn[ik, iband2] < 2.0 and E_diff_nm > 1e-10:
                        f_nm = fn[ik, iband1] - fn[ik, iband2]
                        jdos += f_nm * gaussian(E_diff_nm, ene, intersmear) * kweights[ik]
                        count += f_nm

    elif jdos_smeartype == 'lorentz':
        for ik in range(ini_ik, end_ik):
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

    # The normalization must use the global oscillator-strength sum so it does
    # not depend on how the k-points were split across ranks.
    count = comm.allreduce(count, op=MPI.SUM)

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
