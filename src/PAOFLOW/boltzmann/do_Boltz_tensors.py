import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_Boltz_tensors(data_controller, smearing, temp, ene, velkp, ispin, channels, weights):
    r"""Compute the Boltzmann transport tensors L0, L1, and L2.

    The three generalized transport tensors are defined by the BZ integral

    .. math::

        L^{(\\alpha)}_{ij}(\\varepsilon) =
        \\sum_{n\\mathbf{k}} \\tau_{n\\mathbf{k}}\\,
        v^i_{n\\mathbf{k}}\\, v^j_{n\\mathbf{k}}\\,
        \\left(-\\frac{\\partial f}{\\partial\\varepsilon}\\right)
        (E_{n\\mathbf{k}} - \\varepsilon)^\\alpha

    for ``alpha`` = 0, 1, 2.  From these, the electrical conductivity
    (\u03c3 ~ L0), Seebeck coefficient (S ~ L1/L0), and electronic thermal
    conductivity (\u03ba ~ L2 - L1\u00b2/L0) are derived.

    When ``smearing`` is ``None`` the Fermi-Dirac derivative
    ``1/(4T cosh\u00b2(...))`` is evaluated analytically.  When adaptive
    smearing is enabled (``'gauss'`` or ``'m-p'``) the integration is
    performed on an extended energy grid and convolved with the Fermi window
    via Simpson quadrature.

    Only the upper-triangular components of each tensor are computed; the
    lower-triangular entries are filled by symmetry before returning.

    Parameters
    ----------
    data_controller : DataController
        Provides ``E_k``, ``deltakp``, ``scattering_tau``, ``nkpnts``, ``bnd``.
    smearing : str or None
        Smearing type: ``None`` (Fermi-Dirac), ``'gauss'``, or ``'m-p'``.
    temp : float
        Temperature in eV (k\u0432T).
    ene : ndarray, shape (ne,)
        Chemical-potential (energy) grid in eV.
    velkp : ndarray, shape (nkpts_local, 3, bnd, nspin)
        Band velocities (diagonal elements of the momentum matrix).
    ispin : int
        Spin channel index.
    channels : list
        Scattering channel identifiers (strings or :class:`TauModel` objects).
    weights : list of float
        Harmonic-sum weights for each scattering channel.

    Returns
    -------
    L0, L1, L2 : ndarray, shape (3, 3, ne), or (None, None, None)
        Transport tensors on rank 0; ``(None, None, None)`` on all other ranks.
    """
    from scipy.integrate import simpson

    # Compute the L_alpha tensors for Boltzmann transport
    arrays, _ = data_controller.data_dicts()

    esize = ene.size
    arrays['scattering_tau'] = get_tau(data_controller, temp, channels, weights)

    #### Forced t_tensor to have all components
    t_tensor = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]], dtype=int)

    # Quick call function for L_loop
    fLloop = lambda spol: L_loop(data_controller, temp, smearing, ene, velkp, t_tensor, spol, ispin)

    # Quick call function for Zeros on rank Zero
    zoz = lambda r: np.zeros((3, 3, esize), dtype=float) if r == 0 else None

    if smearing is None:
        L0 = zoz(rank)
        L0aux = fLloop(0)
        comm.Reduce(L0aux, L0, op=MPI.SUM)
        L0aux = None

        L1 = zoz(rank)
        L1aux = fLloop(1)
        comm.Reduce(L1aux, L1, op=MPI.SUM)
        L1aux = None

        L2 = zoz(rank)
        L2aux = fLloop(2)
        comm.Reduce(L2aux, L2, op=MPI.SUM)
        L2aux = None

    else:
        L0 = zoz(rank)
        L1 = zoz(rank)
        L2 = zoz(rank)

        # Fixed threshold
        thresh = 1e-9
        dE_max = 2 * temp * np.arccosh(1 / np.sqrt(thresh))

        if len(ene) > 1:
            dE = ene[1] - ene[0]
        else:
            dE = 2 * temp * np.arccosh(1 / np.sqrt(thresh)) * 1e-2 + 1e-8

        lower = np.flip(np.arange(ene[0] - dE, ene[0] - dE_max - dE, -dE))
        upper = np.arange(ene[-1] + dE, ene[-1] + dE_max + dE, dE)
        ene_aux = np.concatenate((lower, ene, upper))

        L0_ext = np.zeros((3, 3, ene_aux.size), dtype=float) if rank == 0 else None
        L0aux = L_loop(data_controller, temp, smearing, ene_aux, velkp, t_tensor, 0, ispin)
        comm.Reduce(L0aux, L0_ext, op=MPI.SUM)
        L0aux = None

        if rank == 0:
            # Interpolate
            ene_int = np.linspace(ene_aux[0], ene_aux[-1], 2 * ene_aux.size - 1)
            L0_int = np.zeros((3, 3, ene_int.size), dtype=float)
            for t_comp in t_tensor:
                L0_int[t_comp[0], t_comp[1], :] = np.interp(
                    ene_int, ene_aux, L0_ext[t_comp[0], t_comp[1], :]
                )

            for i, ef in enumerate(ene):
                fermi_smear = 1 / (4 * temp * (np.cosh((ene_int - ef) / (2 * temp)) ** 2))

                for t_comp in t_tensor:
                    L_smear_aux = L0_int[t_comp[0], t_comp[1], :] * fermi_smear
                    L0[t_comp[0], t_comp[1], i] = simpson(L_smear_aux, ene_int)
                    L1[t_comp[0], t_comp[1], i] = simpson(L_smear_aux * (ene_int - ef), ene_int)
                    L2[t_comp[0], t_comp[1], i] = simpson(
                        L_smear_aux * (ene_int - ef) ** 2, ene_int
                    )

    if rank == 0:
        # Assign lower triangular to upper triangular
        sym = lambda L: (L[0, 1], L[0, 2], L[1, 2])
        L0[1, 0], L0[2, 0], L0[2, 1] = sym(L0)
        L1[1, 0], L1[2, 0], L1[2, 1] = sym(L1)
        L2[1, 0], L2[2, 0], L2[2, 1] = sym(L2)

    return (L0, L1, L2) if rank == 0 else (None, None, None)


def do_Boltz_tensors_hall(data_controller, smearing, temp, ene, velkp, ispin, channels, weights):
    r"""Compute the anomalous (Hall) Boltzmann transport tensors L0_hall and L1_hall.

    Evaluates the rank-3 Hall conductivity kernel

    .. math::

        L^{\\rm Hall}_{ijp}(\\varepsilon) =
        \\sum_{n\\mathbf{k}} \\tau^2_{n\\mathbf{k}}
        \\sum_{qr} \\epsilon_{pqr}\\,
        v^i_{n\\mathbf{k}}\\, v^r_{n\\mathbf{k}}\\,
        M^{-1}_{jq,n\\mathbf{k}}\\,
        \\left(-\\frac{\\partial f}{\\partial\\varepsilon}\\right)

    where :math:`\\epsilon_{pqr}` is the Levi-Civita symbol and
    :math:`M^{-1}_{jq}` is the inverse effective-mass tensor from
    ``arry['d2Ed2k']``.

    ``L0_hall`` is the zeroth energy moment of this kernel and ``L1_hall``
    its first moment, weighted by :math:`(E_{n\\mathbf{k}} - \\varepsilon)`.

    When ``smearing`` is ``None`` the Fermi-Dirac derivative
    ``1/(4T cosh\u00b2(...))`` is evaluated analytically and each moment is
    obtained from a separate :func:`L_loop_hall` call, selected through its
    ``alpha`` argument.  When adaptive smearing is enabled (``'gauss'`` or
    ``'m-p'``) the kernel is evaluated once on an extended energy grid and
    both moments are obtained by convolution with the Fermi window via
    Simpson quadrature.

    Parameters
    ----------
    data_controller : DataController
        Provides ``E_k``, ``deltakp``, ``d2Ed2k``, ``scattering_tau``,
        ``nkpnts``, ``bnd``, ``nspin``.
    smearing : str or None
        Smearing type: ``None`` (Fermi-Dirac), ``'gauss'``, or ``'m-p'``.
    temp : float
        Temperature in eV (k\u0432T).
    ene : ndarray, shape (ne,)
        Chemical-potential grid in eV.
    velkp : ndarray, shape (nkpts_local, 3, bnd, nspin)
        Band velocities.
    ispin : int
        Spin channel index.
    channels : list
        Scattering channel identifiers.
    weights : list of float
        Harmonic-sum weights for each scattering channel.

    Returns
    -------
    L0_hall, L1_hall : ndarray, shape (3, 3, 3, ne), or (None, None)
        Hall transport tensor on rank 0; ``(None, None)`` on all other ranks.
    """
    from scipy.integrate import simpson

    arrays, _ = data_controller.data_dicts()

    esize = ene.size
    arrays['scattering_tau'] = get_tau(data_controller, temp, channels, weights)

    #### Forced t_tensor to have all components
    t_tensor = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]], dtype=int)

    # Quick call function for Zeros on rank Zero
    zoz = lambda r: np.zeros((3, 3, 3, esize), dtype=float) if r == 0 else None

    if smearing is None:
        L0_hall = zoz(rank)
        L0_hall_aux = L_loop_hall(data_controller, temp, smearing, ene, velkp, t_tensor, 0, ispin)
        comm.Reduce(L0_hall_aux, L0_hall, op=MPI.SUM)
        L0_hall_aux = None

        L1_hall = zoz(rank)
        L1_hall_aux = L_loop_hall(data_controller, temp, smearing, ene, velkp, t_tensor, 1, ispin)
        comm.Reduce(L1_hall_aux, L1_hall, op=MPI.SUM)
        L1_hall_aux = None
    else:
        L0_hall = zoz(rank)
        L1_hall = zoz(rank)

        # Fixed threshold
        thresh = 1e-9
        dE_max = 2 * temp * np.arccosh(1 / np.sqrt(thresh))

        if len(ene) > 1:
            dE = ene[1] - ene[0]
        else:
            dE = 2 * temp * np.arccosh(1 / np.sqrt(thresh)) * 1e-2 + 1e-8

        lower = np.flip(np.arange(ene[0] - dE, ene[0] - dE_max - dE, -dE))
        upper = np.arange(ene[-1] + dE, ene[-1] + dE_max + dE, dE)
        ene_aux = np.concatenate((lower, ene, upper))

        L0_hall_ext = np.zeros((3, 3, 3, ene_aux.size), dtype=float) if rank == 0 else None
        L0_hall_aux = L_loop_hall(
            data_controller, temp, smearing, ene_aux, velkp, t_tensor, 0, ispin
        )
        comm.Reduce(L0_hall_aux, L0_hall_ext, op=MPI.SUM)
        L0_hall_aux = None

        if rank == 0:
            # Interpolate
            ene_int = np.linspace(ene_aux[0], ene_aux[-1], 2 * ene_aux.size - 1)
            L0_hall_int = np.zeros((3, 3, 3, ene_int.size), dtype=float)
            for h_indx in np.ndindex((3, 3, 3)):
                L0_hall_int[h_indx[0], h_indx[1], h_indx[2], :] = np.interp(
                    ene_int, ene_aux, L0_hall_ext[h_indx[0], h_indx[1], h_indx[2], :]
                )

            for i, ef in enumerate(ene):
                fermi_smear = 1 / (4 * temp * (np.cosh((ene_int - ef) / (2 * temp)) ** 2))

                for h_indx in np.ndindex((3, 3, 3)):
                    L_hall_smear_aux = L0_hall_int[h_indx[0], h_indx[1], h_indx[2], :] * fermi_smear
                    L0_hall[h_indx[0], h_indx[1], h_indx[2], i] = simpson(L_hall_smear_aux, ene_int)
                    L1_hall[h_indx[0], h_indx[1], h_indx[2], i] = simpson(
                        L_hall_smear_aux * (ene_int - ef), ene_int
                    )
    return (L0_hall, L1_hall) if rank == 0 else (None, None)


def get_tau(data_controller, temp, channels, weights):
    """Compute the relaxation time \u03c4(n, k) from a list of scattering channels.

    Constructs the total scattering rate as a harmonic sum of individual
    channel contributions

    .. math::

        \\frac{1}{\\tau_{n\\mathbf{k}}} =
        \\sum_c \\frac{w_c}{\\tau^{(c)}_{n\\mathbf{k}}}

    and returns :math:`\\tau = 1 / \\sum_c (w_c / \\tau^{(c)})`.  When no
    channels are provided, a constant relaxation time of unity is used
    (constant-\u03c4 approximation).

    Each channel entry in ``channels`` can be either a built-in model name
    string (dispatched to :func:`do_tau_models.builtin_tau_model`) or a
    :class:`TauModel` instance.

    Parameters
    ----------
    data_controller : DataController
        Provides ``E_k``, ``bnd``, and ``tau_dict`` (model parameters).
    temp : float
        Temperature in eV (k\u0432T), passed to each model's ``evaluate`` method.
    channels : list
        Scattering channel identifiers (strings or :class:`TauModel` objects).
        Pass an empty list or ``None`` for constant-\u03c4.
    weights : list of float
        Per-channel harmonic-sum weights.  Defaults to all-ones when empty.

    Returns
    -------
    tau : ndarray, shape (nkpts_local, bnd, nspin)
        Relaxation time in arbitrary units (consistent with ``tau_dict``).

    Raises
    ------
    Exception
        If a required parameter is missing from ``attr['tau_dict']``.
    """
    import numpy as np

    from .do_tau_models import builtin_tau_model
    from .TauModel import TauModel

    arry, attr = data_controller.data_dicts()
    snktot = arry['E_k'].shape[0]

    taus = []
    for c in channels:
        if c == 'acoustic':
            a_tau = np.ones((snktot), dtype=float)
            taus.append(a_tau)

    bnd = attr['bnd']
    eigs = np.abs(arry['E_k'][:, :bnd, :])
    snktot, _, nspin = eigs.shape

    models = []
    if channels != None:
        if len(weights) == 0:
            weights = np.ones(len(channels))
        elif len(weights) != len(models):
            raise Exception('Length of weights does not match the number of channels.')
        for i, c in enumerate(channels):
            if isinstance(c, str):
                models.append(builtin_tau_model(c, attr['tau_dict'], weights[i]))
            elif isinstance(c, TauModel):
                c.weight = weights[i]
                models.append(c)
            else:
                print('Invalid channel type.')

    if len(models) == 0:
        # Constant relaxation time approximation with tau = 1
        tau = np.ones((snktot, bnd, nspin), dtype=float)

    else:
        # Compute tau as a harmonic sum of scattering channel contributions.
        tau = np.zeros((snktot, bnd, nspin), dtype=float)
        for m in models:
            try:
                tau += m.weight / m.evaluate(temp, eigs)
            except KeyError as e:
                from ..utils.report_exception import report_exception

                print(
                    'Ensure that all required parameters are specified in the provided dictionary.'
                )
                report_exception()
                raise e
        tau = 1 / tau

    return tau


def L_loop(data_controller, temp, smearing, ene, velkp, t_tensor, alpha, ispin):
    r"""Inner BZ summation loop for one L\u1d45 transport tensor.

    Evaluates the energy-resolved integral

    .. math::

        L^{(\\alpha)}_{ij}(\\varepsilon) =
        \\frac{1}{N_k} \\sum_{n\\mathbf{k}}
        \\tau_{n\\mathbf{k}}\\,
        v^i_{n\\mathbf{k}}\\, v^j_{n\\mathbf{k}}\\,
        \\sigma(E_{n\\mathbf{k}}, \\varepsilon, \\delta_k)\\,
        (E_{n\\mathbf{k}} - \\varepsilon)^\\alpha

    where :math:`\\sigma` is the smearing kernel (Fermi-Dirac derivative,
    Gaussian, or Methfessel-Paxton depending on ``smearing``).

    The loop runs over the k-point slice held by the local MPI rank.  Results
    are subsequently reduced across ranks by the caller.

    Parameters
    ----------
    data_controller : DataController
        Provides ``E_k``, ``deltakp``, ``scattering_tau``, ``nkpnts``, ``bnd``.
    temp : float
        Temperature in eV.
    smearing : str or None
        ``None`` uses the Fermi-Dirac window; ``'gauss'`` or ``'m-p'`` uses
        adaptive smearing widths from ``deltakp``.
    ene : ndarray, shape (ne,)
        Energy (chemical-potential) grid.
    velkp : ndarray, shape (nkpts_local, 3, bnd, nspin)
        Band velocities.
    t_tensor : ndarray, shape (nc, 2), int
        Tensor component pairs ``[i, j]`` to evaluate (upper triangle).
    alpha : int
        Power of the ``(E - \u03b5)`` kernel: 0, 1, or 2.
    ispin : int
        Spin channel index.

    Returns
    -------
    L : ndarray, shape (3, 3, ne)
        Local contribution to L\u1d45 from this MPI rank's k-points.
    """
    from ..utils.smearing import gaussian, metpax
    # We assume tau=1 in the constant relaxation time approximation

    arrays, attributes = data_controller.data_dicts()

    esize = ene.size

    bnd = attributes['bnd']
    kq_wght = 1.0 / attributes['nkpnts']
    if smearing is not None and smearing != 'gauss' and smearing != 'm-p':
        print('%s Smearing Not Implemented.' % smearing)
        comm.Abort()

    L = np.zeros((3, 3, esize), dtype=float)

    # Vectorised over (k, band, energy). Eaux[k, n, e] = E_k - ene; the smearing
    # window and (E - eps)^alpha share that grid, contracted against the per-(k,n)
    # tau * v_i * v_j prefactor. Matches the former band loop up to sum order.
    Ek = arrays['E_k'][:, :bnd, ispin]
    tau = arrays['scattering_tau'][:, :bnd, ispin]
    Ediff = Ek[:, :, None] - ene[None, None, :]
    EtoAlpha = np.power(Ediff, alpha)
    if smearing is None:
        smearA = 1.0 / (4 * temp * (np.cosh(Ediff / (2 * temp)) ** 2))
    else:
        delk = arrays['deltakp'][:, :bnd, ispin][:, :, None]
        if smearing == 'gauss':
            smearA = gaussian(Ek[:, :, None], ene[None, None, :], delk)
        elif smearing == 'm-p':
            smearA = metpax(Ek[:, :, None], ene[None, None, :], delk)

    S = smearA * EtoAlpha  # (k, bnd, ne)
    for ll in range(t_tensor.shape[0]):
        i = t_tensor[ll][0]
        j = t_tensor[ll][1]
        pref = kq_wght * tau * velkp[:, i, :bnd, ispin] * velkp[:, j, :bnd, ispin]
        L[i, j, :] = np.tensordot(pref, S, axes=([0, 1], [0, 1]))
    """
  # noise reduction using a running average (correlation function)
  # Only possible for sigma vs chemical potential
  win = int(esize*0.025)
  N = esize
  for l in range(t_tensor.shape[0]):
    i = t_tensor[l][0]
    j = t_tensor[l][1]
    L[i,j,:] = signal.correlate(L[i,j,:] , np.ones(win), mode='same', method='fft')/win
  """
    return L


def L_loop_hall(data_controller, temp, smearing, ene, velkp, t_tensor, alpha, ispin):
    r"""Inner BZ summation loop for the Hall transport tensor.

    Computes the rank-3 Hall kernel

    .. math::

        L^{\\rm Hall}_{ijp}(\\varepsilon) =
        \\frac{1}{N_k} \\sum_{n\\mathbf{k}}
        \\tau^2_{n\\mathbf{k}}
        \\left(\\sum_{qr} \\epsilon_{pqr}\\,
        v^i_{n\\mathbf{k}}\\, v^r_{n\\mathbf{k}}\\,
        M^{-1}_{jq,n\\mathbf{k}}\\right)
        \\sigma(E_{n\\mathbf{k}}, \\varepsilon, \\delta_k)\\,
        (E_{n\\mathbf{k}} - \\varepsilon)^\\alpha

    The Levi-Civita symbol :math:`\\epsilon_{pqr}` is a static rank-3
    tensor; the inverse effective-mass tensor components are read
    from ``arry['d2Ed2k']`` and assembled into a full 3\u00d73 matrix.

    The :math:`(E - \\varepsilon)^\\alpha` factor is applied only in the
    Fermi-Dirac branch; with adaptive smearing the kernel is returned for
    ``alpha`` = 0 irrespective of the argument, and the energy moment is
    taken by the caller.

    Parameters
    ----------
    data_controller : DataController
        Provides ``E_k``, ``deltakp``, ``d2Ed2k``, ``scattering_tau``,
        ``nkpnts``, ``bnd``, ``nspin``.
    temp : float
        Temperature in eV.
    smearing : str or None
        ``None`` uses the Fermi-Dirac window; ``'gauss'`` or ``'m-p'`` uses
        adaptive smearing widths from ``deltakp``.
    ene : ndarray, shape (ne,)
        Energy (chemical-potential) grid.
    velkp : ndarray, shape (nkpts_local, 3, bnd, nspin)
        Band velocities.
    t_tensor : ndarray, shape (nc, 2), int
        Tensor component pairs (used to unpack the symmetric effective-mass
        tensor stored as 6 independent components).
    alpha : int
        Power of the ``(E - \u03b5)`` kernel: 0 or 1.  Honoured only when
        ``smearing`` is ``None``.
    ispin : int
        Spin channel index.

    Returns
    -------
    L_hall : ndarray, shape (3, 3, 3, ne)
        Local contribution to the Hall tensor from this MPI rank's k-points.
    """
    from ..utils.smearing import gaussian, metpax

    arrays, attributes = data_controller.data_dicts()

    esize = ene.size

    snktot = arrays['E_k'].shape[0]
    bnd = attributes['bnd']
    nspin = attributes['nspin']
    kq_wght = 1.0 / attributes['nkpnts']
    if smearing is not None and smearing != 'gauss' and smearing != 'm-p':
        print('%s Smearing Not Implemented.' % smearing)
        comm.Abort()

    L_hall = np.zeros((3, 3, 3, esize), dtype=float)

    M_inv = np.zeros((3, 3, snktot, bnd, nspin))
    eff_mass_inv = arrays['d2Ed2k']
    for l in range(t_tensor.shape[0]):
        i = t_tensor[l][0]
        j = t_tensor[l][1]
        if i == j:
            M_inv[i, j] = eff_mass_inv[i]
        elif i == 0 and j == 1:
            M_inv[i, j] = eff_mass_inv[3]
            M_inv[j, i] = eff_mass_inv[3]
        elif i == 0 and j == 2:
            M_inv[i, j] = eff_mass_inv[4]
            M_inv[j, i] = eff_mass_inv[4]
        elif i == 1 and j == 2:
            M_inv[i, j] = eff_mass_inv[5]
            M_inv[j, i] = eff_mass_inv[5]

    # Vectorised over (k, band, energy). The Levi-Civita contraction
    # sig[i,j,p] = sum_qr eps_pqr v_i v_r Minv_jq is precomputed once, then
    # reduced against tau^2 * smearing over (k, band). Matches the former
    # 5-deep band loop up to floating-point summation order.
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

    Ek = arrays['E_k'][:, :bnd, ispin]
    tau2 = arrays['scattering_tau'][:, :bnd, ispin] ** 2
    v = velkp[:, :, :bnd, ispin]  # (k, 3, bnd)
    Minv = M_inv[:, :, :, :, ispin]  # (j, q, k, bnd)
    if smearing is None:
        Ediff = Ek[:, :, None] - ene[None, None, :]
        EtoAlpha = np.power(Ediff, alpha)
        smearA = 1.0 / (4 * temp * (np.cosh(Ediff / (2 * temp)) ** 2))
        smearA = smearA * EtoAlpha
    else:
        delk = arrays['deltakp'][:, :bnd, ispin][:, :, None]
        if smearing == 'gauss':
            smearA = gaussian(Ek[:, :, None], ene[None, None, :], delk)
        elif smearing == 'm-p':
            smearA = metpax(Ek[:, :, None], ene[None, None, :], delk)

    # A[j,p,k,n] = sum_qr eps_pqr Minv_jq v_r ; sig[i,j,p,k,n] = v_i A[j,p,k,n]
    A = np.einsum('pqr,jqkn,krn->jpkn', eps, Minv, v, optimize=True)
    pref = kq_wght * np.einsum('kn,kin,jpkn->ijpkn', tau2, v, A, optimize=True)
    L_hall = np.tensordot(pref, smearA, axes=([3, 4], [0, 1]))
    return L_hall
