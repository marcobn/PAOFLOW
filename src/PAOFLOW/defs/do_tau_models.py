import numpy as np
from scipy.constants import hbar

me = 9.10938e-31
e = 1.60217662e-19
ev2j = 1.60217662e-19
epso = 8.854187817e-12


def acoustic_model(temp, eigs, params):
    """Compute electron lifetimes for acoustic phonon deformation-potential scattering.

    Parameters
    ----------
    temp : float
        Temperature (K).
    eigs : np.ndarray
        Band eigenvalues (eV) measured from the band edge.
    params : dict
        Model parameters:

        - ``'v'`` : float — sound velocity (m/s).
        - ``'rho'`` : float — mass density (kg/m³).
        - ``'ms'`` : float — effective mass in units of :math:`m_e`.
        - ``'D_ac'`` : float — acoustic deformation potential (eV).

    Returns
    -------
    np.ndarray
        Electron lifetime :math:`\\tau_{\\rm ac}` (s) for each eigenvalue.

    Notes
    -----
    Formula from Fiorentini *et al.* applied to Mg₃Sb₂-type materials.
    The scattering rate scales as :math:`\\tau^{-1} \\propto k_BT \\sqrt{E}`.
    """
    # Formula from fiorentini paper on Mg3Sb2
    temp *= ev2j
    E = eigs * ev2j  # Eigenvalues in J
    v = params['v']  # Velocity in m/s
    rho = params['rho']  # Mass density kg/m^3
    ms = params['ms'] * me  # effective mass tensor in kg
    D_ac = params['D_ac'] * ev2j  # Acoustic deformation potential in J

    return (2 * np.pi * rho * (hbar**2 * v) ** 2) / (
        (2 * ms) ** 1.5 * (D_ac**2) * np.sqrt(E) * temp
    )


def optical_model(temp, eigs, params):
    """Compute electron lifetimes for optical phonon deformation-potential scattering.

    Parameters
    ----------
    temp : float
        Temperature (K).
    eigs : np.ndarray
        Band eigenvalues (eV) measured from the band edge.
    params : dict
        Model parameters:

        - ``'hwlo'`` : array_like of float — LO phonon energies (eV).
        - ``'rho'`` : float — mass density (kg/m³).
        - ``'D_op'`` : float — optical deformation potential (eV).
        - ``'ms'`` : float — effective mass in units of :math:`m_e`.

    Returns
    -------
    np.ndarray
        Electron lifetime :math:`\\tau_{\\rm op}` (s) for each eigenvalue.

    Notes
    -----
    Formula from Jacoboni, *Theory of Electron Transport in Semiconductors*.
    The emission and absorption channels are weighted by the Bose-Einstein
    distribution :math:`N_{\\rm op} = [e^{\\hbar\\omega/k_BT}-1]^{-1}`.
    """
    # Formula from jacoboni theory of electron transport in semiconductors
    temp *= ev2j
    E = eigs * ev2j
    hwlo = np.array(params['hwlo']) * ev2j  # Phonon freq
    rho = params['rho']  # Mass density kg/m^3
    D_op = params['D_op'] * ev2j  # Acoustic deformation potential in J
    ms = params['ms'] * me  # effective mass tensor in kg

    x = E / temp
    x0 = hwlo / temp
    X = x - x0
    X[X < 0] = 0

    Nop = 1 / (np.exp(x0) - 1)

    return (np.sqrt(2 * temp) * np.pi * x0 * rho * hbar**2) / (
        (ms**1.5) * (D_op**2) * (Nop * np.sqrt(x + x0) + (Nop + 1) * np.sqrt(X))
    )


def polar_acoustic_model(temp, eigs, params):
    """Compute electron lifetimes for polar acoustic phonon (piezoelectric) scattering.

    Parameters
    ----------
    temp : float
        Temperature (K).
    eigs : np.ndarray
        Band eigenvalues (eV) measured from the band edge.
    params : dict
        Model parameters:

        - ``'piezo'`` : float — piezoelectric constant (C/m²).
        - ``'doping_conc'`` : float — doping concentration (cm⁻³).
        - ``'eps_0'`` : float — low-frequency dielectric constant (in units
          of :math:`\\varepsilon_0`).
        - ``'eps_inf'`` : float — high-frequency dielectric constant.
        - ``'ms'`` : float — effective mass in units of :math:`m_e`.
        - ``'rho'`` : float — mass density (kg/m³).
        - ``'v'`` : float — sound velocity (m/s).

    Returns
    -------
    np.ndarray
        Electron lifetime :math:`\\tau_{\\rm pac}` (s) for each eigenvalue.
    """
    temp *= ev2j
    E = eigs * ev2j
    piezo = params['piezo']  # Piezoelectric constant
    nd = np.abs(params['doping_conc']) * 1e6  # Doping concentration in /m^3
    eps_0 = params['eps_0'] * epso  # Low freq dielectric const
    eps_inf = params['eps_inf'] * epso  # High freq dielectirc const
    ms = params['ms'] * me  # effective mass tensor in kg
    rho = params['rho']
    v = params['v']

    eps = eps_inf + eps_0
    qo = np.sqrt(abs(nd) * e**2 / (eps * temp))
    eps_o = ((hbar * qo) ** 2) / (2 * ms)
    P_pac = (
        ((piezo * e) ** 2 * ms**0.5 * temp)
        / (np.sqrt(2 * E) * 2 * np.pi * eps**2 * hbar**2 * rho * v**2)
    ) * (1 - (eps_o / (2 * E)) * np.log(1 + 4 * E / eps_o) + 1 / (1 + 4 * E / eps_o))
    P_pac[np.isnan(P_pac)] = 0
    return 1 / P_pac


def polar_optical_model(temp, eigs, params):
    """Compute electron lifetimes for polar optical (Fr\u00f6hlich) phonon scattering.

    Parameters
    ----------
    temp : float
        Temperature (K).
    eigs : np.ndarray
        Band eigenvalues (eV) measured from the band edge.
    params : dict
        Model parameters:

        - ``'Ef'`` : float — Fermi energy (eV).
        - ``'hwlo'`` : array_like of float — LO phonon energies (eV).
        - ``'eps_0'`` : float — low-frequency dielectric constant.
        - ``'eps_inf'`` : float — high-frequency dielectric constant.
        - ``'ms'`` : float — effective mass in units of :math:`m_e`.

    Returns
    -------
    np.ndarray
        Electron lifetime :math:`\\tau_{\\rm pol}` (s) for each eigenvalue.

    Notes
    -----
    Formula from Fiorentini *et al.*, Phys. Rev. B (Mg₃Sb₂ paper).  The
    coupling involves both phonon emission and absorption weighted by Bose
    and Fermi-Dirac statistics.
    """
    # Formula from fiorentini paper on Mg3Sb2
    temp *= ev2j
    E = eigs * ev2j
    Ef = params['Ef'] * ev2j  # fermi energy
    hwlo = np.array(params['hwlo']) * ev2j  # Phonon freq
    eps_0 = params['eps_0'] * epso  # low freq dielectric const
    eps_inf = params['eps_inf'] * epso  # high freq dielectirc const
    ms = params['ms'] * me  # effective mass tensor in kg

    fermi = lambda E, Ef, T: 1 / (np.exp((E - Ef) / T) + 1.0)
    planck = lambda hwlo, T: 1 / (np.exp(hwlo / T) - 1)

    P_pol = 0.0
    eps = eps_inf + eps_0
    eps_inv = 1 / eps_inf - 1 / eps

    for hw in hwlo:
        f = fermi(E, Ef, temp)
        fp = fermi(E + hw, Ef, temp)
        fm = fermi(E - hw, Ef, temp)
        n = planck(hw, temp)
        Wo = e**2 / (4 * np.pi * hbar**2) * np.sqrt(2 * ms * hw) * eps_inv
        Z = 2 / (Wo * np.sqrt(hw))

        def remove_NaN(arr):
            arr[np.isnan(arr)] = 0
            return arr

        A = remove_NaN(
            (n + 1) * fp / f * ((2 * E + hw) * np.arcsinh(np.sqrt(E / hw)) - np.sqrt(E * (E + hw)))
        )
        B = remove_NaN(
            np.heaviside(E - hw, 1)
            * n
            * fm
            / f
            * ((2 * E - hw) * np.arccosh(np.sqrt(E / hw)) - np.sqrt(E * (E - hw)))
        )
        t1 = remove_NaN((n + 1) * fp / f * np.arcsinh(np.sqrt(E / hw)))
        t2 = remove_NaN(np.heaviside(E - hw, 1) * n * fm / f * np.arccosh(np.sqrt(E / hw)))
        C = 2 * E * (t1 + t2)
        P = (C - A - B) / (Z * E**1.5)
        P_pol += P

    return 1 / P_pol


def impurity_model(temp, eigs, params):
    """Compute electron lifetimes for ionized impurity scattering.

    Parameters
    ----------
    temp : float
        Temperature (K).
    eigs : np.ndarray
        Band eigenvalues (eV) measured from the band edge.
    params : dict
        Model parameters:

        - ``'nI'`` : float — impurity concentration (cm⁻³).
        - ``'Zi'`` : int — charge number of the impurity.
        - ``'ms'`` : float — effective mass in units of :math:`m_e`.
        - ``'eps_0'`` : float — low-frequency dielectric constant.
        - ``'eps_inf'`` : float — high-frequency dielectric constant.

    Returns
    -------
    np.ndarray
        Electron lifetime :math:`\\tau_{\\rm imp}` (s) for each eigenvalue.

    Notes
    -----
    Formula from Fiorentini *et al.* (Mg₃Sb₂ paper).  The Brooks-Herring
    screened Coulomb potential is used with a Thomas-Fermi screening
    wave-vector :math:`q_0 = \\sqrt{e^2 n_I / (\\varepsilon k_BT)}`.
    """
    # formula from fiorentini paper on Mg3Sb2
    temp *= ev2j
    E = eigs * ev2j
    nI = np.abs(params['nI']) * 1e6  # impurity conc in /m^3
    Zi = params['Zi']
    ms = params['ms'] * me  # effective mass tensor in kg
    eps_0 = params['eps_0'] * epso  # low freq dielectric const
    eps_inf = params['eps_inf'] * epso  # high freq dielectirc const

    eps = eps_inf + eps_0
    qo = np.sqrt(e**2 * nI / (eps * temp))
    x = (hbar * qo) ** 2 / (8 * ms * E)
    P_imp = np.pi * nI * Zi**2 * e**4 / (E**1.5 * np.sqrt(2 * ms) * (4 * np.pi * eps) ** 2)
    return 1 / (P_imp * (np.log(1 + 1.0 / x) - 1.0 / (1 + x)))


def builtin_tau_model(label, params, weight):
    """Instantiate a built-in scattering rate model as a :class:`TauModel` object.

    Parameters
    ----------
    label : str
        Name of the built-in model.  Supported values:

        - ``'acoustic'`` — :func:`acoustic_model`
        - ``'optical'`` — :func:`optical_model`
        - ``'polar_optical'`` — :func:`polar_optical_model`
        - ``'polar_acoustic'`` — :func:`polar_acoustic_model`
        - ``'impurity'`` — :func:`impurity_model`
    params : dict
        Parameter dictionary forwarded to the chosen model function.
    weight : float
        Multiplicative weight applied to the model scattering rate during
        Matthiessen\'s-rule combination.

    Returns
    -------
    TauModel or None
        A :class:`~PAOFLOW.defs.TauModel.TauModel` instance with
        ``.function`` set to the appropriate callable, or ``None`` if
        ``label`` is not recognised.
    """
    from .TauModel import TauModel

    model = TauModel(params=params, weight=weight)

    if label == 'acoustic':
        model.function = acoustic_model
    elif label == 'optical':
        model.function = optical_model
    elif label == 'polar_optical':
        model.function = polar_optical_model
    elif label == 'polar_acoustic':
        model.function = polar_acoustic_model
    elif label == 'impurity':
        model.function = impurity_model
    else:
        print('Model not implemented.')
        return None

    return model
