def gaussian(eig, ene, delta):
    """Evaluate the normalised Gaussian smearing function.

    Parameters
    ----------
    eig : float or np.ndarray
        Band eigenvalue(s) (eV).
    ene : float or np.ndarray
        Energy point(s) at which to evaluate the function (eV).
    delta : float or np.ndarray
        Smearing width parameter (eV).

    Returns
    -------
    float or np.ndarray
        Gaussian broadening:

        .. math::

            g(E; \\varepsilon, \\delta) =
            \\frac{1}{\\sqrt{\\pi}\\,\\delta}
            \\exp\\!\\left(-\\left(\\frac{E - \\varepsilon}{\\delta}\\right)^2\\right)
    """
    import numpy as np

    # Gaussian Smearing
    return (np.exp(-(((ene - eig) / delta) ** 2)) / delta) / np.sqrt(np.pi)


def metpax(eig, ene, delta):
    """Evaluate the Methfessel-Paxton smearing function.

    Parameters
    ----------
    eig : float or np.ndarray
        Band eigenvalue(s) (eV).
    ene : float or np.ndarray
        Energy point(s) at which to evaluate the function (eV).
    delta : float or np.ndarray
        Smearing width parameter (eV).

    Returns
    -------
    float or np.ndarray
        Methfessel-Paxton approximation to the delta function at order
        ``nh = 5``:

        .. math::

            D_N(x) = \\sum_{n=0}^{N}
            \\frac{(-1)^n}{n!\\, 4^n \\sqrt{\\pi}}
            H_{2n}(x) \\exp(-x^2)

        where :math:`x = (E - \\varepsilon)/\\delta` and
        :math:`H_{2n}` are Hermite polynomials.  See Methfessel and
        Paxton, Phys. Rev. B **40**, 3616 (1989).
    """
    import numpy as np
    from math import factorial

    import numpy as np
    from numpy.polynomial.hermite import hermval

    # Methfessel and Paxton smearing
    nh = 5
    coeff = np.zeros(2 * nh)
    coeff[0] = 1.0
    for n in range(2, 2 * nh, 2):
        m = n // 2
        coeff[n] = (-1.0) ** m / (factorial(m) * (4.0**m) * np.sqrt(np.pi))

    x = (ene - eig) / delta
    return hermval(x, coeff) * np.exp(-((x) ** 2)) / (delta * np.sqrt(np.pi))


def intgaussian(eig, ene, delta):
    """Evaluate the error-function approximation to the Fermi-Dirac step.

    Parameters
    ----------
    eig : float or np.ndarray
        Fermi energy or eigenvalue(s) (eV).
    ene : float or np.ndarray
        Energy point(s) (eV).
    delta : float or np.ndarray
        Smearing width parameter (eV).

    Returns
    -------
    float or np.ndarray
        Occupation approximated by

        .. math::

            f(\\varepsilon; E, \\delta) =
            \\frac{1 - {\\rm erf}\\bigl((\\varepsilon - E)/\\delta\\bigr)}{2}
    """
    from scipy.special import erf

    # integral of the gaussian function as approximation of the Fermi-Dirac distribution
    return (1.0 - erf((eig - ene) / delta)) / 2.0


def intmetpax(eig, ene, delta):
    """Evaluate the Methfessel-Paxton approximation to the Fermi-Dirac occupation.

    Parameters
    ----------
    eig : float or np.ndarray
        Eigenvalue(s) (eV).
    ene : float or np.ndarray
        Energy point(s) (eV).
    delta : float or np.ndarray
        Smearing width parameter (eV).

    Returns
    -------
    float or np.ndarray
        Methfessel-Paxton occupation function at order ``nh = 5``:

        .. math::

            f_N(x) = \\frac{1 - {\\rm erf}(x)}{2}
                + \\sum_{n=1}^{N} A_n H_{2n-1}(x) \\exp(-x^2)

        where :math:`x = (\\varepsilon - E)/\\delta`.  See Methfessel
        and Paxton, Phys. Rev. B **40**, 3616 (1989).
    """
    import numpy as np
    from math import factorial

    import numpy as np
    from numpy.polynomial.hermite import hermval
    from scipy.special import erf

    # Methfessel and Paxton correction to the Fermi-Dirac distribution
    nh = 5
    coeff = np.zeros(2 * nh)
    coeff[0] = 0.0
    for n in range(2, 2 * nh, 2):
        m = n // 2
        coeff[n - 1] = (-1.0) ** m / (factorial(m) * (4.0**m) * np.sqrt(np.pi))

    x = (eig - ene) / delta
    return (1.0 - erf(x)) / 2.0 + hermval(x, coeff) * np.exp(-(x**2)) / np.sqrt(np.pi)
