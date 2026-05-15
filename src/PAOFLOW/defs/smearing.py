def gaussian(eig, ene, delta):
    import numpy as np

    # Gaussian Smearing
    return (np.exp(-(((ene - eig) / delta) ** 2)) / delta) / np.sqrt(np.pi)


def metpax(eig, ene, delta):
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
    from scipy.special import erf

    # integral of the gaussian function as approximation of the Fermi-Dirac distribution
    return (1.0 - erf((eig - ene) / delta)) / 2.0


def intmetpax(eig, ene, delta):
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
