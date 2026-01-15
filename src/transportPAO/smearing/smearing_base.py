import numpy as np


def smearing_func(x: float, smearing_type: str) -> float:
    """
    Evaluate a normalized smearing function F(x) such that (1/δ)·F(x) integrates to 1.

    Parameters
    ----------
    x : float
        Dimensionless energy shift (typically (E - E₀) / δ).
    smearing_type : str
        Type of smearing:
        'lorentzian', 'gaussian', 'fermi-dirac' (or 'fd'),
        'methfessel-paxton' (or 'mp'), or 'marzari-vanderbilt' (or 'mv').

    Returns
    -------
    float
        Value of the smearing function F(x).

    Raises
    ------
    ValueError
        If the `smearing_type` is not recognized.

    Notes
    -----
    Implemented smearing types:

    - Lorentzian:
        F(x) = 1 / π(1 + x²)

    - Gaussian:
        F(x) = 1 / √π · exp(−x²)

    - Fermi-Dirac:
        F(x) = 1/2 · 1 / (1 + cosh(x))

    - Methfessel-Paxton (N=1):
        F(x) = 1 / √π · exp(−x²) · (3/2 − x²)

    - Marzari-Vanderbilt:
        F(x) = 1 / √π · exp(−(x − 1/√2)²) · (2 − √2 x)
    """
    smearing_type = smearing_type.strip().lower()

    if smearing_type == "lorentzian":
        return 1.0 / (np.pi * (1.0 + x**2))

    if smearing_type == "gaussian":
        return np.exp(-(x**2)) / np.sqrt(np.pi)

    if smearing_type in {"fermi-dirac", "fd"}:
        return 0.5 / (1.0 + np.cosh(x))

    if smearing_type in {"methfessel-paxton", "mp"}:
        return np.exp(-(x**2)) / np.sqrt(np.pi) * (1.5 - x**2)

    if smearing_type in {"marzari-vanderbilt", "mv"}:
        return (
            np.exp(-((x - 1.0 / np.sqrt(2.0)) ** 2))
            / np.sqrt(np.pi)
            * (2.0 - np.sqrt(2.0) * x)
        )

    raise ValueError(f"Unsupported smearing_type: '{smearing_type}'")
