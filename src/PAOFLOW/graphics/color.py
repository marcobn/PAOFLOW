"""Derive the perceived visible color of a material from its optical spectra.

This module turns a normal-incidence reflectivity spectrum ``R(E)`` (as written
by :func:`PAOFLOW.response.do_epsilon.refractive_index` into ``refl_*.dat``
files) into an sRGB color, following the standard colorimetric pipeline:

    R(lambda) x illuminant -> CIE XYZ tristimulus -> linear sRGB -> gamma sRGB.

The CIE 1931 2-deg color-matching functions are evaluated with the compact
multi-lobe Gaussian analytic fit of Wyman, Sloan & Shirley, *Simple Analytic
Approximations to the CIE XYZ Color Matching Functions* (JCGT, 2013), so no
tabulated data files are required.
"""

import numpy as np

# Visible sampling grid (nm). 5 nm steps over 380-780 nm is ample for color.
_LAMBDA_MIN = 380.0
_LAMBDA_MAX = 780.0
_LAMBDA_STEP = 5.0

# eV * nm:  lambda(nm) = _EV_NM / E(eV)
_EV_NM = 1239.84198

# Linear-sRGB (D65) from CIE XYZ. IEC 61966-2-1.
_XYZ_TO_SRGB = np.array(
    [
        [3.2406255, -1.5372080, -0.4986286],
        [-0.9689307, 1.8757561, 0.0415175],
        [0.0557101, -0.2040211, 1.0569959],
    ]
)

# CIE D65 reference white tristimulus (Y normalized to 1) for the sRGB matrix.
_D65_WHITE = np.array([0.95047, 1.0, 1.08883])


def _gaussian_lobe(x, mu, sigma_lo, sigma_hi):
    """Piecewise (asymmetric) Gaussian used by the Wyman et al. CMF fit."""
    sigma = np.where(x < mu, sigma_lo, sigma_hi)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def cie_cmf(lam):
    """CIE 1931 2-deg color-matching functions x-bar, y-bar, z-bar.

    Multi-lobe Gaussian analytic approximation (Wyman, Sloan & Shirley, 2013).

    Parameters
    ----------
    lam : ndarray
        Wavelength grid in nanometers.

    Returns
    -------
    xbar, ybar, zbar : ndarray
        Color-matching function values on ``lam``.
    """
    lam = np.asarray(lam, dtype=float)
    xbar = (
        1.056 * _gaussian_lobe(lam, 599.8, 37.9, 31.0)
        + 0.362 * _gaussian_lobe(lam, 442.0, 16.0, 26.7)
        - 0.065 * _gaussian_lobe(lam, 501.1, 20.4, 26.2)
    )
    ybar = 0.821 * _gaussian_lobe(lam, 568.8, 46.9, 40.5) + 0.286 * _gaussian_lobe(
        lam, 530.9, 16.3, 31.1
    )
    zbar = 1.217 * _gaussian_lobe(lam, 437.0, 11.8, 36.0) + 0.681 * _gaussian_lobe(
        lam, 459.0, 26.0, 13.8
    )
    return xbar, ybar, zbar


def _planck_spd(lam, temperature):
    """Relative blackbody spectral power distribution vs wavelength (nm).

    Only the spectral shape matters here (overall scale cancels in the XYZ
    normalization), so physical constants are folded into a single factor.
    """
    # h c / k_B in nm*K.
    hc_over_k = 1.4387768775e7  # nm K
    lam = np.asarray(lam, dtype=float)
    return lam**-5 / np.expm1(hc_over_k / (lam * temperature))


def illuminant_spd(lam, illuminant='E'):
    """Relative spectral power distribution of the chosen illuminant.

    Parameters
    ----------
    lam : ndarray
        Wavelength grid in nanometers.
    illuminant : {'E', 'D65'} or float
        ``'E'`` equal-energy (flat) illuminant -- the material's *intrinsic*
        color. ``'D65'`` a 6504 K Planckian daylight approximation. A float is
        treated as a blackbody temperature in kelvin.

    Returns
    -------
    ndarray
        Relative SPD on ``lam`` (arbitrary overall scale).
    """
    lam = np.asarray(lam, dtype=float)
    if isinstance(illuminant, str):
        key = illuminant.upper()
        if key == 'E':
            return np.ones_like(lam)
        if key == 'D65':
            return _planck_spd(lam, 6504.0)
        raise ValueError("illuminant must be 'E', 'D65' or a temperature (K)")
    return _planck_spd(lam, float(illuminant))


def _linear_to_srgb(c):
    """Apply the sRGB gamma transfer to linear-light components in [0, 1]."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def xyz_to_srgb(xyz):
    """Convert a CIE XYZ tristimulus triple to gamma-encoded sRGB in [0, 1].

    Out-of-gamut negative components are clipped to zero before encoding.
    """
    rgb_lin = _XYZ_TO_SRGB @ np.asarray(xyz, dtype=float)
    rgb_lin = np.clip(rgb_lin, 0.0, None)
    peak = rgb_lin.max()
    if peak > 1.0:
        rgb_lin = rgb_lin / peak
    return _linear_to_srgb(rgb_lin)


def reflectance_to_srgb(ene, refl, illuminant='E'):
    """Perceived sRGB color of an opaque material from its reflectivity.

    Parameters
    ----------
    ene : array_like, shape (n,)
        Photon-energy grid in eV (monotonically increasing).
    refl : array_like, shape (n,)
        Normal-incidence reflectivity on ``ene`` (values in [0, 1]).
    illuminant : {'E', 'D65'} or float, optional
        Illuminant under which the color is observed (see
        :func:`illuminant_spd`). Default equal-energy ``'E'``.

    Returns
    -------
    rgb01 : ndarray, shape (3,)
        Gamma-encoded sRGB components in [0, 1].
    rgb255 : ndarray, shape (3,)
        8-bit integer sRGB components in [0, 255].
    hexstr : str
        Hex color string, e.g. ``'#rrggbb'``.

    Notes
    -----
    The energy grid should span the visible range (about 1.59-3.26 eV, i.e.
    380-780 nm). Reflectivity outside the supplied grid is held at the edge
    value; a truncated grid biases the color.
    """
    ene = np.asarray(ene, dtype=float)
    refl = np.clip(np.asarray(refl, dtype=float), 0.0, 1.0)

    lam = np.arange(_LAMBDA_MIN, _LAMBDA_MAX + 0.5 * _LAMBDA_STEP, _LAMBDA_STEP)
    # Energies (eV) corresponding to the wavelength grid, ascending for interp.
    ene_of_lam = _EV_NM / lam
    refl_lam = np.interp(ene_of_lam, ene, refl)

    spd = illuminant_spd(lam, illuminant)
    xbar, ybar, zbar = cie_cmf(lam)

    norm = np.trapezoid(spd * ybar, lam)
    if norm <= 0.0:
        norm = 1.0

    # White point of the illuminant (perfect reflector, R == 1). White-balancing
    # to this point (von Kries) guarantees a flat/white reflector maps to neutral
    # white and removes the small bias of the analytic CMF fit.
    Xw = np.trapezoid(spd * xbar, lam) / norm
    Yw = np.trapezoid(spd * ybar, lam) / norm
    Zw = np.trapezoid(spd * zbar, lam) / norm
    white = np.array([Xw, Yw, Zw])
    white[white <= 0.0] = 1.0

    weight = spd * refl_lam
    X = np.trapezoid(weight * xbar, lam) / norm
    Y = np.trapezoid(weight * ybar, lam) / norm
    Z = np.trapezoid(weight * zbar, lam) / norm

    # Adapt from the illuminant white to the sRGB D65 reference white.
    xyz = np.array([X, Y, Z]) / white * _D65_WHITE

    rgb01 = xyz_to_srgb(xyz)
    rgb255 = np.clip(np.rint(rgb01 * 255.0), 0, 255).astype(int)
    hexstr = '#{:02x}{:02x}{:02x}'.format(*rgb255)
    return rgb01, rgb255, hexstr


def visible_grid_covered(ene):
    """Return True if the energy grid spans the full visible range."""
    ene = np.asarray(ene, dtype=float)
    return ene.min() <= _EV_NM / _LAMBDA_MAX and ene.max() >= _EV_NM / _LAMBDA_MIN
