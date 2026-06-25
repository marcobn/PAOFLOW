"""Tests for the reflectance-to-sRGB color derivation."""

import numpy as np

from PAOFLOW.graphics.color import (
    cie_cmf,
    reflectance_to_srgb,
    visible_grid_covered,
    xyz_to_srgb,
)


# Energy grid (eV) spanning the full visible range (~1.59-3.26 eV).
ENE = np.linspace(1.4, 3.4, 400)


def test_perfect_white_reflector():
    """R == 1 under equal-energy illuminant E maps to white (255, 255, 255)."""
    refl = np.ones_like(ENE)
    _, rgb255, hexstr = reflectance_to_srgb(ENE, refl, illuminant='E')
    np.testing.assert_array_equal(rgb255, [255, 255, 255])
    assert hexstr == '#ffffff'


def test_black_absorber():
    """R == 0 maps to black (0, 0, 0)."""
    refl = np.zeros_like(ENE)
    _, rgb255, hexstr = reflectance_to_srgb(ENE, refl, illuminant='E')
    np.testing.assert_array_equal(rgb255, [0, 0, 0])
    assert hexstr == '#000000'


def test_gray_is_neutral():
    """A flat gray reflectance gives equal R, G, B components."""
    refl = np.full_like(ENE, 0.5)
    rgb01, _, _ = reflectance_to_srgb(ENE, refl, illuminant='E')
    np.testing.assert_allclose(rgb01[0], rgb01[1], atol=1e-4)
    np.testing.assert_allclose(rgb01[1], rgb01[2], atol=1e-4)


def test_red_band_is_reddish():
    """Reflectance peaked at long wavelengths (low E) reads as red-dominant."""
    # Gaussian reflectance peak around 1.8 eV (~690 nm, red).
    refl = np.exp(-0.5 * ((ENE - 1.8) / 0.12) ** 2)
    rgb255, = (reflectance_to_srgb(ENE, refl, illuminant='E')[1],)
    assert rgb255[0] > rgb255[1]
    assert rgb255[0] > rgb255[2]


def test_blue_band_is_bluish():
    """Reflectance peaked at short wavelengths (high E) reads as blue-dominant."""
    refl = np.exp(-0.5 * ((ENE - 3.1) / 0.12) ** 2)
    rgb255 = reflectance_to_srgb(ENE, refl, illuminant='E')[1]
    assert rgb255[2] > rgb255[0]


def test_components_in_range():
    rng = np.random.default_rng(0)
    refl = np.clip(rng.uniform(0.0, 1.0, ENE.size), 0.0, 1.0)
    rgb01, rgb255, hexstr = reflectance_to_srgb(ENE, refl, illuminant='D65')
    assert np.all(rgb01 >= 0.0) and np.all(rgb01 <= 1.0)
    assert np.all(rgb255 >= 0) and np.all(rgb255 <= 255)
    assert hexstr.startswith('#') and len(hexstr) == 7


def test_cmf_nonnegative_peaks():
    """y-bar (luminous efficiency) peaks near 555 nm and is non-negative."""
    lam = np.arange(380.0, 781.0, 1.0)
    _, ybar, _ = cie_cmf(lam)
    assert np.all(ybar >= -1e-9)
    assert 540.0 <= lam[np.argmax(ybar)] <= 570.0


def test_xyz_to_srgb_white_point():
    """D65 white point (normalized) maps to near-white sRGB."""
    rgb = xyz_to_srgb((0.95047, 1.0, 1.08883))
    np.testing.assert_allclose(rgb, [1.0, 1.0, 1.0], atol=1e-2)


def test_visible_grid_coverage():
    assert visible_grid_covered(np.linspace(1.4, 3.4, 10))
    assert not visible_grid_covered(np.linspace(0.0, 2.0, 10))
