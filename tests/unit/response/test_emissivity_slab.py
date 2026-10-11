"""Slab emissivity, Planck coverage and Kramers-Kronig helpers of response/do_epsilon."""

import numpy as np
import pytest

from PAOFLOW.response.do_epsilon import (
    directional_reflectivity,
    fresnel_reflectivities,
    kramers_kronig_eps1,
    planck_coverage,
    refractive_index,
    slab_directional_absorptance,
    spectral_hemispherical_emissivity,
)

ENE = np.linspace(0.05, 3.0, 60)
EPSR = 11.7 + 0.5 * ENE
EPSI = np.geomspace(1.0e-5, 2.0, ENE.size)


def test_directional_reflectivity_is_the_s_p_average():
    for theta in (0.0, 0.4, 1.2):
        r_s, r_p = fresnel_reflectivities(EPSR, EPSI, theta)
        np.testing.assert_array_equal(
            directional_reflectivity(EPSR, EPSI, theta), 0.5 * (r_s + r_p)
        )


@pytest.mark.parametrize('theta', [0.0, 0.7, 1.4])
def test_slab_absorptance_limits(theta):
    opaque = 1.0 - directional_reflectivity(EPSR, EPSI, theta)
    np.testing.assert_allclose(
        slab_directional_absorptance(ENE, EPSR, EPSI, theta, np.inf), opaque, rtol=1e-14
    )
    np.testing.assert_allclose(
        slab_directional_absorptance(ENE, EPSR, EPSI, theta, 1.0e4), opaque, rtol=1e-12
    )  # alpha d > 1e3 everywhere: opaque
    thin = slab_directional_absorptance(ENE, EPSR, EPSI, theta, 1.0e-12)
    assert np.all(np.abs(thin) < 1e-5)
    mid = slab_directional_absorptance(ENE, EPSR, EPSI, theta, 1.0e-5)
    assert np.all((mid >= 0.0) & (mid <= opaque + 1e-15))


def test_slab_absorptance_normal_incidence_formula():
    from PAOFLOW.utils.constants import HBAR, SPEED_OF_LIGHT

    d = 2.0e-5
    n, _, _, refl = refractive_index(ENE, EPSI, EPSR)
    kappa = EPSI / (2.0 * n)  # exact; avoids the |eps| - eps1 cancellation for tiny eps2
    alpha = 2.0 * (ENE / HBAR) * kappa / SPEED_OF_LIGHT
    tau = np.exp(-alpha * d)
    expected = (1.0 - refl) * (1.0 - tau) / (1.0 - refl * tau)
    np.testing.assert_allclose(
        slab_directional_absorptance(ENE, EPSR, EPSI, 0.0, d), expected, rtol=1e-10
    )


def test_hemispherical_slab_tends_to_opaque():
    opaque = spectral_hemispherical_emissivity(EPSR, EPSI, 64)
    thick = spectral_hemispherical_emissivity(EPSR, EPSI, 64, ENE, np.inf)
    np.testing.assert_allclose(thick, opaque, rtol=1e-12)
    thin = spectral_hemispherical_emissivity(EPSR, EPSI, 64, ENE, 1.0e-6)
    assert np.all(thin <= opaque + 1e-12)
    with pytest.raises(ValueError):
        spectral_hemispherical_emissivity(EPSR, EPSI, 64, thickness_m=1.0e-3)


def test_kramers_kronig_lorentz_oscillator():
    ene = np.arange(0.0, 200.0 + 1e-9, 0.01)
    e0, gamma, strength = 3.0, 0.3, 20.0
    denominator = (e0**2 - ene**2) ** 2 + (gamma * ene) ** 2
    eps2 = strength * gamma * ene / denominator
    eps1 = 1.0 + strength * (e0**2 - ene**2) / denominator
    got = kramers_kronig_eps1(ene, eps2)
    np.testing.assert_allclose(got[ene < 10.0], eps1[ene < 10.0], atol=1e-5)
    stacked = kramers_kronig_eps1(ene, np.stack([eps2, 2.0 * eps2]))
    np.testing.assert_allclose(stacked[1] - 1.0, 2.0 * (got - 1.0), atol=1e-12)
    with pytest.raises(ValueError):
        kramers_kronig_eps1(ene[1:], eps2[1:])


def test_planck_coverage():
    kt = 1.38066e-23 / 1.6021892e-19 * 500.0
    assert planck_coverage(np.array([1.0e-5, 60.0 * kt]), 500.0) == pytest.approx(1.0, abs=1e-6)
    low = planck_coverage(np.array([1.0e-5, 2.0 * kt]), 500.0)
    high = planck_coverage(np.array([2.0 * kt, 60.0 * kt]), 500.0)
    assert low + high == pytest.approx(1.0, abs=1e-5)
    # 15/pi^4 * integral_0^2 x^3 / (e^x - 1) dx
    assert low == pytest.approx(0.181144683, abs=1e-6)


def test_slab_absorptance_finite_at_grazing_emission():
    transparent = np.zeros_like(EPSI)  # R = 1 and tau = 1 at theta = pi/2: A -> 0, not 0/0
    grazing = slab_directional_absorptance(ENE, EPSR, transparent, 0.5 * np.pi, 1.0e-3)
    np.testing.assert_array_equal(grazing, 0.0)
    hemi = spectral_hemispherical_emissivity(EPSR, transparent + 1e-9, 32, ENE, 1.0e-3)
    assert np.all(np.isfinite(hemi))
