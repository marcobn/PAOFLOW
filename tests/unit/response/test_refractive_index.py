"""Tests for the refractive-index post-processing helper."""

import numpy as np
import pytest

from PAOFLOW.response.do_epsilon import refractive_index


def test_lossless_limit():
    """epsi -> 0, epsr > 0: n = sqrt(epsr), kappa = 0, R = ((n-1)/(n+1))**2."""
    ene = np.linspace(0.1, 5.0, 50)
    epsr = np.full_like(ene, 11.7)
    epsi = np.zeros_like(ene)

    n, kappa, alpha, refl = refractive_index(ene, epsi, epsr)

    np.testing.assert_allclose(n, np.sqrt(epsr), atol=1e-12)
    np.testing.assert_allclose(kappa, 0.0, atol=1e-12)
    np.testing.assert_allclose(alpha, 0.0, atol=1e-12)
    n0 = np.sqrt(11.7)
    np.testing.assert_allclose(refl, ((n0 - 1) / (n0 + 1)) ** 2, atol=1e-12)


def test_reciprocity():
    """tilde_n^2 = epsilon => n^2 - kappa^2 = epsr, 2 n kappa = epsi."""
    rng = np.random.default_rng(0)
    ene = np.linspace(0.1, 8.0, 30)
    epsr = rng.uniform(-5.0, 15.0, ene.size)
    epsi = rng.uniform(1e-3, 8.0, ene.size)

    n, kappa, _, _ = refractive_index(ene, epsi, epsr)

    np.testing.assert_allclose(n * n - kappa * kappa, epsr, atol=1e-10)
    np.testing.assert_allclose(2.0 * n * kappa, epsi, atol=1e-10)


def test_reflectivity_bounded():
    rng = np.random.default_rng(1)
    ene = np.linspace(0.1, 8.0, 20)
    epsr = rng.uniform(-2.0, 12.0, ene.size)
    epsi = rng.uniform(0.0, 5.0, ene.size)

    _, _, _, refl = refractive_index(ene, epsi, epsr)
    assert np.all(refl >= 0.0)
    assert np.all(refl <= 1.0)


def test_alpha_positive_units():
    """alpha = 2 omega kappa / c has units 1/m; positive for absorbing media."""
    ene = np.linspace(0.1, 5.0, 10)
    epsr = np.full_like(ene, 4.0)
    epsi = np.full_like(ene, 1.0)

    _, kappa, alpha, _ = refractive_index(ene, epsi, epsr)
    assert np.all(kappa > 0.0)
    assert np.all(alpha > 0.0)
    # alpha should grow with photon energy at fixed kappa.
    assert alpha[-1] > alpha[0]
