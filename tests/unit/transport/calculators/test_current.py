"""Unit tests for transport current utilities."""

import numpy as np
import pytest

from PAOFLOW.transport.calculators.current import (
    build_bias_grid,
    compute_current_vs_bias,
    fermi_dirac,
)


@pytest.mark.unit
def test_build_bias_grid_linspace():
    """Bias grid should be a linear space with requested endpoints."""
    grid = build_bias_grid(-1.0, 1.0, 5)

    np.testing.assert_allclose(grid, [-1.0, -0.5, 0.0, 0.5, 1.0])


@pytest.mark.unit
def test_fermi_dirac_midpoint():
    """At E==mu, the Fermi-Dirac occupation is exactly 0.5."""
    values = fermi_dirac(np.array([1.0]), mu=1.0, sigma=0.2)

    np.testing.assert_allclose(values, [0.5])


@pytest.mark.unit
def test_compute_current_zero_when_potentials_equal():
    """When mu_L == mu_R, the integrand vanishes and current is zero."""
    egrid = np.linspace(-1.0, 1.0, 9)
    transm = np.ones_like(egrid)
    vgrid = np.array([0.0, 0.2, 0.5])

    currents = compute_current_vs_bias(
        egrid=egrid,
        transm=transm,
        vgrid=vgrid,
        mu_L=0.5,
        mu_R=0.5,
        sigma=0.1,
    )

    np.testing.assert_allclose(currents, 0.0, atol=1e-12)


@pytest.mark.unit
def test_fermi_dirac_is_the_occupation():
    """f is 1 well below mu and 0 well above it, without overflow far from mu."""
    energies = np.array([-1.0, 1.0, -50.0, 50.0])
    with np.errstate(over='raise'):
        values = fermi_dirac(energies, mu=0.0, sigma=0.01)

    np.testing.assert_allclose(values, [1.0, 0.0, 1.0, 0.0], atol=1e-40)


@pytest.mark.unit
def test_compute_current_sign_follows_the_landauer_formula():
    """mu_L > mu_R gives a positive current: I = T (mu_L - mu_R) for a constant T."""
    egrid = np.linspace(-1.0, 1.0, 201)
    transm = np.full_like(egrid, 0.5)
    vgrid = np.array([-0.2, 0.2])

    currents = compute_current_vs_bias(egrid, transm, vgrid, mu_L=0.5, mu_R=-0.5, sigma=0.02)

    np.testing.assert_allclose(currents, 0.5 * vgrid, rtol=1e-4)
