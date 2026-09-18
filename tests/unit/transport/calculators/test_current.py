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
