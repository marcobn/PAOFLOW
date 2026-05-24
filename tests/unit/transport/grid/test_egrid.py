"""Unit tests for energy grid initialization."""

import numpy as np
import pytest

from PAOFLOW.transport.grid.egrid import initialize_energy_grid


@pytest.mark.unit
def test_initialize_energy_grid_linear_spacing():
    """Energy grid should be linearly spaced between endpoints."""
    egrid = initialize_energy_grid(-1.0, 1.0, 5, carriers='electrons')

    np.testing.assert_allclose(egrid, [-1.0, -0.5, 0.0, 0.5, 1.0])


@pytest.mark.unit
def test_initialize_energy_grid_phonons_zero_shift():
    """Phonon grids avoid an exact zero at the first point."""
    egrid = initialize_energy_grid(0.0, 2.0, 5, carriers='phonons')

    assert egrid[0] == pytest.approx(egrid[1] / 100.0)


@pytest.mark.unit
def test_initialize_energy_grid_requires_two_points():
    """Grid creation requires at least two points."""
    with pytest.raises(ValueError):
        initialize_energy_grid(-1.0, 1.0, 1, carriers='electrons')
