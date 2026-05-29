"""Unit tests for numerical smearing grid initialization."""

import numpy as np
import pytest

from PAOFLOW.transport.smearing.smearing_T import SmearingData, initialize_smearing_grid


@pytest.mark.unit
def test_initialize_smearing_grid_shapes():
    """Smearing grid should return matching xgrid and g_smear lengths."""
    xgrid, g_smear = initialize_smearing_grid(
        smearing_type='lorentzian',
        delta=0.5,
        delta_ratio=0.5,
        xmax=2.0,
        smearing_func=lambda x, _: np.exp(-(x**2)),
    )

    assert xgrid.shape == g_smear.shape
    assert xgrid.size > 0


@pytest.mark.unit
def test_smearing_data_memory_usage_nonzero():
    """Memory usage should be positive once arrays are initialized."""
    data = SmearingData(lambda x, _: np.exp(-(x**2)))
    data.initialize('lorentzian', delta=0.5, delta_ratio=0.5, xmax=2.0)

    assert data.memory_usage() > 0.0
