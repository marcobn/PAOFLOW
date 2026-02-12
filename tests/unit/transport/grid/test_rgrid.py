"""Unit tests for real-space grid construction."""

import numpy as np
import pytest

from PAOFLOW.transport.grid.rgrid import get_rgrid


@pytest.mark.unit
def test_get_rgrid_weight_sum_rule():
    """Weights should sum to nr1 * nr2 * nr3 as enforced in the routine."""
    r_points, weights = get_rgrid((2, 1, 1))

    assert weights.sum() == pytest.approx(2.0)
    assert r_points.shape[1] == 3


@pytest.mark.unit
def test_get_rgrid_contains_negative_counterparts():
    """Every R should have a -R counterpart after completion."""
    r_points, _ = get_rgrid((2, 1, 1))
    assert any(np.all(r == [1, 0, 0]) for r in r_points)
    assert any(np.all(r == [-1, 0, 0]) for r in r_points)


@pytest.mark.unit
def test_get_rgrid_rejects_invalid_dims():
    """Mesh dimensions must be positive integers."""
    with pytest.raises(ValueError):
        get_rgrid((0, 2, 2))
