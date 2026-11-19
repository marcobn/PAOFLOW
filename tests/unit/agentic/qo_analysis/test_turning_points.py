"""Extrema of the frequency along a chain of slices: a flat top is one extremum."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.agentic.qo_analysis.magnetic_breakdown import turning_points

pytestmark = pytest.mark.unit


def test_strict_extrema_are_the_product_rule():
    values = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 2.2, 2.4])
    old = [k for k in range(1, len(values) - 1)
           if (values[k] - values[k - 1]) * (values[k + 1] - values[k]) < 0]  # fmt: skip
    assert turning_points(values) == old == [2, 4]


def test_flat_top_and_bottom_count_once():
    assert turning_points(np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])) == [2]
    assert turning_points(np.array([5.0, 4.0, 4.0, 4.0, 5.0])) == [2]
    assert turning_points(np.array([1.0, 1.0, 1.0])) == []
    assert turning_points(np.array([1.0, 2.0])) == []


def test_monotonic_plateau_is_not_an_extremum():
    assert turning_points(np.array([1.0, 2.0, 2.0, 3.0])) == []
