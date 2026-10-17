import math

import pandas as pd
import pytest

from PAOFLOW.agentic.qo_analysis.magnetic_breakdown import (
    group_network_copies,
    network_orbits_at_angle,
    single_angle_orbits,
)
from PAOFLOW.agentic.qo_analysis.skeaf_output import AREA_TO_KILOTESLA

pytestmark = pytest.mark.unit


def test_sphere_frequency_from_pyskeaf(two_spheres):
    folder, _ = two_spheres
    radius = 0.135
    orbits = single_angle_orbits(folder / 'Fermi_surf_band_1.bxsf', 0.0, 0.0, num_interpolation=40)
    expected = AREA_TO_KILOTESLA * math.pi * radius**2 * 1000
    assert orbits and orbits[0][0] == pytest.approx(expected, rel=0.03)


def test_touching_spheres_give_sum_network_orbit(two_spheres):
    folder, gap = two_spheres
    paths = {1: folder / 'Fermi_surf_band_1.bxsf', 2: folder / 'Fermi_surf_band_2.bxsf'}
    extrema, stats = network_orbits_at_angle(
        paths, 0.0, 0.0, num_interpolation=60, junction_cutoff=0.05
    )
    assert stats['junctions'] > 0
    network = group_network_copies(pd.DataFrame(extrema))
    single = AREA_TO_KILOTESLA * math.pi * 0.135**2 * 1000
    assert any(
        abs(f - 2 * single) / (2 * single) < 0.05 for f in network.freq_T
    )  # figure-of-eight / sum
