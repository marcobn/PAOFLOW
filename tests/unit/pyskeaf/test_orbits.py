from types import SimpleNamespace

import numpy as np

from PAOFLOW.pyskeaf.orbits import (
    average_orbits,
    ExtremalOrbit,
    find_extremal,
    match_chunks,
    SliceOrbit,
)


def _orbit(slice_index, avg, area=None):
    avg = np.array(avg, dtype=float)
    area = 1.0 + 0.1 * slice_index if area is None else area
    return SliceOrbit(
        slice_index=slice_index,
        contour_xy=np.zeros((4, 2), dtype=float),
        area=area,
        inside_area=1.0,
        frequency_kT=area,
        effective_mass=1.0,
        orbit_type=1,
        avg_xy_frac=avg,
        std_xy_frac=np.array([0.002, 0.002]),
        min_xy_frac=avg - 0.003,
        max_xy_frac=avg + 0.003,
        n_points=4,
        is_open=False,
        is_too_small=False,
    )


def test_match_chunks_relaxed_fallback_links_small_moving_pockets():
    chunks = match_chunks(
        [
            [_orbit(1, [0.50, 0.50])],
            [_orbit(2, [0.53, 0.51])],
            [_orbit(3, [0.56, 0.52])],
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].slice_indices == [1, 2, 3]


def test_match_chunks_links_tiny_pockets_across_short_slice_gaps():
    chunks = match_chunks(
        [
            [_orbit(1, [0.50, 0.50])],
            [],
            [],
            [_orbit(4, [0.505, 0.502])],
            [],
            [_orbit(6, [0.508, 0.503])],
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].slice_indices == [1, 4, 6]


def test_find_extremal_uses_gapped_three_point_neighbourhood():
    chunks = match_chunks(
        [
            [_orbit(1, [0.50, 0.50], area=3.0)],
            [],
            [],
            [_orbit(4, [0.505, 0.502], area=1.0)],
            [],
            [_orbit(6, [0.508, 0.503], area=3.0)],
        ]
    )
    geom = SimpleNamespace(
        numx=6,
        zkseparation=0.1,
        maxlreciplat=1.0,
        rotation=np.eye(3),
        plr_inverse=np.eye(3),
    )

    extrema = find_extremal(chunks, geom)

    assert len(extrema) == 1
    assert extrema[0].slice_orbit.slice_index == 4
    assert extrema[0].curvature_kT_A2 > 0.0


def test_average_orbits_unwraps_periodic_ruc_coordinates():
    extrema = [
        ExtremalOrbit(_orbit(1, [0.5, 0.5]), 0, 0, 1.0, np.array([0.99, 0.01, 0.99])),
        ExtremalOrbit(_orbit(2, [0.5, 0.5]), 0, 1, 1.0, np.array([0.01, 0.99, 0.01])),
    ]

    averaged = average_orbits(extrema, freq_same_frac=0.2, avg_same_frac=0.05)

    assert len(averaged) == 1
    assert np.all(np.abs(averaged[0].avg_xyz_ruc - np.round(averaged[0].avg_xyz_ruc)) < 0.02)
