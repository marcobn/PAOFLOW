"""Plane-wave blocks and FFT-grid transforms of the native DFPT vertex."""

import numpy as np
import pytest

from PAOFLOW.elphon.plane_waves import (
    g_sphere,
    interpolate_to_grid,
    periodic_part_on_grid,
    project_on_plane_waves,
)

pytestmark = pytest.mark.unit


def test_periodic_part_round_trip():
    rng = np.random.default_rng(1)
    nr = (6, 5, 7)
    mill = np.array([[0, 1, -1, 2, 0], [0, 0, 1, -2, 2], [0, 1, 1, 0, -3]])
    c = rng.standard_normal((3, 5)) + 1j * rng.standard_normal((3, 5))
    u = periodic_part_on_grid(c, mill, nr)
    x = np.stack(np.meshgrid(*[np.arange(n) / n for n in nr], indexing='ij'), axis=-1)
    direct = np.einsum(
        'nG,G...->n...', c, np.exp(2j * np.pi * np.einsum('Gi,...i->G...', mill.T, x))
    )
    assert np.allclose(u, direct)
    assert np.allclose(project_on_plane_waves(u, mill, nr), c)
    # umklapp gather: the component at G'' - G0
    shifted = project_on_plane_waves(u, mill, nr, umklapp=(0, 1, 0))
    expect = np.zeros_like(c)
    for j in range(5):
        target = mill[:, j] - np.array([0, 1, 0])
        hit = [i for i in range(5) if np.array_equal(mill[:, i], target)]
        if hit:
            expect[:, j] = c[:, hit[0]]
    assert np.allclose(shifted, expect)


def test_interpolate_to_grid_keeps_low_components():
    rng = np.random.default_rng(2)
    nr_in, nr_out = (8, 8, 8), (6, 6, 6)
    mill = np.array([[0, 1, -1], [0, 0, 2], [0, 1, 0]])
    c = rng.standard_normal((1, 3)) + 1j * rng.standard_normal((1, 3))
    u = periodic_part_on_grid(c, mill, nr_in)
    v = interpolate_to_grid(u, nr_out)
    assert np.allclose(project_on_plane_waves(v, mill, nr_out), c)


def test_g_sphere_counts():
    mill, g = g_sphere((8, 8, 8), np.eye(3), 4.0)
    assert mill.shape[1] == g.shape[0]
    assert (np.einsum('gi,gi->g', g, g) <= 4.0 + 1e-9).all()
    # all 33 Miller vectors with |m|^2 <= 4 (counting (0,0,0), six of length 1, twelve of sqrt 2, eight of sqrt 3, six of 2)
    assert mill.shape[1] == 1 + 6 + 12 + 8 + 6
