"""Disk cache of g(R_e, R_p)."""

import numpy as np
import pytest

from PAOFLOW.elphon.coupling_cache import (
    dense_vertex_cache_key,
    load_dense_vertex,
    store_dense_vertex,
)

pytestmark = pytest.mark.unit


def test_cache_round_trip_and_key_stability(tmp_path):
    coupling = tmp_path / 'ph'
    coupling.mkdir()
    (coupling / 'x.dvscf_q1').write_bytes(b'0' * 16)
    key = dense_vertex_cache_key(
        str(coupling), 'dfpt', (2, 2, 2), (4, 4, 4), [1.0], (6, 4, 64), 'basis'
    )
    assert key == dense_vertex_cache_key(
        str(coupling), 'dfpt', (2, 2, 2), (4, 4, 4), [1.0], (6, 4, 64), 'basis'
    )
    assert key != dense_vertex_cache_key(
        str(coupling), 'dfpt', (2, 2, 2), (4, 4, 4), [1.0], (6, 4, 64), 'other'
    )
    (coupling / 'x.dvscf_q1').write_bytes(b'0' * 32)
    assert key != dense_vertex_cache_key(
        str(coupling), 'dfpt', (2, 2, 2), (4, 4, 4), [1.0], (6, 4, 64), 'basis'
    )
    rng = np.random.default_rng(0)
    g = rng.standard_normal((2, 2, 3, 2, 2, 2, 2, 2, 2)) + 0j
    cache = str(tmp_path / 'cache')
    out = np.zeros_like(g)
    assert not load_dense_vertex(cache, key, out)
    store_dense_vertex(cache, key, g, {'note': 'test'})
    assert load_dense_vertex(cache, key, out) and np.array_equal(out, g)
    with pytest.raises(ValueError):
        load_dense_vertex(cache, key, np.zeros((2, 2, 3, 2, 2, 2, 2, 2, 1), complex))
