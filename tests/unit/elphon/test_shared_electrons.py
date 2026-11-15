"""Node-shared dense electron cache: identical to private arrays, on-the-fly vertex phases."""

import numpy as np
import pytest

mpi4py = pytest.importorskip('mpi4py')

from PAOFLOW.elphon.elph_bloch import (  # noqa: E402
    precompute_dense_electrons,
    release_dense_electrons,
    vertex_phases,
)


def _random_hamiltonian(nawf=4, grid=4, seed=0):
    rng = np.random.default_rng(seed)
    shape = (nawf, nawf, grid, grid, grid, 1)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.mark.parametrize('keep', [None, lambda energies: np.array([1, 2])])
def test_shared_cache_matches_private(keep):
    from mpi4py import MPI

    HRs = _random_hamiltonian()
    common = dict(keep_bands=keep, kblock=50)
    private = precompute_dense_electrons(HRs, np.eye(3), 6, [0.05], None, (4, 4, 4), **common)
    shared = precompute_dense_electrons(
        HRs, np.eye(3), 6, [0.05], None, (4, 4, 4), comm=MPI.COMM_WORLD, **common
    )
    np.testing.assert_array_equal(shared['E'], private['E'])
    np.testing.assert_array_equal(shared['V'], private['V'])
    np.testing.assert_array_equal(shared['bands'], private['bands'])
    assert not shared['V'].flags.writeable  # read-only view of the shared window
    ks = np.array([0, 5, 77, 200])
    table = np.exp(2j * np.pi * (private['K'] @ private['Nintg'].T))
    np.testing.assert_array_equal(vertex_phases(shared, ks), table[ks])
    release_dense_electrons(shared)
    assert 'V' not in shared and shared['windows'] == []
    release_dense_electrons(private)  # no window: a no-op
    assert 'V' in private


def test_vertex_phases_uses_explicit_table():
    electrons = {'phkg': np.arange(12).reshape(4, 3)}
    np.testing.assert_array_equal(vertex_phases(electrons, [1, 3]), [[3, 4, 5], [9, 10, 11]])
