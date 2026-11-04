"""Single-precision storage of the node-shared g(R_e, R_p) (``cell_dtype``)."""

import numpy as np
import pytest

from PAOFLOW.elphon.coupling_cache import load_dense_vertex, store_dense_vertex
from PAOFLOW.elphon.do_pao_eph_dense_q import CELL_DTYPES, _broadcast_cells
from PAOFLOW.elphon.workflows.system import ElphonSystem

pytestmark = pytest.mark.unit


def _table(dtype, seed=0):
    rng = np.random.default_rng(seed)
    shape = (2, 2, 3, 2, 1, 1, 2, 1, 1)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)


@pytest.mark.parametrize('dtype', CELL_DTYPES)
def test_cache_round_trip_keeps_the_storage_precision(tmp_path, dtype):
    table = _table(dtype)
    store_dense_vertex(str(tmp_path), 'key', table, {'source': 'test'})
    out = np.zeros_like(table)
    assert load_dense_vertex(str(tmp_path), 'key', out)
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(out, table)


def test_cache_refuses_a_different_precision(tmp_path):
    store_dense_vertex(str(tmp_path), 'key', _table('complex128'), {})
    with pytest.raises(ValueError, match='complex64'):
        load_dense_vertex(str(tmp_path), 'key', _table('complex64'))


def test_single_precision_table_is_a_faithful_copy():
    table = _table('complex128')
    single = table.astype('complex64')
    assert np.abs(single - table).max() < 1e-6 * np.abs(table).max()


def test_broadcast_cells_uses_the_window_precision():
    MPI = pytest.importorskip('mpi4py.MPI')
    table = _table('complex64')
    labels = np.array([[0, 0, 0], [1, 0, 0]])
    before = table.copy()
    _broadcast_cells(MPI.COMM_WORLD, table, labels, {(0, 0, 0): 0, (1, 0, 0): 0})
    np.testing.assert_array_equal(table, before)


def test_system_passes_the_precision_to_the_dense_drivers():
    assert ElphonSystem.__dataclass_fields__['vertex_dtype'].default == 'complex128'
    import inspect

    from PAOFLOW.elphon import carrier_scattering, phonon_assisted_absorption, self_energy
    from PAOFLOW.elphon.do_pao_eph_dense_q import eliashberg_dense_q, prepare_dense_vertex

    for fn in (
        prepare_dense_vertex,
        eliashberg_dense_q,
        carrier_scattering.carrier_scattering_dense_q,
        phonon_assisted_absorption.phonon_assisted_absorption_dense_q,
        self_energy.fan_migdal_self_energy,
    ):
        assert inspect.signature(fn).parameters['cell_dtype'].default == 'complex128'
