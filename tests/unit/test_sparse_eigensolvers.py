import numpy as np
import pytest

from PAOFLOW.sparse.containers import SparseHamiltonian
from PAOFLOW.sparse.eigensolvers import _estimate_window_bands, solve_path


class _DC:
    def __init__(self, arrays, attributes):
        self._arrays = arrays
        self._attributes = attributes

    def data_dicts(self):
        return self._arrays, self._attributes


def _diagonal_sparse_hamiltonian(nawf):
    rows = np.arange(nawf, dtype=np.int32)
    cols = np.arange(nawf, dtype=np.int32)
    vals = np.arange(1, nawf + 1, dtype=float).astype(complex)
    ridx = np.zeros(nawf, dtype=np.int64)
    return SparseHamiltonian(
        nawf,
        1,
        np.zeros((1, 3), dtype=float),
        [rows],
        [cols],
        [vals],
        [ridx],
        1.0,
        0.0,
    )


def test_estimated_window_band_count_stays_selected_for_small_system():
    dc = _DC({}, {'nawf': 8, 'nelec': 8})

    assert _estimate_window_bands(dc, 2.2) == 5


def test_solve_path_rejects_near_full_sparse_arpack_request():
    dc = _DC({'sparse_H': _diagonal_sparse_hamiltonian(8)}, {})

    with pytest.raises(NotImplementedError, match='almost full-spectrum'):
        solve_path(dc, np.zeros((1, 3)), n_eigs=6)


def test_solve_path_selected_request_reports_progress():
    dc = _DC({'sparse_H': _diagonal_sparse_hamiltonian(8)}, {})
    progress = []

    eig = solve_path(
        dc,
        np.zeros((1, 3)),
        n_eigs=3,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert eig.solver == 'eigsh(SA)'
    assert eig.E_k.shape == (1, 3, 1)
    np.testing.assert_allclose(eig.E_k[0, :, 0], [1.0, 2.0, 3.0])
    assert progress == [(1, 1)]
