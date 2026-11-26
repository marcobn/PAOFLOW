import numpy as np
import pytest

from PAOFLOW.transport.grid.rgrid import get_rgrid
from PAOFLOW.transport.io.write_data import populate_real_space_hamiltonian


class _Store:
    def __init__(self, arrays: dict, attr: dict) -> None:
        self.arrays, self.attr = arrays, attr

    def data_dicts(self) -> tuple[dict, dict]:
        return self.arrays, self.attr


@pytest.mark.unit
def test_hexagonal_cell_recovers_the_hoppings() -> None:
    """H(R) of a non-orthogonal cell: a_vectors are rows, so R = n @ a (wrong for ibrav 4 before)."""
    a_vectors = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.6]])
    alat = 7.46
    shape = (4, 4, 2)
    k_crystal = np.stack(
        np.meshgrid(*[np.arange(n) / n for n in shape], indexing='ij'), axis=-1
    ).reshape(-1, 3)
    hoppings = {(0, 0, 0): 0.3, (1, 0, 0): -1.0, (0, 1, 0): 0.7, (1, 1, 1): 0.2}
    Hk = sum(t * np.exp(2j * np.pi * k_crystal @ np.array(n)) for n, t in hoppings.items())
    ivr, _ = get_rgrid(shape)
    arrays = {
        'a_vectors': a_vectors,
        'vkpts_cartesian': (k_crystal @ np.linalg.inv(a_vectors).T).T * (2 * np.pi / alat),
        'wk': np.full(len(k_crystal), 1.0 / len(k_crystal)),
    }
    populate_real_space_hamiltonian(
        _Store(arrays, {'alat': alat}), {'Hk': Hk.reshape(1, -1, 1, 1), 'ivr': ivr}, False
    )
    for n, t in hoppings.items():
        index = np.flatnonzero(np.all(ivr == np.array(n), axis=1))[0]
        assert arrays['HRs'][index, 0, 0] == pytest.approx(t)
