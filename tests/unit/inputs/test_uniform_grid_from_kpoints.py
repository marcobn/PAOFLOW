"""Grid recovery for explicit K_POINTS lists (no monkhorst_pack element)."""

import numpy as np
import pytest

from PAOFLOW.inputs.read_QE_xml import uniform_grid_from_kpoints

# fcc reciprocal vectors (rows, 2 pi / alat)
B_FCC = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])


def _grid(n, offset=0.0, image='positive'):
    ax = [(np.arange(m) + offset) / m for m in n]
    k = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)  # third index fastest
    if image == 'centred':
        k = k - np.round(k)  # (-1/2, 1/2] like K_POINTS automatic
    return k @ B_FCC


@pytest.mark.parametrize('image', ['positive', 'centred'])
def test_gamma_centred_grid_recovered(image):
    assert uniform_grid_from_kpoints(_grid((6, 6, 6), image=image), B_FCC) == ((6, 6, 6), (0, 0, 0))


def test_anisotropic_and_shifted_grid():
    assert uniform_grid_from_kpoints(_grid((4, 3, 2), offset=0.5), B_FCC) == ((4, 3, 2), (1, 1, 1))


def test_incomplete_grid_rejected():
    with pytest.raises(ValueError, match='complete'):
        uniform_grid_from_kpoints(_grid((4, 4, 4))[:-1], B_FCC)


def test_wrong_order_rejected():
    k = _grid((3, 3, 3)).reshape(3, 3, 3, 3).transpose(2, 1, 0, 3).reshape(-1, 3)  # first fastest
    with pytest.raises(ValueError, match='order'):
        uniform_grid_from_kpoints(k, B_FCC)
