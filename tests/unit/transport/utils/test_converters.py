import numpy as np
import pytest

from PAOFLOW.transport.utils.converters import cartesian_to_crystal, crystal_to_cartesian


@pytest.mark.unit
def test_roundtrip_cartesian_crystal():
    basis = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.5, 1.5, 0.0],
            [0.0, 0.1, 1.2],
        ]
    )
    coords = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    crystal = cartesian_to_crystal(coords, basis)
    cart = crystal_to_cartesian(crystal, basis)

    np.testing.assert_allclose(cart, coords)


@pytest.mark.unit
def test_converters_reject_invalid_shape():
    basis = np.eye(3)
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError):
        cartesian_to_crystal(coords, basis)

    with pytest.raises(ValueError):
        crystal_to_cartesian(coords, basis)


@pytest.mark.unit
def test_cartesian_to_crystal_rejects_singular_basis():
    basis = np.zeros((3, 3))
    coords = np.array([[1.0], [2.0], [3.0]])

    with pytest.raises(ValueError):
        cartesian_to_crystal(coords, basis)
