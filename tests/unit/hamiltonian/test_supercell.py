"""Regression tests for real-space supercells."""

from typing import Any

import numpy as np
import pytest

from PAOFLOW.hamiltonian.supercell import (
    build_supercell_real_space_hamiltonian,
    check_supercell_prerequisites,
    lattice_vector_grid_map,
    update_supercell_metadata,
)

pytestmark = pytest.mark.unit


class _Controller:
    def __init__(self, arrays: dict[str, Any], attributes: dict[str, Any]) -> None:
        self.arrays, self.attributes = arrays, attributes

    def data_dicts(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.arrays, self.attributes


def _primitive_hamiltonian() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    onsite = np.array([[0.2, 0.04j], [-0.04j, -0.1]])
    hopping = np.array([[-0.3, 0.07 + 0.02j], [-0.01j, -0.2]])
    hamiltonian = np.zeros((2, 2, 5, 1, 1, 1), dtype=complex)
    hamiltonian[:, :, 0, 0, 0, 0] = onsite
    hamiltonian[:, :, 4, 0, 0, 0] = hopping
    hamiltonian[:, :, 1, 0, 0, 0] = hopping.conj().T
    return hamiltonian, onsite, hopping


def _bloch(hamiltonian: np.ndarray, wavevector: float) -> np.ndarray:
    result = np.zeros(hamiltonian.shape[:2], dtype=complex)
    for vector, (index, weight) in lattice_vector_grid_map(hamiltonian.shape[2]).items():
        result += (
            weight * hamiltonian[:, :, index, 0, 0, 0] * np.exp(2j * np.pi * wavevector * vector)
        )
    return result


@pytest.mark.parametrize('wavevector', [0.0, 0.13, 0.37])
def test_supercell_preserves_folded_bands(wavevector: float) -> None:
    primitive, _, _ = _primitive_hamiltonian()
    supercell = build_supercell_real_space_hamiltonian(primitive, (3, 1, 1))
    bloch_supercell = _bloch(supercell, wavevector)
    expected = np.sort(
        np.concatenate(
            [np.linalg.eigvalsh(_bloch(primitive, (wavevector + shift) / 3)) for shift in range(3)]
        )
    )
    np.testing.assert_allclose(bloch_supercell, bloch_supercell.conj().T, atol=1e-14, rtol=0)
    np.testing.assert_allclose(np.linalg.eigvalsh(bloch_supercell), expected, atol=1e-13, rtol=0)


def test_finite_supercell_has_open_boundaries() -> None:
    primitive, onsite, hopping = _primitive_hamiltonian()
    supercell = build_supercell_real_space_hamiltonian(
        primitive, (3, 1, 1), finite_directions=(True, False, False)
    )
    expected = (
        np.kron(np.eye(3), onsite)
        + np.kron(np.diag(np.ones(2), 1), hopping)
        + np.kron(np.diag(np.ones(2), -1), hopping.conj().T)
    )
    assert supercell.shape == (6, 6, 1, 1, 1, 1)
    np.testing.assert_allclose(supercell[:, :, 0, 0, 0, 0], expected, atol=1e-14, rtol=0)


def test_metadata_scales_cell_and_removes_stale_distances() -> None:
    controller = _Controller(
        dict(
            tau=np.zeros((1, 3)),
            a_vectors=np.eye(3),
            b_vectors=np.eye(3),
            atoms=['X'],
            naw=np.array([2]),
            Dnm=np.zeros((2, 2, 3)),
        ),
        dict(alat=1.0, nawf=2, natoms=1, nelec=2, nbnds=2, bnd=2),
    )
    update_supercell_metadata(controller, (2, 3, 1), build_distance_matrix=False)
    np.testing.assert_allclose(controller.arrays['a_vectors'], np.diag([2.0, 3.0, 1.0]))
    np.testing.assert_allclose(
        controller.arrays['a_vectors'] @ controller.arrays['b_vectors'].T,
        np.eye(3),
        atol=1e-14,
        rtol=0,
    )
    assert controller.attributes['nawf'] == 12
    assert controller.attributes['natoms'] == 6
    assert controller.arrays['tau'].shape == (6, 3)
    assert len(controller.arrays['atoms']) == 6
    assert 'Dnm' not in controller.arrays


def test_supercell_requires_real_space_hamiltonian() -> None:
    with pytest.raises(RuntimeError, match='call pao_hamiltonian'):
        check_supercell_prerequisites(_Controller({}, {}))
