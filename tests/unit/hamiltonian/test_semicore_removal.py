"""Regression tests for semicore identification and valence projection."""

from typing import Any

import numpy as np
import pytest
from scipy.linalg import expm

from PAOFLOW.hamiltonian.semicore_removal import (
    check_semicore_prerequisites,
    find_semicore_orbitals,
    project_valence_hamiltonian,
)

pytestmark = pytest.mark.unit


class _Controller:
    def __init__(self, arrays: dict[str, Any], attributes: dict[str, Any]) -> None:
        self.arrays, self.attributes = arrays, attributes

    def data_dicts(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.arrays, self.attributes


def _mixed_core_model() -> _Controller:
    generator = np.zeros((4, 4), dtype=complex)
    generator[0, 2], generator[2, 0] = 0.01j, 0.01j
    generator[1, 3], generator[3, 1] = 0.015, -0.015
    unitary = expm(generator)
    hamiltonian = np.zeros((4, 4, 5, 1, 1, 1), dtype=complex)
    for index in range(5):
        cosine = np.cos(2 * np.pi * index / 5)
        eigenvalues = np.array(
            [-30 + 0.1 * cosine, -29 + 0.2 * cosine, -0.2 * cosine, 1 + 0.3 * cosine]
        )
        hamiltonian[:, :, index, 0, 0, 0] = (unitary * eigenvalues) @ unitary.conj().T
    return _Controller(
        dict(
            HRs=np.fft.ifftn(hamiltonian, axes=(2, 3, 4)),
            Hks=hamiltonian.copy(),
            basis=[dict(atom='X', label='core' if index < 2 else 'valence') for index in range(4)],
        ),
        dict(nawf=4, nspin=1, dftSO=True),
    )


def test_identifies_complete_semicore_shell() -> None:
    controller = _mixed_core_model()
    is_semicore = find_semicore_orbitals(controller, emin=-20.0, weight_threshold=0.5)
    np.testing.assert_array_equal(is_semicore, [True, True, False, False])


def test_projection_preserves_mixed_valence_bands() -> None:
    controller = _mixed_core_model()
    original = controller.arrays['Hks'].copy()
    is_semicore = find_semicore_orbitals(controller, emin=-20.0, weight_threshold=0.5)
    minimum_overlap = project_valence_hamiltonian(controller, is_semicore, emin=-20.0)
    projected = np.fft.fftn(controller.arrays['HRs'], axes=(2, 3, 4))
    assert minimum_overlap > 0.99
    assert projected.shape == (2, 2, 5, 1, 1, 1)
    for index in range(5):
        expected = np.linalg.eigvalsh(original[:, :, index, 0, 0, 0])[2:]
        actual = np.linalg.eigvalsh(projected[:, :, index, 0, 0, 0])
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)


def test_rejects_shell_split_by_energy_cutoff() -> None:
    controller = _mixed_core_model()
    for orbital in controller.arrays['basis']:
        orbital['label'] = 'shared_shell'
    with pytest.raises(RuntimeError, match='partially below'):
        find_semicore_orbitals(controller, emin=-20.0, weight_threshold=0.5)


def test_semicore_removal_requires_real_space_hamiltonian() -> None:
    with pytest.raises(RuntimeError, match='call pao_hamiltonian'):
        check_semicore_prerequisites(_Controller({}, {}))
