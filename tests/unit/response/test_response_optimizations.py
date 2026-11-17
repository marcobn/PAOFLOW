"""Compare response buffers against band-expectation-value formulas."""

from typing import Any

import numpy as np
import pytest

from PAOFLOW.response.do_conductivity import do_conductivity
from PAOFLOW.response.do_rashba_edelstein import do_rashba_edelstein_intra
from PAOFLOW.utils.constants import BOHR_RADIUS_CM, ELECTRONVOLT_SI, HBAR

pytestmark = pytest.mark.unit


class _Controller:
    def __init__(self, adaptive: bool) -> None:
        rng = np.random.default_rng(1729)
        velocities = rng.normal(size=(3, 3, 2, 2, 1)) + 1j * rng.normal(size=(3, 3, 2, 2, 1))
        velocities = velocities + velocities.swapaxes(2, 3).conj()
        vectors = np.empty((3, 2, 2, 1), dtype=complex)
        for k in range(3):
            angle, phase = 0.2 * (k + 1), np.exp(0.3j * (k + 1))
            c, s = np.cos(angle), np.sin(angle)
            vectors[k, :, :, 0] = [[c, -s * phase.conjugate()], [s * phase, c]]
        self.data_arrays = dict(
            v_k=vectors,
            dHksp=velocities,
            pksp=velocities.copy(),
            E_k=np.array([[[-0.3 + 0.04 * k], [0.45 - 0.05 * k]] for k in range(3)]),
            deltakp=np.full((3, 2, 1), 0.13),
            degen=[[[], [], []]],
        )
        self.data_attributes = dict(
            nawf=2,
            nspin=1,
            nkpnts=3,
            shift=10.0,
            smearing='gauss' if adaptive else None,
            ree_reg=1e-30,
            ree_twoD=True,
            ree_lt=10.0,
            ree_st=2.0,
        )
        self.outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def data_dicts(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.data_arrays, self.data_attributes

    def write_file_row_col(self, name: str, energy: np.ndarray, values: np.ndarray) -> None:
        self.outputs[name] = (energy.copy(), values.copy())


def _expectations(controller: _Controller, operators: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.diag(vectors.conj().T @ operator @ vectors).real
            for vectors, operator in zip(controller.data_arrays['v_k'][:, :, :, 0], operators)
        ]
    )


def _weights(controller: _Controller, energy: float, broadening: float) -> np.ndarray:
    width = (
        controller.data_arrays['deltakp'][:, :, 0]
        if controller.data_attributes['smearing']
        else broadening
    )
    return np.exp(-(((energy - controller.data_arrays['E_k'][:, :, 0]) / width) ** 2)) / (
        width * np.sqrt(np.pi)
    )


@pytest.mark.parametrize('adaptive', [False, True])
def test_off_diagonal_conductivity_matches_band_sum(adaptive: bool) -> None:
    controller = _Controller(adaptive)
    do_conductivity(controller, -0.4, 0.5, 7, 0.1, 0, 1)
    energies, actual = controller.outputs['cond_xy_0.dat']
    velocity_x = _expectations(controller, controller.data_arrays['dHksp'][:, 0, :, :, 0])
    velocity_y = _expectations(controller, controller.data_arrays['dHksp'][:, 1, :, :, 0])
    expected = [
        np.sum(_weights(controller, energy, 0.1) * velocity_x * velocity_y) / 3
        for energy in energies
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize('adaptive', [False, True])
def test_projected_ree_preserves_normalization_and_component_filename(adaptive: bool) -> None:
    controller = _Controller(adaptive)
    spin = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]) / 2
    projector = np.diag([1.0, 0.0])
    energies = np.linspace(-0.4, 0.5, 7)
    do_rashba_edelstein_intra(controller, 'spin', energies, 0.1, 0, 1, spin, projector)
    assert set(controller.outputs) == {'spin_reeEf_yx.dat'}
    _, actual = controller.outputs['spin_reeEf_yx.dat']
    operator = (projector @ spin[1] + spin[1] @ projector) / 2
    spin_y = _expectations(controller, np.broadcast_to(operator, (3, 2, 2)))
    velocity_x = _expectations(controller, controller.data_arrays['dHksp'][:, 0, :, :, 0])
    expected = []
    for energy in energies:
        weights = _weights(controller, energy, 0.1)
        # Fixed-width REE omits 1/delta. Keep that scale because the current
        # regularizer prevents it from cancelling exactly in the ratio.
        if not adaptive:
            weights *= 0.1
        numerator = np.sum(weights * spin_y * velocity_x)
        current = np.sum(weights * velocity_x**2)
        expected.append(
            -HBAR * numerator / (current * ELECTRONVOLT_SI * BOHR_RADIUS_CM + 1e-30) * 5
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-20)
