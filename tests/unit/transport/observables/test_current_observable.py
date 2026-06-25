import numpy as np
import pytest

from PAOFLOW.transport.observables import current as current_observable


@pytest.mark.unit
def test_compute_current_delegates_to_existing_calculator(monkeypatch):
    captured = {}

    def fake_compute_current_vs_bias(*, egrid, transm, vgrid, mu_L, mu_R, sigma):
        captured['egrid'] = egrid
        captured['transm'] = transm
        captured['vgrid'] = vgrid
        captured['mu_L'] = mu_L
        captured['mu_R'] = mu_R
        captured['sigma'] = sigma
        return np.array([0.0, 1.0, 2.0])

    monkeypatch.setattr(current_observable, 'compute_current_vs_bias', fake_compute_current_vs_bias)

    energy_grid = np.array([-1.0, 0.0, 1.0])
    transmission = np.array([0.2, 0.8, 0.4])
    bias_grid = np.array([-0.5, 0.0, 0.5])

    current = current_observable.compute_current(
        energy_grid=energy_grid,
        transmission=transmission,
        bias_grid=bias_grid,
        mu_L=-0.5,
        mu_R=0.5,
        sigma=0.05,
    )

    np.testing.assert_allclose(current, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(captured['egrid'], energy_grid)
    np.testing.assert_allclose(captured['transm'], transmission)
    np.testing.assert_allclose(captured['vgrid'], bias_grid)
    assert captured['mu_L'] == pytest.approx(-0.5)
    assert captured['mu_R'] == pytest.approx(0.5)
    assert captured['sigma'] == pytest.approx(0.05)
