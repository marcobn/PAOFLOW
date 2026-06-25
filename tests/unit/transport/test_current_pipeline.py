import numpy as np
import pytest

from PAOFLOW.transport import current_pipeline


@pytest.mark.unit
def test_run_current_from_file_uses_observable_level_current(monkeypatch, tmp_path):
    transmission_file = tmp_path / 'conductance.dat'
    np.savetxt(transmission_file, np.array([[-1.0, 0.1], [0.0, 0.2], [1.0, 0.3]]))

    out_file = tmp_path / 'current.dat'
    captured = {}

    def fake_compute_current(*, energy_grid, transmission, bias_grid, mu_L, mu_R, sigma):
        captured['energy_grid'] = energy_grid.copy()
        captured['transmission'] = transmission.copy()
        captured['bias_grid'] = bias_grid.copy()
        captured['mu_L'] = mu_L
        captured['mu_R'] = mu_R
        captured['sigma'] = sigma
        return np.full_like(bias_grid, 1.23, dtype=float)

    monkeypatch.setattr(current_pipeline, 'compute_current', fake_compute_current)

    currents = current_pipeline.run_current_from_file(
        data={'fileout': str(out_file), 'mu_L': -0.5, 'mu_R': 0.5, 'sigma': 0.05},
        filein=str(transmission_file),
        bias_min=-1.0,
        bias_max=1.0,
        nbias=5,
    )

    np.testing.assert_allclose(currents, 1.23)
    np.testing.assert_allclose(captured['energy_grid'], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(captured['transmission'], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(captured['bias_grid'], np.linspace(-1.0, 1.0, 5))
    assert captured['mu_L'] == pytest.approx(-0.5)
    assert captured['mu_R'] == pytest.approx(0.5)
    assert captured['sigma'] == pytest.approx(0.05)

    out_data = np.loadtxt(out_file)
    np.testing.assert_allclose(out_data[:, 0], np.linspace(-1.0, 1.0, 5))
    np.testing.assert_allclose(out_data[:, 1], np.full(5, 1.23))
