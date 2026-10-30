"""Baseline fits and the Boltzmann channel bridge."""

import numpy as np
import pytest

from PAOFLOW.agentic.transport import baselines as bl
from PAOFLOW.agentic.transport.channels import (
    carrier_channels,
    impurity_channel,
    impurity_rate_from_mobility,
    metal_channels,
)


def test_bloch_gruneisen_limits_and_fit():
    temps = np.array([10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 500.0])
    theta = 105.0
    shape = bl.bloch_gruneisen(temps, theta)
    # linear above theta: shape -> T / (4 theta) for n = 5 (J_5(y) -> y^4 / 4)
    assert shape[-1] / (500.0 / (4 * theta)) == pytest.approx(1.0, rel=0.03)
    # T^5 far below theta: J_5(inf) = 5! zeta(5) = 124.43 (at T = 10 K the integral is 95 % of it)
    assert bl.bloch_gruneisen([2.0], theta)[0] / ((2.0 / theta) ** 5 * 124.431) == pytest.approx(
        1.0, rel=1e-3
    )
    assert shape[0] / ((10.0 / theta) ** 5 * 124.431) == pytest.approx(0.951, rel=1e-2)
    rho = 0.3 + 60.0 * shape
    fit = bl.fit_bloch_gruneisen(temps, rho)
    assert fit['theta_k'] == pytest.approx(theta, rel=1e-3)
    assert fit['rho0'] == pytest.approx(0.3, abs=1e-6) and fit['amplitude'] == pytest.approx(
        60.0, rel=1e-4
    )
    fixed = bl.fit_bloch_gruneisen(temps, rho, theta_k=theta, rho0=0.0)
    assert fixed['rho0'] == 0.0 and fixed['rms'] > 0.0


def test_power_law_linear_and_matthiessen_offset():
    x = np.array([100.0, 200.0, 300.0, 400.0])
    fit = bl.fit_power_law(x, 1300.0 * (x / 300.0) ** -2.4)
    assert fit['exponent'] == pytest.approx(-2.4) and fit['rms_log10'] < 1e-12
    line = bl.fit_linear(x, 1.0 + 0.05 * x, window=(150.0, 450.0))
    assert line['slope'] == pytest.approx(0.05) and line['n'] == 3
    model_t = np.array([50.0, 100.0, 200.0, 300.0])
    model_rho = 0.05 * model_t
    exp_t = np.array([80.0, 150.0, 250.0, 400.0])
    out = bl.matthiessen_offset(exp_t, 0.05 * exp_t + 0.7, model_t, model_rho)
    assert out['n'] == 3 and out['rho0'] == pytest.approx(0.7)
    assert out['rms_after'] < 1e-12 and out['rms_before'] == pytest.approx(0.7)
    assert out['slope_ratio'] == pytest.approx(1.0)
    assert np.allclose(bl.matthiessen_mobility(2000.0, 2000.0), 1000.0)


def test_metal_channels_from_eliashberg(tmp_path):
    omega = np.linspace(0.001, 0.02, 40)
    path = tmp_path / 'eliashberg.npz'
    np.savez(path, omega=omega, a2F_tr=0.5 * np.ones(40), omega_plasma_ev=8.0)
    channels = metal_channels(path, rho0_uohm_cm=0.5, b_ee_per_ev=1e-3)
    assert [c.name for c in channels] == ['e-ph (a2F_tr)', 'residual', 'electron-electron']
    assert channels[0].lambda_tr == pytest.approx(2 * np.trapezoid(0.5 / omega, omega))
    assert metal_channels(path) == channels[:1] or len(metal_channels(path)) == 1
    np.savez(tmp_path / 'bare.npz', omega=omega, a2F=omega)
    with pytest.raises(FileNotFoundError):
        metal_channels(tmp_path / 'bare.npz')


def test_carrier_channels_and_impurity(tmp_path):
    energy = np.linspace(-1.0, 1.0, 21)
    path = tmp_path / 'scattering_rates.npz'
    np.savez(path, energy_ev=energy, temps_k=np.array([300.0]), rate_serta=np.ones((1, 21)) * 1e13,
             rate_mrta=np.ones((1, 21)) * 2e13, vbm_ev=-0.3, cbm_ev=0.3, metal=False)  # fmt: skip
    rate = impurity_rate_from_mobility(1000.0, 0.26)
    assert rate == pytest.approx(1.602176634e-19 / (0.26 * 9.1093837015e-31 * 0.1), rel=1e-6)
    extra = impurity_channel(1000.0, 0.26)
    channels = carrier_channels(path, kind='mrta', band_edges=(-0.5, 0.6), extra=[extra])
    assert (
        len(channels) == 2 and channels[0].name == 'e-ph (MRTA)' and channels[1].name == 'impurity'
    )
    assert channels[0].target_edges == (-0.5, 0.6)
    assert float(np.atleast_1d(extra.rate(300.0))[0]) == pytest.approx(rate)
