"""Scattering channels: Matthiessen rule, Allen rates, Boltzmann (get_tau) integration."""

import numpy as np
import pytest

from PAOFLOW.boltzmann.do_Boltz_tensors import get_tau
from PAOFLOW.boltzmann.do_tau_models import acoustic_model, impurity_model
from PAOFLOW.boltzmann.scattering_channels import (
    EPSILON0_SI,
    HBAR_EV_S,
    KB_EV_K,
    ConstantRate,
    FermiLiquidRate,
    PhenomenologicalChannel,
    ResidualResistivity,
    TransportSpectralChannel,
    matthiessen,
    total_mass_enhancement,
)
from PAOFLOW.boltzmann.TauModel import TauModel

PARAMS = {'D_ac': 7.0, 'rho': 5.31e3, 'v': 5.2e3, 'ms': 0.7, 'nI': 3.5e17, 'Zi': 1,
          'eps_inf': 11.6, 'eps_0': 13.5}  # fmt: skip


class _DC:
    def __init__(self, energies, tau_dict):
        self.arrays = {'E_k': energies}
        self.attributes = {'bnd': energies.shape[1], 'tau_dict': tau_dict}

    def data_dicts(self):
        return self.arrays, self.attributes


def _energies(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4, 0.4, (20, 3, 1))


def _einstein(omega0=0.01, area=0.006, width=0.0003):
    w = np.linspace(1e-5, 0.03, 3000)
    a2f = area * np.exp(-0.5 * ((w - omega0) / width) ** 2) / (width * np.sqrt(2 * np.pi))
    return TransportSpectralChannel(w, a2f), omega0, area


def test_phenomenological_channel_reproduces_get_tau():
    dc = _DC(_energies(), PARAMS)
    kt = KB_EV_K * 300.0
    reference = get_tau(dc, kt, ['acoustic', 'impurity'], [])
    via_channels = get_tau(
        dc, kt, [PhenomenologicalChannel('acoustic', PARAMS), PhenomenologicalChannel(impurity_model, PARAMS)], []
    )  # fmt: skip
    np.testing.assert_allclose(via_channels, reference, rtol=1e-12)
    mixed = get_tau(dc, kt, ['acoustic', PhenomenologicalChannel('impurity', PARAMS)], [])
    np.testing.assert_allclose(mixed, reference, rtol=1e-12)
    tau_model = TauModel(function=acoustic_model, params=PARAMS)
    np.testing.assert_allclose(
        PhenomenologicalChannel(tau_model).rate(300.0, energy_ev=dc.arrays['E_k']),
        1.0 / acoustic_model(kt, np.abs(dc.arrays['E_k']), PARAMS),
    )


def test_get_tau_weights_and_rate_channels():
    dc = _DC(_energies(1), PARAMS)
    kt = KB_EV_K * 300.0
    one = get_tau(dc, kt, ['acoustic'], [])
    # weights now work (the length check used to compare with an empty list)
    np.testing.assert_allclose(get_tau(dc, kt, ['acoustic'], [2.0]), one / 2.0, rtol=1e-12)
    with pytest.raises(Exception):
        get_tau(dc, kt, ['acoustic'], [1.0, 2.0])
    extra = 1.0e13
    combined = get_tau(dc, kt, ['acoustic', ConstantRate(extra)], [])
    np.testing.assert_allclose(1.0 / combined, 1.0 / one + extra, rtol=1e-12)
    only_rate = get_tau(dc, kt, [ConstantRate(extra)], [])
    np.testing.assert_allclose(only_rate, 1.0 / extra)


def test_matthiessen_and_simple_channels():
    omega = np.array([0.01, 0.1])
    channels = [ConstantRate(1e13), FermiLiquidRate(2.0e-2)]
    kt = KB_EV_K * 400.0
    optical = matthiessen(channels, 400.0, omega_ev=omega)
    expected = 1e13 + 2.0e-2 * (omega**2 + (2 * np.pi * kt) ** 2) / HBAR_EV_S
    np.testing.assert_allclose(optical, expected)
    dc = matthiessen(channels, 400.0, energy_ev=np.array([0.0, 0.05]))
    np.testing.assert_allclose(
        dc, 1e13 + 2.0e-2 * (np.array([0.0, 0.05]) ** 2 + (np.pi * kt) ** 2) / HBAR_EV_S
    )
    residual = ResidualResistivity(1.0e-8, 10.0)
    assert residual.rate(300.0) == pytest.approx(EPSILON0_SI * (10.0 / HBAR_EV_S) ** 2 * 1.0e-8)
    np.testing.assert_array_equal(total_mass_enhancement(channels, omega, 300.0), 0.0)
    with pytest.raises(ValueError):
        PhenomenologicalChannel('acoustic', PARAMS).rate(300.0, omega_ev=omega)


def test_allen_rates_einstein_spectrum():
    channel, omega0, area = _einstein()
    lam = 2.0 * area / omega0
    assert channel.lambda_tr == pytest.approx(lam, rel=2e-3)
    hot = 2000.0
    assert channel.dc_rate(hot) == pytest.approx(
        2 * np.pi * channel.lambda_tr * KB_EV_K * hot / HBAR_EV_S, rel=2e-3
    )
    temp = 100.0
    x = omega0 / (2 * KB_EV_K * temp)
    exact = 4 * np.pi * KB_EV_K * temp * area / omega0 * (x / np.sinh(x)) ** 2 / HBAR_EV_S
    assert channel.dc_rate(temp) == pytest.approx(exact, rel=3e-3)
    # omega -> 0 limit of the optical rate is the DC rate
    assert channel.rate(temp, omega_ev=np.array([1e-6]))[0] == pytest.approx(
        channel.dc_rate(temp), rel=1e-6
    )
    # T = 0: (2 pi / omega) int_0^omega (omega - Omega) a2F
    omega = np.array([0.005, 0.02, 0.05])
    zero_t = channel.optical_rate_ev(omega, 0.0)
    np.testing.assert_allclose(
        zero_t, np.where(omega > omega0, 2 * np.pi * area * (omega - omega0) / omega, 0.0),
        rtol=2e-3, atol=1e-9,
    )  # fmt: skip


def test_mass_enhancement_einstein_spectrum():
    channel, omega0, area = _einstein()
    omega = np.array([1e-4, 0.005, 0.02, 0.05])
    lam = channel.mass_enhancement(omega, 0.0)

    def exact(w):
        return (
            4
            * area
            * (
                np.log(abs((omega0 + w) / (omega0 - w))) / (2 * w)
                + omega0 / (2 * w**2) * np.log(abs(1 - w**2 / omega0**2))
            )
        )

    assert lam[0] == pytest.approx(channel.lambda_tr, rel=5e-3)  # lambda(0, T=0) = lambda_tr
    np.testing.assert_allclose(lam[1:], [exact(w) for w in omega[1:]], rtol=5e-3)
    # temperature suppresses the low-frequency mass enhancement
    assert channel.mass_enhancement(np.array([1e-4]), 1000.0)[0] < lam[0]
