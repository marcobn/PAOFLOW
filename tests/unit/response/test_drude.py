"""Drude / extended Drude response, resistivity, plasma frequency and metal emissivity."""

import numpy as np
import pytest

from PAOFLOW.boltzmann.scattering_channels import (
    EPSILON0_SI,
    HBAR_EV_S,
    ConstantRate,
    TransportSpectralChannel,
)
from PAOFLOW.response.drude import (
    E2_EV_BOHR,
    dc_resistivity,
    drude_dielectric,
    extended_drude_dielectric,
    plasma_frequency_tensor,
)
from PAOFLOW.response.emissivity import emissivity_from_dielectric, metal_thermal_emissivity

PLASMA = 7.0  # eV
GAMMA = 0.05  # eV


def _einstein(omega0=0.01, area=0.006, width=0.0005):
    w = np.linspace(1e-5, 0.03, 1500)
    a2f = area * np.exp(-0.5 * ((w - omega0) / width) ** 2) / (width * np.sqrt(2 * np.pi))
    return TransportSpectralChannel(w, a2f)


def test_constant_rate_extended_equals_simple_drude():
    omega = np.linspace(0.01, 2.0, 50)
    rate = GAMMA / HBAR_EV_S
    simple = drude_dielectric(omega, PLASMA, rate)
    extended = extended_drude_dielectric(omega, PLASMA, [ConstantRate(rate)], 300.0)
    np.testing.assert_allclose(extended, simple, rtol=1e-12)
    np.testing.assert_allclose(simple, -(PLASMA**2) / (omega * (omega + 1j * GAMMA)))


def test_drude_f_sum_rule():
    omega = np.linspace(1e-4, 400.0, 400001)
    eps = drude_dielectric(omega, PLASMA, GAMMA / HBAR_EV_S)
    f_sum = np.trapezoid(omega * eps.imag, omega)
    assert f_sum == pytest.approx(0.5 * np.pi * PLASMA**2, rel=2e-3)


def test_extended_drude_holstein_absorption():
    channel = _einstein()
    temp = 20.0
    omega = np.array([0.003, 0.05, 0.2])
    dc = drude_dielectric(omega, PLASMA, channel.dc_rate(temp))
    ext = extended_drude_dielectric(omega, PLASMA, [channel], temp)
    # above the phonon energy the optical rate exceeds the DC one (Holstein)
    assert ext.imag[1] > dc.imag[1] and ext.imag[2] > dc.imag[2]
    # with no scattering channel the response is that of free carriers
    free = extended_drude_dielectric(omega, PLASMA, [], temp)
    np.testing.assert_allclose(free, -(PLASMA**2) / omega**2)


def test_dc_resistivity():
    rates = [ConstantRate(1.0e13, 'imp'), ConstantRate(3.0e13, 'phonon')]
    out = dc_resistivity(PLASMA, rates, 300.0)
    omega_p = PLASMA / HBAR_EV_S
    assert out['rho_ohm_m'] == pytest.approx(4.0e13 / (EPSILON0_SI * omega_p**2))
    assert out['channels']['imp'][1] + out['channels']['phonon'][1] == pytest.approx(
        out['rho_ohm_m']
    )


def test_plasma_frequency_free_electrons():
    # free electrons in a cubic box of side L (bohr), Fermi sphere well inside the grid
    side, nk, kf = 6.0, 64, 0.3  # k_F well inside the zone (pi / side = 0.52)
    kmax = np.pi / side
    axis = (np.arange(nk) - nk // 2) * (2 * kmax / nk)
    k = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3) * 1.0
    hbar2_2m = 13.605693122994  # eV bohr^2
    energies = hbar2_2m * (k**2).sum(1) - hbar2_2m * kf**2
    sigma = 0.15
    delta = np.exp(-((energies / sigma) ** 2)) / (sigma * np.sqrt(np.pi))
    velocities = 2.0 * hbar2_2m * k  # d e / d k, eV bohr
    tensor = plasma_frequency_tensor(velocities[:, None, :], delta[:, None], side**3)
    density = kf**3 / (3 * np.pi**2)  # both spins, bohr^-3
    expected = 4 * np.pi * density * E2_EV_BOHR * 2 * hbar2_2m  # 4 pi n e^2 hbar^2 / m
    np.testing.assert_allclose(np.diag(tensor), expected, rtol=0.03)
    assert abs(tensor[0, 1]) < 1e-3 * expected


def test_metal_emissivity_hagen_rubens_and_opacity():
    omega = np.concatenate([np.linspace(1e-4, 1e-3, 10), np.linspace(0.002, 1.0, 400)])
    rate = GAMMA / HBAR_EV_S
    out = metal_thermal_emissivity(
        omega, [300.0], PLASMA, [ConstantRate(rate)], thickness_m=np.inf, ntheta=48,
        angles_deg=(0.0,), extended=False,
    )  # fmt: skip
    np.testing.assert_allclose(out['hemispherical_slab'], out['hemispherical_opaque'])
    sigma_dc = EPSILON0_SI * (PLASMA / HBAR_EV_S) ** 2 / rate  # S/m
    w = omega[:10] / HBAR_EV_S  # omega <= 0.02 / tau: Hagen-Rubens regime
    hagen_rubens = 2.0 * np.sqrt(2.0 * EPSILON0_SI * w / sigma_dc)
    np.testing.assert_allclose(out['directional_slab'][0, 0, :10], hagen_rubens, rtol=0.03)
    assert out['rho_ohm_m'][0] == pytest.approx(1.0 / sigma_dc)
    assert 0.0 < out['total_opaque'][0] < 0.1  # a good metal emits little
    # a dirtier metal emits more
    dirty = metal_thermal_emissivity(
        omega, [300.0], PLASMA, [ConstantRate(5 * rate)], ntheta=48, extended=False
    )
    assert dirty['total_opaque'][0] > out['total_opaque'][0]


def test_metal_emissivity_interband_and_errors():
    omega = np.linspace(0.01, 2.0, 100)
    inter = (np.linspace(0.0, 3.0, 301), np.full(301, 5.0), np.zeros(301))
    with_inter = metal_thermal_emissivity(omega, [300.0], PLASMA, [ConstantRate(1e14)], inter)
    bare = metal_thermal_emissivity(omega, [300.0], PLASMA, [ConstantRate(1e14)])
    np.testing.assert_allclose(with_inter['eps1'] - bare['eps1'], 4.0)
    with pytest.raises(ValueError):
        metal_thermal_emissivity(
            omega, [300.0], PLASMA, [], (inter[0][:50], inter[1][:50], inter[2][:50])
        )


def test_emissivity_from_dielectric_shapes():
    omega = np.linspace(0.01, 1.0, 50)
    eps1 = np.full((2, 50), 12.0)
    eps2 = np.full((2, 50), 1e-3)
    out = emissivity_from_dielectric(omega, eps1, eps2, [300.0, 600.0], 1e-3, (0.0, 45.0), 24)
    assert out['directional_slab'].shape == (2, 2, 50)
    assert np.all(out['hemispherical_slab'] <= out['hemispherical_opaque'] + 1e-12)
