"""Unit tests for the SERTA/MRTA carrier scattering rates and the RTA transport."""

import numpy as np
import pytest

pytest.importorskip('scipy')

from PAOFLOW.boltzmann.do_Boltz_tensors import get_tau
from PAOFLOW.boltzmann.scattering_channels import EnergyResolvedRate
from PAOFLOW.elphon.carrier_scattering import (
    BOHR_M,
    ELEMENTARY_CHARGE,
    HBAR_EV_S,
    _window_bands,
    energy_resolved_rates,
    fermi_levels_for_doping,
    rta_transport,
    scattering_rate_kernel,
    write_mobility_outputs,
)
from PAOFLOW.elphon.elph_bloch import RY_TO_EV
from PAOFLOW.elphon.phonon_assisted_absorption import (
    KELVIN_TO_EV,
    bose_occupation,
    fermi_occupation,
    intrinsic_fermi_levels,
)

ELECTRON_MASS = 9.1093837015e-31


def _random_block(seed=0, nb=3, nband=4, nmode=3, ntemp=2):
    rng = np.random.default_rng(seed)
    e_k = rng.uniform(-0.1, 0.1, (nb, nband))
    e_kq = rng.uniform(-0.1, 0.1, (nb, nband))
    v_k = rng.normal(size=(nb, nband, 3))
    v_kq = rng.normal(size=(nb, nband, 3))
    g2 = rng.uniform(0.0, 1.0e-3, (nb, nmode, nband, nband))
    omega = np.array([0.002, 0.03, 0.06])[:nmode]
    temps = np.array([0.0259, 0.05])[:ntemp]
    occ_energy = rng.uniform(-0.1, 0.1, (ntemp, nb, nband))
    return e_k, e_kq, v_k, v_kq, g2, omega, temps, occ_energy


def _literal_rates(e_k, e_kq, v_k, v_kq, g2, omega, temps, occ_energy, sigma, eps_ac, weight):
    nb, nband = e_k.shape
    serta = np.zeros((temps.size, nb, nband))
    mrta = np.zeros((temps.size, nb, nband))

    def delta(x):
        return np.exp(-((x / sigma) ** 2)) / (sigma * np.sqrt(np.pi)) if abs(x) < 6 * sigma else 0.0

    for it, kt in enumerate(temps):
        for b in range(nb):
            for n in range(nband):
                for nu, w in enumerate(omega):
                    if w <= eps_ac:
                        continue
                    n_ph = bose_occupation(np.array(w), kt)
                    for m in range(nband):
                        f = fermi_occupation(np.array(occ_energy[it, b, m]), kt)
                        x = e_kq[b, m] - e_k[b, n]
                        term = g2[b, nu, m, n] * (
                            (n_ph + f) * delta(x - w) + (n_ph + 1 - f) * delta(x + w)
                        )
                        rate = 2 * np.pi / HBAR_EV_S * weight * term
                        v2 = v_k[b, n] @ v_k[b, n]
                        serta[it, b, n] += rate
                        mrta[it, b, n] += rate * (1.0 - (v_k[b, n] @ v_kq[b, m]) / v2)
    return serta, mrta


def test_kernel_matches_literal_fan_migdal_sum():
    e_k, e_kq, v_k, v_kq, g2, omega, temps, occ = _random_block()
    args = (e_k, e_kq, v_k, v_kq, g2, omega, temps, occ, 0.02, 0.005)
    serta, mrta = scattering_rate_kernel(*args, weight=0.125)
    ref_s, ref_m = _literal_rates(*args, 0.125)
    assert np.abs(ref_s).max() > 0.0
    np.testing.assert_allclose(serta, ref_s, rtol=1e-10, atol=1e-6 * ref_s.max())
    np.testing.assert_allclose(mrta, ref_m, rtol=1e-10, atol=1e-6 * np.abs(ref_m).max())


def test_kernel_mrta_limits_and_acoustic_skip():
    e_k, e_kq, v_k, _, g2, omega, temps, occ = _random_block(seed=1)
    common = dict(smearing=0.02, eps_acoustic=0.005)
    # forward scattering into a state with the same velocity relaxes no momentum
    same_v = np.broadcast_to(v_k[:, :1], v_k.shape).copy()
    serta, mrta = scattering_rate_kernel(e_k, e_kq, same_v, same_v, g2, omega, temps, occ, **common)
    assert serta.max() > 0.0
    np.testing.assert_allclose(mrta, 0.0, atol=1e-12 * serta.max())
    # back scattering (v' = -v) doubles the SERTA rate
    serta, mrta = scattering_rate_kernel(
        e_k, e_kq, same_v, -same_v, g2, omega, temps, occ, **common
    )
    np.testing.assert_allclose(mrta, 2.0 * serta, rtol=1e-12)
    # only modes above eps_acoustic contribute
    only_low = scattering_rate_kernel(
        e_k, e_kq, v_k, v_k, g2, omega, temps, occ, smearing=0.02, eps_acoustic=0.1
    )
    assert not only_low[0].any() and not only_low[1].any()


def test_fermi_levels_intrinsic_and_doped():
    rng = np.random.default_rng(3)
    nk = 400
    valence = -0.5 + 0.4 * rng.random((nk, 2))
    conduction = 0.6 + 0.5 * rng.random((nk, 2))
    energies = np.hstack([valence, conduction])
    temps = [300.0, 900.0]
    volume = 270.0  # bohr^3
    intrinsic = fermi_levels_for_doping(energies, temps, nelec=4.0)
    reference = intrinsic_fermi_levels(energies, np.array(temps) * KELVIN_TO_EV, 4.0)
    np.testing.assert_allclose(intrinsic, reference, atol=1e-10)
    doping = 1.0e19
    n_type = fermi_levels_for_doping(energies, temps, 4.0, doping, volume)
    p_type = fermi_levels_for_doping(energies, temps, 4.0, -doping, volume)
    assert np.all(n_type > intrinsic) and np.all(p_type < intrinsic)
    omega_cm3 = volume * (BOHR_M * 1e2) ** 3
    for it, temp in enumerate(temps):
        f = fermi_occupation(energies - n_type[it], KELVIN_TO_EV * temp)
        n_e = 2.0 * f[:, 2:].sum() / (nk * omega_cm3)
        n_h = 2.0 * (1.0 - f[:, :2]).sum() / (nk * omega_cm3)
        np.testing.assert_allclose(n_e - n_h, doping, rtol=1e-6)
    with pytest.raises(ValueError):
        fermi_levels_for_doping(energies, temps, 4.0, doping)
    # a scissor on the conduction bands moves the intrinsic level up by half of it
    shifted = fermi_levels_for_doping(
        energies, [300.0], 4.0, conduction=np.array([0, 0, 1, 1], bool), scissor_ev=[0.4]
    )
    assert shifted[0] > intrinsic[0]


def test_window_bands_semiconductor_and_metal():
    e = np.array([[-3.0, -0.5, -0.1, 0.8, 1.0, 4.0], [-2.9, -0.25, 0.0, 1.0, 1.1, 4.2]])
    kept = _window_bands(e / RY_TO_EV, 3, 0.3, None)
    np.testing.assert_array_equal(kept, [1, 2, 3, 4])
    kept = _window_bands(e / RY_TO_EV, 3, 0.3, 0.85)
    np.testing.assert_array_equal(kept, [3, 4])


def _parabolic_results(mass=0.25, tau=1.0e-14, ngrid=48, alat=12.0, mu_below=0.1):
    """Single isotropic conduction band in a simple-cubic cell, constant tau."""
    axis = (np.arange(ngrid) - ngrid // 2) / ngrid * 2 * np.pi / (alat * BOHR_M)  # 1/m
    k = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    hbar_js = HBAR_EV_S * ELEMENTARY_CHARGE
    m = mass * ELECTRON_MASS
    energy = (hbar_js * k) ** 2 / (2 * m) / ELEMENTARY_CHARGE
    energy = energy.sum(axis=1)[:, None]  # eV
    velocity = (hbar_js * k / m)[:, None, :]
    temps = np.array([300.0])
    mu = np.array([-mu_below])
    f = fermi_occupation(energy - mu[0], KELVIN_TO_EV * temps[0])
    volume = alat**3
    n_e = 2.0 * f.sum() / (energy.shape[0] * volume * (BOHR_M * 1e2) ** 3)
    rates = np.full((1,) + energy.shape, 1.0 / tau)
    return {
        'energies_ev': energy,
        'velocities_ms': velocity,
        'in_window': np.ones(energy.shape, bool),
        'carrier': np.array([1]),
        'rates_serta': rates,
        'rates_mrta': rates,
        'fermi_levels_ev': mu,
        'scissor_ev': np.zeros(1),
        'n_e_cm3': np.array([n_e]),
        'n_h_cm3': np.zeros(1),
        'temps_k': temps,
        'cell_volume_bohr3': volume,
        'spin_degeneracy': 2.0,
        'vbm_ev': -1.0,
        'cbm_ev': 0.0,
    }


def test_rta_transport_parabolic_band_drude_mobility():
    mass, tau = 0.25, 1.0e-14
    results = _parabolic_results(mass, tau)
    out = rta_transport(results, 'serta')
    drude_mobility = ELEMENTARY_CHARGE * tau / (mass * ELECTRON_MASS) * 1e4  # cm^2/Vs
    np.testing.assert_allclose(out['mobility_e'][0], drude_mobility, rtol=0.02)
    sigma = results['n_e_cm3'][0] * 1e6 * ELEMENTARY_CHARGE**2 * tau / (mass * ELECTRON_MASS)
    np.testing.assert_allclose(out['sigma_avg'][0], sigma, rtol=0.02)
    np.testing.assert_allclose(np.diag(out['sigma'][0]), sigma, rtol=0.02)
    assert np.isnan(out['mobility_h'][0])
    # non-degenerate electrons: S = -(k_B/e)(5/2 + (E_c - mu)/kT) for constant tau
    kt = KELVIN_TO_EV * 300.0
    seebeck = -(2.5 + 0.1 / kt) * KELVIN_TO_EV * 1e6
    np.testing.assert_allclose(out['seebeck_uv_k'][0], seebeck, rtol=0.03)


def test_energy_resolved_rates_reproduce_k_resolved_conductivity():
    results = _parabolic_results()
    rng = np.random.default_rng(5)
    # a k-dependent (anisotropic) rate: only the transport-weighted average matters
    rates = 1.0e14 * (1.0 + rng.random(results['energies_ev'].shape))
    results['rates_mrta'] = rates[None]
    reference = rta_transport(results, 'mrta')['sigma_avg'][0]
    table = energy_resolved_rates(results, 'mrta', bin_ev=0.001)
    channel = EnergyResolvedRate(table['energy_ev'], table['temps_k'], table['rate_per_s'])
    bridged = dict(results)
    bridged['rates_mrta'] = channel.rate(300.0, energy_ev=results['energies_ev'])[None]
    approx = rta_transport(bridged, 'mrta')['sigma_avg'][0]
    np.testing.assert_allclose(approx, reference, rtol=0.02)


def test_bridge_through_get_tau(tmp_path):
    results = _parabolic_results()
    results['rates_mrta'] = np.stack([results['rates_mrta'][0], 2 * results['rates_mrta'][0]])
    results['rates_serta'] = results['rates_mrta']
    results['temps_k'] = np.array([300.0, 400.0])
    results['fermi_levels_ev'] = np.array([-0.1, -0.1])
    results['scissor_ev'] = np.zeros(2)
    results['n_e_cm3'] = np.repeat(results['n_e_cm3'], 2)
    results['n_h_cm3'] = np.zeros(2)
    written = write_mobility_outputs(results, str(tmp_path))
    assert any(p.endswith('scattering_rates.npz') for p in written)
    channel = EnergyResolvedRate.from_npz(str(tmp_path / 'scattering_rates.npz'), 'mrta')

    class _DC:
        arrays = {'E_k': np.array([[[-0.9], [0.2]]])}
        attributes = {'bnd': 2, 'tau_dict': {}, 'nelec': 2, 'dftSO': False}

        def data_dicts(self):
            return self.arrays, self.attributes

    tau = get_tau(_DC(), KELVIN_TO_EV * 350.0, [channel], [])
    np.testing.assert_allclose(tau, 1.0 / 1.5e14, rtol=1e-6)
    # get_tau aligned the table (vbm -1, cbm 0) to the band edges of the run
    assert channel.band_edges == (-1.0, 0.0) and channel.target_edges == (-0.9, 0.2)


def test_kernel_mrta_velocity_floor():
    e_k, e_kq, v_k, v_kq, g2, omega, temps, occ = _random_block(seed=2)
    slow = 1.0e-3 * v_k
    serta, mrta = scattering_rate_kernel(
        e_k, e_kq, slow, v_kq, g2, omega, temps, occ, 0.02, 0.005, velocity_floor=1.0e-2
    )
    np.testing.assert_allclose(mrta, serta, rtol=1e-12)


def test_fermi_levels_resolve_tiny_carrier_densities():
    # wide gap at low T: carriers ~1e-30 per cell, far below the total-electron resolution
    rng = np.random.default_rng(7)
    nk = 1500
    energies = np.hstack([-0.4 + 0.4 * rng.random((nk, 4)), 1.8 + 0.5 * rng.random((nk, 4))])
    conduction = np.arange(8) >= 4
    volume, temps = 275.7, [150.0, 300.0]
    omega_cm3 = volume * (BOHR_M * 1e2) ** 3

    def densities(mu, temp):
        kt = KELVIN_TO_EV * temp
        n_e = 2.0 * fermi_occupation(energies[:, conduction] - mu, kt).sum() / (nk * omega_cm3)
        n_h = 2.0 * fermi_occupation(mu - energies[:, ~conduction], kt).sum() / (nk * omega_cm3)
        return n_e, n_h

    mu_i = fermi_levels_for_doping(energies, temps, 8, conduction=conduction)
    np.testing.assert_allclose(
        mu_i, intrinsic_fermi_levels(energies, np.array(temps) * KELVIN_TO_EV, 8), atol=1e-12
    )
    for it, temp in enumerate(temps):
        n_e, n_h = densities(mu_i[it], temp)
        # below the ~2e8 cm^-3 resolution of counting all 8 electrons per cell
        assert 0.0 < n_e < 1e8 and np.isclose(n_e, n_h, rtol=1e-8)
    mu_n = fermi_levels_for_doping(energies, temps, 8, 1e16, volume, conduction=conduction)
    for it, temp in enumerate(temps):
        n_e, n_h = densities(mu_n[it], temp)
        n_i = densities(mu_i[it], temp)[0]
        np.testing.assert_allclose(n_e, 1e16, rtol=1e-8)
        np.testing.assert_allclose(n_h, n_i**2 / 1e16, rtol=1e-4)  # mass-action law (Boltzmann limit)
