"""Electron-phonon self-energies: Eliashberg form, Fan-Migdal kernel and the ARPES adapter."""

import numpy as np
import pytest

from PAOFLOW.elphon.phonon_assisted_absorption import (
    KELVIN_TO_EV,
    bose_occupation,
    fermi_occupation,
)
from PAOFLOW.elphon.self_energy import (
    TabulatedSelfEnergy,
    eliashberg_self_energy,
    fan_migdal_kernel,
    mass_enhancement,
)


def _einstein_a2f(lam=0.8, omega_e=0.03, width=0.0008, n=2001):
    omega = np.linspace(1e-4, 0.08, n)
    a2f = (
        lam * omega_e / 2.0 * np.exp(-(((omega - omega_e) / width) ** 2)) / (width * np.sqrt(np.pi))
    )
    return omega, a2f


@pytest.mark.parametrize('temperature', [0.0, 2.0])
def test_eliashberg_self_energy_einstein_model(temperature):
    lam, omega_e = 0.8, 0.03
    omega, a2f = _einstein_a2f(lam, omega_e)
    assert np.isclose(2.0 * np.trapezoid(a2f / omega, omega), lam, rtol=1e-3)
    w = np.linspace(-0.1, 0.1, 2001)
    sigma = eliashberg_self_energy(omega, a2f, w, temperature)
    # mass enhancement from the slope at E_F
    lam_fit, z = mass_enhancement(w, sigma)
    assert np.isclose(lam_fit, lam, rtol=0.03) and np.isclose(z, 1.0 / (1.0 + lam), rtol=0.03)
    # Im Sigma: zero inside the phonon energy, -pi lambda omega_E / 2 beyond it
    inside = np.abs(w) < 0.8 * omega_e
    outside = np.abs(w) > 1.3 * omega_e
    assert np.all(sigma.imag <= 0.0)
    assert np.abs(sigma.imag[inside]).max() < 1e-3 * np.pi * lam * omega_e / 2.0 + 1e-9
    assert np.allclose(sigma.imag[outside], -np.pi * lam * omega_e / 2.0, rtol=2e-3)
    # odd real part, even imaginary part
    assert np.allclose(sigma.real, -sigma.real[::-1], atol=1e-10)
    assert np.allclose(sigma.imag, sigma.imag[::-1], atol=1e-10)


def test_eliashberg_broadcasts_over_states():
    omega, a2f = _einstein_a2f()
    stack = np.stack([a2f, 2.0 * a2f])
    w = np.linspace(-0.05, 0.05, 101)
    sigma = eliashberg_self_energy(omega, stack, w, 10.0)
    assert sigma.shape == (2, 101)
    assert np.allclose(sigma[1], 2.0 * sigma[0])


def _block(seed=0, nb=3, nkq=5, nband=2, nmode=3):
    rng = np.random.default_rng(seed)
    e_kq = rng.uniform(-0.3, 0.3, size=(nb, nkq))
    g2 = rng.uniform(0.0, 1e-3, size=(nb, nmode, nkq, nband))
    omega = np.array([0.01, 0.03, 0.05])[:nmode]
    temps = [0.02]
    occ = e_kq[None] - 0.01
    return e_kq, g2, omega, temps, occ


def test_fan_migdal_kernel_matches_a_literal_sum():
    e_kq, g2, omega, temps, occ = _block()
    w = np.linspace(-0.5, 0.5, 51)
    eta = 0.004
    out = fan_migdal_kernel(e_kq, g2, omega, temps, occ, w, eta, 1e-4, weight=0.3)
    assert out.shape == (1, 3, 2, 51)
    literal = np.zeros_like(out)
    kt = temps[0]
    for b in range(3):
        for n in range(2):
            for m in range(5):
                f = fermi_occupation(np.array([occ[0, b, m]]), kt)[0]
                for nu in range(3):
                    nq = bose_occupation(np.array([omega[nu]]), kt)[0]
                    literal[0, b, n] += (
                        0.3
                        * g2[b, nu, m, n]
                        * (
                            (nq + f) / (w - e_kq[b, m] + omega[nu] + 1j * eta)
                            + (nq + 1.0 - f) / (w - e_kq[b, m] - omega[nu] + 1j * eta)
                        )
                    )
    assert np.allclose(out, literal, rtol=1e-12, atol=1e-18)
    assert np.all(out.imag <= 1e-15)
    # per-state widths are accepted
    widths = np.full(e_kq.shape, eta)
    assert np.allclose(
        fan_migdal_kernel(e_kq, g2, omega, temps, occ, w, widths, 1e-4, weight=0.3), out
    )


def test_on_shell_imaginary_part_is_the_lorentzian_scattering_rate():
    """-Im Sigma_n(e_n) / pi is the SERTA sum with a Lorentzian delta of the same width."""
    rng = np.random.default_rng(5)
    nb, nkq, nband, nmode = 4, 40, 2, 2
    e_k = rng.uniform(-0.1, 0.1, size=(nb, nband))
    e_kq = rng.uniform(-0.6, 0.6, size=(nb, nkq))
    g2 = rng.uniform(0.0, 1e-3, size=(nb, nmode, nkq, nband))
    omega = np.array([0.02, 0.04])
    kt = 0.03
    occ = e_kq[None] - 0.02
    eta = 0.05
    sigma_on_shell = np.zeros((nb, nband), dtype=complex)
    for n in range(nband):
        for b in range(nb):
            sig = fan_migdal_kernel(
                e_kq[b : b + 1],
                g2[b : b + 1],
                omega,
                [kt],
                occ[:, b : b + 1],
                e_k[b : b + 1, n],
                eta,
                1e-4,
            )
            sigma_on_shell[b, n] = sig[0, 0, n, 0]
    # Lorentzian delta of the same width in the SERTA sum: sum_m g2 [(n+f) L(e_n - e_m + w) + (n+1-f) L(e_n - e_m - w)]
    literal = np.zeros((nb, nband))
    f = fermi_occupation(occ[0], kt)
    for nu in range(nmode):
        nq = bose_occupation(np.array([omega[nu]]), kt)[0]
        for n in range(nband):
            x_abs = e_k[:, n][:, None] - e_kq + omega[nu]
            x_emi = e_k[:, n][:, None] - e_kq - omega[nu]
            lor_abs = eta / np.pi / (x_abs**2 + eta**2)
            lor_emi = eta / np.pi / (x_emi**2 + eta**2)
            literal[:, n] += np.sum(
                g2[:, nu, :, n] * ((nq + f) * lor_abs + (nq + 1.0 - f) * lor_emi), axis=1
            )
    assert np.allclose(-sigma_on_shell.imag / np.pi, literal, rtol=1e-10)


def test_tabulated_self_energy_maps_blocks_and_bands():
    kpts = np.array([[0.0, 0.0, 0.1], [0.5, 0.0, 0.1], [1.0, 0.0, 0.1]])
    w_grid = np.linspace(-1.0, 1.0, 21)
    sigma = np.zeros((3, 2, 21), dtype=complex)
    sigma[:, 0, :] = -0.1j
    sigma[:, 1, :] = -0.2j + 0.05 * w_grid[None, :]
    table = TabulatedSelfEnergy(kpts, bands=[3, 4], w_grid=w_grid, sigma=sigma)
    w = np.array([-0.5, 0.0, 0.5])
    out = table(w, None, kpts[[2, 0]], bands=[2, 3, 4])
    assert out.shape == (2, 3, 3)
    assert np.allclose(out[:, 0, :], 0.0)  # band 2 not tabulated
    assert np.allclose(out[:, 1, :], -0.1j)
    assert np.allclose(out[:, 2, :], -0.2j + 0.05 * w)
    with pytest.raises(ValueError):
        table(w, None, np.array([[0.25, 0.0, 0.1]]), bands=[3])
    result = {'bands': np.array([3, 4]), 'w_ev': w_grid, 'sigma_ev': sigma[None]}
    assert np.allclose(
        TabulatedSelfEnergy.from_result(result, kpts)(w, None, kpts[:1], [4])[0, 0],
        -0.2j + 0.05 * w,
    )
    assert KELVIN_TO_EV > 0


def test_eliashberg_grid_starting_at_zero_frequency():
    """An a2F table that starts at omega = 0 (as eliashberg_dense_q writes it) stays finite."""
    omega, a2f = _einstein_a2f()
    omega = np.concatenate([[0.0], omega])
    a2f = np.concatenate([[1e-6], a2f])  # a tiny spurious value at the origin
    w = np.linspace(-0.1, 0.1, 401)
    for temperature in (10.0, 300.0):
        sigma = eliashberg_self_energy(omega, a2f, w, temperature)
        assert np.all(np.isfinite(sigma)) and np.abs(sigma).max() < 0.1
    lam, _ = mass_enhancement(w, eliashberg_self_energy(omega, a2f, w, 1.0))
    assert np.isclose(lam, 0.8, rtol=0.05)


def test_group_velocities_at_arbitrary_k_match_the_cosine_band():
    from PAOFLOW.elphon.elph_bloch import group_velocities_at, precompute_dense_electrons

    ng, t, alat = (3, 3, 3), 1.0, 5.0
    HRs = np.zeros((1, 1) + ng + (1,), dtype=complex)
    for axis in range(3):
        for sign in (1, -1):
            idx = [0, 0, 0]
            idx[axis] = sign % 3
            HRs[0, 0, idx[0], idx[1], idx[2], 0] = -t
    electrons = precompute_dense_electrons(HRs, np.eye(3), 6, [0.01], 1.0, ng)
    k = np.array([[0.13, 0.27, -0.31], [0.5, 0.0, 0.0]])
    v = group_velocities_at(electrons, k, np.eye(3), alat)  # Ry bohr
    # e(k) = -2t sum cos(2 pi k_a) (eV) -> de/dk_cart = 2 t alat sin(2 pi k_a) (eV bohr)
    expected = 2.0 * t * alat * np.sin(2.0 * np.pi * k) / 13.605693122994
    assert np.allclose(v[:, 0, :], expected, atol=1e-6)
