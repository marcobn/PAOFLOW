"""Adaptive smearing of the electron-phonon deltas."""

import numpy as np
import pytest

from PAOFLOW import elphon_native
from PAOFLOW.elphon.adaptive_smearing import (
    fermi_level_adaptive,
    grid_spacing,
    pair_widths,
    state_widths,
    summarize_widths,
)
from PAOFLOW.elphon.carrier_scattering import scattering_rate_kernel
from PAOFLOW.elphon.elph_bloch import RY_TO_EV, precompute_dense_electrons
from PAOFLOW.elphon.phonon_assisted_absorption import (
    direct_absorption_kernel,
    indirect_absorption_kernel,
)


def test_grid_spacing_is_paoflow_dk():
    omega, n = 270.0, 12
    assert np.isclose(grid_spacing(omega, n), (8.0 * np.pi**3 / omega / n**3) ** (1.0 / 3.0))


def test_state_and_pair_widths_and_floor():
    v = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0]])  # |v| = 5, 0
    w = state_widths(v, 0.1, 0.7, 0.01)
    assert np.allclose(w, [0.35, 0.01])
    vk = np.array([[1.0, 0.0, 0.0]])
    vkq = np.array([[0.0, 2.0, 0.0]])
    # |vk - vkq|^2 dk^2 + |vkq|^2 dq^2 = 5 * 0.01 + 4 * 0.04 = 0.21
    assert np.allclose(pair_widths(vk, vkq, 0.1, 0.2, 1.0, 0.0), np.sqrt(0.21))
    assert np.allclose(pair_widths(vk, vkq, 0.1, 0.0, 1.0, 0.0), np.sqrt(0.05))
    assert pair_widths(vk, vk, 0.1, 0.0, 1.0, 0.3) == 0.3
    summary = summarize_widths(w, 1000.0)
    assert summary['min'] == 10.0 and summary['max'] == 350.0


def test_fermi_level_adaptive_half_filled_symmetric_band():
    rng = np.random.default_rng(0)
    e = rng.uniform(-1.0, 1.0, size=(200, 3))
    e = np.concatenate([e, -e])  # symmetric spectrum
    widths = rng.uniform(0.01, 0.1, size=e.shape)
    widths = np.concatenate([widths[:200], widths[:200]])
    mu = fermi_level_adaptive(e, widths, nelec=3.0, spin_degeneracy=2.0)
    assert abs(mu) < 1e-9
    mu_fixed = fermi_level_adaptive(e, np.full(e.shape, 0.05), nelec=3.0)
    assert abs(mu_fixed) < 1e-9


def _rate_block(seed=0, nb=3, nband=4, nmode=3, ntemp=2):
    rng = np.random.default_rng(seed)
    e_k = rng.uniform(-0.2, 0.2, size=(nb, nband))
    e_kq = e_k + rng.uniform(-0.05, 0.05, size=(nb, nband))
    v_k = rng.normal(size=(nb, nband, 3))
    v_kq = rng.normal(size=(nb, nband, 3))
    g2 = rng.uniform(0.0, 1.0, size=(nb, nmode, nband, nband))
    omega = np.array([0.01, 0.03, 0.05])[:nmode]
    temps = [0.02, 0.04][:ntemp]
    occ = np.stack([e_kq - 0.01 * (it + 1) for it in range(ntemp)])
    return e_k, e_kq, v_k, v_kq, g2, omega, temps, occ


def test_scattering_kernel_constant_width_array_equals_scalar():
    e_k, e_kq, v_k, v_kq, g2, omega, temps, occ = _rate_block()
    s1, m1 = scattering_rate_kernel(e_k, e_kq, v_k, v_kq, g2, omega, temps, occ, 0.02, 1e-4)
    widths = np.full(e_kq.shape, 0.02)
    s2, m2 = scattering_rate_kernel(e_k, e_kq, v_k, v_kq, g2, omega, temps, occ, widths, 1e-4)
    assert np.array_equal(s1, s2) and np.array_equal(m1, m2)


def test_scattering_kernel_per_final_state_widths():
    """Per-final-state widths: the kernel is additive over the final states m."""
    e_k, e_kq, v_k, v_kq, g2, omega, temps, occ = _rate_block(seed=3)
    rng = np.random.default_rng(1)
    widths = rng.choice([0.01, 0.03], size=e_kq.shape)
    s_ad, m_ad = scattering_rate_kernel(e_k, e_kq, v_k, v_kq, g2, omega, temps, occ, widths, 1e-4)
    s_sum, m_sum = 0.0, 0.0
    for sigma in (0.01, 0.03):
        g2_masked = g2 * (widths == sigma)[:, None, :, None]  # keep the final states of this width
        s, m = scattering_rate_kernel(
            e_k, e_kq, v_k, v_kq, g2_masked, omega, temps, occ, sigma, 1e-4
        )
        s_sum, m_sum = s_sum + s, m_sum + m
    assert np.allclose(s_ad, s_sum, rtol=1e-12, atol=0.0)
    assert np.allclose(m_ad, m_sum, rtol=1e-12, atol=0.0)
    assert np.abs(s_ad).max() > 0.0


def _absorption_case(seed=0, nk=8, nb=1, nmode=2):
    rng = np.random.default_rng(seed)
    e_k = rng.uniform(-0.03, 0.0, size=(nk, nb))
    e_kq = rng.uniform(0.04, 0.1, size=(nk, nb))
    vk = rng.normal(size=(nk, 3, nb, nb)) + 1j * rng.normal(size=(nk, 3, nb, nb))
    vkq = rng.normal(size=(nk, 3, nb, nb)) + 1j * rng.normal(size=(nk, 3, nb, nb))
    g = rng.normal(size=(nk, nmode, nb, nb)) + 1j * rng.normal(size=(nk, nmode, nb, nb))
    omega_ph = np.array([0.002, 0.004])[:nmode]
    photon = np.linspace(0.03, 0.14, 23)
    return e_k, e_kq, vk, vkq, g, omega_ph, photon


@pytest.mark.parametrize('native', [False, True])
def test_indirect_kernel_constant_width_array_equals_scalar(monkeypatch, native):
    if native and not elphon_native.available():
        pytest.skip('native kernel not built')
    monkeypatch.setattr(elphon_native, 'available', lambda: native)
    e_k, e_kq, vk, vkq, g, omega_ph, photon = _absorption_case()
    args = (omega_ph, [0.002], photon, [0.003, 0.006])
    kw = dict(fsthick=1.0, eps_acoustic=1e-5, occupation_tol=0.0)
    g1, l1 = indirect_absorption_kernel(e_k, e_kq, vk, vkq, g, *args, 0.004, **kw)
    widths = np.full((e_k.shape[0], 1, 1), 0.004)
    g2, l2 = indirect_absorption_kernel(e_k, e_kq, vk, vkq, g, *args, widths, **kw)
    assert np.array_equal(g1, g2) and np.array_equal(l1, l2)


@pytest.mark.parametrize('native', [False, True])
def test_indirect_kernel_per_pair_widths_are_additive(monkeypatch, native):
    """One band per k: each (k, i, j) pair is selected by zeroing the coupling at the other k."""
    if native and not elphon_native.available():
        pytest.skip('native kernel not built')
    monkeypatch.setattr(elphon_native, 'available', lambda: native)
    e_k, e_kq, vk, vkq, g, omega_ph, photon = _absorption_case(seed=5)
    nk = e_k.shape[0]
    widths = np.where((np.arange(nk) % 2 == 0)[:, None, None], 0.003, 0.009)
    args = (omega_ph, [0.002], photon, [0.003])
    kw = dict(fsthick=1.0, eps_acoustic=1e-5, occupation_tol=0.0)
    g_ad, l_ad = indirect_absorption_kernel(e_k, e_kq, vk, vkq, g, *args, widths, **kw)
    g_sum, l_sum = 0.0, 0.0
    for sigma in (0.003, 0.009):
        g_masked = g * (widths[:, 0, 0] == sigma)[:, None, None, None]
        gs, ls = indirect_absorption_kernel(e_k, e_kq, vk, vkq, g_masked, *args, sigma, **kw)
        g_sum, l_sum = g_sum + gs, l_sum + ls
    assert np.allclose(g_ad, g_sum, rtol=1e-10, atol=1e-300)
    assert np.allclose(l_ad, l_sum, rtol=1e-10, atol=1e-300)
    assert np.abs(g_ad).max() > 0.0


def test_direct_kernel_per_pair_widths_are_additive():
    rng = np.random.default_rng(2)
    nk, nb = 6, 3
    e_k = np.sort(rng.uniform(-0.1, 0.1, size=(nk, nb)), axis=1)
    vk = rng.normal(size=(nk, 3, nb, nb)) + 1j * rng.normal(size=(nk, 3, nb, nb))
    photon = np.linspace(0.005, 0.25, 50)
    widths = rng.choice([0.004, 0.012], size=(nk, nb, nb))
    g1, l1 = direct_absorption_kernel(e_k, vk, [0.002], photon, 0.004, 1.0)
    g2, l2 = direct_absorption_kernel(e_k, vk, [0.002], photon, np.full(widths.shape, 0.004), 1.0)
    assert np.array_equal(g1, g2) and np.array_equal(l1, l2)
    g_ad, l_ad = direct_absorption_kernel(e_k, vk, [0.002], photon, widths, 1.0)
    g_sum, l_sum = 0.0, 0.0
    for sigma in (0.004, 0.012):
        mask = (widths == sigma).transpose(0, 2, 1)  # v2 uses velocity_k[k, :, j, i]
        gs, ls = direct_absorption_kernel(e_k, vk * mask[:, None], [0.002], photon, sigma, 1.0)
        g_sum, l_sum = g_sum + gs, l_sum + ls
    assert np.allclose(g_ad, g_sum, rtol=1e-10, atol=1e-300)
    assert np.allclose(l_ad, l_sum, rtol=1e-10, atol=1e-300)


def _simple_cubic_band(hopping_ev=1.0):
    """One s orbital per cell with nearest-neighbour hopping: e(k) = -2t sum_a cos(2 pi k_a)."""
    ng = (3, 3, 3)
    HRs = np.zeros((1, 1) + ng + (1,), dtype=complex)
    for axis in range(3):
        for sign in (1, -1):
            idx = [0, 0, 0]
            idx[axis] = sign % 3
            HRs[0, 0, idx[0], idx[1], idx[2], 0] = -hopping_ev
    at = np.eye(3)
    return HRs, at, 5.0  # alat (bohr)


def test_adaptive_fermi_surface_dos_converges_on_a_coarse_grid():
    """Simple-cubic s band at half filling: the adaptive DOS of a 14^3 grid matches a converged
    90^3 reference (fixed 0.15 eV smearing, above its 0.14 eV level spacing), where the fixed
    smearing on 14^3 with the same 0.1 meV floor does not."""
    HRs, at, alat = _simple_cubic_band()
    reference = precompute_dense_electrons(HRs, at, 90, [0.15 / RY_TO_EV], 1.0, (3, 3, 3))
    floor = 1e-4 / RY_TO_EV
    adaptive = precompute_dense_electrons(
        HRs, at, 14, [floor], 1.0, (3, 3, 3), velocities=True, alat=alat,
        diagonal_velocities=True, adaptive_smearing=0.7,
    )  # fmt: skip
    fixed = precompute_dense_electrons(HRs, at, 14, [floor], 1.0, (3, 3, 3))
    assert abs(reference['ef_sig'][0]) < 1e-6 and abs(adaptive['ef_sig'][0]) < 1e-6
    assert adaptive['widths'].shape == (1, 14**3, 1)
    assert adaptive['widths'].min() >= floor
    # widths: 0.7 |de/dk| dk with de/dk_a = 2 t a sin(2 pi k_a) (Ry bohr), dk = 2 pi / (a N);
    # the largest |sin| on the 14-point grid is sin(2 pi 3 / 14)
    speed_max = 2.0 * (1.0 / RY_TO_EV) * alat * np.sqrt(3.0) * np.sin(2.0 * np.pi * 3.0 / 14.0)
    expected = 0.7 * speed_max * 2.0 * np.pi / (alat * 14)
    assert np.isclose(adaptive['widths'].max(), expected, rtol=1e-6)
    err_adaptive = abs(adaptive['dos'][0] / reference['dos'][0] - 1.0)
    err_fixed = abs(fixed['dos'][0] / reference['dos'][0] - 1.0)
    assert err_adaptive < 0.05, (adaptive['dos'][0], reference['dos'][0])
    assert err_fixed > 0.5
    # the shell is the set of k with a band within fs_window local widths of E_F
    near = np.min(np.abs(adaptive['E'] - adaptive['ef_sig'][0]) / adaptive['widths'][0], axis=1)
    assert np.array_equal(adaptive['shell'][0], near < adaptive['fs_window'])


def test_precompute_adaptive_requires_velocities():
    HRs, at, alat = _simple_cubic_band()
    with pytest.raises(ValueError, match='adaptive_smearing'):
        precompute_dense_electrons(HRs, at, 6, [0.001], 1.0, (3, 3, 3), adaptive_smearing=0.7)
