"""Unit tests for the isotropic Migdal-Eliashberg solver (``elphon/migdal_eliashberg.py``)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.elphon import migdal_eliashberg as me
from PAOFLOW.elphon.eph_kq import THZ_TO_EV, eliashberg_from_modes

KB = me.KB_EV


def _einstein(
    lam: float, w_e: float = 0.01, width: float = 1.0e-4, nqstep: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Narrow-Gaussian Einstein a2F with total coupling ``lam`` at ``w_e`` (eV)."""
    return me.a2f_from_modes([[lam]], [[w_e / THZ_TO_EV]], None, width, nqstep)


def test_matsubara_grid_matches_epw_count() -> None:
    # EPW tutorial 04: 616 Matsubara frequencies at 0.3 K with wscut = 0.1 eV.
    wn = me.matsubara_grid(0.3, 0.1)
    assert wn.size == 616
    assert wn[-1] <= 0.1
    np.testing.assert_allclose(wn[0], np.pi * KB * 0.3)


def test_a2f_from_modes_conserves_lambda() -> None:
    rng = np.random.default_rng(0)
    lam_qv = rng.uniform(0.1, 1.0, (7, 3))
    om_thz = rng.uniform(0.5, 2.0, (7, 3))
    wq = rng.uniform(1.0, 3.0, 7)
    ref = eliashberg_from_modes(lam_qv, om_thz, q_weights=wq)['lambda']
    omega, a2F = me.a2f_from_modes(lam_qv, om_thz, wq, degaussq_ev=1.5e-4)
    np.testing.assert_allclose(np.sum(2.0 * a2F * omega[0] / omega), ref, rtol=1.0e-3)
    np.testing.assert_allclose(omega[-1], 1.1 * om_thz.max() * THZ_TO_EV)


def test_matsubara_lambda_einstein_is_analytic() -> None:
    lam, w_e, T = 0.8, 0.01, 2.0
    omega, a2F = _einstein(lam, w_e)
    nu = 2.0 * np.pi * KB * T * np.arange(50)
    np.testing.assert_allclose(
        me.matsubara_lambda(omega, a2F, T, 49), lam * w_e**2 / (w_e**2 + nu**2), rtol=1.0e-3
    )


def test_uniform_a2f_interpolates_grids_starting_at_zero() -> None:
    omega = np.linspace(0.0, 0.02, 401)
    a2F = np.exp(-(((omega - 0.01) / 0.002) ** 2))
    nu, au = me._uniform_a2f(omega, a2F)
    np.testing.assert_allclose(nu, 0.02 / 400 * np.arange(1, 401))
    np.testing.assert_allclose(au, a2F[1:], atol=1.0e-12)


def test_normal_state_above_tc() -> None:
    lam, T = 0.5, 10.0
    omega, a2F = _einstein(lam)
    sol = me.solve_imag_iso(omega, a2F, T, 0.1, wscut=1.0)  # wide cutoff: negligible tail
    assert sol['rho_max'] < 1.0 and sol['converged']
    assert not np.any(sol['delta'])
    # Z_n = 1 + (pi T / w_n) [lambda(0) + 2 sum_{m=1}^{n} lambda(m)] (cutoff tail negligible at low n).
    lamn = me.matsubara_lambda(omega, a2F, T, 10)
    n = np.arange(5)
    zref = 1.0 + np.pi * KB * T / sol['wn'][n] * np.array(
        [lamn[0] + 2.0 * lamn[1 : k + 1].sum() for k in n]
    )
    np.testing.assert_allclose(sol['Z'][n], zref, rtol=1.0e-3)


def test_pade_reproduces_rational_function() -> None:
    poles = np.array([0.3 - 0.05j, -0.2 - 0.1j, 0.05 - 0.02j])
    res = np.array([1.0, 0.5, 0.2])

    def rational(z: np.ndarray) -> np.ndarray:
        return np.sum(res / (z[:, None] - poles), axis=1)

    z = 1j * (2 * np.arange(12) + 1) * 0.01
    w = np.linspace(-0.5, 0.5, 31) + 0.01j
    coefficients = me.pade_coefficients(z, rational(z))
    np.testing.assert_allclose(me.pade_eval(coefficients, z, w), rational(w), rtol=1.0e-8)


def test_lambda_real_minus_imag_matches_direct_sum() -> None:
    omega, a2F = _einstein(1.0, nqstep=60)
    nu, au = me._uniform_a2f(omega, a2F)
    w = nu[0] * np.arange(1, 150)
    wn = me.matsubara_grid(5.0, 0.01)
    lam = me._lambda_real_minus_imag(nu, au * nu[0], w, wn)
    z = w[:, None] - 1j * wn[None, :]
    direct = np.sum(au * nu[0] * 2.0 * nu / (nu**2 - z[..., None] ** 2), axis=-1)
    np.testing.assert_allclose(lam, direct, rtol=1.0e-9, atol=1.0e-12)


def test_gap_edge_and_bcs_quasiparticle_dos() -> None:
    w = 1.0e-5 * np.arange(1, 1001)
    delta = np.full(w.size, 1.0e-3 + 0j)
    assert me.gap_edge(w, delta) == pytest.approx(1.0e-3, rel=1.0e-9)
    assert me.gap_edge(w, np.zeros_like(delta)) == 0.0
    dos = me.quasiparticle_dos(w, delta)
    above, below = w > 1.001e-3, w < 0.999e-3  # skip the singular point w = Delta
    np.testing.assert_allclose(dos[above], w[above] / np.sqrt(w[above] ** 2 - 1.0e-6))
    np.testing.assert_allclose(dos[below], 0.0, atol=1.0e-12)


def test_retarded_sqrt_branch() -> None:
    w = np.array([1.0, 1.0, 0.5, -1.0])
    x = np.array([1.0 - 1.0e-14j, 1.0 + 0.1j, -1.0 - 1.0e-14j, 1.0 + 0.1j])
    s = me._retarded_sqrt(x, w)
    assert np.all(s[:3].real >= 0.0) and np.all(s[:3].imag >= 0.0)
    np.testing.assert_allclose(s[0], 1.0, atol=1.0e-12)  # no sign flip from rounding
    np.testing.assert_allclose(s[3], -np.conj(s[1]))


def test_weak_coupling_bcs_ratio() -> None:
    omega, a2F = _einstein(0.35)
    temps = np.linspace(0.5, 4.0, 36)
    tc = me.linearized_eigenvalues(omega, a2F, temps, 0.0)['Tc_linear']
    sol = me.solve_imag_iso(omega, a2F, tc / 10.0, 0.0)
    assert 2.0 * sol['delta'][0] / (KB * tc) == pytest.approx(3.53, rel=0.03)


def test_tc_estimates_and_continuations_agree() -> None:
    omega, a2F = _einstein(1.0, w_e=0.008, width=2.0e-4)
    lin = me.linearized_eigenvalues(omega, a2F, np.linspace(3.0, 9.0, 61), 0.1)
    tc = lin['Tc_linear']
    assert np.isfinite(tc)
    temps = tc * np.array([0.1, 0.8, 0.9, 0.95, 1.05])
    res = me.migdal_eliashberg_iso(omega, a2F, temps, 0.1)
    assert res['converged'].all()
    assert res['gap0_imag'][-1] == 0.0 and res['gap0_imag'][-2] > 0.0
    assert res['Tc_gap'] == pytest.approx(tc, rel=0.01)
    # Low T: the real-axis gap edges agree and are close to Delta(i w_0).
    assert res['gap_acon'][0] == pytest.approx(res['gap_pade'][0], rel=0.01)
    assert res['gap_acon'][0] == pytest.approx(res['gap0_imag'][0], rel=0.05)
    qdos = res['qdos'][0]
    w = res['w_real']
    assert qdos[w < 0.9 * res['gap_acon'][0]].max() < 0.05  # gapped
    assert qdos.max() > 2.0  # coherence peak


def test_tc_helpers() -> None:
    assert me.tc_from_eigenvalues([1.0, 2.0, 3.0], [1.4, 1.2, 0.8]) == pytest.approx(2.5)
    assert np.isnan(me.tc_from_eigenvalues([1.0, 2.0], [0.9, 0.8]))
    # Delta^2 = 1 - T/5 -> Tc = 5.
    T = np.array([1.0, 2.0, 3.0, 6.0])
    gap = np.sqrt(np.clip(1.0 - T / 5.0, 0.0, None))
    assert me.tc_from_gap(T, gap) == pytest.approx(5.0)


def test_write_me_outputs(tmp_path: Path) -> None:
    omega, a2F = _einstein(1.0, w_e=0.008, width=2.0e-4)
    res = me.migdal_eliashberg_iso(omega, a2F, [0.5, 20.0], 0.1, wscut=0.05)
    lin = me.linearized_eigenvalues(omega, a2F, [0.5, 20.0], 0.1, wscut=0.05)
    me.write_me_outputs(res, str(tmp_path), 'pb', lin)
    names = set(os.listdir(tmp_path))
    for kind in ('imag', 'pade', 'acon', 'qdos'):
        assert 'pb.%s_iso_000.50' % kind in names
    assert 'pb.imag_iso_020.00' in names and 'pb.pade_iso_020.00' not in names
    assert {'gap_vs_T.dat', 'max_eigenvalue.dat', 'migdal_eliashberg.npz'} <= names
    pade = np.loadtxt(tmp_path / 'pb.pade_iso_000.50')
    assert pade.shape == (res['w_real'].size, 5)
    d = np.load(tmp_path / 'migdal_eliashberg.npz')
    np.testing.assert_allclose(d['max_eigenvalue'], lin['max_eigenvalue'])
    np.testing.assert_allclose(np.loadtxt(tmp_path / 'gap_vs_T.dat')[:, 1], res['gap0_imag'] * 1e3)


def test_a2f_from_npz_rebuilds_from_modes(tmp_path: Path) -> None:
    lam_qv = np.array([[0.4, 0.6]])
    om = np.array([[1.0, 2.0]])
    path = tmp_path / 'eliashberg.npz'
    np.savez(path, **eliashberg_from_modes(lam_qv, om), omega_qv_thz=om)
    omega, a2F = me.a2f_from_npz(str(path), degaussq_ev=1.0e-4)
    assert omega.size == 500
    np.testing.assert_allclose(np.sum(2.0 * a2F * omega[0] / omega), 1.0, rtol=1.0e-3)


def test_default_temperatures_bracket_the_migdal_eliashberg_tc() -> None:
    omega, a2F = _einstein(1.0, w_e=0.008, width=2.0e-4)
    temps = me.default_temperatures(omega, a2F, 0.1)
    assert temps.size == 25 and temps[0] > 0.0
    np.testing.assert_allclose(temps[-1], 1.5 * me.allen_dynes_tc(omega, a2F, 0.1))
    tc = me.linearized_eigenvalues(omega, a2F, temps, 0.1)['Tc_linear']
    assert temps[0] < tc < temps[-1]


def test_default_wscut_follows_the_phonon_spectrum() -> None:
    # EPW tutorial 04: 0.1 eV for Pb (phonons up to 9 meV), 0.5 eV for MgB2 (100 meV).
    assert me.default_wscut(*_einstein(1.0, w_e=0.009)) == pytest.approx(0.1)
    assert me.default_wscut(*_einstein(0.6, w_e=0.1)) == pytest.approx(0.5)
