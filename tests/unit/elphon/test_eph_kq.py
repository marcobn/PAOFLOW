"""Unit tests for the Bloch-basis e-ph assembly and Eliashberg helpers (no QE)."""

import numpy as np
import pytest

from PAOFLOW.elphon import gkq
from PAOFLOW.elphon.eph_kq import (
    _AMU_KG,
    _BOHR_M,
    _HBAR_JS,
    EV_TO_K,
    THZ_TO_EV,
    assemble_g_bloch,
    bloch_hamiltonian,
    eliashberg,
    estates_on_grid,
    fourier_dHdu,
    fourier_dHdu_on_grid,
    mcmillan_allen_dynes_tc,
    phonon_moments,
    primitive_eigenstates,
    zero_point_amplitude,
)


def _hermitian_HR(nawf, n, rng):
    """Random real-space H with H(-R) = H(R)^dagger (so H(k) is Hermitian)."""
    HR = rng.standard_normal((nawf, nawf, n, n, n, 1)) + 1j * rng.standard_normal(
        (nawf, nawf, n, n, n, 1)
    )
    HR_mR = np.conj(HR[:, :, ::-1, ::-1, ::-1, :])
    HR_mR = np.roll(np.roll(np.roll(HR_mR, 1, 2), 1, 3), 1, 4)
    HR = 0.5 * (HR + np.transpose(HR_mR, (1, 0, 2, 3, 4, 5)))
    return HR


def test_bloch_hamiltonian_constant_for_onsite():
    # H(R) nonzero only at R=0 -> H(k) is k-independent and equal to that block.
    nawf, n = 3, 4
    block = np.diag([1.0, 2.0, 3.0]).astype(complex)
    HR = np.zeros((nawf, nawf, n, n, n, 1), dtype=complex)
    HR[:, :, 0, 0, 0, 0] = block
    Hk = bloch_hamiltonian(HR)
    for idx in [(0, 0, 0), (1, 2, 3), (3, 3, 3)]:
        assert np.allclose(Hk[:, :, idx[0], idx[1], idx[2], 0], block)


def test_bloch_hamiltonian_hermitian():
    rng = np.random.default_rng(0)
    HR = _hermitian_HR(4, 6, rng)
    Hk = bloch_hamiltonian(HR)
    for idx in [(0, 0, 0), (1, 2, 3), (5, 5, 5)]:
        M = Hk[:, :, idx[0], idx[1], idx[2], 0]
        assert np.linalg.norm(M - M.conj().T) < 1e-10


def test_primitive_eigenstates_shapes_and_values():
    rng = np.random.default_rng(1)
    HR = _hermitian_HR(3, 4, rng)
    E, V = primitive_eigenstates(HR)
    assert E.shape == (4, 4, 4, 1, 3)
    assert V.shape == (4, 4, 4, 1, 3, 3)
    # eigenvalues ascending and reconstruct H(k).
    Hk = bloch_hamiltonian(HR)
    M = Hk[:, :, 1, 2, 3, 0]
    w, _ = np.linalg.eigh(M)
    assert np.allclose(E[1, 2, 3, 0], np.sort(w))


def test_fourier_dHdu_gamma_is_Rp_sum():
    rng = np.random.default_rng(2)
    g_R = rng.standard_normal((3, 2, 2, 4, 4, 4, 2, 2, 2, 1)) + 1j * rng.standard_normal(
        (3, 2, 2, 4, 4, 4, 2, 2, 2, 1)
    )
    d = fourier_dHdu(g_R)
    assert d.shape == (3, 2, 2, 4, 4, 4, 2, 2, 2, 1)
    # q=Gamma component == sum over R_p; k=Gamma == sum over R_e.
    gamma = d[:, :, :, 0, 0, 0, 0, 0, 0, :]
    expected = g_R.sum(axis=(3, 4, 5, 6, 7, 8))
    assert np.allclose(gamma, expected)


def test_zero_point_amplitude_value_and_floor():
    # sqrt(hbar / (2 M omega)) in Bohr for M = 1 amu, omega = 1 THz.
    M, f = 1.0, 1.0
    omega_rad = 2.0 * np.pi * f * 1e12
    expect_bohr = np.sqrt(_HBAR_JS / (2.0 * M * _AMU_KG * omega_rad)) / _BOHR_M
    got = zero_point_amplitude(M, f)
    assert np.isclose(got, expect_bohr, rtol=1e-12)
    # sub-floor frequencies (acoustic Gamma) give zero amplitude.
    assert zero_point_amplitude(M, 1e-9) == 0.0


def test_assemble_g_bloch_shapes_and_identity_projection():
    rng = np.random.default_rng(3)
    natom, nawf, nmode = 1, 4, 3
    dHdu = rng.standard_normal((3, nawf, nawf)) + 1j * rng.standard_normal((3, nawf, nawf))
    v = np.eye(nawf, dtype=complex)
    eigvec_q = rng.standard_normal((natom, 3, nmode)) + 0j
    omega = np.array([2.0, 4.0, 6.0])  # THz, all above floor
    masses = np.array([26.98])
    g = assemble_g_bloch(dHdu, v, v, eigvec_q, omega, masses)
    assert g.shape == (nmode, nawf, nawf)
    # With identity eigenvectors, g is the amplitude-weighted mode combination of dHdu.
    amp = zero_point_amplitude(masses[:, None], omega[None, :])  # (1, nmode)
    pattern = np.moveaxis((eigvec_q * amp[:, None, :]).reshape(3, nmode), 0, 1)  # (nmode,3)
    expected = np.einsum('vc,cij->vij', pattern, dHdu)
    assert np.allclose(g, expected)


def test_acoustic_mode_zero_amplitude_zeros_g():
    rng = np.random.default_rng(4)
    dHdu = rng.standard_normal((3, 2, 2)) + 0j
    v = np.eye(2, dtype=complex)
    eigvec_q = rng.standard_normal((1, 3, 3)) + 0j
    omega = np.array([0.0, 5.0, 7.0])  # first mode acoustic at Gamma
    masses = np.array([1.0])
    g = assemble_g_bloch(dHdu, v, v, eigvec_q, omega, masses)
    assert np.allclose(g[0], 0.0)  # acoustic Gamma mode -> zero coupling


# --- generalized (arbitrary supercell / interpolated k-grid) path ------------


def test_embed_fftfreq_interpolation_matches_coarse_grid():
    # H(k) on a native n^3 grid, then interpolated to a multiple 2n^3; the coarse
    # k-samples (every other point) must reproduce the native eigenvalues.
    rng = np.random.default_rng(10)
    HR = _hermitian_HR(3, 4, rng)
    E_native, _ = primitive_eigenstates(HR)  # (4,4,4,1,3)
    E_fine, _ = estates_on_grid(HR, 8)  # (8,8,8,1,3)
    np.testing.assert_allclose(E_fine[::2, ::2, ::2], E_native, atol=1e-9)


def test_estates_on_grid_native_matches_primitive():
    rng = np.random.default_rng(11)
    HR = _hermitian_HR(3, 4, rng)
    E0, _ = primitive_eigenstates(HR)
    E1, _ = estates_on_grid(HR, 4)  # native size -> identical
    np.testing.assert_allclose(E1, E0, atol=1e-10)


def test_fourier_dHdu_on_grid_native_matches_plain():
    rng = np.random.default_rng(12)
    g_R = rng.standard_normal((3, 2, 2, 4, 4, 4, 2, 2, 2, 1)) + 0j
    d0 = fourier_dHdu(g_R)
    d1 = fourier_dHdu_on_grid(g_R, 4)  # native R_e size
    np.testing.assert_allclose(d1, d0, atol=1e-10)


class _EphController:
    def __init__(self, HR, efermi):
        self._arry = {'HRs': HR}
        self._attr = {'Efermi': float(efermi)}
        self.rank = 0

    def data_dicts(self):
        return self._arry, self._attr


def _fake_phonon_modes(natom, nmode):
    def _modes(phonon, qpoints, with_eigenvectors=True):
        nq = len(qpoints)
        rng = np.random.default_rng(99)
        freqs = np.abs(rng.standard_normal((nq, nmode))) * 5.0 + 1.0  # THz, all > floor
        freqs[0, 0:3] = 0.0  # acoustic modes vanish at Gamma (q=0 is first)
        eig = rng.standard_normal((nq, natom, 3, nmode)) + 0j
        masses = np.full(natom, 26.98)
        return freqs, eig, masses

    return _modes


def test_eliashberg_S3_runs_on_interpolated_grid(monkeypatch):
    # g_R from a 3x3x3 supercell: R_e grid 6^3 (!= electronic native), 27 q-points.
    natom, nawf, nmode = 1, 2, 3
    rng = np.random.default_rng(20)
    g_R = rng.standard_normal((natom * 3, nawf, nawf, 6, 6, 6, 3, 3, 3, 1)) + 0j
    HR = _hermitian_HR(nawf, 4, rng)  # native electronic grid 4^3 differs from R_e 6^3
    dc = _EphController(HR, efermi=0.0)
    monkeypatch.setattr(gkq, 'phonon_modes', _fake_phonon_modes(natom, nmode))

    out = eliashberg(dc, g_R, phonon=None, smearing_ev=0.5, nk_electron=6)
    assert out['nk_electron'] == 6  # divisible by 3
    assert out['lambda_qv'].shape == (27, nmode)
    assert np.isfinite(out['lambda'])
    # Acoustic modes were set to zero frequency at Gamma -> zero coupling there.
    assert out['gamma_acoustic'] == 0.0


def test_eliashberg_requires_divisible_grid(monkeypatch):
    natom, nawf, nmode = 1, 2, 3
    rng = np.random.default_rng(21)
    g_R = rng.standard_normal((natom * 3, nawf, nawf, 6, 6, 6, 3, 3, 3, 1)) + 0j
    HR = _hermitian_HR(nawf, 4, rng)
    dc = _EphController(HR, efermi=0.0)
    monkeypatch.setattr(gkq, 'phonon_modes', _fake_phonon_modes(natom, nmode))
    with pytest.raises(ValueError, match='divisible'):
        eliashberg(dc, g_R, phonon=None, nk_electron=8)  # 8 not divisible by 3


# --------------------------------------------------------------------------- #
# McMillan / Allen-Dynes critical temperature
# --------------------------------------------------------------------------- #
def test_phonon_moments_single_mode():
    # A single populated mode: omega_log and omega_2 both equal that frequency.
    w_thz = 5.0
    lam_qv = np.array([[0.7]])
    omega = np.array([[w_thz]])
    w_log, w2 = phonon_moments(lam_qv, omega)
    expected = w_thz * THZ_TO_EV
    np.testing.assert_allclose(w_log, expected, rtol=1e-12)
    np.testing.assert_allclose(w2, expected, rtol=1e-12)


def test_phonon_moments_weighted_average():
    # log-average weighted by lambda; RMS weighted by lambda; skips w<=0.
    w = np.array([[2.0, 8.0, 0.0]])
    lam = np.array([[1.0, 3.0, 5.0]])  # the zero-frequency mode is ignored
    w_log, w2 = phonon_moments(lam, w)
    tot = 1.0 + 3.0
    exp_log = np.exp((1.0 * np.log(2.0) + 3.0 * np.log(8.0)) / tot) * THZ_TO_EV
    exp_w2 = np.sqrt((1.0 * 2.0**2 + 3.0 * 8.0**2) / tot) * THZ_TO_EV
    np.testing.assert_allclose(w_log, exp_log, rtol=1e-12)
    np.testing.assert_allclose(w2, exp_w2, rtol=1e-12)


def test_phonon_moments_zero_coupling():
    w_log, w2 = phonon_moments(np.zeros((2, 3)), np.ones((2, 3)))
    assert w_log == 0.0 and w2 == 0.0


def test_mcmillan_tc_matches_closed_form():
    # McMillan Tc against a hand evaluation (single mode -> omega_2 = omega_log,
    # so f2 = 1; f1 supplies the only Allen-Dynes correction).
    lam, mu = 1.0, 0.10
    w_log_ev = 0.025  # eV
    res = mcmillan_allen_dynes_tc(lam, w_log_ev, w_log_ev, mu_star=mu)
    denom = lam - mu * (1.0 + 0.62 * lam)
    expect_mcm = (w_log_ev * EV_TO_K / 1.2) * np.exp(-1.04 * (1.0 + lam) / denom)
    np.testing.assert_allclose(res['Tc_mcmillan_K'], expect_mcm, rtol=1e-10)
    # omega_2 == omega_log -> f2 exactly 1.
    np.testing.assert_allclose(res['f2'], 1.0, rtol=1e-12)
    lam1 = 2.46 * (1.0 + 3.8 * mu)
    f1 = (1.0 + (lam / lam1) ** 1.5) ** (1.0 / 3.0)
    np.testing.assert_allclose(res['f1'], f1, rtol=1e-10)
    np.testing.assert_allclose(res['Tc_allen_dynes_K'], f1 * expect_mcm, rtol=1e-10)


def test_allen_dynes_reduces_to_mcmillan_weak_coupling():
    # Weak coupling: f1, f2 -> 1 so Allen-Dynes ~ McMillan.
    res = mcmillan_allen_dynes_tc(0.3, 0.02, 0.02, mu_star=0.12)
    np.testing.assert_allclose(res['f1'], 1.0, atol=5e-2)
    np.testing.assert_allclose(res['f2'], 1.0, atol=1e-12)
    np.testing.assert_allclose(
        res['Tc_allen_dynes_K'], res['f1'] * res['Tc_mcmillan_K'], rtol=1e-10
    )


def test_tc_zero_when_denominator_nonpositive():
    # lambda - mu*(1 + 0.62 lambda) <= 0 -> no superconductivity in this model.
    res = mcmillan_allen_dynes_tc(0.10, 0.02, 0.02, mu_star=0.13)
    assert res['Tc_mcmillan_K'] == 0.0
    assert res['Tc_allen_dynes_K'] == 0.0


def test_eliashberg_reports_tc(monkeypatch):
    natom, nawf, nmode = 1, 2, 3
    rng = np.random.default_rng(7)
    g_R = rng.standard_normal((natom * 3, nawf, nawf, 6, 6, 6, 3, 3, 3, 1)) + 0j
    HR = _hermitian_HR(nawf, 4, rng)
    dc = _EphController(HR, efermi=0.0)
    monkeypatch.setattr(gkq, 'phonon_modes', _fake_phonon_modes(natom, nmode))

    out = eliashberg(dc, g_R, phonon=None, smearing_ev=0.5, nk_electron=6, mu_star=0.12)
    for key in ('omega_log', 'omega_2', 'mu_star', 'Tc_mcmillan', 'Tc_allen_dynes', 'f1', 'f2'):
        assert key in out
    assert out['mu_star'] == 0.12
    assert np.isfinite(out['Tc_mcmillan']) and out['Tc_mcmillan'] >= 0.0
    assert np.isfinite(out['Tc_allen_dynes']) and out['Tc_allen_dynes'] >= 0.0
    # Moments consistent with the standalone helper on the returned mode data.
    w_log, w2 = phonon_moments(out['lambda_qv'], out['omega_q'])
    np.testing.assert_allclose(out['omega_log'], w_log, rtol=1e-12)
    np.testing.assert_allclose(out['omega_2'], w2, rtol=1e-12)
