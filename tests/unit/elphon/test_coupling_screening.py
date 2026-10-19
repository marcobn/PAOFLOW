"""Poncé-Resta screening correction of the electron-phonon coupling."""

import warnings

import numpy as np
import pytest

from PAOFLOW.elphon.coupling_screening import (
    CouplingScreening,
    folded_q_norm,
    resolve_coupling_screening,
    resta_dielectric,
    resta_screening_radius,
    screening_report,
    thomas_fermi_wavevector,
)

SI_ALAT = 10.3318  # bohr, the PBE lattice of the Si example
SI_BG = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])  # fcc, 2 pi/alat
SI_VOLUME = SI_ALAT**3 / 4.0


def test_thomas_fermi_wavevector_and_resta_radius_for_si():
    q_tf = thomas_fermi_wavevector(8.0, SI_VOLUME)
    assert q_tf == pytest.approx(1.1002, abs=1e-4)
    radius = resta_screening_radius(11.94, q_tf)
    assert np.sinh(q_tf * radius) / (q_tf * radius) == pytest.approx(11.94, rel=1e-10)
    assert radius == pytest.approx(4.296, abs=1e-3)
    with pytest.raises(ValueError, match='above 1'):
        resta_screening_radius(1.0, q_tf)
    with pytest.raises(ValueError, match='positive'):
        thomas_fermi_wavevector(0.0, SI_VOLUME)


def test_resta_dielectric_limits_and_monotonic_decay():
    q_tf = thomas_fermi_wavevector(8.0, SI_VOLUME)
    q = np.linspace(0.0, 3.0, 301)
    eps = resta_dielectric(q, 11.94, q_tf)
    assert eps[0] == pytest.approx(11.94, rel=1e-12)
    assert np.all(np.diff(eps) < 0.0)
    assert float(resta_dielectric(100.0, 11.94, q_tf)) == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize(
    'bg',
    [SI_BG, np.array([[1.0, 0.57735027, 0.0], [0.0, 1.15470054, 0.0], [0.0, 0.0, 0.6]])],
    ids=['fcc', 'hexagonal'],
)
def test_folded_q_norm_is_the_shortest_image(bg):
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.5, 1.5, size=(50, 3))
    wide = np.stack(np.meshgrid(*(np.arange(-3, 4),) * 3, indexing='ij'), -1).reshape(-1, 3)
    brute = np.linalg.norm((q[:, None, :] + wide[None]) @ bg, axis=-1).min(axis=1)
    np.testing.assert_allclose(folded_q_norm(q, bg, SI_ALAT), brute * 2 * np.pi / SI_ALAT)
    shifted = q + rng.integers(-2, 3, size=q.shape)  # lattice-equivalent q
    np.testing.assert_allclose(folded_q_norm(shifted, bg, SI_ALAT), folded_q_norm(q, bg, SI_ALAT))
    assert float(folded_q_norm(np.zeros(3), bg, SI_ALAT)) == 0.0


def test_si_factors_match_ponce_magnitudes():
    screening = CouplingScreening(12.97, 11.94, thomas_fermi_wavevector(8.0, SI_VOLUME))
    assert float(screening.ratio(0.0)) == pytest.approx(12.97 / 11.94, rel=1e-12)
    x_point = screening.factor([0.5, 0.5, 0.0], SI_BG, SI_ALAT)  # (0, 0, 1) 2 pi/a
    assert float(folded_q_norm([0.5, 0.5, 0.0], SI_BG, SI_ALAT)) == pytest.approx(
        2 * np.pi / SI_ALAT
    )
    assert 1.0 < x_point < 1.015  # |g|^2 about +2 % at X: intervalley scattering barely moves
    assert float(screening.ratio(50.0)) == pytest.approx(1.0, abs=1e-4)
    info = screening.summary(SI_BG, SI_ALAT)
    assert info['coupling_screening'] == 'ponce-resta'
    assert info['screening_ratio_gamma'] ** 2 == pytest.approx(1.18, abs=0.005)
    assert info['screening_q_boundary'] == pytest.approx(np.sqrt(3) / 2 * 2 * np.pi / SI_ALAT)  # L
    assert 'x1.180 at Gamma' in screening_report(info)
    assert 'no correction' in screening_report({'coupling_screening': 'none'})


def test_resolve_coupling_screening():
    epsi = np.diag([12.0, 13.0, 14.0])
    assert resolve_coupling_screening(None, None, epsi, 8.0, SI_VOLUME) is None
    resolved = resolve_coupling_screening(11.94, None, epsi, 8.0, SI_VOLUME)
    assert resolved.eps_dft == pytest.approx(13.0) and resolved.eps_target == 11.94
    explicit = resolve_coupling_screening(11.94, 12.89, np.zeros((3, 3)), 8.0, SI_VOLUME)
    assert explicit.eps_dft == 12.89
    with pytest.raises(ValueError, match='no dielectric tensor'):
        resolve_coupling_screening(11.94, None, np.zeros((3, 3)), 8.0, SI_VOLUME)
    with pytest.raises(ValueError, match='needs screening_eps'):
        resolve_coupling_screening(None, 12.9, epsi, 8.0, SI_VOLUME)
    with pytest.raises(ValueError, match='above 1'):
        resolve_coupling_screening(0.5, None, epsi, 8.0, SI_VOLUME)


# --------------------------------------------------------------------------- #
# End to end on the synthetic EPW system of test_pao_electrons
# --------------------------------------------------------------------------- #
def _quiet(run, *args, **kwargs):
    with warnings.catch_warnings():  # quadrupole warning of a non-polar insulator
        warnings.simplefilter('ignore', UserWarning)
        return run(*args, **kwargs)


def test_absorption_screening_scales_the_phonon_assisted_term(tmp_path, monkeypatch):
    pytest.importorskip('mpi4py')
    from .test_pao_electrons import _gapped_HRs, _run, _synthetic_epw

    _synthetic_epw(tmp_path, 2, epsi=5.0 * np.eye(3))
    HRs = _gapped_HRs()
    plain = _quiet(_run, tmp_path, HRs)
    assert plain['coupling_screening'] == 'none'

    # the same constant on both sides: factor exactly 1, nothing changes
    same = _quiet(_run, tmp_path, HRs, screening_eps=5.0)
    assert same['screening_eps_dft'] == 5.0 and same['coupling_screening'] == 'ponce-resta'
    for key in ('eps2_indirect', 'eps2_direct', 'alpha_cm'):
        np.testing.assert_array_equal(same[key], plain[key])

    # Resta factor between 1 and its Gamma value: the indirect term grows within those bounds
    screened = _quiet(_run, tmp_path, HRs, screening_eps=4.0)
    ratio2 = screened['screening_ratio_gamma'] ** 2
    indirect, base = np.asarray(screened['eps2_indirect']), np.asarray(plain['eps2_indirect'])
    assert np.all(indirect >= base * (1 - 1e-12)) and np.all(indirect <= base * ratio2 + 1e-30)
    assert indirect.sum() > base.sum()
    np.testing.assert_array_equal(screened['eps2_direct'], plain['eps2_direct'])

    # a constant factor c scales the phonon-assisted term by c^2 exactly
    monkeypatch.setattr(CouplingScreening, 'factor', lambda self, q, bg, alat: 1.3)
    constant = _quiet(_run, tmp_path, HRs, screening_eps=4.0)
    np.testing.assert_allclose(constant['eps2_indirect'], 1.69 * base, rtol=1e-12, atol=1e-30)


def test_absorption_screening_needs_the_dft_constant(tmp_path):
    pytest.importorskip('mpi4py')
    from .test_pao_electrons import _gapped_HRs, _run, _synthetic_epw

    _synthetic_epw(tmp_path, 2)  # ph.x without epsil: no tensor in the .epb
    with pytest.raises(ValueError, match='no dielectric tensor'):
        _run(tmp_path, _gapped_HRs(), screening_eps=4.0)
    explicit = _run(tmp_path, _gapped_HRs(), screening_eps=4.0, screening_eps_dft=5.0)
    assert explicit['screening_eps_dft'] == 5.0
    with pytest.raises(ValueError, match='needs screening_eps'):
        _run(tmp_path, _gapped_HRs(), screening_eps_dft=5.0)


def test_mobility_rates_scale_with_the_squared_factor(tmp_path, monkeypatch):
    pytest.importorskip('mpi4py')
    from PAOFLOW.elphon.carrier_scattering import carrier_scattering_dense_q

    from .test_pao_electrons import NAWF, _gapped_HRs, _grid, _synthetic_epw

    _synthetic_epw(tmp_path, 2, epsi=5.0 * np.eye(3))
    A = np.repeat(np.eye(NAWF, dtype=complex)[:, :, None], 8, axis=2)

    def rates(**extra):
        return _quiet(
            carrier_scattering_dense_q,
            A, _gapped_HRs(), _grid(2), np.eye(3), np.eye(3), 10.0, 1000.0, str(tmp_path),
            (2, 2, 2), (2, 2, 2), masses_amu=[28.0], nelec=4, nk_dense=4, nq_dense=2,
            temps_k=(300.0,), window_ev=1.0, smearing_ev=0.1, eps_acoustic_cm=0.1, **extra,
        )  # fmt: skip

    plain = rates()
    assert plain['coupling_screening'] == 'none' and plain['rates_serta'].max() > 0.0
    monkeypatch.setattr(CouplingScreening, 'factor', lambda self, q, bg, alat: 1.3)
    scaled = rates(screening_eps=4.0)
    assert scaled['coupling_screening'] == 'ponce-resta'
    for key in ('rates_serta', 'rates_mrta'):
        np.testing.assert_allclose(scaled[key], 1.69 * plain[key], rtol=1e-12, atol=1e-6)
