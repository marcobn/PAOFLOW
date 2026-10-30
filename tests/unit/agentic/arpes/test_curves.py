"""MDC analysis on synthetic spectral functions with a known self-energy."""

import numpy as np
import pytest

from PAOFLOW.agentic.arpes import curves

V_BARE, K_FERMI = 3.0, 0.2  # eV A, 1/A


def spectral_map(lam=0.0, gamma=0.005, k=None, w=None):
    """A(k, w) of a linear band E = v (k - kF) with Re Sigma = -lam w, Im Sigma = -gamma."""
    k = np.linspace(0.0, 0.4, 2001) if k is None else k
    w = np.linspace(-0.1, 0.02, 241) if w is None else w
    e_b = V_BARE * (k - K_FERMI)
    denominator = (w[None, :] - e_b[:, None] + lam * w[None, :]) ** 2 + gamma**2
    return k, w, gamma / np.pi / denominator


def test_peak_refinement_and_widths():
    x = np.linspace(-1.0, 1.0, 201)
    y = curves.lorentzian(x, 0.123, 0.2, 1.0, 0.1)
    assert curves.peak_position(x, y) == pytest.approx(0.123, abs=2e-4)
    fwhm, lo, hi = curves.half_maximum_width(x, y, baseline=0.1)
    assert fwhm == pytest.approx(0.2, rel=2e-2) and lo < 0.123 < hi
    fit = curves.fit_lorentzian(x, y)
    assert fit['center'] == pytest.approx(0.123, abs=1e-6) and fit['fwhm'] == pytest.approx(
        0.2, rel=1e-6
    )
    assert fit['offset'] == pytest.approx(0.1, abs=1e-6) and fit['rms'] < 1e-9
    assert np.isnan(curves.half_maximum_width(x[:2], y[:2])[0])


def test_mdc_dispersion_recovers_band_and_width():
    k, w, intensity = spectral_map()
    d = curves.mdc_dispersion(k, w, intensity, energy_window=(-0.08, -0.01))
    assert d['energy_ev'].size > 0
    expected_k = K_FERMI + d['energy_ev'] / V_BARE
    assert np.allclose(d['k'], expected_k, atol=2e-4)
    assert np.allclose(d['fwhm_k'], 2 * 0.005 / V_BARE, rtol=0.01)
    fit = curves.mdc_dispersion(k, w, intensity, energy_window=(-0.05, -0.02), method='lorentzian')
    assert np.allclose(fit['fwhm_k'], 2 * 0.005 / V_BARE, rtol=1e-3)
    assert curves.fermi_momentum(d['energy_ev'], d['k']) is not None
    full = curves.mdc_dispersion(k, w, intensity)
    assert curves.fermi_momentum(full['energy_ev'], full['k']) == pytest.approx(K_FERMI, abs=2e-4)
    band = curves.linear_band(full['energy_ev'], full['k'], (-0.08, -0.01))
    assert band['velocity'] == pytest.approx(V_BARE, rel=1e-3) and band['k_fermi'] == pytest.approx(
        K_FERMI, abs=1e-4
    )
    e = curves.edc_dispersion(k, w, intensity, k_window=(0.17, 0.19))
    assert np.allclose(e['energy_ev'], V_BARE * (e['k'] - K_FERMI), atol=1e-3)
    with pytest.raises(ValueError):
        curves.mdc_dispersion(k, w, intensity.T)


def test_self_energy_from_dispersion_and_kink():
    lam, gamma = 0.5, 0.004
    k, w, intensity = spectral_map(lam=lam, gamma=gamma)
    d = curves.mdc_dispersion(k, w, intensity, energy_window=(-0.08, -0.01))
    bare = (k, V_BARE * (k - K_FERMI))
    sigma = curves.self_energy_from_dispersion(d['energy_ev'], d['k'], d['fwhm_k'], bare)
    assert np.allclose(sigma['re_sigma_ev'], -lam * d['energy_ev'], atol=1e-3)
    assert np.allclose(sigma['im_sigma_ev'], -gamma, rtol=0.05)
    assert np.allclose(sigma['bare_velocity'], V_BARE)
    callable_sigma = curves.self_energy_from_dispersion(
        d['energy_ev'], d['k'], d['fwhm_k'], lambda q: V_BARE * (q - K_FERMI)
    )
    assert np.allclose(callable_sigma['re_sigma_ev'], sigma['re_sigma_ev'], atol=1e-6)
    dressed = curves.linear_band(d['energy_ev'], d['k'], (-0.08, -0.01))
    assert curves.mass_enhancement_from_velocities(V_BARE, dressed['velocity']) == pytest.approx(
        lam, rel=1e-2
    )
    wd = 0.02
    ww = np.linspace(-0.1, 0.1, 401)
    re = -0.3 * ww * np.exp(-0.5 * (ww / wd) ** 2)
    assert curves.kink_energy(ww, re) == pytest.approx(-wd, abs=1e-3)
    assert curves.kink_energy(ww, re, side='empty') == pytest.approx(wd, abs=1e-3)
    assert np.isnan(curves.kink_energy(ww[ww > 0], re[ww > 0], side='occupied'))
