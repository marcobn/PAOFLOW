"""Unit tests for the dielectric-spectrum metric extractor.

These tests do not depend on PAOFLOW or QE outputs — they synthesize
small spectra in-memory or write tiny test files to ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.integration.qe.dielectric_metrics import (
    DielectricMetrics,
    Spectrum,
    load_paoflow_spectrum,
    load_qe_spectrum,
    mean_in_window,
    metrics_from_paoflow_output,
    peak_in_window,
    static_value,
)

# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------


def _flat_spectrum(energy: np.ndarray, value: float) -> Spectrum:
    return Spectrum(energy=energy, values=np.full((energy.size, 3), value))


def _per_channel_spectrum(energy, x_vals, y_vals, z_vals) -> Spectrum:
    return Spectrum(
        energy=np.asarray(energy),
        values=np.stack([np.asarray(x_vals), np.asarray(y_vals), np.asarray(z_vals)], axis=1),
    )


def test_static_value_linear_extrapolation():
    # Channel-average is the line y = 1 + 2x → at x=0 returns 1.
    e = np.array([1.0, 2.0, 3.0])
    v = 1.0 + 2.0 * e
    s = _per_channel_spectrum(e, v, v, v)
    assert static_value(s) == pytest.approx(1.0)


def test_peak_in_window_interior_max_beats_window_edge():
    e = np.linspace(0.0, 10.0, 101)
    v = -((e - 4.0) ** 2) + 5.0  # parabola peaking at e=4, value=5
    s = _per_channel_spectrum(e, v, v, v)
    peak_e, peak_h = peak_in_window(s, 1.0, 9.0)
    assert peak_e == pytest.approx(4.0, abs=0.1)
    assert peak_h == pytest.approx(5.0, abs=1e-3)


def test_peak_in_window_falls_back_to_argmax_when_monotonic():
    # Monotonically increasing spectrum has no interior local maximum;
    # peak_in_window should return the window's right edge.
    e = np.linspace(0.0, 10.0, 21)
    v = e.copy()
    s = _per_channel_spectrum(e, v, v, v)
    peak_e, peak_h = peak_in_window(s, 2.0, 7.0)
    assert peak_e == pytest.approx(7.0)
    assert peak_h == pytest.approx(7.0)


def test_peak_in_window_empty_window_raises():
    e = np.array([1.0, 2.0])
    s = _per_channel_spectrum(e, e, e, e)
    with pytest.raises(ValueError):
        peak_in_window(s, 10.0, 20.0)


def test_mean_in_window_constant_spectrum():
    e = np.linspace(0.0, 10.0, 21)
    s = _flat_spectrum(e, value=3.5)
    assert mean_in_window(s, 2.0, 8.0) == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------


def _write_paoflow_file(path: Path, energy, value) -> None:
    with path.open('w') as fh:
        for e, v in zip(energy, value):
            fh.write(f'{e:.6f} {v:.6e}\n')


def test_load_paoflow_spectrum_round_trip(tmp_path):
    e = np.linspace(0.0, 5.0, 11)
    vx = e * 1.0
    vy = e * 2.0
    vz = e * 3.0
    _write_paoflow_file(tmp_path / 'epsi_xx.dat', e, vx)
    _write_paoflow_file(tmp_path / 'epsi_yy.dat', e, vy)
    _write_paoflow_file(tmp_path / 'epsi_zz.dat', e, vz)

    s = load_paoflow_spectrum(tmp_path, 'epsi')
    np.testing.assert_allclose(s.energy, e)
    np.testing.assert_allclose(s.values[:, 0], vx)
    np.testing.assert_allclose(s.values[:, 1], vy)
    np.testing.assert_allclose(s.values[:, 2], vz)


def test_load_paoflow_spectrum_energy_grid_mismatch_raises(tmp_path):
    e1 = np.linspace(0.0, 5.0, 11)
    e2 = np.linspace(0.1, 5.0, 11)
    _write_paoflow_file(tmp_path / 'epsi_xx.dat', e1, e1)
    _write_paoflow_file(tmp_path / 'epsi_yy.dat', e2, e2)
    _write_paoflow_file(tmp_path / 'epsi_zz.dat', e1, e1)
    with pytest.raises(ValueError, match='energy grid disagrees'):
        load_paoflow_spectrum(tmp_path, 'epsi')


def test_load_qe_spectrum_ignores_comments(tmp_path):
    path = tmp_path / 'epsi_test.dat'
    path.write_text(
        '# energy grid [eV]   epsi_x  epsi_y  epsi_z\n'
        '#\n'
        '   0.5   10.0   11.0   12.0\n'
        '   1.0   20.0   21.0   22.0\n'
    )
    s = load_qe_spectrum(path)
    np.testing.assert_allclose(s.energy, [0.5, 1.0])
    np.testing.assert_allclose(s.values, [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])


# ---------------------------------------------------------------------------
# End-to-end metric bundle
# ---------------------------------------------------------------------------


def test_metrics_from_paoflow_output_bundle(tmp_path):
    e = np.linspace(0.5, 10.0, 96)

    # epsr: linear from 5.0 at the lowest energy to 1.0 at the highest.
    eps1_vals = 5.0 - 4.0 / 9.5 * (e - 0.5)
    # epsi: lorentzian-ish peak at 3.0 eV, height 8.0
    eps2_vals = 8.0 / (1.0 + ((e - 3.0) / 0.3) ** 2)
    # eels: gaussian-ish peak at 6.0 eV, height 0.5
    eels_vals = 0.5 * np.exp(-((e - 6.0) ** 2) / 0.5)

    for basename, vals in (('epsr', eps1_vals), ('epsi', eps2_vals), ('eels', eels_vals)):
        for axis in ('xx', 'yy', 'zz'):
            _write_paoflow_file(tmp_path / f'{basename}_{axis}.dat', e, vals)

    metrics = metrics_from_paoflow_output(
        tmp_path,
        eps2_peak_window=(1.0, 5.0),
        eels_peak_window=(4.0, 8.0),
    )

    assert isinstance(metrics, DielectricMetrics)
    # Static eps1: linear extrapolation from a linear function recovers the intercept
    # (loose absolute tolerance to absorb the 6-decimal precision of _write_paoflow_file).
    assert metrics.eps1_static == pytest.approx(5.0 + (4.0 / 9.5) * 0.5, abs=1e-4)
    assert metrics.eps2_peak_energy == pytest.approx(3.0, abs=0.15)
    assert metrics.eps2_peak_height == pytest.approx(8.0, abs=0.05)
    assert metrics.eels_peak_energy == pytest.approx(6.0, abs=0.15)
    assert metrics.eels_peak_height == pytest.approx(0.5, abs=0.05)
