"""Canonical datasets, unit conversions and comparison metrics."""

import numpy as np
import pytest

from PAOFLOW.agentic.thermo_optics import metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import (
    COLUMNS,
    dataset_header_lines,
    make_dataset,
    parse_conditions,
    read_dataset,
    write_dataset,
)
from PAOFLOW.agentic.thermo_optics.units import convert_photon


def test_dataset_round_trip_keeps_header_and_hash_in_fields(tmp_path):
    frame = make_dataset('A_300K', 'absorption_coefficient', [1.1, 1.2], [3.5, 64.0], 'eV', 'cm-1', 'digitised',
                         'Ref #1', 'Fig. #2', temperature_k=300, conditions={'thickness_m': 5e-4, 'sample': 'n'},
                         y_rel_uncertainty=0.05, note='marker #3')  # fmt: skip
    path = write_dataset(tmp_path / 'a.csv', frame, ['source line', 'calibration line'])
    back = read_dataset(path)
    assert list(back.columns) == list(COLUMNS)
    assert np.allclose(back.y, [3.5, 64.0])
    assert back.note.iloc[0] == 'marker #3' and back.locator.iloc[0] == 'Fig. #2'
    assert dataset_header_lines(path) == ['source line', 'calibration line']
    assert parse_conditions(back.conditions.iloc[0]) == {'thickness_m': 5e-4, 'sample': 'n'}


def test_dataset_rejects_unknown_provenance():
    with pytest.raises(ValueError):
        make_dataset('A', 'q', [1], [1], 'eV', '1', 'guessed', 'ref')


def test_photon_unit_round_trip():
    energies = np.array([0.1, 1.0, 3.0])
    um = convert_photon(energies, 'eV', 'um')
    assert np.allclose(um, 1.23984198 / energies, rtol=1e-6)
    assert np.allclose(
        convert_photon(convert_photon(energies, 'eV', 'cm-1'), 'cm-1', 'eV'), energies
    )


def test_curve_deviation_log10_and_shift():
    x = np.linspace(1.0, 2.0, 51)
    y_exp = np.exp(5 * x)
    stats = mt.curve_deviation(x, y_exp, x, 2.0 * y_exp, 'log10')
    assert stats['bias'] == pytest.approx(np.log10(2.0)) and stats['median_ratio'] == pytest.approx(
        2.0
    )
    shifted = mt.best_shift(
        x, y_exp, x + 0.1, y_exp, np.arange(-0.2, 0.201, 0.01), 'log10', window=(1.2, 1.8)
    )
    assert shifted['shift'] == pytest.approx(-0.1, abs=1e-9) and shifted['rms_shifted'] < 1e-9


def test_best_shift_without_overlap_returns_nan():
    x = np.linspace(1.0, 2.0, 11)
    out = mt.best_shift(x, np.exp(x), x + 5.0, np.exp(x), np.arange(-0.5, 0.51, 0.1), 'log10')
    assert np.isnan(out['shift']) and np.isnan(out['rms_shifted']) and out['n'] == 0


def test_threshold_arrhenius_and_matching():
    x = np.linspace(0.0, 1.0, 11)
    assert mt.threshold_crossing(x, x**2, 0.25) == pytest.approx(0.5, abs=0.01)
    temps = np.array([500.0, 600.0, 700.0])
    fit = mt.arrhenius_fit(temps, 3.0 * np.exp(-0.6 / (mt.KB_EV * temps)))
    assert fit['activation_ev'] == pytest.approx(0.6) and fit['prefactor'] == pytest.approx(3.0)
    pairs = mt.match_conditions([291, 363], [300, 450], tolerance=40)
    assert pairs[0][1] == 300 and pairs[1][1] is None


def test_interpolate_in_condition_is_exact_for_arrhenius():
    temps = np.array([600.0, 750.0])
    curves = np.exp(-0.5 / (mt.KB_EV * temps))[:, None] * np.ones((1, 3))
    out = mt.interpolate_in_condition(temps, curves, 690.0, log_y=True, transform=lambda t: 1 / t)
    assert np.allclose(out, np.exp(-0.5 / (mt.KB_EV * 690.0)))
