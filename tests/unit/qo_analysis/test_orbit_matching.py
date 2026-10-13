import numpy as np
import pandas as pd
import pytest

from PAOFLOW.qo_analysis.orbit_matching import (
    anisotropy_candidates,
    chance_baselines,
    combination_test,
    fermi_shift_pairs,
    match_rate,
    onsager_slope,
)

pytestmark = pytest.mark.unit


def test_exact_matches_beat_chance():
    angles = np.repeat(np.arange(0.0, 91.0, 5.0), 3)
    frequencies = np.tile([30.0, 600.0, 4000.0], 19)
    rate, _ = match_rate(angles, frequencies, angles, frequencies, 0.1, 0.03)
    baselines = chance_baselines(
        angles, frequencies, angles, frequencies, 0.1, 0.03, np.unique(angles), trials=50
    )
    assert rate == 1.0
    assert baselines['loguniform'][0] < 0.3


def test_dense_candidates_give_chance_level_rates():
    rng = np.random.default_rng(1)
    angles = np.repeat(np.arange(0.0, 91.0, 5.0), 40)
    frequencies = np.exp(rng.uniform(np.log(15), np.log(8000), len(angles)))
    peaks = np.exp(rng.uniform(np.log(15), np.log(8000), 60))
    peak_angles = rng.choice(np.arange(0.0, 91.0, 5.0), 60)
    rate, _ = match_rate(angles, frequencies, peak_angles, peaks, 0.1, 0.05)
    baselines = chance_baselines(
        angles, frequencies, peak_angles, peaks, 0.1, 0.05, np.unique(angles), trials=100
    )
    assert abs(rate - baselines['loguniform'][0]) < 4 * baselines['loguniform'][1] + 0.02


def test_anisotropy_without_common_label_is_empty_with_columns():
    sheets = pd.DataFrame(dict(group=['1'], freq_T=[100.0], mstar_me=[0.5]))
    lk_a = pd.DataFrame(dict(label=['alpha'], freq_T=[100.0], mstar_me=[1.0]))
    lk_b = pd.DataFrame(dict(label=['beta'], freq_T=[100.0], mstar_me=[1.0]))
    ranked = anisotropy_candidates(sheets, sheets, lk_a, lk_b)
    assert ranked.empty and 'ratio_calc' in ranked.columns


def test_onsager_slope_signs():
    assert onsager_slope([1.0, 1.0], ['e', 'h']).tolist() == pytest.approx([8.637, -8.637])


def test_fermi_shift_pairs_recover_onsager():
    def run(shift_meV):
        frequency = 500.0 + onsager_slope(0.5, 'h') * shift_meV
        return pd.DataFrame(
            dict(
                polar_deg=[0.0],
                freq_T=[float(frequency)],
                mstar_abs=[0.5],
                carrier=['h'],
                orbit_type=[-1.0],
            )
        )

    pairs = fermi_shift_pairs(run(-5.0), run(0.0), run(5.0), 5.0)
    assert pairs.ratio.iloc[0] == pytest.approx(1.0)


def test_weak_combinations_are_found():
    peaks = pd.DataFrame(
        dict(
            angle_deg=[0, 0, 0, 0, 5, 5],
            freq_T=[100.0, 470.0, 570.0, 200.0, 300.0, 977.0],
            rel_mag=[0.5, 0.5, 0.05, 0.05, 0.6, 0.05],
        )
    )
    table, summary = combination_test(peaks, trials=20)
    found = dict(zip(table.freq_T, table.combination))
    assert found[570.0] == '100 + 470' and found[200.0] == '2 x 100' and found[977.0] == ''
    assert summary['harmonics'] == 1
