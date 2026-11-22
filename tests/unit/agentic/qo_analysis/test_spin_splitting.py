"""Spin factors of the calculated orbits (qo_analysis.spin_splitting and the driver step)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PAOFLOW.agentic.qo_analysis import QuantumOscillationAnalysis
from PAOFLOW.agentic.qo_analysis.spin_splitting import (
    LEVELS,
    attach_orbits,
    lk_branch_table,
    match_peaks,
    rank_correlation_test,
    read_spin_splitting,
    sheet_of_pao_band,
)

pytestmark = pytest.mark.unit

COLUMNS = (
    ['azimuthal', 'polar', 'frequency_kT', 'm_star']
    + [
        f'{kind}_{level}{suffix}'
        for level in LEVELS
        for kind, suffix in (('g', ''), ('Rs', ''), ('g', '_prop'), ('absRs', '_prop'))
    ]
    + [
        'berry_phase_mean',
        'berry_rotation',
        'max_pair_splitting_meV',
        'max_energy_error_meV',
        'npoints',
    ]
)


def _write_table(path: Path, rows: list[dict]) -> Path:
    """A table in the layout of PAOFLOW.orbit_spin_splitting."""
    with open(path, 'w') as handle:
        handle.write('# ' + ' '.join(COLUMNS) + '\n')
        for row in rows:
            handle.write(' '.join(f'{float(row.get(c, 0.0)):.6g}' for c in COLUMNS) + '\n')
    return path


def _row(polar: float, frequency_T: float, mass: float, g: float) -> dict:
    gm2 = 0.5 * g * mass
    row = dict(
        azimuthal=30.0, polar=polar, frequency_kT=frequency_T / 1e3, m_star=mass, npoints=300
    )
    for level in LEVELS:
        row.update({f'g_{level}': g, f'Rs_{level}': np.cos(np.pi * gm2),
                    f'g_{level}_prop': g, f'absRs_{level}_prop': abs(np.cos(np.pi * gm2))})  # fmt: skip
    return row


def _fake_bxsf(folder: Path, sheet: int, pao_band: int) -> None:
    (folder / f'Fermi_surf_band_{sheet}.bxsf').write_text(
        f'BEGIN_INFO\nEND_INFO\n  BAND: {pao_band:5d}\n'
    )


def test_tables_band_map_and_orbit_properties(frequency_run, tmp_path):
    tables = tmp_path / 'oss'
    tables.mkdir()
    _write_table(tables / 'orbit_spin_splitting_21.dat',
                 [_row(p, 500.0 + p, 0.5, 2.0) for p in (0.0, 45.0, 90.0)])  # fmt: skip
    _write_table(tables / 'orbit_spin_splitting_23.dat', [_row(0.0, 1200.0, 1.1, 1.8)])
    spin = read_spin_splitting(tables)
    assert sorted(spin.pao_band.unique()) == [21, 23]
    assert np.allclose(spin.gm_total, spin.g_total * spin.m_star)
    _fake_bxsf(frequency_run, 1, 21)
    _fake_bxsf(frequency_run, 2, 23)
    mapping = sheet_of_pao_band(frequency_run)
    assert mapping == {21: 1, 23: 2}
    from PAOFLOW.agentic.qo_analysis.skeaf_output import read_orbits

    orbits = read_orbits(frequency_run, None, '0', {1: '1', 2: '2'})
    attached = attach_orbits(spin, orbits, mapping, 'polar')
    assert attached.num_orbit_copies.notna().all()
    assert set(attached.group) == {'1', '2'}
    assert (attached.loc[attached.pao_band == 21, 'orbit_type'] == -1).all()


def test_rank_test_finds_a_monotonic_relation_and_rejects_noise():
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0, 1, 40)
    strong = rank_correlation_test(2.0 * predicted + 0.01, predicted, trials=500)
    noise = rank_correlation_test(
        np.random.default_rng(42).uniform(0, 1, 40), predicted, trials=500
    )
    assert strong['rho'] == pytest.approx(1.0) and strong['p_value'] < 0.01
    assert noise['p_value'] > 0.01


def test_peak_matching_and_lk_branches():
    spin = pd.DataFrame([dict(_row(0.0, 500.0, 0.5, 2.0), freq_T=500.0, group='1'),
                         dict(_row(0.0, 1200.0, 1.1, 1.8), freq_T=1200.0, group='2')])  # fmt: skip
    peaks = pd.DataFrame(dict(calc_angle=[0.0, 0.0, 0.0, 30.0], freq_T=[510.0, 1190.0, 800.0, 500.0],
                              rel_mag=[0.4, 0.2, 0.1, 0.3]))  # fmt: skip
    matched = match_peaks(spin, peaks, 'polar', 0.05)
    assert list(matched.freq_calc_T) == [500.0, 1200.0]
    lk = lk_branch_table(spin, pd.DataFrame(dict(label=['a', 'b', 'c'], freq_T=[495.0, 1210.0, 3000.0],
                                                 mstar_me=[1.0, 2.2, 3.0])), 0.0)  # fmt: skip
    assert lk.enhancement.iloc[0] == pytest.approx(1.0) and lk.enhancement.iloc[1] == pytest.approx(
        1.0
    )
    assert np.isnan(lk.freq_calc_T.iloc[2])


def test_driver_step_writes_its_outputs(frequency_run, tmp_path):
    tables = tmp_path / 'oss'
    tables.mkdir()
    rows = [_row(p, 500.0 + p, 0.5 + 0.2 * k, 2.0) for k, p in enumerate((0.0, 45.0, 90.0))]
    _write_table(tables / 'orbit_spin_splitting_21.dat', rows)
    _fake_bxsf(frequency_run, 1, 21)
    _fake_bxsf(frequency_run, 2, 23)
    peaks = pd.DataFrame(dict(angle_deg=[0.0, 45.0, 90.0], freq_T=[500.0, 545.0, 590.0],
                              rel_mag=[0.9, 0.5, 0.1]))  # fmt: skip
    peaks.to_csv(tmp_path / 'peaks.csv', index=False)
    qo = QuantumOscillationAnalysis(frequency_run, output_dir=tmp_path / 'out', degenerate_groups=1,
                                    primitive_cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # fmt: skip
    qo.read_experiment(sweep_peaks=tmp_path / 'peaks.csv')
    result = qo.spin_splitting(tables, trials=200)
    assert len(result['table']) == 3 and len(result['matched']) == 3
    assert set(result['correlation']) == {'adiabatic', 'propagator'}
    for name in (
        'spin_splitting_table.csv',
        'fig_spin_factors.png',
        'matched_peaks.csv',
        'summary.md',
    ):
        assert (tmp_path / 'out' / 'spin_splitting' / name).exists()


def test_lk_branches_follow_their_assigned_group():
    from PAOFLOW.agentic.qo_analysis.spin_splitting import assigned_group

    spin = pd.DataFrame([dict(_row(0.0, 100.0, 0.3, 2.0), freq_T=100.0, group='1,2'),
                         dict(_row(0.0, 93.0, 0.4, 1.9), freq_T=93.0, group='3,4')])  # fmt: skip
    assignments = pd.DataFrame(dict(label=['beta1, beta2', 'beta1'], status=['assigned', 'assigned'],
                                    group=['3,4', '1,2'], arrow_angle_deg=[5.0, 88.0]))  # fmt: skip
    assert assigned_group(assignments, 'beta1', 0.0) == '3,4'
    assert assigned_group(assignments, 'beta1', 90.0) == '1,2'
    lk = lk_branch_table(spin, pd.DataFrame(dict(label=['beta1'], freq_T=[101.0], mstar_me=[0.8])), 0.0,
                         assignments=assignments)  # fmt: skip
    assert lk.group.iloc[0] == '3,4' and lk.freq_calc_T.iloc[0] == 93.0
