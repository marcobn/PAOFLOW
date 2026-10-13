"""Regression test: the NbSn2 example (bundled, compressed data) reproduces its reference numbers.

Runtime about 2 minutes (the breakdown gap map dominates); the breakdown network is not run.
"""

import json
from pathlib import Path

import pytest

from PAOFLOW.qo_analysis import QuantumOscillationAnalysis

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).resolve().parents[3] / 'examples' / 'qo_analysis_examples' / 'NbSn2'
EXPECTED = json.loads((EXAMPLE / 'expected_results.json').read_text()) if EXAMPLE.exists() else {}
DIRECTIONS = [
    dict(label='[001]', hkl=[0, 0, 1], azimuthal=29.95, polar=0.0, day=3),
    dict(label='[11y]', hkl=[1, 1, 3.902], azimuthal=29.95, polar=45.0),
    dict(label='[110]', hkl=[1, 1, 0], azimuthal=29.95, polar=90.0, day=3),
]


@pytest.fixture(scope='module')
def analysis(tmp_path_factory):
    if not EXAMPLE.exists():
        pytest.skip('NbSn2 example data not available')
    qo = QuantumOscillationAnalysis(
        EXAMPLE / 'calc' / 'soc', output_dir=tmp_path_factory.mktemp('qo'), bands=list(range(11, 19)),
        sweep_angle='polar', fixed_angle=29.95,
        primitive_cell=[[0.5, 0.0, 1.690651], [0.5, 0.867802, 0.0], [0.0, 0.867802, 1.690651]],
        conventional_from_primitive=[[1, 1, -1], [-1, 1, 1], [1, -1, 1]],
        parent_bands={11: 6, 12: 6, 13: 7, 14: 7, 15: 8, 16: 8, 17: 9, 18: 9})  # fmt: skip
    qo.add_reference_run('SR', EXAMPLE / 'calc' / 'sr_81_320', bands=[6, 7, 8, 9])
    qo.field_directions(DIRECTIONS)
    qo.read_experiment(lk_table=EXAMPLE / 'data' / 'exp_LK_table.csv')
    return qo


def test_inventory(analysis):
    inventory = analysis.fermi_surface_inventory()
    assert inventory['groups'] == EXPECTED['degenerate_groups']
    assert not inventory['missing_bands'] and not inventory['empty_bands']
    pieces = dict(zip(inventory['pieces'].group, inventory['pieces'].pieces))
    assert pieces == EXPECTED['fermi_surface_pieces']


def test_digitisation_and_sweep(analysis):
    digitised = analysis.digitize_fft_map(
        EXAMPLE / 'data' / 'exp_torque_FFT_vs_angle_slide15.png', frequency_axis='x',
        frequency_pixels=(807, 2991), frequency_values=(10.0, 10000.0), angle_pixels=(1496, 272),
        angle_values=(0.0, 90.0), data_pixel_range=(630, 2991),
        colorbar=dict(x=3454, y0=1470, y1=228, vmax=0.001435), angles=list(range(0, 91, 5)),
        prominence=6.0e-5, min_frequency=15.0, pixel_nudges={'0': -4})  # fmt: skip
    assert len(digitised['peaks']) == EXPECTED['digitised_peaks']
    assert digitised['validation'].dev_pct.abs().median() == pytest.approx(
        EXPECTED['digitisation_median_abs_dev_pct'], abs=0.01)  # fmt: skip
    assert analysis.sweep_comparison(min_copies=50)['rate'] == pytest.approx(
        EXPECTED['sweep_rate_N50']
    )
    assert analysis.sweep_comparison(min_copies=100)['rate'] == pytest.approx(
        EXPECTED['sweep_rate_N100']
    )


def test_soc_vs_scalar_relativistic(analysis):
    shifts = analysis.compare_runs()['SR']['shifts']
    measured = {str(row.ref_band): row.median_shift_pct for row in shifts.itertuples()}
    assert measured == pytest.approx(EXPECTED['soc_minus_sr_median_shift_pct'], abs=0.01)


def test_fermi_level_scan(analysis):
    scan = analysis.fermi_level_scan(EXAMPLE / 'calc' / 'sr_band7_fermi_scan', bands=[7])
    assert len(scan['pairs']) == EXPECTED['fermi_scan_pairs']
    assert scan['median_ratio'] == pytest.approx(EXPECTED['fermi_scan_median_ratio'], abs=0.005)


def test_breakdown_gaps_and_weak_peaks(analysis):
    contacts = analysis.breakdown_gaps(bands=[11, 13, 15, 17])
    assert contacts.groupby('pair').k_gap_invA.min().to_dict() == pytest.approx(
        EXPECTED['min_k_gap_invA']
    )
    weak = analysis.weak_peak_analysis()
    assert weak['weak']['rate'] == pytest.approx(EXPECTED['weak_combination_rate'])
    assert weak['weak']['harmonics'] == EXPECTED['weak_combination_harmonics']
    assert weak['parents'] == EXPECTED['combination_parents']
