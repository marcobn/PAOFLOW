import pandas as pd
import pytest

from PAOFLOW.agentic.qo_analysis.orbit_tables import (
    angular_coverage,
    copy_count_histogram,
    merge_into_sheets,
    orbit_shifts,
    orbits_at_angle,
    per_angle_listing,
    robust_orbits,
)
from PAOFLOW.agentic.qo_analysis.skeaf_output import read_orbits

pytestmark = pytest.mark.unit


def orbit_rows(frequencies, group='1'):
    return pd.DataFrame(
        dict(
            group=group,
            freq_T=frequencies,
            mstar_abs=0.5,
            carrier='e',
            extremum='max',
            num_orbit_copies=150,
            amp_factor=0.1,
        )
    )


def test_robust_cut_removes_wall_artefacts(frequency_run):
    orbits = read_orbits(frequency_run)
    assert (robust_orbits(orbits, 50).num_orbit_copies >= 50).all()
    assert len(robust_orbits(orbits, 50)) == 9
    assert copy_count_histogram(orbits).entries.sum() == len(orbits)


def test_merge_compares_with_anchor_not_last_member():
    sheets = merge_into_sheets(orbit_rows([100.0, 105.0, 110.0, 115.0]), merge_fraction=0.06)
    assert sheets.freq_T.tolist() == pytest.approx([102.5, 112.5])


def test_orbits_at_angle_and_listing(frequency_run):
    orbits = robust_orbits(read_orbits(frequency_run))
    assert set(orbits_at_angle(orbits, 44.0).polar_deg) == {45.0}
    assert orbits_at_angle(orbits, 200.0).empty  # outside the sweep grid
    peaks = pd.DataFrame(dict(calc_angle=[45.0], freq_T=[545.0], rel_mag=[0.5]))
    listing = per_angle_listing(orbits, peaks, [45.0])
    assert '545' in listing and '545h1' in listing
    assert angular_coverage(orbits, 1000, 1300) == [0, 45, 90]


def test_shifts_between_runs(frequency_run):
    orbits = robust_orbits(read_orbits(frequency_run))
    shifted = orbits.assign(freq_T=orbits.freq_T * 1.02)
    shifts = orbit_shifts(orbits, shifted, [0.0, 45.0, 90.0])
    assert shifts.shift_pct.median() == pytest.approx(2.0)
