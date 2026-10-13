import gzip
import shutil

import numpy as np
import pytest

from PAOFLOW.qo_analysis.skeaf_output import (
    fermi_tokens,
    find_frequency_files,
    frequency_file_inventory,
    read_bxsf_grid,
    read_frequency_table,
    read_orbits,
    sweep_angle_name,
)

pytestmark = pytest.mark.unit


def test_frequency_table_units_and_derived_columns(frequency_run):
    table = read_frequency_table(frequency_run / 'qo_EF_0_freqvsangle_2.out')
    assert table.freq_T.iloc[0] == pytest.approx(1200.0)  # kT -> T
    assert set(table.extremum) == {'min'}  # positive curvature
    assert set(table.carrier) == {'e'}
    assert table.amp_factor.iloc[0] == pytest.approx(300.0**-0.5)


def test_empty_file_is_reported_not_parsed(frequency_run):
    assert read_frequency_table(frequency_run / 'qo_EF_0_freqvsangle_3.out').empty
    inventory = frequency_file_inventory(frequency_run)
    assert inventory.set_index('band').loc[3, 'is_empty']
    assert not inventory.set_index('band').loc[1, 'is_empty']


def test_gzip_files_and_tokens(frequency_run):
    source = frequency_run / 'qo_EF_0_freqvsangle_1.out'
    for token in ('-0.005', '0.005'):
        with (
            open(source, 'rb') as plain,
            gzip.open(frequency_run / f'qo_EF_{token}_freqvsangle_1.out.gz', 'wb') as packed,
        ):
            shutil.copyfileobj(plain, packed)
    assert fermi_tokens(frequency_run) == ['-0.005', '0', '0.005']
    shifted = read_orbits(frequency_run, [1], '0.005')
    assert len(shifted) == 9 and set(shifted.band) == {1}


def test_groups_and_sweep_angle(frequency_run):
    orbits = read_orbits(frequency_run, None, '0', {1: '1,2', 2: '1,2'})
    assert set(orbits.group) == {'1,2'}
    assert sweep_angle_name(orbits) == 'polar'
    assert sorted(find_frequency_files(frequency_run)) == [1, 2, 3]


def test_bxsf_gzip_is_decompressed_once(pocket_bands, tmp_path):
    source = pocket_bands / 'Fermi_surf_band_1.bxsf'
    packed = tmp_path / 'Fermi_surf_band_1.bxsf.gz'
    with open(source, 'rb') as plain, gzip.open(packed, 'wb') as out:
        shutil.copyfileobj(plain, out)
    grid = read_bxsf_grid(packed, tmp_path / 'cache')
    assert grid['energies'].shape == (16, 16, 16)  # periodic end points dropped
    assert np.allclose(grid['recip_ang'], np.eye(3))
    assert (tmp_path / 'cache' / 'Fermi_surf_band_1.bxsf').exists()
