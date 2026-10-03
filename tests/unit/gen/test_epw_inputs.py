"""EPW input generation for the PAO electron-phonon route."""

import numpy as np
import pytest

from PAOFLOW.elphon.qe_elph_io import read_epw_ukk
from PAOFLOW.gen.epw_inputs import (
    epw_input,
    exclude_bands_string,
    parse_exclude_bands,
    kpoints_card,
    uniform_kpoint_list,
    write_placeholder_ukk,
)
from PAOFLOW.inputs.read_QE_xml import uniform_grid_from_kpoints


def test_uniform_kpoint_list_is_full_grid_in_paoflow_order():
    rows = uniform_kpoint_list((4, 3, 2))
    assert rows.shape == (24, 4)
    assert np.all((rows[:, :3] >= 0) & (rows[:, :3] < 1))
    np.testing.assert_allclose(rows[:, 3].sum(), 1.0)
    assert uniform_grid_from_kpoints(rows[:, :3], np.eye(3)) == ((4, 3, 2), (0, 0, 0))


def test_kpoints_card_header():
    card = kpoints_card((2, 2, 2)).splitlines()
    assert card[0] == 'K_POINTS crystal' and card[1] == '8' and len(card) == 10


def test_placeholder_ukk_round_trips(tmp_path):
    path = tmp_path / 'pb.ukk'
    write_placeholder_ukk(str(path), nbnd=5, nk_total=8, nbndsub=3)
    out = read_epw_ukk(str(path), 8)
    assert (out['nbndep'], out['nbndskip'], out['nbnd'], out['nwann']) == (5, 0, 5, 3)
    np.testing.assert_array_equal(out['ibndkept'], np.arange(5))
    assert out['lwin'].all() and not out['exband'].any()
    np.testing.assert_array_equal(out['u'][0], np.eye(5, 3))


def test_placeholder_ukk_rejects_too_many_wannier(tmp_path):
    with pytest.raises(ValueError):
        write_placeholder_ukk(str(tmp_path / 'x.ukk'), nbnd=4, nk_total=1, nbndsub=5)


def test_epw_input_requires_commensurate_grids():
    text = epw_input('pb', [207.2], (6, 6, 6), (3, 3, 3), nbndsub=16)
    assert 'epbwrite    = .true.' in text and 'wannierize  = .false.' in text
    with pytest.raises(ValueError, match='divide'):
        epw_input('pb', [207.2], (9, 9, 9), (6, 6, 6), nbndsub=16)


def test_exclude_bands_string_compresses_ranges():
    assert exclude_bands_string([1, 2, 3, 4, 5]) == 'exclude_bands = 1:5'
    assert exclude_bands_string([8, 1, 2, 10, 11]) == 'exclude_bands = 1:2, 8, 10:11'


def test_placeholder_ukk_with_excluded_semicore_matches_epw_bookkeeping(tmp_path):
    """Pb: 16 bands, 14 electrons, semicore 1-5 excluded -> EPW wrote '11 5', bands 6-16."""
    path = tmp_path / 'pb.ukk'
    write_placeholder_ukk(str(path), nbnd=16, nk_total=8, exclude_bands=[1, 2, 3, 4, 5], nelec=14)
    out = read_epw_ukk(str(path), 8)
    assert (out['nbndep'], out['nbndskip'], out['nwann']) == (11, 5, 11)
    np.testing.assert_array_equal(out['ibndkept'], np.arange(5, 16))
    np.testing.assert_array_equal(out['exband'], np.arange(16) < 5)
    assert out['lwin'].shape == (8, 11) and out['lwin'].all()


def test_placeholder_ukk_counts_only_occupied_excluded_bands(tmp_path):
    path = tmp_path / 'x.ukk'
    write_placeholder_ukk(str(path), nbnd=10, nk_total=1, exclude_bands=[1, 9, 10], nelec=8)
    out = read_epw_ukk(str(path), 1)
    assert (out['nbndep'], out['nbndskip']) == (7, 1)  # bands 9, 10 are empty (nelec / 2 = 4)


def test_placeholder_ukk_requires_nelec_with_exclusion(tmp_path):
    with pytest.raises(ValueError, match='nelec'):
        write_placeholder_ukk(str(tmp_path / 'x.ukk'), nbnd=6, nk_total=1, exclude_bands=[1])


def test_epw_input_writes_bands_skipped():
    text = epw_input('pb', [207.2], (6, 6, 6), (6, 6, 6), nbndsub=11, exclude_bands=[1, 2, 3, 4, 5])
    assert "bands_skipped = 'exclude_bands = 1:5'" in text
    assert 'bands_skipped' not in epw_input('pb', [207.2], (6, 6, 6), (6, 6, 6), nbndsub=16)


def test_parse_exclude_bands():
    assert parse_exclude_bands('') == []
    assert parse_exclude_bands('1:5') == [1, 2, 3, 4, 5]
    assert parse_exclude_bands('exclude_bands = 1-3, 8 10:11') == [1, 2, 3, 8, 10, 11]
    with pytest.raises(ValueError):
        parse_exclude_bands('5:1')
