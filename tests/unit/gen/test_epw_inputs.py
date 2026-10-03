"""EPW input generation for the PAO electron-phonon route."""

import numpy as np
import pytest

from PAOFLOW.elphon.qe_elph_io import read_epw_ukk
from PAOFLOW.gen.epw_inputs import (
    epw_input,
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
