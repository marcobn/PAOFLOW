import pytest

from PAOFLOW.agentic.qo_analysis.fermi_sheets import (
    bands_at_fermi_level,
    breakdown_field,
    degenerate_groups,
    fermi_surface_pieces,
    group_labels,
    sheet_contacts,
)

pytestmark = pytest.mark.unit


def test_crossing_bands_and_kramers_pairs(pocket_bands, tmp_path):
    windows = bands_at_fermi_level(pocket_bands, tmp_path / 'cache')
    assert windows.loc[windows.crosses_EF, 'band'].tolist() == [1, 2, 3]
    groups = degenerate_groups(windows)
    assert groups == [[1, 2], [3]]
    assert group_labels(groups) == {1: '1,2', 2: '1,2', 3: '3'}


def test_pocket_at_cell_corner_is_one_piece(pocket_bands, tmp_path):
    electron = fermi_surface_pieces(pocket_bands / 'Fermi_surf_band_1.bxsf', tmp_path / 'cache')
    hole = fermi_surface_pieces(pocket_bands / 'Fermi_surf_band_3.bxsf', tmp_path / 'cache')
    assert electron['kind'].startswith('electron') and electron['n_pieces'] == 1  # periodic merge
    assert hole['kind'].startswith('hole') and hole['n_pieces'] == 1


def test_breakdown_field_scale():
    assert breakdown_field(0.01) == pytest.approx(10.34, rel=1e-3)


def test_sheet_contact_recovers_known_gap(two_spheres, tmp_path):
    folder, gap = two_spheres
    contacts = sheet_contacts(folder, [1, 2], tmp_path / 'cache', upsampling=3, max_gap=0.05)
    assert len(contacts) == 1
    assert contacts.k_gap_invA.iloc[0] == pytest.approx(gap, abs=contacts.voxel_invA.iloc[0])
    assert contacts.B0_T.iloc[0] == pytest.approx(
        breakdown_field(contacts.k_gap_invA.iloc[0]), rel=1e-2
    )
