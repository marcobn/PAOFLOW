"""Unit tests for :func:`PAOFLOW.inputs.read_pao_output.read_band_path_PAO`."""

from __future__ import annotations

from PAOFLOW.inputs.read_pao_output import read_band_path_PAO


def test_jump_label_sits_on_the_jump(tmp_path):
    # fcc path of Tutorial 01: ... W-L-K | U-X, the jump is between K and U
    kpath = tmp_path / 'kpath_points.txt'
    kpath.write_text(
        'gG 312\nX 156\nW 111\nK 331\ngG 270\nL 191\nU 111\nW 221\nL 191\nK 0\nU 111\nX 0\n\n'
        '0.0 0.0 0.0\n'
    )
    index, labels = read_band_path_PAO(str(kpath))
    assert labels == [r'$\Gamma$', 'X', 'W', 'K', r'$\Gamma$', 'L', 'U', 'W', 'L', 'K|U', 'X']
    assert index == [0, 312, 468, 579, 910, 1180, 1371, 1482, 1703, 1894, 2005]
