"""Unit tests for the ibrav detection wired into :mod:`PAOFLOW.gen.aflow_qe`.

These tests exercise the lattice-detection helpers and ``build_input`` without
any network access: synthetic CONTCAR text and a throwaway pseudopotential
folder are constructed in-process.
"""

from __future__ import annotations

import json

import numpy as np

from PAOFLOW.gen.aflow_qe import (
    build_input,
    cell_rows_to_matrix,
    detect_ibrav,
    format_celldm_lines,
    parse_contcar_qe,
    remap_atomic_positions,
    suggest_intersite_cutoff,
)
from PAOFLOW.inputs.lattice_format import BOHR_RADIUS_ANGS, lattice_format_QE


def _contcar_text(cell_rows, cell_unit, pos_rows, pos_unit):
    lines = ['CELL_PARAMETERS ({})'.format(cell_unit)]
    lines += ['  {:.12f} {:.12f} {:.12f}'.format(*r) for r in cell_rows]
    lines.append('ATOMIC_POSITIONS ({})'.format(pos_unit))
    lines += pos_rows
    return '\n'.join(lines) + '\n'


def _fcc_contcar(a_ang=5.43):
    """A 1-atom fcc CONTCAR (angstrom cell, crystal positions)."""
    a_bohr = a_ang / BOHR_RADIUS_ANGS
    lat_bohr = lattice_format_QE(2, [a_bohr, 0, 0, 0, 0, 0])
    cell_ang = lat_bohr * BOHR_RADIUS_ANGS
    text = _contcar_text(cell_ang, 'angstrom', ['Si 0.0 0.0 0.0'], 'crystal')
    return parse_contcar_qe(text)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_format_celldm_lines_slots():
    celldm = [5.0, 1.2, 1.7, 0.1, 0.2, 0.3]
    # fcc: only celldm(1)
    assert format_celldm_lines(2, celldm) == ['    celldm(1) = 5.0000000000,']
    # hexagonal: celldm(1) and celldm(3)
    lines = format_celldm_lines(4, celldm)
    assert lines[0].startswith('    celldm(1)')
    assert lines[1].startswith('    celldm(3)')
    # -12 monoclinic unique-axis-b: celldm(1,2,3,5)
    slots = [ln.split('(')[1].split(')')[0] for ln in format_celldm_lines(-12, celldm)]
    assert slots == ['1', '2', '3', '5']


def test_cell_rows_to_matrix_units():
    a_bohr = 7.0
    lat_bohr = lattice_format_QE(1, [a_bohr, 0, 0, 0, 0, 0])
    cell_ang = lat_bohr * BOHR_RADIUS_ANGS

    ang = parse_contcar_qe(_contcar_text(cell_ang, 'angstrom', ['Si 0 0 0'], 'crystal'))
    np.testing.assert_allclose(cell_rows_to_matrix(ang), lat_bohr, atol=1e-9)

    boh = parse_contcar_qe(_contcar_text(lat_bohr, 'bohr', ['Si 0 0 0'], 'crystal'))
    np.testing.assert_allclose(cell_rows_to_matrix(boh), lat_bohr, atol=1e-9)

    alat = parse_contcar_qe(_contcar_text(lat_bohr, 'alat', ['Si 0 0 0'], 'crystal'))
    assert cell_rows_to_matrix(alat) is None


def test_detect_ibrav_fcc():
    contcar = _fcc_contcar()
    aflow = {'Bravais_lattice_relax': 'FCC'}
    res = detect_ibrav(aflow, contcar, symprec=1e-4)
    assert res is not None
    assert res['ibrav'] == 2
    assert len(res['pos_rows']) == 1


def test_detect_ibrav_skew_returns_none():
    skew = [[5.0, 0.3, 0.1], [0.2, 5.5, 0.4], [0.1, 0.2, 6.0]]
    contcar = parse_contcar_qe(_contcar_text(skew, 'angstrom', ['Si 0 0 0'], 'crystal'))
    res = detect_ibrav({'Bravais_lattice_relax': 'CUB'}, contcar, symprec=1e-5)
    assert res is None


def test_remap_positions_preserves_geometry():
    a_ang = 5.43
    a_bohr = a_ang / BOHR_RADIUS_ANGS
    lat_bohr = lattice_format_QE(2, [a_bohr, 0, 0, 0, 0, 0])
    cell_ang = lat_bohr * BOHR_RADIUS_ANGS
    contcar = parse_contcar_qe(_contcar_text(cell_ang, 'angstrom', ['Si 0.1 0.2 0.3'], 'crystal'))
    # identity map: remap with M = I must reproduce the same fractional coords
    rows = remap_atomic_positions(contcar, lat_bohr, np.eye(3))
    vals = [float(x) for x in rows[0].split()[1:4]]
    np.testing.assert_allclose(vals, [0.1, 0.2, 0.3], atol=1e-9)


def test_remap_positions_minimum_image():
    # 0.75 is the primitive-cell value of the Si basis atom; it must be wrapped
    # to the minimum-image representative -0.25 so the bond stays in the cell.
    a_ang = 5.43
    a_bohr = a_ang / BOHR_RADIUS_ANGS
    lat_bohr = lattice_format_QE(2, [a_bohr, 0, 0, 0, 0, 0])
    cell_ang = lat_bohr * BOHR_RADIUS_ANGS
    contcar = parse_contcar_qe(
        _contcar_text(cell_ang, 'angstrom', ['Si 0.0 0.0 0.0', 'Si 0.75 0.75 0.75'], 'crystal')
    )
    rows = remap_atomic_positions(contcar, lat_bohr, np.eye(3))
    second = [float(x) for x in rows[1].split()[1:4]]
    np.testing.assert_allclose(second, [-0.25, -0.25, -0.25], atol=1e-9)
    # every coordinate must lie in the minimum-image range [-0.5, 0.5)
    for row in rows:
        for c in (float(x) for x in row.split()[1:4]):
            assert -0.5 <= c < 0.5


def test_suggest_intersite_cutoff_silicon():
    a_ang = 5.43
    a_bohr = a_ang / BOHR_RADIUS_ANGS
    lat_bohr = lattice_format_QE(2, [a_bohr, 0, 0, 0, 0, 0])
    frac = np.array([[0.0, 0.0, 0.0], [0.75, 0.75, 0.75]])
    d_nn, cutoff = suggest_intersite_cutoff(lat_bohr, frac)
    # nearest-neighbour Si-Si bond is sqrt(3)/4 * a; 2nd shell is a/sqrt(2).
    np.testing.assert_allclose(d_nn, np.sqrt(3) / 4 * a_ang, atol=1e-3)
    assert d_nn < cutoff < a_ang / np.sqrt(2)


# --------------------------------------------------------------------------- #
# build_input integration (synthetic pseudo folder, no network)
# --------------------------------------------------------------------------- #
def _make_pseudo_dir(tmp_path):
    (tmp_path / 'Si.upf').write_text('<PP_CHI l="0"/>\n<PP_CHI l="1"/>\n')
    (tmp_path / 'PeriodicTableJSON.json').write_text(
        json.dumps({'elements': [{'symbol': 'Si', 'atomic_mass': 28.085}]})
    )
    (tmp_path / 'reference.json').write_text(json.dumps({'Si': {'hn': 12.0}}))
    return str(tmp_path)


def _aflow():
    return {
        'compound': 'Si',
        'species': ['Si'],
        'composition': [1],
        'natoms': 1,
        'Egap': 0.6,
        'spinD': [],
        'kpoints_static': [8, 8, 8],
        'Bravais_lattice_relax': 'FCC',
    }


def test_build_input_auto_emits_ibrav(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path)
    text = build_input(
        _aflow(),
        _fcc_contcar(),
        pseudo_dir,
        soc=False,
        degauss=0.05,
        ibrav_mode='auto',
        symprec=1e-4,
    )
    assert 'ibrav = 2,' in text
    assert 'celldm(1)' in text
    assert 'ATOMIC_POSITIONS (crystal)' in text
    assert 'CELL_PARAMETERS' not in text


def test_build_input_mode_zero_keeps_cell(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path)
    text = build_input(
        _aflow(),
        _fcc_contcar(),
        pseudo_dir,
        soc=False,
        degauss=0.05,
        ibrav_mode='0',
        symprec=1e-4,
    )
    assert 'ibrav = 0,' in text
    assert 'CELL_PARAMETERS' in text
    assert 'celldm' not in text


def test_build_input_emits_cutoff_comment(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path)
    text = build_input(
        _aflow(),
        _fcc_contcar(),
        pseudo_dir,
        soc=False,
        degauss=0.05,
        ibrav_mode='auto',
        symprec=1e-4,
    )
    assert 'intersite V cutoff' in text
    # the comment block must precede the &control namelist and use '!'
    assert text.lstrip().startswith('!')
    assert text.index('intersite V cutoff') < text.index('&control')
