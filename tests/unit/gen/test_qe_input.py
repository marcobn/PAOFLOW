"""Unit tests for the database-agnostic QE-input generator.

These tests exercise the lattice-detection helpers, the C2DB/AFLOW adapters and
``build_qe_input`` without any network access: synthetic geometries and a
throwaway pseudopotential folder are constructed in-process.
"""

from __future__ import annotations

import json

import numpy as np

from PAOFLOW.gen.qe_input.record import MaterialRecord
from PAOFLOW.gen.qe_input.sources.aflow import parse_contcar_qe
from PAOFLOW.gen.qe_input.sources.c2db import (
    C2dbSource,
    _geometry_from_atoms,
    _ordered_species,
    resolve_uid,
    scrape_overview,
)
from PAOFLOW.gen.qe_input.writer import (
    build_qe_input,
    cell_rows_to_matrix,
    detect_ibrav,
    detect_ibrav_2d,
    format_celldm_lines,
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


def _fcc_geometry(a_ang=5.43):
    """A 1-atom fcc geometry (angstrom cell, crystal positions)."""
    a_bohr = a_ang / BOHR_RADIUS_ANGS
    lat_bohr = lattice_format_QE(2, [a_bohr, 0, 0, 0, 0, 0])
    cell_ang = lat_bohr * BOHR_RADIUS_ANGS
    text = _contcar_text(cell_ang, 'angstrom', ['Si 0.0 0.0 0.0'], 'crystal')
    return parse_contcar_qe(text)


def _record(
    geometry,
    *,
    species,
    bravais_hint=None,
    natoms=None,
    metallic=True,
    magnetic=False,
    dimensionality='3D',
    kpoints=None,
    energy_cutoff=None,
    compound='Si',
    source='aflow',
):
    return MaterialRecord(
        compound=compound,
        geometry=geometry,
        species=species,
        natoms=natoms if natoms is not None else len(geometry['atom_order']),
        metallic=metallic,
        magnetic=magnetic,
        dimensionality=dimensionality,
        kpoints=kpoints,
        energy_cutoff=energy_cutoff,
        bravais_hint=bravais_hint,
        spacegroup=None,
        source=source,
    )


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
    record = _record(_fcc_geometry(), species=[('Si', 1)], bravais_hint='FCC')
    res = detect_ibrav(record, symprec=1e-4)
    assert res is not None
    assert res['ibrav'] == 2
    assert len(res['pos_rows']) == 1


def test_detect_ibrav_skew_returns_none():
    skew = [[5.0, 0.3, 0.1], [0.2, 5.5, 0.4], [0.1, 0.2, 6.0]]
    geometry = parse_contcar_qe(_contcar_text(skew, 'angstrom', ['Si 0 0 0'], 'crystal'))
    record = _record(geometry, species=[('Si', 1)], bravais_hint='CUB')
    res = detect_ibrav(record, symprec=1e-5)
    assert res is None


def test_remap_positions_preserves_geometry():
    a_ang = 5.43
    a_bohr = a_ang / BOHR_RADIUS_ANGS
    lat_bohr = lattice_format_QE(2, [a_bohr, 0, 0, 0, 0, 0])
    cell_ang = lat_bohr * BOHR_RADIUS_ANGS
    geometry = parse_contcar_qe(_contcar_text(cell_ang, 'angstrom', ['Si 0.1 0.2 0.3'], 'crystal'))
    # identity map: remap with M = I must reproduce the same fractional coords
    rows = remap_atomic_positions(geometry, lat_bohr, np.eye(3))
    vals = [float(x) for x in rows[0].split()[1:4]]
    np.testing.assert_allclose(vals, [0.1, 0.2, 0.3], atol=1e-9)


def test_remap_positions_minimum_image():
    # 0.75 is the primitive-cell value of the Si basis atom; it must be wrapped
    # to the minimum-image representative -0.25 so the bond stays in the cell.
    a_ang = 5.43
    a_bohr = a_ang / BOHR_RADIUS_ANGS
    lat_bohr = lattice_format_QE(2, [a_bohr, 0, 0, 0, 0, 0])
    cell_ang = lat_bohr * BOHR_RADIUS_ANGS
    geometry = parse_contcar_qe(
        _contcar_text(cell_ang, 'angstrom', ['Si 0.0 0.0 0.0', 'Si 0.75 0.75 0.75'], 'crystal')
    )
    rows = remap_atomic_positions(geometry, lat_bohr, np.eye(3))
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
# C2DB adapter (offline)
# --------------------------------------------------------------------------- #
_GRAPHENE_ATOMS = {
    'numbers': [6, 6],
    'positions': [
        [6.5e-18, 3.5e-17, 7.4999999732],
        [1.2336156302, 0.7122283161, 7.5000000054],
    ],
    'cell': [
        [2.4672312604, -1.75e-20, 0.0],
        [-1.2336156302, 2.1366849485, 0.0],
        [1.1e-17, 0.0, 15.0],
    ],
    'pbc': [True, True, False],
}


def _graphene_record():
    geometry = _geometry_from_atoms(_GRAPHENE_ATOMS)
    return _record(
        geometry,
        species=_ordered_species(geometry['atom_order']),
        compound='2C-1',
        dimensionality='2D',
        source='c2db',
    )


def test_c2db_geometry_species_order():
    geometry = _geometry_from_atoms(_GRAPHENE_ATOMS)
    assert geometry['atom_order'] == ['C', 'C']
    assert _ordered_species(geometry['atom_order']) == [('C', 2)]


def test_c2db_resolve_and_match():
    src = C2dbSource()
    assert src.matches('2C-1')
    assert src.matches('https://c2db.fysik.dtu.dk/material/2C-1')
    assert resolve_uid('c2db:MoS2-165798be3c80') == 'MoS2-165798be3c80'
    assert resolve_uid('https://c2db.fysik.dtu.dk/material/2C-1') == '2C-1'


def test_c2db_scrape_overview():
    html = '<td>Band gap (PBE) [eV]</td><td>0.000</td> <td>Magnetic</td><td>No</td>'
    gap, magnetic = scrape_overview(html)
    assert gap == 0.0
    assert magnetic is False


def test_detect_ibrav_2d_graphene():
    res = detect_ibrav_2d(_graphene_record(), symprec=1e-4)
    assert res['ibrav'] == 4
    a_bohr = 2.4672312604 / BOHR_RADIUS_ANGS
    np.testing.assert_allclose(res['celldm'][0], a_bohr, rtol=1e-6)
    np.testing.assert_allclose(res['celldm'][2], 15.0 / 2.4672312604, rtol=1e-6)
    assert len(res['pos_rows']) == 2


# --------------------------------------------------------------------------- #
# build_qe_input integration (synthetic pseudo folder, no network)
# --------------------------------------------------------------------------- #
def _make_pseudo_dir(tmp_path, elements=('Si',), masses=None):
    masses = masses or {'Si': 28.085, 'C': 12.011}
    for el in elements:
        (tmp_path / '{}.upf'.format(el)).write_text('<PP_CHI l="0"/>\n<PP_CHI l="1"/>\n')
    (tmp_path / 'PeriodicTableJSON.json').write_text(
        json.dumps({'elements': [{'symbol': el, 'atomic_mass': masses[el]} for el in elements]})
    )
    (tmp_path / 'reference.json').write_text(json.dumps({el: {'hn': 12.0} for el in elements}))
    return str(tmp_path)


def test_build_input_auto_emits_ibrav(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path)
    record = _record(_fcc_geometry(), species=[('Si', 1)], bravais_hint='FCC', kpoints=(8, 8, 8))
    text = build_qe_input(
        record, pseudo_dir, soc=False, degauss=0.05, ibrav_mode='auto', symprec=1e-4
    )
    assert 'ibrav = 2,' in text
    assert 'celldm(1)' in text
    assert 'ATOMIC_POSITIONS (crystal)' in text
    assert 'CELL_PARAMETERS' not in text


def test_build_input_mode_zero_keeps_cell(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path)
    record = _record(_fcc_geometry(), species=[('Si', 1)], bravais_hint='FCC', kpoints=(8, 8, 8))
    text = build_qe_input(record, pseudo_dir, soc=False, degauss=0.05, ibrav_mode='0', symprec=1e-4)
    assert 'ibrav = 0,' in text
    assert 'CELL_PARAMETERS' in text
    assert 'celldm' not in text


def test_build_input_emits_cutoff_comment(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path)
    record = _record(_fcc_geometry(), species=[('Si', 1)], bravais_hint='FCC', kpoints=(8, 8, 8))
    text = build_qe_input(
        record, pseudo_dir, soc=False, degauss=0.05, ibrav_mode='auto', symprec=1e-4
    )
    assert 'intersite V cutoff' in text
    # the comment block must precede the &control namelist and use '!'
    assert text.lstrip().startswith('!')
    assert text.index('intersite V cutoff') < text.index('&control')


def test_build_input_2d_graphene(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path, elements=('C',))
    text = build_qe_input(
        _graphene_record(), pseudo_dir, soc=False, degauss=0.05, ibrav_mode='auto', symprec=1e-4
    )
    assert 'ibrav = 4,' in text
    assert "assume_isolated = '2D'" in text
    # out-of-plane k must collapse to a single point for a 2D system
    lines = text.splitlines()
    kline = lines[lines.index('K_POINTS {automatic}') + 1]
    assert kline.split()[2] == '1'


def _kpoint_line(text):
    lines = text.splitlines()
    return lines[lines.index('K_POINTS {automatic}') + 1]


def test_kpoints_use_source_grid_when_present(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path)
    record = _record(_fcc_geometry(), species=[('Si', 1)], bravais_hint='FCC', kpoints=(8, 8, 8))
    text = build_qe_input(record, pseudo_dir, soc=False, degauss=0.05, ibrav_mode='auto')
    assert _kpoint_line(text).split() == ['8', '8', '8', '0', '0', '0']
    assert 'CHECK K-POINT CONVERGENCE' not in text


def test_kpoints_default_metal_3d_with_caveat(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path)
    record = _record(_fcc_geometry(), species=[('Si', 1)], bravais_hint='FCC', metallic=True)
    text = build_qe_input(record, pseudo_dir, soc=False, degauss=0.05, ibrav_mode='auto')
    assert _kpoint_line(text).split() == ['18', '18', '18', '1', '1', '0']
    assert 'CHECK K-POINT CONVERGENCE' in text


def test_kpoints_default_insulator_2d(tmp_path):
    pseudo_dir = _make_pseudo_dir(tmp_path, elements=('C',))
    geometry = _geometry_from_atoms(_GRAPHENE_ATOMS)
    record = _record(
        geometry,
        species=_ordered_species(geometry['atom_order']),
        compound='2C-1',
        dimensionality='2D',
        metallic=False,
        source='c2db',
    )
    text = build_qe_input(record, pseudo_dir, soc=False, degauss=0.05, ibrav_mode='auto')
    assert _kpoint_line(text).split() == ['12', '12', '1', '0', '0', '0']
    assert 'CHECK K-POINT CONVERGENCE' in text
