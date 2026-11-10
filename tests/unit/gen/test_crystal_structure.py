"""Crystal structure of a working folder (Setup tab): readers of saves and pw.x inputs, figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.gen.crystal_structure import (
    BOHR_ANGSTROM,
    element_of,
    folder_structure,
    structure_figure,
    structure_from_pw_input,
    structure_from_save,
)

pytestmark = pytest.mark.unit

EXAMPLE = Path(__file__).resolve().parents[3] / 'examples' / 'qe_examples' / 'example01'

HEADER = """&control
  calculation = 'scf'
/
&system
  %s
  nat = 2, ntyp = 2, ecutwfc = 40
/
&electrons
/
ATOMIC_SPECIES
Ga 69.7 Ga.upf
N 14.0 N.upf
"""


def _write(tmp_path, name, system, cards):
    path = tmp_path / name
    path.write_text(HEADER % system + cards + 'K_POINTS automatic\n4 4 4 0 0 0\n')
    return str(path)


def test_save_and_input_of_the_same_run_agree():
    save = structure_from_save(str(EXAMPLE / 'silicon.save'))
    text = structure_from_pw_input(str(EXAMPLE / 'scf.in'))
    assert save['species'] == text['species'] == ['Si', 'Si']
    assert np.allclose(save['cell'], text['cell'], atol=1e-4)
    assert np.allclose(save['positions'], text['positions'], atol=1e-4)
    assert np.isclose(np.linalg.norm(save['cell'][0]), 10.599478 * BOHR_ANGSTROM / 2**0.5)


def test_input_lattice_and_position_conventions(tmp_path):
    a, c = 3.19, 5.19
    hexagonal = _write(
        tmp_path, 'abc.in', 'ibrav = 4, A = %g, C = %g' % (a, c),
        'ATOMIC_POSITIONS crystal\nGa 0.3333333333 0.6666666667 0.0\nN 0.3333333333 0.6666666667 0.377\n',
    )  # fmt: skip
    vectors = 'CELL_PARAMETERS angstrom\n%g 0 0\n%g %g 0\n0 0 %g\n' % (
        a, -a / 2, a * 3**0.5 / 2, c,
    )  # fmt: skip
    explicit = _write(
        tmp_path, 'explicit.in', 'ibrav = 0',
        vectors + 'ATOMIC_POSITIONS {angstrom}\nGa %.10f %.10f 0.0\nN %.10f %.10f %.10f\n'
        % (0.0, a / 3**0.5, 0.0, a / 3**0.5, 0.377 * c),
    )  # fmt: skip
    one, two = structure_from_pw_input(hexagonal), structure_from_pw_input(explicit)
    assert np.allclose(one['cell'], two['cell'], atol=1e-6)
    assert np.allclose(one['positions'], two['positions'], atol=1e-6)
    # CELL_PARAMETERS in alat with celldm(1), positions in bohr
    alat = a / BOHR_ANGSTROM
    in_alat = _write(
        tmp_path, 'alat.in', 'ibrav = 0, celldm(1) = %.10f' % alat,
        'CELL_PARAMETERS alat\n1 0 0\n-0.5 %.10f 0\n0 0 %.10f\n' % (3**0.5 / 2, c / a)
        + 'ATOMIC_POSITIONS bohr\nGa 0 %.10f 0\nN 0 %.10f %.10f\n'
        % (alat / 3**0.5, alat / 3**0.5, 0.377 * c / BOHR_ANGSTROM),
    )  # fmt: skip
    three = structure_from_pw_input(in_alat)
    assert np.allclose(three['cell'], two['cell'], atol=1e-6)
    assert np.allclose(three['positions'], two['positions'], atol=1e-6)
    # what is not read
    assert structure_from_pw_input(str(tmp_path / 'missing.in')) is None
    assert structure_from_pw_input(_write(tmp_path, 'none.in', 'ibrav = 0', '')) is None


def test_figure_counts_cells_bonds_and_species(tmp_path):
    structure = structure_from_save(str(EXAMPLE / 'silicon.save'))
    one = structure_figure(structure, 1)
    assert one['formula'] == 'Si2' and one['nat'] == 2 and one['repeat'] == 1
    # the two atoms and the seven images of the one at the origin on the corners
    assert one['shown'] == 9
    assert one['species'] == [{'label': 'Si', 'element': 'Si', 'color': '#f0c8a0', 'count': 2}]
    assert np.isclose(one['cell']['alpha'], 60.0) and np.isclose(
        one['cell']['volume'], 44.116, atol=1e-3
    )
    names = [trace['name'] for trace in one['figure']['data']]
    assert names == ['cell', 'bonds', 'Si', 'axes']
    bonds = one['figure']['data'][1]['x']
    assert (
        bonds.count(None) == 32
    )  # every atom of the block with its four neighbours, whole or half
    two = structure_figure(structure, 2)
    assert two['shown'] == 35 and two['repeat'] == 2
    # a large cell is shown with fewer copies
    big = {
        'cell': np.eye(3) * 13.0,
        'species': ['Nb', 'Xq'] * 64,
        'positions': np.random.default_rng(1).random((128, 3)) * 13.0,
    }
    figure = structure_figure(big, 3)
    assert figure['repeat'] == 2 and figure['species'][1]['color'] == '#ff1493'
    assert element_of('Fe1') == element_of('Fe_up') == 'Fe' and element_of('Xq') == 'Xq'
    # a folder: a save or an input
    assert folder_structure(str(EXAMPLE), 'silicon.save')['kind'] == 'save'
    assert folder_structure(str(EXAMPLE), 'scf.in')['kind'] == 'pw.x input'
    assert folder_structure(str(tmp_path), 'nothing.in') is None
