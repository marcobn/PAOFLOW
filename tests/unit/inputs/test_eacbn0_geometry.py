"""Round-trip test: ``eACBN0._geometry_from_cards`` with ``ibrav != 0``.

Verifies that reconstructing the cell from ``ibrav`` + ``celldm`` (no
``CELL_PARAMETERS`` card) yields the same lattice and Cartesian positions as
the equivalent ``ibrav = 0`` template with an explicit ``CELL_PARAMETERS``
card.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.ACBN0 import eACBN0
from PAOFLOW.inputs.lattice_format import BOHR_RADIUS_ANGS, lattice_format_QE


def _make(blocks, cards):
    """Build an eACBN0 instance without running the heavy __init__."""
    obj = eACBN0.__new__(eACBN0)
    obj.blocks = blocks
    obj.cards = cards
    return obj


def test_geometry_ibrav_matches_explicit_cell():
    a_bohr = 7.5  # celldm(1) in Bohr
    celldm = np.array([a_bohr, 0.0, 0.0, 0.0, 0.0, 0.0])
    lattice_ang = lattice_format_QE(2, celldm) * BOHR_RADIUS_ANGS  # fcc

    # Two atoms in crystal coordinates.
    crystal = [('Ni', [0.0, 0.0, 0.0]), ('O', [0.5, 0.5, 0.5])]
    atomic_pos = ['ATOMIC_POSITIONS (crystal)'] + [f'{s} {c[0]} {c[1]} {c[2]}' for s, c in crystal]

    # ibrav != 0 template: no CELL_PARAMETERS card.
    obj_ibrav = _make(
        blocks={'system': {'ibrav': '2', 'celldm(1)': str(a_bohr), 'nat': '2', 'ntyp': '2'}},
        cards={'ATOMIC_POSITIONS': list(atomic_pos)},
    )

    # ibrav = 0 template: explicit CELL_PARAMETERS in angstrom.
    cell_card = ['CELL_PARAMETERS (angstrom)'] + [f'{v[0]} {v[1]} {v[2]}' for v in lattice_ang]
    obj_cell = _make(
        blocks={'system': {'ibrav': '0', 'nat': '2', 'ntyp': '2'}},
        cards={'CELL_PARAMETERS': cell_card, 'ATOMIC_POSITIONS': list(atomic_pos)},
    )

    lat_i, pos_i, sp_i = obj_ibrav._geometry_from_cards()
    lat_c, pos_c, sp_c = obj_cell._geometry_from_cards()

    np.testing.assert_allclose(lat_i, lat_c, atol=1e-10)
    np.testing.assert_allclose(pos_i, pos_c, atol=1e-10)
    assert sp_i == sp_c == ['Ni', 'O']


def test_geometry_ibrav_with_A_convention():
    # A in Ångström; orthorhombic ibrav=8.
    a_ang = 4.0
    b_ang = 5.2
    c_ang = 6.8
    obj = _make(
        blocks={'system': {'ibrav': '8', 'a': str(a_ang), 'b': str(b_ang), 'c': str(c_ang)}},
        cards={
            'ATOMIC_POSITIONS': [
                'ATOMIC_POSITIONS (crystal)',
                'Fe 0.0 0.0 0.0',
            ]
        },
    )
    lat, pos, sp = obj._geometry_from_cards()
    expected = np.array([[a_ang, 0, 0], [0, b_ang, 0], [0, 0, c_ang]])
    np.testing.assert_allclose(lat, expected, atol=1e-10)
    np.testing.assert_allclose(pos, [[0, 0, 0]], atol=1e-10)
    assert sp == ['Fe']


def test_geometry_ibrav0_without_cell_raises():
    obj = _make(
        blocks={'system': {'ibrav': '0'}},
        cards={'ATOMIC_POSITIONS': ['ATOMIC_POSITIONS (crystal)', 'Fe 0 0 0']},
    )
    with pytest.raises(ValueError):
        obj._geometry_from_cards()


def test_geometry_missing_ibrav_and_cell_raises():
    obj = _make(
        blocks={'system': {'nat': '1'}},
        cards={'ATOMIC_POSITIONS': ['ATOMIC_POSITIONS (crystal)', 'Fe 0 0 0']},
    )
    with pytest.raises(ValueError):
        obj._geometry_from_cards()
