"""Per-atom orbital counts of the transport partition from the projection basis records."""

from __future__ import annotations

import numpy as np

from PAOFLOW.transport.partition.orbital_inventory import _orbital_counts_by_atom


def _records(atoms, taus, shells):
    records = []
    for atom, tau in zip(atoms, taus):
        for l in shells[atom]:
            for m in range(2 * l + 1):
                records.append(
                    {'atom': atom, 'tau': np.asarray(tau), 'l': l, 'm': m + 1, 'label': 'x'}
                )
    return records


def test_counts_follow_the_basis_records_atom_by_atom():
    atoms = ['Ga', 'As', 'Ga']
    taus = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    arrays = {'atomic_basis': _records(atoms, taus, {'Ga': [2, 0, 1], 'As': [0, 1]})}
    counts = _orbital_counts_by_atom(arrays, {'dft': 'QE'}, atoms, 9 + 4 + 9)
    assert counts == [9, 4, 9]


def test_two_consecutive_atoms_of_one_species_are_told_apart_by_their_position():
    atoms = ['Si', 'Si']
    taus = [[0.0, 0.0, 0.0], [1.3, 1.3, 1.3]]
    arrays = {'atomic_basis': _records(atoms, taus, {'Si': [0, 1]})}
    assert _orbital_counts_by_atom(arrays, {}, atoms, 8) == [4, 4]


class _Controller:
    """Minimal stand-in for DataController (``data_dicts`` only)."""

    def __init__(self, arrays: dict, attributes: dict) -> None:
        self._arrays, self._attributes = arrays, attributes

    def data_dicts(self) -> tuple[dict, dict]:
        return self._arrays, self._attributes


def _gaas_controller(spin_blocks: int, copies: int = 1) -> _Controller:
    """Ga (s, p, d: 9 rows) and As (s, p: 4 rows) on a cell, spin added after projection."""
    atoms = ['Ga', 'As']
    taus = [[0.0, 0.0, 0.0], [2.67, 2.67, 2.67]]
    basis = _records(atoms, taus, {'Ga': [0, 1, 2], 'As': [0, 1]})
    shift = np.array([0.0, 0.0, 10.0])
    all_taus = np.array([np.asarray(t) + c * shift for c in range(copies) for t in taus])
    arrays = {'atoms': atoms * copies, 'tau': all_taus, 'basis': basis}
    return _Controller(arrays, {'nawf': 13 * spin_blocks * copies})


def test_spin_block_rows_of_adhoc_spin_orbit():
    from PAOFLOW.transport.partition.orbital_inventory import build_atom_orbital_inventory

    inventory = build_atom_orbital_inventory(_gaas_controller(spin_blocks=2))
    assert [atom.orbital_ranges for atom in inventory] == [((1, 9), (14, 22)), ((10, 13), (23, 26))]


def test_cell_doubled_after_spin_orbit_repeats_both_spin_blocks():
    from PAOFLOW.transport.partition.orbital_inventory import build_atom_orbital_inventory

    inventory = build_atom_orbital_inventory(_gaas_controller(spin_blocks=2, copies=2))
    assert [atom.orbital_ranges for atom in inventory] == [
        ((1, 9), (14, 22)),
        ((10, 13), (23, 26)),
        ((27, 35), (40, 48)),
        ((36, 39), (49, 52)),
    ]


def test_without_spin_blocks_every_atom_keeps_one_range():
    from PAOFLOW.transport.partition.orbital_inventory import build_atom_orbital_inventory

    inventory = build_atom_orbital_inventory(_gaas_controller(spin_blocks=1, copies=2))
    assert [atom.orbital_ranges for atom in inventory] == [
        ((1, 9),),
        ((10, 13),),
        ((14, 22),),
        ((23, 26),),
    ]


def test_layer_selectors_and_block_order_with_spin_blocks():
    from PAOFLOW.transport.partition.layers import resolve_layer_partition
    from PAOFLOW.transport.partition.types import block_atom_index

    controller = _gaas_controller(spin_blocks=2, copies=2)
    partition = resolve_layer_partition(
        controller, left_lead_layers=2, right_lead_layers=2, transport_direction='z'
    )
    selectors = partition.selectors
    assert partition.dim_c == 52 and partition.dim_l == 26
    assert selectors['H00_C']['rows'] == '1-52'
    assert selectors['H00_L']['rows'] == '1-26'
    assert selectors['H01_L']['rows'] == '27-52'
    # rows of the conductor block: Ga up, As up, Ga down, As down, then the second cell
    owner = block_atom_index(partition.central_atoms)
    expected = np.concatenate([np.full(n, i) for i, n in [(0, 9), (1, 4), (0, 9), (1, 4)]])
    assert np.array_equal(owner, np.concatenate([expected, expected + 2]))
