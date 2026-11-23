"""Orbitals per atom for the transport partition: internal projection bases of any preset."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.transport.partition.orbital_inventory import build_atom_orbital_inventory

pytestmark = pytest.mark.unit


class DummyDataController:
    def __init__(self, arrays: dict, attributes: dict) -> None:
        self.arrays, self.attributes = arrays, attributes

    def data_dicts(self) -> tuple[dict, dict]:
        return self.arrays, self.attributes


def basis_records(
    atoms: list[str], positions: np.ndarray, orbitals_per_atom: list[int]
) -> list[dict]:
    return [
        {'atom': atom, 'tau': position.copy(), 'l': 0, 'm': 1, 'label': '3S'}
        for atom, position, count in zip(atoms, positions, orbitals_per_atom)
        for _ in range(count)
    ]


def test_internal_basis_counts_any_preset():
    atoms = ['Al', 'Si', 'Al']
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.8], [0.0, 0.0, 7.6]])
    counts = [13, 26, 13]  # e.g. Al standard, Si extended-like
    arrays = {
        'atoms': atoms,
        'tau': positions,
        'species': [('Al', 'Al.upf'), ('Si', 'Si.upf')],  # the UPF route would give 4 per atom
        'atomic_basis': basis_records(atoms, positions, counts),
    }
    inventory = build_atom_orbital_inventory(DummyDataController(arrays, {'nawf': sum(counts)}))
    assert [atom.dimension for atom in inventory] == counts
    assert [atom.orbital_ranges for atom in inventory] == [((1, 13),), ((14, 39),), ((40, 52),)]


def test_basis_of_another_size_falls_back():
    atoms = ['Al', 'Al']
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.8]])
    arrays = {
        'atoms': atoms,
        'tau': positions,
        'atomic_basis': basis_records(atoms, positions, [13, 13]),
    }
    inventory = build_atom_orbital_inventory(
        DummyDataController(arrays, {'nawf': 8})
    )  # projwfc-sized H
    assert [atom.dimension for atom in inventory] == [4, 4]


def test_basis_off_the_atoms_or_unordered_raises():
    atoms = ['Al', 'Al']
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.8]])
    records = basis_records(atoms, positions, [2, 2])
    records[1]['tau'] = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match='exactly one atom'):
        build_atom_orbital_inventory(
            DummyDataController(
                {'atoms': atoms, 'tau': positions, 'atomic_basis': records}, {'nawf': 4}
            )
        )
    records = basis_records(atoms, positions, [2, 2])
    records[0], records[3] = records[3], records[0]
    with pytest.raises(ValueError, match='atom by atom'):
        build_atom_orbital_inventory(
            DummyDataController(
                {'atoms': atoms, 'tau': positions, 'atomic_basis': records}, {'nawf': 4}
            )
        )


def test_model_positions_in_alat_units_reach_the_partition_in_bohr():
    """Model loaders keep tau in alat units (tau_units = 'alat'): the layers of a doubled model are
    alat apart along the transport axis, not 1 bohr (regression of the alat-unit doubling shift)."""
    import contextlib
    import io

    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import single_band_model

    model = single_band_model({(0, 0, 1): -1.0}, np.eye(3), 5.0, 1.0, (4, 4, 4))
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, verbose=False)
        p.doubling_Hamiltonian(0, 0, 3)
    assert p.data_controller.data_attributes['tau_units'] == 'alat'
    inventory = build_atom_orbital_inventory(p.data_controller)
    z = np.sort([atom.position[2] for atom in inventory])
    assert np.allclose(np.diff(np.unique(np.round(z, 8))), 5.0)
