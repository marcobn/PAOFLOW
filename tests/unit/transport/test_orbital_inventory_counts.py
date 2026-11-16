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
