from __future__ import annotations

from pathlib import Path

import numpy as np

from PAOFLOW.DataController import DataController
from PAOFLOW.inputs.read_upf import UPF
from PAOFLOW.transport.partition.types import AtomOrbitals


def build_atom_orbital_inventory(data_controller: DataController) -> list[AtomOrbitals]:
    arrays, attributes = data_controller.data_dicts()
    atoms = list(arrays['atoms'])
    positions = np.asarray(arrays['tau'], dtype=float)
    total_orbitals = int(attributes['nawf'])
    orbital_counts = _orbital_counts_by_atom(
        arrays, attributes, atoms, total_orbitals, data_controller
    )

    inventory: list[AtomOrbitals] = []
    first_orbital = 1
    for atom_index, (element, position, orbital_count) in enumerate(
        zip(atoms, positions, orbital_counts), start=1
    ):
        last_orbital = first_orbital + orbital_count - 1
        inventory.append(
            AtomOrbitals(
                atom_index=atom_index,
                element=str(element),
                position=np.array(position, dtype=float),
                first_orbital=first_orbital,
                last_orbital=last_orbital,
            )
        )
        first_orbital = last_orbital + 1

    if inventory[-1].last_orbital != total_orbitals:
        raise ValueError(
            'Resolved atom orbital count does not match PAOFLOW atomic wavefunction count: '
            f'{inventory[-1].last_orbital} != {total_orbitals}.'
        )

    return inventory


def _orbital_counts_by_atom(
    arrays: dict,
    attributes: dict,
    atoms: list[str],
    total_orbitals: int,
    data_controller: DataController | None = None,
) -> list[int]:
    atomic_basis = arrays.get('atomic_basis')
    if atomic_basis:
        # the records of the projection basis, one per orbital, stored atom by atom: exact
        # for any basis route and both DFT codes
        counts = [0] * len(atoms)
        for site in _sites_of_basis_records(atomic_basis, atoms):
            counts[site] += 1
        if sum(counts) == total_orbitals:
            return counts
    if 'species' not in arrays:
        if total_orbitals % len(atoms) != 0:
            raise ValueError('Cannot infer per-atom orbital counts without species metadata.')
        return [total_orbitals // len(atoms)] * len(atoms)

    if data_controller is not None and attributes.get('dft', 'QE') == 'VASP':
        from PAOFLOW.inputs.pseudopotentials import load_pseudopotentials

        species_counts = {
            str(element): _count_orbitals(data)
            for element, data in load_pseudopotentials(data_controller).items()
        }
    else:
        species_counts = {
            str(element): _count_orbitals_from_pseudopotential(attributes, pseudo_file)
            for element, pseudo_file in arrays['species']
        }
    orbital_counts = [species_counts[str(atom)] for atom in atoms]
    if sum(orbital_counts) != total_orbitals:
        raise ValueError(
            'Pseudopotential-derived orbital count does not match QE atomic wavefunction count: '
            f'{sum(orbital_counts)} != {total_orbitals}.'
        )
    return orbital_counts


def _sites_of_basis_records(atomic_basis: list, atoms: list[str]) -> list[int]:
    """Atom site of every basis record (records are stored atom by atom, in site order)."""
    sites: list[int] = []
    site = -1
    for index, record in enumerate(atomic_basis):
        new_atom = index == 0 or (
            record['atom'] != atomic_basis[index - 1]['atom']
            or not np.array_equal(record['tau'], atomic_basis[index - 1]['tau'])
        )
        if new_atom:
            site += 1
        sites.append(site)
    if site + 1 != len(atoms):
        raise ValueError(
            'atomic_basis records span %d atoms but the run has %d' % (site + 1, len(atoms))
        )
    return sites


def _count_orbitals(upf) -> int:
    if upf.jchia:
        return sum(int(round(2.0 * float(j_value) + 1.0)) for j_value in upf.jchia)
    return sum(2 * int(shell_l) + 1 for shell_l in upf.shells)


def _count_orbitals_from_pseudopotential(attributes: dict, pseudo_file: str) -> int:
    pseudo_path = _find_pseudopotential(attributes, pseudo_file)
    return _count_orbitals(UPF(str(pseudo_path)))


def _find_pseudopotential(attributes: dict, pseudo_file: str) -> Path:
    workpath = Path(attributes.get('workpath', '.'))
    savedir = Path(attributes.get('savedir', '.'))
    candidates = [
        workpath / savedir / pseudo_file,
        savedir / pseudo_file,
        workpath / pseudo_file,
        Path(pseudo_file),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not locate pseudopotential file {pseudo_file!r}.')
