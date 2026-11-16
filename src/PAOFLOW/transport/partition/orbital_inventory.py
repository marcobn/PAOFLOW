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
    """Number of PAO orbitals on each atom, in the order of ``arrays['atoms']``.

    Parameters
    ----------
    arrays : dict
        Data-controller arrays: ``atoms``, ``tau`` (bohr) and, when available, ``atomic_basis``
        (the records of the internal projections) or ``species``.
    attributes : dict
        Data-controller attributes (``workpath``, ``savedir`` locate the UPF files).
    atoms : list of str
        Species label of every atom.
    total_orbitals : int
        ``nawf``.
    data_controller : DataController, optional
        Reads the pseudopotentials (POTCAR) of a VASP run when there is no basis.

    Returns
    -------
    list of int
        Orbitals per atom; they sum to ``total_orbitals``.

    Raises
    ------
    ValueError
        If the counts cannot be resolved or do not add up to ``total_orbitals``.

    Notes
    -----
    The basis of the internal projections (``atomic_basis``) is used first: each record carries
    the position of its atom, so any preset (``minimal``, ``standard``, ``extended``, explicit
    shells) is counted correctly, for QE and VASP alike. Without it (the ``projwfc.x`` route) the
    count comes from the pseudo-orbitals of each species (the UPF files, or the POTCAR of a VASP
    run), and without species metadata from an equal split.
    """
    basis_counts = _orbital_counts_from_basis(arrays, atoms, total_orbitals)
    if basis_counts is not None:
        return basis_counts
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


def _orbital_counts_from_basis(
    arrays: dict, atoms: list[str], total_orbitals: int, tolerance_bohr: float = 1.0e-6
) -> list[int] | None:
    """Orbitals per atom from the records of the internal projection basis.

    Parameters
    ----------
    arrays : dict
        Data-controller arrays with ``tau`` (bohr) and, optionally, ``atomic_basis``.
    atoms : list of str
        Species label of every atom.
    total_orbitals : int
        ``nawf``.
    tolerance_bohr : float, optional
        Largest distance between a record's centre and its atom.

    Returns
    -------
    list of int or None
        Orbitals per atom, or None when the basis does not describe ``total_orbitals`` orbitals
        (for example on the ``projwfc.x`` route). A Hamiltonian built by
        :meth:`PAOFLOW.doubling_Hamiltonian` (``nawf`` a multiple of the basis, the atoms repeated
        cell by cell) gets the counts of the basis cell repeated.

    Raises
    ------
    ValueError
        If a record sits on no atom (or on an atom of another species), or if the records are not
        grouped atom by atom in the order of ``atoms``, which the Hamiltonian blocks assume.
    """
    basis = arrays.get('atomic_basis')
    if basis is None or len(basis) == 0 or total_orbitals % len(basis) != 0:
        return None
    if 'tau' not in arrays:
        # Records alone: those of one atom are consecutive (same species and centre).
        counts = [0] * len(atoms)
        for site in _sites_of_basis_records(basis, atoms):
            counts[site] += 1
        return counts if sum(counts) == total_orbitals else None
    copies = total_orbitals // len(basis)
    if len(atoms) % copies != 0:
        return None
    # doubling_Hamiltonian repeats the cell: the basis then describes the first len(atoms)/copies
    # atoms, and every copy of the cell carries the same orbitals in the same order
    cell_atoms = len(atoms) // copies
    positions = np.asarray(arrays['tau'], dtype=float)[:cell_atoms]
    atom_of_orbital = []
    for index, record in enumerate(basis):
        distance = np.linalg.norm(positions - np.asarray(record['tau'], dtype=float), axis=1)
        matches = np.flatnonzero(distance < tolerance_bohr)
        if matches.size != 1 or str(atoms[matches[0]]) != str(record['atom']).strip():
            raise ValueError(
                f'Basis orbital {index} ({record["atom"]}) at {np.asarray(record["tau"])} bohr '
                'does not sit on exactly one atom of its species.'
            )
        atom_of_orbital.append(int(matches[0]))
    if np.any(np.diff(atom_of_orbital) < 0):
        raise ValueError(
            'The basis orbitals are not grouped atom by atom in the order of the atoms.'
        )
    if any(str(atoms[i]) != str(atoms[i % cell_atoms]) for i in range(len(atoms))):
        raise ValueError('The repeated cells do not repeat the species of the basis cell.')
    cell_counts = np.bincount(atom_of_orbital, minlength=cell_atoms).astype(int).tolist()
    return cell_counts * copies


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
