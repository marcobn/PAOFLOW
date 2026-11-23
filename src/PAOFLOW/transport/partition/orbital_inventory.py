from __future__ import annotations

from pathlib import Path

import numpy as np

from PAOFLOW.DataController import DataController
from PAOFLOW.inputs.read_upf import UPF
from PAOFLOW.transport.partition.types import AtomOrbitals


def build_atom_orbital_inventory(data_controller: DataController) -> list[AtomOrbitals]:
    """The Hamiltonian rows of every atom, in the order of ``arrays['atoms']``.

    Parameters
    ----------
    data_controller : DataController
        Holds ``atoms``, ``tau``, ``nawf`` and, when available, the orbital records ``basis``.

    Returns
    -------
    list of AtomOrbitals
        One entry per atom; their ranges cover rows ``1..nawf`` exactly once.

    Raises
    ------
    ValueError
        If the orbital counts cannot be resolved or do not add up to ``nawf``.

    Notes
    -----
    The rows are laid out as ``[cell copy][spin block][atoms of the cell]``: one spin block unless
    the spin was added to a spin-less basis (``adhoc_spin_orbit``, or ``j_to_lm_hamiltonian``),
    which stacks the copy of the basis for spin down after that for spin up, and one cell copy
    unless ``doubling_Hamiltonian`` repeated the Hamiltonian (after any spin doubling).
    """
    arrays, attributes = data_controller.data_dicts()
    atoms = list(arrays['atoms'])
    positions = positions_bohr(arrays, attributes)
    total_orbitals = int(attributes['nawf'])
    spin_blocks, cell_atoms = _spin_blocks_and_cell_atoms(arrays, atoms, total_orbitals)
    copies = len(atoms) // cell_atoms
    block_size = total_orbitals // (spin_blocks * copies)
    orbital_counts = _orbital_counts_by_atom(
        arrays, attributes, atoms, total_orbitals // spin_blocks, data_controller
    )

    inventory: list[AtomOrbitals] = []
    offset = 0  # rows of the cell block before the atom
    for atom_index, (element, position, orbital_count) in enumerate(
        zip(atoms, positions, orbital_counts), start=1
    ):
        copy, site = divmod(atom_index - 1, cell_atoms)
        if site == 0:
            offset = 0
        first = copy * spin_blocks * block_size + offset + 1
        inventory.append(
            AtomOrbitals(
                atom_index=atom_index,
                element=str(element),
                position=np.array(position, dtype=float),
                orbital_ranges=tuple(
                    (first + spin * block_size, first + spin * block_size + orbital_count - 1)
                    for spin in range(spin_blocks)
                ),
            )
        )
        offset += orbital_count

    covered = sum(atom.dimension for atom in inventory)
    last = max(atom.orbital_ranges[-1][1] for atom in inventory)
    if covered != total_orbitals or last != total_orbitals:
        raise ValueError(
            'Resolved atom orbital count does not match PAOFLOW atomic wavefunction count: '
            f'{covered} != {total_orbitals}.'
        )

    return inventory


def _spin_blocks_and_cell_atoms(
    arrays: dict, atoms: list[str], total_orbitals: int
) -> tuple[int, int]:
    """Spin blocks of the Hamiltonian rows and atoms of the cell the basis describes.

    Parameters
    ----------
    arrays : dict
        Data-controller arrays, with the orbital records ``basis`` (or ``atomic_basis``) when
        available.
    atoms : list of str
        Species of every atom (repeated cell by cell after ``doubling_Hamiltonian``).
    total_orbitals : int
        ``nawf``.

    Returns
    -------
    tuple of int
        ``(spin_blocks, cell_atoms)``: 2 when the rows hold a spin-less basis twice,
        ``[up][down]``, else 1; and the number of atoms the basis records sit on (all atoms when
        the cell was not doubled).

    Notes
    -----
    With ``N`` records on ``n`` atoms and ``len(atoms) = c n`` (``c`` cell copies), the rows hold
    ``nawf / (c N)`` spin blocks. Without records, one block and one cell.
    """
    basis = _basis_records(arrays)
    if not basis:
        return 1, len(atoms)
    cell_atoms = len(set(_sites_of_basis_records(basis, None)))
    if cell_atoms == 0 or len(atoms) % cell_atoms != 0:
        return 1, len(atoms)
    copies = len(atoms) // cell_atoms
    spin_blocks, remainder = divmod(total_orbitals, copies * len(basis))
    if remainder != 0 or spin_blocks not in (1, 2):
        return 1, len(atoms)
    return spin_blocks, cell_atoms


def _basis_records(arrays: dict) -> list | None:
    """Orbital records of the Hamiltonian basis (``basis``, else ``atomic_basis``).

    ``basis`` is set on every route (internal projections, ``read_atomic_proj_QE``,
    ``j_to_lm_hamiltonian``, which replaces the ``j`` records by ``lm`` ones); ``atomic_basis``
    only by ``projections``, and it is not updated by ``j_to_lm_hamiltonian``.
    """
    for key in ('basis', 'atomic_basis'):
        records = arrays.get(key)
        if records is not None and len(records) > 0 and 'tau' in records[0]:
            return records
    return None


def positions_bohr(arrays: dict, attributes: dict) -> np.ndarray:
    """Atomic (or orbital) positions ``tau`` in bohr.

    Parameters
    ----------
    arrays : dict
        Data-controller arrays with ``tau``.
    attributes : dict
        Data-controller attributes: ``tau_units`` (``'alat'`` for the model loaders, absent
        for DFT runs, whose ``tau`` is in bohr) and ``alat`` (bohr).

    Returns
    -------
    ndarray, shape ``(n, 3)``
        The positions in bohr.
    """
    tau = np.asarray(arrays['tau'], dtype=float)
    if attributes.get('tau_units', 'bohr') == 'alat':
        return tau * float(attributes['alat'])
    return tau


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
        Data-controller arrays: ``atoms``, ``tau`` and, when available, the orbital records
        ``basis`` (or ``atomic_basis``) or ``species``.
    attributes : dict
        Data-controller attributes (``workpath``, ``savedir`` locate the UPF files; ``tau_units``
        and ``alat`` give the units of ``tau``).
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
    The orbital records (``basis``, else ``atomic_basis``) are used first: each record carries
    the position of its atom, so any preset (``minimal``, ``standard``, ``extended``, explicit
    shells) is counted correctly, for QE and VASP alike. Without it (the ``projwfc.x`` route) the
    count comes from the pseudo-orbitals of each species (the UPF files, or the POTCAR of a VASP
    run), and without species metadata from an equal split.
    """
    basis_counts = _orbital_counts_from_basis(arrays, attributes, atoms, total_orbitals)
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
    arrays: dict,
    attributes: dict,
    atoms: list[str],
    total_orbitals: int,
    tolerance_bohr: float = 1.0e-6,
) -> list[int] | None:
    """Orbitals per atom from the records of the internal projection basis.

    Parameters
    ----------
    arrays : dict
        Data-controller arrays with ``tau`` and, optionally, ``basis`` / ``atomic_basis``
        (centres in bohr).
    attributes : dict
        Data-controller attributes (``tau_units``, ``alat``: see `positions_bohr`).
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
    basis = _basis_records(arrays)
    if basis is None or total_orbitals % len(basis) != 0:
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
    positions = positions_bohr(arrays, attributes)[:cell_atoms]
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


def _sites_of_basis_records(atomic_basis: list, atoms: list[str] | None) -> list[int]:
    """Atom site of every basis record (records are stored atom by atom, in site order).

    With ``atoms`` the records must span exactly those atoms; ``None`` skips the check.
    """
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
    if atoms is not None and site + 1 != len(atoms):
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
