from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtomOrbitals:
    """The PAO orbitals of one atom.

    Attributes
    ----------
    atom_index : int
        1-based index of the atom in ``arrays['atoms']``.
    element : str
        Species label.
    position : ndarray, shape ``(3,)``
        Position in bohr.
    orbital_ranges : tuple of (int, int)
        1-based inclusive ranges of the atom's rows in the Hamiltonian. One range when the
        orbitals of every atom are consecutive; one per spin block when the Hamiltonian is laid
        out as ``[all spin up][all spin down]`` (``adhoc_spin_orbit``, ``j_to_lm_hamiltonian``).
    """

    atom_index: int
    element: str
    position: np.ndarray
    orbital_ranges: tuple[tuple[int, int], ...]

    @property
    def first_orbital(self) -> int:
        """Lowest 1-based Hamiltonian row of the atom."""
        return self.orbital_ranges[0][0]

    @property
    def orbitals(self) -> tuple[int, ...]:
        """Every 1-based Hamiltonian row of the atom, in increasing order."""
        return tuple(
            index for first, last in self.orbital_ranges for index in range(first, last + 1)
        )

    @property
    def dimension(self) -> int:
        return sum(last - first + 1 for first, last in self.orbital_ranges)


def block_atom_index(atoms: Sequence[AtomOrbitals]) -> np.ndarray:
    """Position in ``atoms`` of the atom of every orbital of their block.

    Parameters
    ----------
    atoms : sequence of AtomOrbitals
        The atoms of a Hamiltonian block (for example
        ``HamiltonianBlockPartition.central_atoms``).

    Returns
    -------
    ndarray of int, shape ``(sum of dimensions,)``
        Entry ``i`` is the position in ``atoms`` of the atom owning row ``i`` of the block. The
        block holds the rows of the atoms in increasing Hamiltonian order (the selectors of
        :func:`PAOFLOW.transport.partition.layers.resolve_layer_partition`), so with spin blocks
        the rows of one atom are not consecutive.
    """
    owners = sorted(
        (orbital, position) for position, atom in enumerate(atoms) for orbital in atom.orbitals
    )
    return np.array([position for _, position in owners], dtype=int)


@dataclass(frozen=True)
class HamiltonianBlockPartition:
    dim_c: int
    dim_l: int | None
    dim_r: int | None
    transport_direction: str
    selectors: dict[str, dict[str, str]]
    central_atoms: tuple[AtomOrbitals, ...] = ()  # sorted by their first Hamiltonian row
