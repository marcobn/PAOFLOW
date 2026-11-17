"""Centres of the orbitals of ``HRs`` in bohr, for DFT runs and model Hamiltonians alike."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _atom_orbital_counts(arrays: dict, natoms: int) -> NDArray[np.int64] | None:
    """Orbitals per atom from the Slater-Koster model shells or from ``naw``, if available."""
    shells = arrays.get('shells')
    if (
        isinstance(shells, (list, tuple))
        and len(shells) == natoms
        and all(isinstance(shell, dict) and 'orbitals' in shell for shell in shells)
    ):
        return np.array([len(shell['orbitals']) for shell in shells], dtype=int)
    if arrays.get('naw') is not None and len(arrays['naw']) == natoms:
        return np.asarray(arrays['naw'], dtype=int)
    return None


def orbital_centres_bohr(arrays: dict, attributes: dict) -> NDArray[np.float64]:
    """Cartesian centre of every orbital (row) of ``HRs``, in bohr.

    Parameters
    ----------
    arrays : dict
        DataController arrays: ``HRs`` (its first dimension is the number of orbitals ``nawf``),
        and ``basis`` (the projection basis, centres in bohr) or ``tau`` (atomic or orbital
        positions), with ``shells`` (Slater-Koster models) or ``naw`` (orbitals per atom).
    attributes : dict
        DataController attributes: ``alat`` (bohr) and ``tau_units`` (``'alat'`` for the model
        loaders; absent, i.e. bohr, for DFT runs); ``natoms``.

    Returns
    -------
    ndarray, shape ``(nawf, 3)``
        The centres in bohr.

    Raises
    ------
    ValueError
        If no centre can be assigned to every orbital.

    Notes
    -----
    The sources, in order: the projection ``basis`` (one entry per orbital); ``tau`` with one row
    per orbital (the hopping-list and PythTB models); ``tau`` per atom expanded with the orbital
    counts of the Slater-Koster ``shells`` or of ``naw``, when they add up; ``tau`` per atom
    with ``nawf // natoms`` consecutive orbitals each (the former rule of ``add_ext_field``). A
    list that covers half of the orbitals is repeated once: the block-ordered spinors made by
    ``adhoc_spin_orbit`` (all orbitals spin up, then all spin down).
    """
    nawf = arrays['HRs'].shape[0]

    def full(per_orbital: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
        if per_orbital is None:
            return None
        if per_orbital.shape[0] == nawf:
            return per_orbital
        if 2 * per_orbital.shape[0] == nawf:
            return np.vstack([per_orbital, per_orbital])
        return None

    basis = arrays.get('basis')
    if basis is not None:
        centres = full(np.array([orbital['tau'] for orbital in basis], dtype=float))
        if centres is not None:
            return centres

    tau = np.atleast_2d(np.asarray(arrays['tau'], dtype=float))
    scale = float(attributes['alat']) if attributes.get('tau_units', 'bohr') == 'alat' else 1.0
    tau_bohr = tau * scale
    if tau.shape[0] == nawf:
        return tau_bohr

    natoms = tau.shape[0]
    counts = _atom_orbital_counts(arrays, natoms)
    if counts is not None:
        centres = full(np.repeat(tau_bohr, counts, axis=0))
        if centres is not None:
            return centres
    if nawf % natoms == 0:
        return np.repeat(tau_bohr, nawf // natoms, axis=0)
    raise ValueError(
        f'cannot assign a centre to each of the {nawf} orbitals from {natoms} positions'
    )
