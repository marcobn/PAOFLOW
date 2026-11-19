"""Neighbour-shell ("star") geometry of the folded R grid.

A cutoff is a radius in Bohr (``rcut``) or a shell count (``bond_order``: keep every bond up to
the n-th distinct interatomic distance). Symmetry-equivalent bonds have equal length, so a
cutoff in the gap between shells keeps the point group of the truncated ``H(R)``.
:func:`shell_cutoff` turns a count into such a radius and :func:`snap_cutoff` moves an explicit
radius into the gap above its outermost shell; ``from_data_controller`` applies it to the bond
length ``|alat*R + tau_i - tau_j|``.

Only bonds the ``nk1 x nk2 x nk3`` supercell represents unambiguously count: longer than its
inradius (:func:`aliasing_safe_radius`) a bond aliases onto a shorter image, and within
``|tau_i - tau_j|`` of it a bond can be stored at a non-shortest image
(:func:`misplaced_bond_length`). The accepted cutoff, in either form, stays below both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .hamiltonian import _minus_R_index, folded_R_triples

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController


def aliasing_safe_radius(lattice_vectors: np.ndarray, grid_shape: Sequence[int]) -> float:
    """Inradius (Bohr) of the ``grid_shape`` supercell: half its smallest interplanar spacing.

    Longer bonds can fold onto the grid cell of a shorter one. ``lattice_vectors`` are rows,
    in Bohr.
    """
    supercell = np.asarray(lattice_vectors, dtype=float) * np.asarray(grid_shape)[:, None]
    dual = np.linalg.inv(supercell).T
    return 0.5 * float(np.min(1.0 / np.linalg.norm(dual, axis=1)))


def misplaced_bond_length(
    offsets: np.ndarray,
    lattice_vectors: np.ndarray,
    grid_shape: Sequence[int],
    distance_tol: float = 1.0e-3,
) -> float:
    """Shortest bond whose stored grid image is not its shortest image.

    Parameters
    ----------
    offsets : np.ndarray, shape (..., 3)
        Intra-cell offsets ``tau_i - tau_j`` (``Dnm``, Bohr); only distinct rows matter.
    lattice_vectors : np.ndarray, shape (3, 3)
        Primitive lattice vectors in Bohr (rows).
    grid_shape : sequence of int
        ``(nk1, nk2, nk3)``.
    distance_tol : float, optional
        Excess length (Bohr) still counted as the shortest image (shells would merge them).

    Returns
    -------
    float
        True length (Bohr) of that bond, ``inf`` if none. A cutoff at or beyond it keeps
        only part of its shell.

    Notes
    -----
    ``R`` is folded into ``[-nk/2, nk/2)`` from ``R`` alone and the mask measures
    ``|alat*R + tau_i - tau_j|`` there (or at the shorter ``(j, i, -R)`` partner), so within
    ``|tau_i - tau_j|`` of the safe radius one bond of a star can be measured at a longer
    image than the rest. Searching one supercell vector each way finds the shortest image of
    every bond the safe radius admits.
    """
    from itertools import product

    lattice = np.asarray(lattice_vectors, dtype=float)
    R_int = folded_R_triples(*grid_shape)
    Rcart = R_int.astype(float) @ lattice
    minus = _minus_R_index(R_int, grid_shape)
    supercell = lattice * np.asarray(grid_shape)[:, None]
    images = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float) @ supercell

    shortest_bad = np.inf
    for d in np.unique(np.asarray(offsets, dtype=float).reshape(-1, 3), axis=0):
        bond = Rcart + d
        stored = np.minimum(np.linalg.norm(bond, axis=1), np.linalg.norm(Rcart[minus] - d, axis=1))
        shortest = np.linalg.norm(bond[:, None, :] + images[None], axis=2).min(axis=1)
        bad = stored > shortest + distance_tol
        if bad.any():
            shortest_bad = min(shortest_bad, float(shortest[bad].min()))
    return shortest_bad


def compute_star_shells(
    atomic_positions: np.ndarray,
    lattice_vectors: np.ndarray,
    grid_shape: Sequence[int],
    distance_tol: float = 1.0e-3,
) -> np.ndarray:
    """Sorted neighbour-shell distances (Bohr) up to :func:`aliasing_safe_radius`, no on-site.

    ``atomic_positions`` (natoms, 3) and ``lattice_vectors`` (rows) in Bohr; distances within
    ``distance_tol`` (Bohr) merge into one shell.
    """
    max_radius = aliasing_safe_radius(lattice_vectors, grid_shape)

    tau = np.asarray(atomic_positions, dtype=float)
    Rcart = folded_R_triples(*grid_shape).astype(float) @ np.asarray(lattice_vectors, dtype=float)
    pair = tau[None, :, :] - tau[:, None, :]  # tau_j - tau_i
    dist = np.linalg.norm(pair[:, :, None, :] + Rcart[None, None, :, :], axis=3).ravel()
    dist = np.unique(np.round(dist[(dist > distance_tol) & (dist <= max_radius)], 6))
    return _shell_starts(dist, distance_tol)


def _shell_starts(dist: np.ndarray, distance_tol: float) -> np.ndarray:
    """First distance of each shell in sorted distinct ``dist``.

    A shell is every distance within ``distance_tol`` of its first member, so the next shell
    starts more than ``distance_tol`` past it and ``start + distance_tol`` is in the gap above.
    """
    starts = []
    i = 0
    while i < dist.size:
        starts.append(dist[i])
        i = int(np.searchsorted(dist, dist[i] + distance_tol, side='right'))
    return np.array(starts, dtype=float)


def snap_cutoff(distances: np.ndarray, rcut: float, distance_tol: float = 1.0e-3) -> float:
    """Move a radius into the gap above the outermost shell it reaches.

    Parameters
    ----------
    distances : np.ndarray
        Bond lengths the cutoff is compared against (Bohr), any shape.
    rcut : float
        Requested cutoff radius (Bohr).
    distance_tol : float, optional
        Shell-merging tolerance (Bohr); a shell starting within it of ``rcut`` is reached.

    Returns
    -------
    float
        ``start + distance_tol`` of the outermost reached shell, or ``rcut`` if none.

    Notes
    -----
    Equivalent bonds agree in length only to rounding (~1e-12 Bohr or the input precision),
    so a radius on a shell distance keeps an arbitrary part of it and breaks symmetry. A
    radius already in a gap keeps the same bonds. Shells come from ``distances`` themselves,
    so the snap also holds beyond the aliasing-safe radius.
    """
    rcut = float(rcut)
    dist = np.unique(np.ravel(distances))
    # shells starting past this are dropped whole, and the starts below
    # it do not depend on the distances above it
    dist = dist[dist <= rcut + distance_tol]
    if dist.size == 0:
        return rcut
    return float(_shell_starts(dist, distance_tol)[-1]) + distance_tol


def assign_shell_order(
    distances: np.ndarray, shell_distances: np.ndarray, distance_tol: float = 1.0e-3
) -> np.ndarray:
    """1-based index of the nearest shell for each distance (Bohr), 0 below ``distance_tol``.

    ``shell_distances`` is from :func:`compute_star_shells`; returns int32 of the shape of
    ``distances``, the last shell for anything beyond it.
    """
    shell_order = np.zeros(np.shape(distances), dtype=np.int32)
    offsite = distances > distance_tol
    if shell_distances.size == 0 or not offsite.any():
        return shell_order

    selected = distances[offsite]
    ins = np.searchsorted(shell_distances, selected)
    lower = np.clip(ins - 1, 0, shell_distances.size - 1)
    upper = np.clip(ins, 0, shell_distances.size - 1)
    nearer_upper = np.abs(shell_distances[upper] - selected) < np.abs(
        selected - shell_distances[lower]
    )
    shell_order[offsite] = np.where(nearer_upper, upper, lower) + 1
    return shell_order


def shell_cutoff(
    data_controller: DataController,
    bond_order: int,
    distance_tol: float = 1.0e-3,
) -> tuple[float, np.ndarray]:
    """Radius that keeps every bond up to and including shell ``bond_order``.

    Parameters
    ----------
    data_controller : DataController
        Must carry ``tau``, ``a_vectors``, ``alat`` and ``nk1..nk3``.
    bond_order : int
        Last shell kept; ``1`` is nearest neighbours.
    distance_tol : float, optional
        Shell-merging tolerance (Bohr), added to the radius so rounding keeps the last shell.

    Returns
    -------
    (cutoff, shells) : tuple
        The cutoff in Bohr and all representable shell distances.

    Raises
    ------
    ValueError
        If ``bond_order`` < 1 or exceeds the shells the grid can represent.
    """
    arry, attr = data_controller.data_dicts()
    grid = (int(attr['nk1']), int(attr['nk2']), int(attr['nk3']))
    lattice = np.asarray(arry['a_vectors'], dtype=float) * float(attr['alat'])
    shells = compute_star_shells(arry['tau'], lattice, grid, distance_tol=distance_tol)

    bond_order = int(bond_order)
    if bond_order < 1:
        raise ValueError(f'bond_order must be a positive shell count; got {bond_order}.')
    if bond_order > shells.size:
        raise ValueError(
            f'bond_order={bond_order} exceeds the {shells.size} neighbour shells representable '
            f'on the {grid} grid (aliasing-safe radius '
            f'{aliasing_safe_radius(lattice, grid):.3f} Bohr). Use a denser k-grid or '
            'a smaller bond_order.'
        )
    return float(shells[bond_order - 1]) + distance_tol, shells
