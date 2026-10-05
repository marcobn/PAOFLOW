"""Neighbour-shell ("star") geometry of the folded R grid.

A real-space cutoff can be given either as a radius in Bohr (``rcut``) or as
a neighbour-shell count (``bond_order``): keep every bond up to and
including the n-th distinct interatomic distance.  The shell form is the
natural one for a crystal.  Symmetry-equivalent bonds have the same length,
so a cutoff placed between two shells keeps or drops each star as a whole
and the truncated ``H(R)`` retains the point group of the lattice.  This
module turns a shell count into a radius (:func:`shell_cutoff`) and moves
an explicit radius into the gap above the shell it reaches
(:func:`snap_cutoff`), so neither form can cut through a shell;
:meth:`~PAOFLOW.sparse.hamiltonian.SparseHamiltonian.from_data_controller`
then applies the radius.

The bond carrying ``H_ij(R)`` has length ``|alat*R + tau_i - tau_j|``, the
same quantity ``from_data_controller`` cuts on (``Dnm_ij = tau_i - tau_j``
on the QE projection path).

Only distances the FFT supercell can represent unambiguously are counted.
The R grid is an ``nk1 x nk2 x nk3`` supercell, and a bond longer than the
inradius of that supercell (:func:`aliasing_safe_radius`) lands on the same
grid cell as a shorter periodic image of itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .hamiltonian import folded_R_triples

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController


def aliasing_safe_radius(lattice_vectors: np.ndarray, grid_shape: Sequence[int]) -> float:
    """Largest bond length representable on the FFT supercell without wrapping.

    Parameters
    ----------
    lattice_vectors : np.ndarray, shape (3, 3)
        Primitive lattice vectors in Bohr (rows).
    grid_shape : sequence of int
        The ``(nk1, nk2, nk3)`` grid, i.e. the supercell repeat.

    Returns
    -------
    float
        Inradius of the supercell in Bohr: half its smallest interplanar
        spacing.

    Notes
    -----
    Bonds longer than this can fold onto the same grid cell as a shorter
    bond and cannot be told apart from it.
    """
    supercell = np.asarray(lattice_vectors, dtype=float) * np.asarray(grid_shape)[:, None]
    dual = np.linalg.inv(supercell).T
    return 0.5 * float(np.min(1.0 / np.linalg.norm(dual, axis=1)))


def compute_star_shells(
    atomic_positions: np.ndarray,
    lattice_vectors: np.ndarray,
    grid_shape: Sequence[int],
    max_radius: float | None = None,
    distance_tol: float = 1.0e-3,
) -> np.ndarray:
    """Distinct interatomic distances (the neighbour shells) on the R grid.

    Parameters
    ----------
    atomic_positions : np.ndarray, shape (natoms, 3)
        Cartesian atomic positions in Bohr.
    lattice_vectors : np.ndarray, shape (3, 3)
        Primitive lattice vectors in Bohr (rows).
    grid_shape : sequence of int
        ``(nk1, nk2, nk3)``.
    max_radius : float, optional
        Search cutoff in Bohr.  Defaults to :func:`aliasing_safe_radius`.
    distance_tol : float, optional
        Distances closer than this are merged into one shell (Bohr).

    Returns
    -------
    np.ndarray, 1-D, float
        Sorted shell distances, excluding the on-site distance (zero).
    """
    if max_radius is None:
        max_radius = aliasing_safe_radius(lattice_vectors, grid_shape)

    tau = np.asarray(atomic_positions, dtype=float)
    Rcart = folded_R_triples(*grid_shape).astype(float) @ np.asarray(lattice_vectors, dtype=float)
    pair = tau[None, :, :] - tau[:, None, :]  # tau_j - tau_i
    dist = np.linalg.norm(pair[:, :, None, :] + Rcart[None, None, :, :], axis=3).ravel()
    dist = np.unique(np.round(dist[(dist > distance_tol) & (dist <= max_radius)], 6))
    return _shell_starts(dist, distance_tol)


def _shell_starts(dist: np.ndarray, distance_tol: float) -> np.ndarray:
    """Group sorted distances into shells and return the first of each.

    Parameters
    ----------
    dist : np.ndarray, 1-D
        Sorted, distinct distances (Bohr).
    distance_tol : float
        A distance within this of a shell's first member joins that shell.

    Returns
    -------
    np.ndarray, 1-D, float
        The smallest distance of every shell, in increasing order.

    Notes
    -----
    Every shell spans at most ``distance_tol``, and the next shell starts
    strictly more than ``distance_tol`` past the previous start, so
    ``start + distance_tol`` always lies in the gap above a shell.  The
    loop runs once per shell, not once per distance.
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
        Shell-merging tolerance (Bohr).  A shell starting within this of
        ``rcut`` counts as reached.

    Returns
    -------
    float
        ``start + distance_tol`` of the outermost shell starting at or below
        ``rcut + distance_tol``, or ``rcut`` itself if no bond is that short.

    Notes
    -----
    Symmetry-equivalent bonds have the same length only up to rounding
    (about 1e-12 Bohr, or the precision of the input positions).  A radius
    placed on a shell distance therefore keeps an arbitrary part of that
    shell and breaks the symmetry of the truncated ``H(R)``.  The snapped
    radius keeps or drops every shell whole, as ``bond_order`` does via
    :func:`shell_cutoff`.  A radius already inside a gap keeps the same
    bonds as before.

    The shells are built from ``distances`` themselves, the exact lengths
    the mask cuts on, so the snap holds beyond the aliasing-safe radius too
    (where :func:`compute_star_shells` stops counting).
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
    """Label each distance with its 1-based shell index (0 marks on-site).

    Parameters
    ----------
    distances : np.ndarray
        Bond lengths in Bohr.
    shell_distances : np.ndarray, 1-D
        Output of :func:`compute_star_shells`.
    distance_tol : float, optional
        Distances below this are on-site.

    Returns
    -------
    np.ndarray, int32
        Same shape as ``distances``.  A distance beyond the last shell gets
        the index of the nearest shell.
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
) -> tuple[float, np.ndarray, float]:
    """Radius that keeps every bond up to and including shell ``bond_order``.

    Parameters
    ----------
    data_controller : DataController
        Must carry ``tau``, ``a_vectors``, ``alat`` and ``nk1..nk3``.
    bond_order : int
        Neighbour shell to keep up to; ``1`` is nearest neighbours.
    distance_tol : float, optional
        Shell-merging tolerance, also added to the radius so the outermost
        shell is kept despite rounding (Bohr).

    Returns
    -------
    (cutoff, shells, safe_radius) : tuple
        The cutoff in Bohr, all representable shell distances, and the
        aliasing-safe radius of the grid.

    Raises
    ------
    ValueError
        If ``bond_order`` is not positive, or exceeds the number of shells
        the grid can represent.
    """
    arry, attr = data_controller.data_dicts()
    grid = (int(attr['nk1']), int(attr['nk2']), int(attr['nk3']))
    lattice = np.asarray(arry['a_vectors'], dtype=float) * float(attr['alat'])
    safe = aliasing_safe_radius(lattice, grid)
    shells = compute_star_shells(
        arry['tau'], lattice, grid, max_radius=safe, distance_tol=distance_tol
    )

    bond_order = int(bond_order)
    if bond_order < 1:
        raise ValueError(f'bond_order must be a positive shell count; got {bond_order}.')
    if bond_order > shells.size:
        raise ValueError(
            f'bond_order={bond_order} exceeds the {shells.size} neighbour shells representable '
            f'on the {grid} grid (aliasing-safe radius {safe:.3f} Bohr). Use a denser k-grid or '
            'a smaller bond_order.'
        )
    return float(shells[bond_order - 1]) + distance_tol, shells, safe
