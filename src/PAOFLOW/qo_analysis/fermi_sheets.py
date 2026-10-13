"""Fermi-sheet bookkeeping from BXSF grids: bands crossing E_F, degenerate groups,
disconnected pockets, and near-contacts between sheets (magnetic-breakdown gaps)."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

from .skeaf_output import RYDBERG_IN_EV, find_bxsf_files, read_bxsf_grid

HBAR_OVER_E = 6.582119569e-16
BREAKDOWN_FIELD_PREFACTOR = 0.5 * np.pi * HBAR_OVER_E * 1e20  # T A^2 (~1.034e5)


def breakdown_field(k_gap: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Breakdown field for a k-space gap.

    Parameters
    ----------
    k_gap : float or ndarray
        Gap between two Fermi sheets in A^-1.

    Returns
    -------
    float or ndarray
        :math:`B_0` in tesla.

    Notes
    -----
    Chambers' estimate with an order-unity geometric factor,

    .. math::

        B_0 \\simeq \\frac{\\pi}{2}\\frac{\\hbar}{e} k_g^2,

    and the tunnelling probability at field :math:`B` is :math:`P = e^{-B_0/B}`.
    """
    return BREAKDOWN_FIELD_PREFACTOR * np.asarray(k_gap) ** 2


def bands_at_fermi_level(bxsf_dir: str | Path, cache_dir: str | Path) -> pd.DataFrame:
    """Energy window of every BXSF band and whether it crosses the Fermi energy.

    Returns
    -------
    pandas.DataFrame
        ``band``, ``source_band``, ``e_min_Ry``, ``e_max_Ry``, ``e_min_eV``, ``e_max_eV``,
        ``occupied_fraction``, ``crosses_EF``, ``grid``.
    """
    rows = []
    for band, path in find_bxsf_files(bxsf_dir).items():
        grid = read_bxsf_grid(path, cache_dir)
        energies = grid['energies']
        rows.append(
            dict(
                band=band,
                source_band=grid['source_band'],
                e_min_Ry=energies.min(),
                e_max_Ry=energies.max(),
                e_min_eV=energies.min() * RYDBERG_IN_EV,
                e_max_eV=energies.max() * RYDBERG_IN_EV,
                occupied_fraction=float((energies < grid['fermi_energy']).mean()),
                crosses_EF=bool(energies.min() < grid['fermi_energy'] < energies.max()),
                grid='x'.join(map(str, grid['shape'])),
            )
        )
    return pd.DataFrame(rows)


def degenerate_groups(band_windows: pd.DataFrame, tolerance_Ry: float = 1e-3) -> list[list[int]]:
    """Group consecutive E_F-crossing bands whose energy windows coincide.

    Notes
    -----
    With spin-orbit coupling in a centrosymmetric, non-magnetic crystal every band is
    a Kramers pair: both members describe one Fermi surface, and their residual
    frequency difference is a free numerical error bar.
    """
    crossing = band_windows.loc[band_windows.crosses_EF].set_index('band').sort_index()
    groups, current = [], []
    for band, window in crossing.iterrows():
        if current:
            previous = crossing.loc[current[-1]]
            same = (
                abs(previous.e_min_Ry - window.e_min_Ry) < tolerance_Ry
                and abs(previous.e_max_Ry - window.e_max_Ry) < tolerance_Ry
            )
            if same:
                current.append(band)
                continue
            groups.append(current)
        current = [band]
    if current:
        groups.append(current)
    return groups


def group_labels(groups: list[list[int]]) -> dict[int, str]:
    """Band index to group label, e.g. ``{11: '11,12', 12: '11,12'}``."""
    return {band: ','.join(map(str, group)) for group in groups for band in group}


def _periodic_labels(region: NDArray[np.bool_]) -> NDArray[np.int64]:
    labels, count = ndimage.label(region)
    parent = list(range(count + 1))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for axis in range(3):
        for first, last in zip(
            np.take(labels, 0, axis=axis).ravel(), np.take(labels, -1, axis=axis).ravel()
        ):
            if first and last:
                parent[root(first)] = root(last)
    roots = np.array([root(node) for node in range(count + 1)])
    merged = roots[labels]
    merged[~region] = 0
    return merged


def fermi_surface_pieces(bxsf_path: str | Path, cache_dir: str | Path) -> dict:
    """Disconnected pieces of the minority region (electron or hole pockets), periodic in k.

    Returns
    -------
    dict
        ``kind``, ``n_pieces``, ``sizes_voxels`` (largest first) and ``occupied_fraction``.
        The number of qo rows per band is the number of extremal orbits, not of pieces.
    """
    grid = read_bxsf_grid(bxsf_path, cache_dir)
    occupied = grid['energies'] < grid['fermi_energy']
    minority = occupied if occupied.mean() < 0.5 else ~occupied
    labels = _periodic_labels(minority)
    _, sizes = np.unique(labels[labels > 0], return_counts=True)
    return dict(
        kind='electron pockets (occupied)' if occupied.mean() < 0.5 else 'hole pockets (empty)',
        n_pieces=int(len(sizes)),
        sizes_voxels=[int(size) for size in np.sort(sizes)[::-1]],
        occupied_fraction=float(occupied.mean()),
    )


def _fermi_vertices(energies: NDArray[np.float64]) -> NDArray[np.float64]:
    from skimage.measure import marching_cubes

    padded = np.pad(energies, ((0, 1), (0, 1), (0, 1)), mode='wrap')
    vertices, _, _, _ = marching_cubes(padded, level=0.0)
    return np.mod(vertices / np.array(energies.shape), 1.0)


def _periodic_tree(fractional: NDArray[np.float64], reciprocal: NDArray[np.float64]) -> cKDTree:
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    return cKDTree(((fractional[None] + shifts[:, None]).reshape(-1, 3)) @ reciprocal)


def sheet_contacts(
    bxsf_dir: str | Path,
    bands: list[int],
    cache_dir: str | Path,
    upsampling: int = 3,
    max_gap: float = 0.04,
    cluster_distance: float = 0.03,
) -> pd.DataFrame:
    """Near-contacts between Fermi sheets of different bands (one band per degenerate group).

    Parameters
    ----------
    bxsf_dir : str or Path
        Folder with the BXSF files.
    bands : list[int]
        One band per degenerate group.
    cache_dir : str or Path
        Folder for decompressed BXSF files.
    upsampling : int, optional
        Periodic cubic-spline upsampling factor of the BXSF grid before marching cubes.
    max_gap : float, optional
        Report contacts closer than this, in A^-1.
    cluster_distance : float, optional
        Linking distance (A^-1) used to group near-contact vertices into regions.

    Returns
    -------
    pandas.DataFrame
        One row per contact region: ``pair``, ``k_gap_invA``, ``B0_T``, ``dE_min_meV``,
        ``n_vertices``, ``frac_k`` (closest point, fractional) and ``voxel_invA``.
        Gaps below the voxel size are resolution-limited.
    """
    files = find_bxsf_files(bxsf_dir)
    energies, vertices, reciprocal = {}, {}, None
    for band in bands:
        grid = read_bxsf_grid(files[band], cache_dir)
        energies[band] = ndimage.zoom(
            grid['energies'] - grid['fermi_energy'], upsampling, order=3, mode='grid-wrap'
        )
        reciprocal = grid['recip_ang']
        vertices[band] = _fermi_vertices(energies[band])
    voxel = float(np.linalg.norm(reciprocal, axis=1).mean() / energies[bands[0]].shape[0])
    rows = []
    for band_a, band_b in combinations(bands, 2):
        distance, _ = _periodic_tree(vertices[band_b], reciprocal).query(
            vertices[band_a] @ reciprocal
        )
        close = distance < max_gap
        points, gaps = vertices[band_a][close], distance[close]
        if not len(points):
            continue
        parent = np.arange(len(points))

        def root(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        tree = _periodic_tree(points, reciprocal)
        for i, j in tree.query_pairs(cluster_distance, output_type='ndarray') % len(points):
            ri, rj = root(i), root(j)
            if ri != rj:
                parent[ri] = rj
        roots = np.array([root(node) for node in range(len(points))])
        for region in np.unique(roots):
            members = np.flatnonzero(roots == region)
            closest = members[np.argmin(gaps[members])]
            coords = (points[members] * energies[band_a].shape[0]).T
            splitting = np.abs(
                ndimage.map_coordinates(energies[band_b], coords, order=1, mode='grid-wrap')
                - ndimage.map_coordinates(energies[band_a], coords, order=1, mode='grid-wrap')
            )
            rows.append(
                dict(
                    pair=f'{band_a}-{band_b}',
                    k_gap_invA=round(float(gaps[closest]), 4),
                    B0_T=round(float(breakdown_field(gaps[closest])), 1),
                    dE_min_meV=round(float(splitting.min()) * RYDBERG_IN_EV * 1000, 1),
                    n_vertices=len(members),
                    frac_k=' '.join(f'{x:.3f}' for x in points[closest]),
                    voxel_invA=round(voxel, 4),
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('k_gap_invA').reset_index(drop=True)
