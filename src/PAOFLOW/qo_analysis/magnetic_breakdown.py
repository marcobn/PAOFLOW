"""Magnetic breakdown on pyskeaf's slicing: open-trajectory contacts and coupled-network orbits.

The slicing, contour finding and area/mass conventions are those of
``PAOFLOW.pyskeaf``, so network frequencies are directly comparable with SKEAF orbits.

Network model (Pippard; Falicov and Stachowiak):

* every slice perpendicular to B is cut through one band per degenerate group;
  all Fermi contours are kept, open ones too, and oriented along the semiclassical
  motion ``dk/dt ~ z x grad E``;
* a junction is a local minimum of the distance between contours of two different
  bands below ``junction_cutoff``; its tunnelling probability is ``P = exp(-B0/B)``;
* at each junction the electron stays on its band (amplitude ``sqrt(1-P)``) or tunnels
  (``sqrt(P)``); closed directed cycles with at least one tunnelling are network
  orbits (paths reaching the supercell boundary are dropped, as in SKEAF);
* orbits are followed from slice to slice and only interior extrema of the area are kept.

Limitation: an extended near-contact (two sheets within ~0.005-0.02 A^-1 over long arcs)
is discretised into many point junctions and yields a dense ladder of network
frequencies. Such families must be judged statistically, against chance.
"""

from __future__ import annotations

import math
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .fermi_sheets import BREAKDOWN_FIELD_PREFACTOR


def _pyskeaf_tools() -> dict:
    from PAOFLOW.pyskeaf.constants import CONV_FSAREA_TO_KT, CONV_FSDADE_TO_MSTAR
    from PAOFLOW.pyskeaf.io_bxsf import read_bxsf
    from PAOFLOW.pyskeaf.orbits import find_closed_orbits_in_slice
    from PAOFLOW.pyskeaf.slice_ops import build_slice, make_slice_geometry

    return dict(read_bxsf=read_bxsf, build_slice=build_slice, geometry=make_slice_geometry,
                contours=find_closed_orbits_in_slice, area_to_kT=CONV_FSAREA_TO_KT,
                mass_factor=CONV_FSDADE_TO_MSTAR)  # fmt: skip


def single_angle_orbits(
    bxsf_path: str | Path,
    azimuthal: float,
    polar: float,
    num_interpolation: int = 200,
    fermi_energy_Ry: float = 0.0,
    frequency_tolerance: float = 0.05,
    orbit_tolerance: float = 0.1,
) -> list[tuple[float, int]]:
    """Run pyskeaf in-process at one field direction for one band.

    Returns
    -------
    list[tuple[float, int]]
        ``(frequency in T, supercell copies)`` of every extremal orbit. If orbits appear
        for a band whose production file was empty, the production job failed.
    """
    from PAOFLOW.pyskeaf.io_bxsf import read_bxsf
    from PAOFLOW.pyskeaf.runner import run_at_angle

    result = run_at_angle(read_bxsf(bxsf_path), math.radians(azimuthal), math.radians(polar),
                          numint=num_interpolation, fermi_energy=fermi_energy_Ry,
                          freq_same_frac=frequency_tolerance, avg_same_frac=orbit_tolerance,
                          allow_near_walls=True)  # fmt: skip
    return sorted(
        (round(orbit.frequency_kT * 1000, 1), orbit.num_copies) for orbit in result.orbits
    )


def open_trajectory_contacts(
    bxsf_paths: dict[int, Path],
    open_band: int,
    other_bands: list[int],
    azimuthal: float,
    polar: float,
    num_interpolation: int = 100,
    gap_thresholds: tuple[float, ...] = (0.02, 0.03),
) -> dict:
    """Count slices in which open contours of ``open_band`` approach other sheets.

    Returns
    -------
    dict
        Number of slices, slices with open contours, minimum gap per partner band (A^-1)
        and, for every threshold, the number of slices with a gap below it. Many such
        slices mean breakdown can close open trajectories at this field direction.
    """
    tools = _pyskeaf_tools()
    grids = {band: tools['read_bxsf'](path) for band, path in bxsf_paths.items()}
    geometry = tools['geometry'](grids[open_band], num_interpolation, math.radians(azimuthal),
                                 math.radians(polar))  # fmt: skip
    hits = {(band, gap): 0 for band in other_bands for gap in gap_thresholds}
    minimum_gap = {band: np.inf for band in other_bands}
    slices_with_open = 0
    for slice_index in range(1, geometry.numx + 1):
        open_contours = [
            contour
            for contour in tools['contours'](
                tools['build_slice'](grids[open_band], geometry, slice_index), 0.0, keep_open=True
            )  # fmt: skip
            if contour.is_open
        ]
        if not open_contours:
            continue
        slices_with_open += 1
        open_points = np.vstack([contour.contour_xy for contour in open_contours])
        for band in other_bands:
            partner = tools['contours'](tools['build_slice'](grids[band], geometry, slice_index), 0.0,
                                        keep_open=True)  # fmt: skip
            if not partner:
                continue
            gap = cKDTree(np.vstack([c.contour_xy for c in partner])).query(open_points)[0].min()
            minimum_gap[band] = min(minimum_gap[band], gap)
            for threshold in gap_thresholds:
                hits[(band, threshold)] += int(gap <= threshold)
    row = dict(
        polar=polar, azimuthal=azimuthal, slices=geometry.numx, slices_with_open=slices_with_open
    )
    for band in other_bands:
        row[f'min_gap_{open_band}-{band}'] = round(float(minimum_gap[band]), 4)
        for threshold in gap_thresholds:
            row[f'slices_{open_band}-{band}_<{threshold}'] = hits[(band, threshold)]
    return row


def _oriented_contours(
    energies: NDArray[np.float64], step_x: float, step_y: float, band: int
) -> list[dict]:
    from skimage.measure import find_contours

    size_x, size_y = energies.shape
    gradient_x = np.gradient(energies, step_x, axis=0, edge_order=2)
    gradient_y = np.gradient(energies, step_y, axis=1, edge_order=2)
    contours = []
    for polygon in find_contours(energies, 0.0):
        if len(polygon) < 3:
            continue
        rows, cols = polygon[:, 0], polygon[:, 1]
        touches = (
            rows.min() <= 0
            or cols.min() <= 0
            or rows.max() >= size_x - 1
            or cols.max() >= size_y - 1
        )
        closed = bool(np.allclose(polygon[0], polygon[-1])) and not touches
        points = np.stack([rows * step_x, cols * step_y], axis=1)
        row_index = np.clip(np.rint(rows).astype(int), 0, size_x - 1)
        col_index = np.clip(np.rint(cols).astype(int), 0, size_y - 1)
        gradient = np.stack(
            [gradient_x[row_index, col_index], gradient_y[row_index, col_index]], axis=1
        )
        motion = np.stack([-gradient[:, 1], gradient[:, 0]], axis=1)  # z x grad E
        if np.sum(np.diff(points, axis=0) * 0.5 * (motion[1:] + motion[:-1])) < 0:
            points, gradient = points[::-1].copy(), gradient[::-1].copy()
        if closed:
            points, gradient = points[:-1], gradient[:-1]
        contours.append(
            dict(band=band, xy=points, gmag=np.linalg.norm(gradient, axis=1), closed=closed)
        )
    return contours


def _junctions(contours: list[dict], step: float, junction_cutoff: float) -> list[dict]:
    by_band: dict[int, list[int]] = {}
    for index, contour in enumerate(contours):
        by_band.setdefault(contour['band'], []).append(index)
    bands = sorted(by_band)
    junctions = []
    for position, band_a in enumerate(bands):
        for band_b in bands[position + 1 :]:
            lengths = [len(contours[i]['xy']) for i in by_band[band_b]]
            partner_points = np.vstack([contours[i]['xy'] for i in by_band[band_b]])
            owner = np.repeat(by_band[band_b], lengths)
            start = dict(zip(by_band[band_b], np.concatenate([[0], np.cumsum(lengths)[:-1]])))
            tree = cKDTree(partner_points)
            for index_a in by_band[band_a]:
                contour = contours[index_a]
                distance, nearest = tree.query(contour['xy'], distance_upper_bound=junction_cutoff)
                finite = np.isfinite(distance)
                if not finite.any():
                    continue
                count = len(distance)
                padded = np.where(finite, distance, np.inf)
                previous = np.roll(padded, 1) if contour['closed'] else np.r_[np.inf, padded[:-1]]
                following = np.roll(padded, -1) if contour['closed'] else np.r_[padded[1:], np.inf]
                minima = np.flatnonzero(finite & (padded <= previous) & (padded <= following))
                kept: list[int] = []
                for i in minima[np.argsort(padded[minima])]:
                    if all(
                        min(abs(i - k), count - abs(i - k)) * step > 2 * junction_cutoff
                        for k in kept
                    ):
                        kept.append(i)
                for i in kept:
                    index_b = int(owner[nearest[i]])
                    junctions.append(dict(ca=index_a, ia=int(i), cb=index_b,
                                          ib=int(nearest[i] - start[index_b]), gap=float(distance[i])))  # fmt: skip
    return junctions


def _network_cycles(
    contours, junctions, reference_field, max_segments, max_cycles, labels, open_band
):
    import networkx as nx

    nodes: dict[int, list] = {}
    for junction_id, junction in enumerate(junctions):
        nodes.setdefault(junction['ca'], []).append((junction['ia'], junction_id))
        nodes.setdefault(junction['cb'], []).append((junction['ib'], junction_id))
    for contour_index in nodes:
        nodes[contour_index].sort()
    position = {(c, j): k for c, entries in nodes.items() for k, (_, j) in enumerate(entries)}
    graph = nx.DiGraph()
    for contour_index, entries in nodes.items():
        count = len(entries)
        for k in range(count):
            if not contours[contour_index]['closed'] and k == count - 1:
                continue
            following = (k + 1) % count
            junction_id = entries[following][1]
            junction = junctions[junction_id]
            partner = junction['cb'] if junction['ca'] == contour_index else junction['ca']
            probability = math.exp(
                -BREAKDOWN_FIELD_PREFACTOR * junction['gap'] ** 2 / reference_field
            )
            graph.add_edge((contour_index, k), (contour_index, following), tunnel=0, P=probability,
                           gap=junction['gap'])  # fmt: skip
            graph.add_edge((contour_index, k), (partner, position[(partner, junction_id)]), tunnel=1,
                           P=probability, gap=junction['gap'])  # fmt: skip
    orbits = []
    for cycle in islice(nx.simple_cycles(graph, length_bound=max_segments), max_cycles):
        edges = [graph.edges[cycle[i], cycle[(i + 1) % len(cycle)]] for i in range(len(cycle))]
        tunnellings = sum(edge['tunnel'] for edge in edges)
        if tunnellings == 0:
            continue
        pieces, inverse_velocity, bands, uses_open = [], 0.0, [], False
        for contour_index, k in cycle:
            entries, contour = nodes[contour_index], contours[contour_index]
            first, last = entries[k][0], entries[(k + 1) % len(entries)][0]
            if len(entries) == 1 and contour['closed']:
                # a single junction on a closed contour: the segment is the whole loop
                index = np.r_[np.arange(first, len(contour['xy'])), np.arange(0, first + 1)]
            elif last >= first:
                index = np.arange(first, last + 1)
            else:
                index = np.r_[np.arange(first, len(contour['xy'])), np.arange(0, last + 1)]
            segment = contour['xy'][index]
            pieces.append(segment)
            if len(segment) > 1:
                lengths = np.linalg.norm(np.diff(segment, axis=0), axis=1)
                speed = 0.5 * (contour['gmag'][index][1:] + contour['gmag'][index][:-1])
                inverse_velocity += float(np.sum(np.where(speed > 0, lengths / speed, 0.0)))
            bands.append(contour['band'])
            uses_open |= contour['band'] == open_band and not contour['closed']
        path = np.vstack(pieces)
        area = 0.5 * float(
            np.dot(path[:, 0], np.roll(path[:, 1], -1))
            - np.dot(path[:, 1], np.roll(path[:, 0], -1))
        )
        log_amplitude = sum(0.5 * math.log(edge['P'] if edge['tunnel'] else max(1 - edge['P'], 1e-300))
                            for edge in edges)  # fmt: skip
        orbits.append(dict(area=abs(area), carrier='e' if area > 0 else 'h', cx=float(path[:, 0].mean()),
                           cy=float(path[:, 1].mean()), n_tunnel=tunnellings,
                           bands='-'.join(str(labels.get(b, b)) for b in sorted(set(bands))),
                           uses_open=uses_open, inverse_velocity=inverse_velocity,
                           R=math.exp(log_amplitude),
                           max_tunnel_gap=max(edge['gap'] for edge in edges if edge['tunnel'])))  # fmt: skip
    return orbits


def network_orbits_at_angle(
    bxsf_paths: dict[int, Path],
    azimuthal: float,
    polar: float,
    num_interpolation: int = 150,
    junction_cutoff: float = 0.03,
    reference_field: float = 30.0,
    max_segments: int = 14,
    max_cycles: int = 20000,
    band_labels: dict[int, object] | None = None,
    open_band: int | None = None,
    link_steps: int = 6,
    min_chain: int = 3,
) -> tuple[list[dict], dict]:
    """Extremal breakdown-network orbits at one field direction.

    Parameters
    ----------
    bxsf_paths : dict[int, Path]
        One plain-text BXSF file per degenerate group.
    azimuthal, polar : float
        Field direction (deg).
    num_interpolation : int, optional
        pyskeaf interpolation (slices = ``4 (num_interpolation - 1) + 1``).
    junction_cutoff : float, optional
        Maximum contour separation (A^-1) treated as a junction.
    reference_field : float, optional
        Field (T) at which tunnelling probabilities and amplitudes are evaluated.
    max_segments, max_cycles : int, optional
        Bounds of the cycle enumeration per slice.
    band_labels : dict or None, optional
        Labels used in the ``bands`` column (e.g. scalar-relativistic parents).
    open_band : int or None, optional
        Band whose open pieces are flagged in ``uses_open``.
    link_steps, min_chain : int, optional
        Slice-to-slice linking radius (in grid steps) and minimum chain length.

    Returns
    -------
    tuple[list[dict], dict]
        Extremal network orbits (``F``, ``mstar``, ``carrier``, ``bands``, ``n_tunnel``,
        ``uses_open``, ``R``, ``max_tunnel_gap``, ``chain_len``) and run statistics.
    """
    tools = _pyskeaf_tools()
    labels = band_labels or {}
    grids = {band: tools['read_bxsf'](path) for band, path in bxsf_paths.items()}
    geometry = tools['geometry'](next(iter(grids.values())), num_interpolation, math.radians(azimuthal),
                                 math.radians(polar))  # fmt: skip
    step_x, step_y = geometry.xkseparation, geometry.ykseparation
    records, junction_count = [], 0
    for slice_index in range(1, geometry.numx + 1):
        contours = []
        for band, grid in grids.items():
            contours += _oriented_contours(tools['build_slice'](grid, geometry, slice_index).energies,
                                           step_x, step_y, band)  # fmt: skip
        junctions = _junctions(contours, step_x, junction_cutoff)
        junction_count += len(junctions)
        if not junctions:
            continue
        for orbit in _network_cycles(contours, junctions, reference_field, max_segments, max_cycles,
                                     labels, open_band):  # fmt: skip
            orbit.update(s=slice_index, F=orbit['area'] * tools['area_to_kT'] * 1000.0,
                         mstar=orbit['inverse_velocity'] * tools['mass_factor'])  # fmt: skip
            records.append(orbit)
    extrema = _extremal_orbits(pd.DataFrame(records), step_x, link_steps, min_chain)
    for orbit in extrema:
        orbit.update(azimuthal=azimuthal, polar=polar)
    return extrema, dict(azimuthal=azimuthal, polar=polar, junctions=junction_count,
                         network_orbits=len(records))  # fmt: skip


def _extremal_orbits(
    records: pd.DataFrame, step: float, link_steps: int, min_chain: int
) -> list[dict]:
    if records.empty:
        return []
    records = records.loc[records.F > 5].copy()
    records['signature'] = (
        records.bands + '|' + records.n_tunnel.astype(str) + '|' + records.uses_open.astype(str)
    )
    extrema = []
    for _, family in records.groupby('signature'):
        family = family.sort_values('s').reset_index(drop=True)
        parent = np.arange(len(family))

        def root(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        by_slice = {s: members for s, members in family.groupby('s')}
        for s, members in by_slice.items():
            previous = by_slice.get(s - 1)
            if previous is None:
                continue
            tree = cKDTree(previous[['cx', 'cy']].to_numpy())
            for i, orbit in members.iterrows():
                best = None
                for j in tree.query_ball_point([orbit.cx, orbit.cy], link_steps * step):
                    candidate = previous.iloc[j]
                    change = abs(candidate.F - orbit.F) / orbit.F
                    if change < 0.1 and (best is None or change < best[0]):
                        best = (change, candidate.name)
                if best is not None:
                    parent[root(i)] = root(best[1])
        family['chain'] = [root(i) for i in range(len(family))]
        for _, chain in family.groupby('chain'):
            chain = chain.sort_values(['s', 'F']).drop_duplicates('s')
            if len(chain) < min_chain:
                continue
            frequencies = chain.F.to_numpy()
            for k in range(1, len(frequencies) - 1):
                if (frequencies[k] - frequencies[k - 1]) * (
                    frequencies[k + 1] - frequencies[k]
                ) < 0:
                    orbit = chain.iloc[k]
                    extrema.append(dict(F=float(frequencies[k]), mstar=orbit.mstar, carrier=orbit.carrier,
                                        bands=orbit.bands, n_tunnel=int(orbit.n_tunnel),
                                        uses_open=bool(orbit.uses_open), R=orbit.R,
                                        max_tunnel_gap=orbit.max_tunnel_gap, chain_len=len(chain)))  # fmt: skip
    return extrema


def group_network_copies(extrema: pd.DataFrame, sweep_angle: str = 'polar') -> pd.DataFrame:
    """Cluster extremal network orbits within 1 % (same angle, bands, tunnellings) into copies."""
    columns = ['angle', 'freq_T', 'copies', 'mstar_band_me', 'carrier', 'bands', 'n_tunnel', 'uses_open',
               'R_ref', 'B0_max_T', 'chain_len']  # fmt: skip
    if extrema.empty:
        return pd.DataFrame(columns=columns)
    rows = []

    def summary(cluster: list) -> dict:
        members = pd.DataFrame(cluster)
        return dict(angle=members[sweep_angle].iloc[0], freq_T=round(members.F.mean(), 1), copies=len(members),
                    mstar_band_me=round(members.mstar.median(), 3), carrier=members.carrier.mode().iloc[0],
                    bands=members.bands.iloc[0], n_tunnel=int(members.n_tunnel.iloc[0]),
                    uses_open=bool(members.uses_open.iloc[0]), R_ref=round(members.R.median(), 4),
                    B0_max_T=round(BREAKDOWN_FIELD_PREFACTOR * members.max_tunnel_gap.max() ** 2, 1),
                    chain_len=int(members.chain_len.median()))  # fmt: skip

    for _, family in extrema.sort_values('F').groupby(
        [sweep_angle, 'bands', 'n_tunnel', 'uses_open']
    ):
        cluster = [family.iloc[0]]
        for _, orbit in list(family.iterrows())[1:]:
            if abs(orbit.F - cluster[0].F) <= 0.01 * cluster[0].F:
                cluster.append(orbit)
            else:
                rows.append(summary(cluster))
                cluster = [orbit]
        rows.append(summary(cluster))
    return pd.DataFrame(rows)


def network_orbits_over_sweep(
    bxsf_paths: dict[int, Path],
    angles: list[float],
    sweep_angle: str = 'polar',
    fixed_angle: float = 0.0,
    n_jobs: int | None = -1,
    **settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Network orbits at every angle of a rotation, one joblib task per angle.

    ``n_jobs`` follows joblib conventions and is clipped to the MPI rank's core budget
    by ``PAOFLOW.utils.parallel_resources.resolve_n_jobs``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Grouped network frequencies (see :func:`group_network_copies`) and per-angle statistics.
    """
    from joblib import Parallel, delayed

    from PAOFLOW.utils.parallel_resources import resolve_n_jobs

    def field(angle: float) -> tuple[float, float]:
        return (fixed_angle, angle) if sweep_angle == 'polar' else (angle, fixed_angle)

    workers = resolve_n_jobs(n_jobs, n_tasks=len(angles))
    results = Parallel(n_jobs=workers, backend='loky', batch_size=1)(
        delayed(network_orbits_at_angle)(bxsf_paths, *field(angle), **settings) for angle in angles
    )
    extrema = pd.DataFrame([orbit for orbits, _ in results for orbit in orbits])
    statistics = pd.DataFrame([stats for _, stats in results])
    grouped = group_network_copies(extrema, sweep_angle) if len(extrema) else pd.DataFrame()
    return grouped, statistics
