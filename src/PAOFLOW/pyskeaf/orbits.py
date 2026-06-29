"""Detection of closed extremal orbits in one 2D slice.

Modernised replacement for the Fortran ``sliceext`` walking algorithm
(skeaf_v1p3p0_r149.F90 lines 3768–4445).  Instead of the pixel-by-pixel
boundary walk we use ``skimage.measure.find_contours`` to obtain the
iso-EF level sets of the slice in one vectorised pass, then compute the
same per-orbit observables (area, centroid, std, bbox, orbit type,
effective mass) that ``sliceext`` populates into the global ``slfs*`` arrays.

Outputs match the Fortran semantics:

* areas in Å^-2
* frequencies in kT (= Tesla × 1000) via ``CONV_FSAREA_TO_KT``
* effective mass in electron masses via ``CONV_FSDADE_TO_MSTAR``
* fractional supercell coordinates ``avg_xy_frac, std_xy_frac, min/max_xy_frac``
  in [0, 1]
* orbit type ``+1`` electron-like, ``-1`` hole-like, ``0`` ambiguous

Phase 4 additions (cross-slice matching, extremum detection, averaging)
appear after :func:`find_closed_orbits_in_slice` below.  They are direct
translations of the main-program loops in skeaf_v1p3p0_r149.F90 lines
1410–2700, preserving the 19-condition matching predicate, the bifurcation
detection (forward + reverse), the floater-displacement loop (capped at
500 iterations), the supercell→RUC centroid mapping, the same-centre
grouping with periodic ±1 wrap, and the freq-similarity averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from PAOFLOW.pyskeaf.constants import CONV_FSAREA_TO_KT, CONV_FSDADE_TO_MSTAR
from PAOFLOW.pyskeaf.slice_ops import Slice2D


@dataclass
class SliceOrbit:
    """One closed FS contour found in a single slice."""

    slice_index: int  # 1-based slice index
    contour_xy: np.ndarray  # (N, 2), Å^-1 in slicing frame (x', y')
    area: float  # Å^-2
    inside_area: float  # Å^-2 (electron < area; hole > area)
    frequency_kT: float
    effective_mass: float  # m_e units
    orbit_type: int  # +1 electron, −1 hole, 0 ambiguous
    avg_xy_frac: np.ndarray  # (2,) supercell-fractional centroid
    std_xy_frac: np.ndarray  # (2,) population std of contour points
    min_xy_frac: np.ndarray  # (2,) bbox lower
    max_xy_frac: np.ndarray  # (2,) bbox upper
    n_points: int  # number of contour vertices
    is_open: bool  # touched supercell boundary
    is_too_small: bool  # area < minarea threshold


def _polygon_area(xy: np.ndarray) -> float:
    """Signed shoelace area of a closed polygon ``xy`` of shape (N, 2).

    The polygon need not have a duplicated closing vertex; it is closed
    implicitly via :func:`numpy.roll`.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _contour_touches_boundary(rows: np.ndarray, cols: np.ndarray, numx: int, numy: int) -> bool:
    """True if any vertex sits on the supercell boundary (within ½ pixel)."""
    eps = 0.5
    return bool(
        np.any(rows < eps)
        or np.any(rows > numx - 1 - eps)
        or np.any(cols < eps)
        or np.any(cols > numy - 1 - eps)
    )


def _bilinear_sample(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of ``field`` (shape (numx, numy)) at fractional pixel coords."""
    numx, numy = field.shape
    r = np.clip(rows, 0.0, numx - 1.000001)
    c = np.clip(cols, 0.0, numy - 1.000001)
    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    r1 = r0 + 1
    c1 = c0 + 1
    fr = r - r0
    fc = c - c0
    return (
        (1 - fr) * (1 - fc) * field[r0, c0]
        + fr * (1 - fc) * field[r1, c0]
        + (1 - fr) * fc * field[r0, c1]
        + fr * fc * field[r1, c1]
    )


def find_closed_orbits_in_slice(
    slice2d: Slice2D,
    fermi_energy: float,
    *,
    keep_too_small: bool = False,
    keep_open: bool = False,
) -> List[SliceOrbit]:
    """Locate every closed FS orbit in one slice and compute its observables.

    Parameters
    ----------
    slice2d : :class:`PAOFLOW.pyskeaf.slice_ops.Slice2D`
        Energies sampled on the (numx × numy) supercell slice.
    fermi_energy : float
        Fermi level in Rydbergs.
    keep_too_small, keep_open : bool
        If True, retain orbits whose area is below ``minarea = 2 Δkx Δky``
        (Fortran ``minarea``) or that touch the supercell boundary.  These
        are flagged via ``is_too_small`` / ``is_open`` while their area /
        frequency / mass fields remain populated for diagnostics.  Default
        behaviour mirrors Fortran (drops them).
    """
    from skimage.measure import find_contours

    geom = slice2d.geometry
    numx, numy = slice2d.energies.shape
    dx = geom.xkseparation
    dy = geom.ykseparation
    minarea = 2.0 * dx * dy

    contours = find_contours(slice2d.energies, level=fermi_energy)

    # Slice energy gradient (Ryd / Å^-1).
    grad_x = np.gradient(slice2d.energies, dx, axis=0, edge_order=2)
    grad_y = np.gradient(slice2d.energies, dy, axis=1, edge_order=2)

    out: List[SliceOrbit] = []
    for poly in contours:
        rows = poly[:, 0]
        cols = poly[:, 1]
        n_pts = rows.size

        is_open = _contour_touches_boundary(rows, cols, numx, numy)
        is_closed = (n_pts >= 3) and bool(np.allclose(poly[0], poly[-1], atol=1e-9))

        if is_open or not is_closed:
            if not keep_open:
                continue
            is_open = True

        # Pixel coords → physical Å^-1 in slicing frame (axis 0 == x').
        x_phys = rows * dx
        y_phys = cols * dy
        xy = np.stack([x_phys, y_phys], axis=1)

        # Drop the duplicated closing vertex for area/centroid.
        xy_open = xy[:-1] if np.allclose(xy[0], xy[-1], atol=1e-12) else xy
        area = abs(_polygon_area(xy_open))
        is_too_small = area < minarea
        if is_too_small and not keep_too_small:
            continue

        frequency_kT = area * CONV_FSAREA_TO_KT

        avg_x = float(np.mean(xy_open[:, 0]))
        avg_y = float(np.mean(xy_open[:, 1]))
        std_x = float(np.sqrt(np.mean((xy_open[:, 0] - avg_x) ** 2)))
        std_y = float(np.sqrt(np.mean((xy_open[:, 1] - avg_y) ** 2)))
        min_x = float(np.min(xy_open[:, 0]))
        max_x = float(np.max(xy_open[:, 0]))
        min_y = float(np.min(xy_open[:, 1]))
        max_y = float(np.max(xy_open[:, 1]))

        # Orbit type: sample energy at contour centroid.
        cx_pixel = avg_x / dx
        cy_pixel = avg_y / dy
        e_centroid = float(
            _bilinear_sample(
                slice2d.energies,
                np.array([cx_pixel]),
                np.array([cy_pixel]),
            )[0]
        )
        if e_centroid < fermi_energy:
            orbit_type = +1
            inside_area = area * 0.95
        elif e_centroid > fermi_energy:
            orbit_type = -1
            inside_area = area * 1.05
        else:
            orbit_type = 0
            inside_area = area

        # Effective mass: m* = (∮ ds / |∇⊥E|) · CONV_FSDADE_TO_MSTAR
        gx = _bilinear_sample(grad_x, rows, cols)
        gy = _bilinear_sample(grad_y, rows, cols)
        gmag = np.sqrt(gx * gx + gy * gy)
        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        gmag_edge = 0.5 * (gmag[:-1] + gmag[1:])
        with np.errstate(divide='ignore', invalid='ignore'):
            integrand = np.where(gmag_edge > 0, ds / gmag_edge, 0.0)
        fsdAde = float(np.sum(integrand))
        effective_mass = fsdAde * CONV_FSDADE_TO_MSTAR

        xlen = geom.xlength
        ylen = geom.ylength
        avg_xy_frac = np.array([avg_x / xlen, avg_y / ylen])
        std_xy_frac = np.array([std_x / xlen, std_y / ylen])
        min_xy_frac = np.array([min_x / xlen, min_y / ylen])
        max_xy_frac = np.array([max_x / xlen, max_y / ylen])

        out.append(
            SliceOrbit(
                slice_index=slice2d.slice_index,
                contour_xy=xy,
                area=area,
                inside_area=inside_area,
                frequency_kT=frequency_kT,
                effective_mass=effective_mass,
                orbit_type=orbit_type,
                avg_xy_frac=avg_xy_frac,
                std_xy_frac=std_xy_frac,
                min_xy_frac=min_xy_frac,
                max_xy_frac=max_xy_frac,
                n_points=n_pts,
                is_open=is_open,
                is_too_small=is_too_small,
            )
        )
    return out


# ============================================================================
# Phase 4: cross-slice matching, extremum detection, averaging
# ============================================================================

from PAOFLOW.pyskeaf.slice_ops import SliceGeometry  # noqa: E402  (local import keeps Phase-3 self-contained)


@dataclass
class Chunk:
    """A trajectory of one orbit through consecutive slices.

    Mirrors the Fortran ``c*`` arrays (carea, cavgx, cnobif, cfsfromslice…)
    indexed by chunk number and per-chunk position.  ``orbits[i]`` is the
    SliceOrbit at the i-th step in the chunk; ``no_bif[i]`` mirrors
    ``cnobif`` at that position.  ``slice_indices[i]`` is the 1-based slice
    that orbit was found on.
    """

    orbits: List[SliceOrbit]
    no_bif: List[bool]

    @property
    def slice_indices(self) -> List[int]:
        return [o.slice_index for o in self.orbits]

    def __len__(self) -> int:
        return len(self.orbits)


@dataclass
class ExtremalOrbit:
    """One extremum found inside a chunk (one local min/max of area vs slice)."""

    slice_orbit: SliceOrbit  # the orbit at the extremum
    chunk_index: int  # 0-based chunk id
    cfs_index: int  # 0-based position within chunk
    curvature_kT_A2: float  # d²A/dk² · CONV_FSAREA_TO_KT  (kT·Å²)
    avg_xyz_ruc: np.ndarray  # (3,) RUC fractional centroid in [0,1)


@dataclass
class AveragedOrbit:
    """One physical extremal orbit (after averaging supercell copies)."""

    frequency_kT: float
    frequency_std_kT: float
    effective_mass: float
    effective_mass_std: float
    curvature_kT_A2: float
    curvature_std_kT_A2: float
    orbit_type: float  # mean of ±1 (may be non-integer)
    orbit_type_std: float
    avg_xyz_ruc: np.ndarray  # (3,) mean RUC centroid
    avg_xyz_ruc_std: np.ndarray  # (3,) sample std
    num_copies: int  # # orbit copies merged
    representative: ExtremalOrbit  # the largest copy in the group


# ----------------------------------------------------------------------------
# 4a. Cross-slice chunk matching (with bifurcation detection + floater loop)
# ----------------------------------------------------------------------------

_MAX_MATCH_SLICE_GAP = 3
_RELAXED_MATCH_BADNESS_MAX = 5.0e-3


def _bbox_overlap(a: SliceOrbit, b: SliceOrbit) -> bool:
    """True iff two orbit bounding boxes overlap in (x', y') supercell coords."""
    return (
        a.min_xy_frac[0] < b.max_xy_frac[0]
        and a.max_xy_frac[0] > b.min_xy_frac[0]
        and a.min_xy_frac[1] < b.max_xy_frac[1]
        and a.max_xy_frac[1] > b.min_xy_frac[1]
    )


def _badness(a: SliceOrbit, b: SliceOrbit) -> float:
    """Sum-of-squared (centroid + bbox-corner) deviations between two orbits.

    Mirrors the ``oldbadness/badness`` accumulators in the Fortran matching
    section (lines 1437–1452 etc.): six squared terms over avg, max, min in x and y.
    """
    da = a.avg_xy_frac - b.avg_xy_frac
    dmax = a.max_xy_frac - b.max_xy_frac
    dmin = a.min_xy_frac - b.min_xy_frac
    return float(da @ da + dmax @ dmax + dmin @ dmin)


def _match_conditions(
    slice_orbit: SliceOrbit, chunk_orbit: SliceOrbit, slice_no_bif: bool, chunk_no_bif: bool
) -> bool:
    """Return True if the 19-condition Fortran predicate holds.

    Implements ``mcond(1)…mcond(19)`` from skeaf lines 1620–1655: each side's
    centroid lies within ±1·std of the other's, each side's bbox corners
    within ±2·std, orbit types match, and both ``nobif`` flags are True.
    """
    a = slice_orbit
    b = chunk_orbit
    sax, say = a.std_xy_frac
    sbx, sby = b.std_xy_frac

    # mcond(1-2): a.avg ∈ b.avg ± 1·b.std
    if not (b.avg_xy_frac[0] - sbx < a.avg_xy_frac[0] < b.avg_xy_frac[0] + sbx):
        return False
    if not (b.avg_xy_frac[1] - sby < a.avg_xy_frac[1] < b.avg_xy_frac[1] + sby):
        return False
    # mcond(3-6): a.bbox ∈ b.bbox ± 2·b.std
    if not (b.max_xy_frac[0] - 2 * sbx < a.max_xy_frac[0] < b.max_xy_frac[0] + 2 * sbx):
        return False
    if not (b.max_xy_frac[1] - 2 * sby < a.max_xy_frac[1] < b.max_xy_frac[1] + 2 * sby):
        return False
    if not (b.min_xy_frac[0] - 2 * sbx < a.min_xy_frac[0] < b.min_xy_frac[0] + 2 * sbx):
        return False
    if not (b.min_xy_frac[1] - 2 * sby < a.min_xy_frac[1] < b.min_xy_frac[1] + 2 * sby):
        return False
    # mcond(7): orbit type equality
    if a.orbit_type != b.orbit_type:
        return False
    # mcond(8-13): symmetric — b.avg ∈ a.avg ± 1·a.std, b.bbox ∈ a.bbox ± 2·a.std
    if not (a.avg_xy_frac[0] - sax < b.avg_xy_frac[0] < a.avg_xy_frac[0] + sax):
        return False
    if not (a.avg_xy_frac[1] - say < b.avg_xy_frac[1] < a.avg_xy_frac[1] + say):
        return False
    if not (a.max_xy_frac[0] - 2 * sax < b.max_xy_frac[0] < a.max_xy_frac[0] + 2 * sax):
        return False
    if not (a.max_xy_frac[1] - 2 * say < b.max_xy_frac[1] < a.max_xy_frac[1] + 2 * say):
        return False
    if not (a.min_xy_frac[0] - 2 * sax < b.min_xy_frac[0] < a.min_xy_frac[0] + 2 * sax):
        return False
    if not (a.min_xy_frac[1] - 2 * say < b.min_xy_frac[1] < a.min_xy_frac[1] + 2 * say):
        return False
    # mcond(14-15): a.avg inside b.bbox
    if not (b.min_xy_frac[0] < a.avg_xy_frac[0] < b.max_xy_frac[0]):
        return False
    if not (b.min_xy_frac[1] < a.avg_xy_frac[1] < b.max_xy_frac[1]):
        return False
    # mcond(16-17): b.avg inside a.bbox
    if not (a.min_xy_frac[0] < b.avg_xy_frac[0] < a.max_xy_frac[0]):
        return False
    if not (a.min_xy_frac[1] < b.avg_xy_frac[1] < a.max_xy_frac[1]):
        return False
    # mcond(18-19): no-bifurcation flags
    return slice_no_bif and chunk_no_bif


def _relaxed_match_conditions(
    slice_orbit: SliceOrbit, chunk_orbit: SliceOrbit, slice_no_bif: bool, chunk_no_bif: bool
) -> bool:
    """Fallback match for small pockets that move farther than their own std."""
    if not (slice_no_bif and chunk_no_bif):
        return False
    if slice_orbit.orbit_type != chunk_orbit.orbit_type:
        return False
    badness = _badness(chunk_orbit, slice_orbit)
    if badness > _RELAXED_MATCH_BADNESS_MAX:
        return False

    avg_delta = np.abs(slice_orbit.avg_xy_frac - chunk_orbit.avg_xy_frac)
    bbox_scale = np.maximum(
        slice_orbit.max_xy_frac - slice_orbit.min_xy_frac,
        chunk_orbit.max_xy_frac - chunk_orbit.min_xy_frac,
    )
    return bool(np.all(avg_delta <= np.maximum(0.05, 4.0 * bbox_scale)))


def _detect_forward_bifurcation(
    chunk_tail: SliceOrbit, slice_orbits: List[SliceOrbit], other_chunk_tails: List[SliceOrbit]
) -> bool:
    """Translate the forward-bifurcation block (Fortran lines 1423–1521).

    Counts how many ``slice_orbits`` give a "unique overlap" with ``chunk_tail``
    — first the orbit whose bbox overlaps the chunk's tail, then any *other*
    slice orbit that overlaps better (lower badness) on a tighter centroid-in-bbox
    test.  If the count exceeds 1 the chunk's tail is flagged as bifurcating.
    """
    n_unique = 0
    for fs_idx, fs in enumerate(slice_orbits):
        if not _bbox_overlap(fs, chunk_tail):
            continue
        oldbad = _badness(chunk_tail, fs)
        # mcond(20): does *another* slice orbit (or another chunk tail) overlap
        # this slice orbit's bbox?  If so, skip — the choice is ambiguous.
        ambiguous = False
        for fs2_idx, fs2 in enumerate(slice_orbits):
            if fs2_idx != fs_idx and _bbox_overlap(fs, fs2):
                ambiguous = True
                break
        if not ambiguous:
            for ch2 in other_chunk_tails:
                if _bbox_overlap(fs, ch2):
                    ambiguous = True
                    break
        if ambiguous:
            continue
        n_unique += 1
        for fs2_idx, fs2 in enumerate(slice_orbits):
            if fs2_idx == fs_idx:
                continue
            if not _bbox_overlap(fs2, chunk_tail):
                continue
            # mcond(25-28): centroids inside opposite bboxes (tighter test)
            if not (chunk_tail.min_xy_frac[0] < fs2.avg_xy_frac[0] < chunk_tail.max_xy_frac[0]):
                continue
            if not (chunk_tail.min_xy_frac[1] < fs2.avg_xy_frac[1] < chunk_tail.max_xy_frac[1]):
                continue
            if not (fs2.min_xy_frac[0] < chunk_tail.avg_xy_frac[0] < fs2.max_xy_frac[0]):
                continue
            if not (fs2.min_xy_frac[1] < chunk_tail.avg_xy_frac[1] < fs2.max_xy_frac[1]):
                continue
            if _badness(chunk_tail, fs2) < oldbad:
                n_unique += 1
    return n_unique > 1


def _detect_reverse_bifurcation(
    slice_orbit: SliceOrbit, chunk_tails: List[SliceOrbit], other_slice_orbits: List[SliceOrbit]
) -> bool:
    """Forward/reverse symmetric: count chunk-tails that point uniquely back at this orbit."""
    n_unique = 0
    for ch_idx, ch in enumerate(chunk_tails):
        if not _bbox_overlap(slice_orbit, ch):
            continue
        oldbad = _badness(ch, slice_orbit)
        ambiguous = False
        for ch2_idx, ch2 in enumerate(chunk_tails):
            if ch2_idx != ch_idx and _bbox_overlap(ch, ch2):
                ambiguous = True
                break
        if not ambiguous:
            for fs2 in other_slice_orbits:
                if _bbox_overlap(ch, fs2):
                    ambiguous = True
                    break
        if ambiguous:
            continue
        n_unique += 1
        for ch2_idx, ch2 in enumerate(chunk_tails):
            if ch2_idx == ch_idx:
                continue
            if not _bbox_overlap(slice_orbit, ch2):
                continue
            if not (ch2.min_xy_frac[0] < slice_orbit.avg_xy_frac[0] < ch2.max_xy_frac[0]):
                continue
            if not (ch2.min_xy_frac[1] < slice_orbit.avg_xy_frac[1] < ch2.max_xy_frac[1]):
                continue
            if not (slice_orbit.min_xy_frac[0] < ch2.avg_xy_frac[0] < slice_orbit.max_xy_frac[0]):
                continue
            if not (slice_orbit.min_xy_frac[1] < ch2.avg_xy_frac[1] < slice_orbit.max_xy_frac[1]):
                continue
            if _badness(ch2, slice_orbit) < oldbad:
                n_unique += 1
    return n_unique > 1


def _find_best_match(
    query: SliceOrbit,
    query_no_bif: bool,
    query_slice: int,
    chunks: List[Chunk],
    prior_badnesses: List[List[float]],
) -> tuple[int | None, bool, float]:
    """Pick the chunk whose latest orbit best matches ``query`` from slice ``query_slice``.

    Returns ``(chunk_idx, occupied, best_badness)`` where ``occupied`` is True
    if the matched chunk already received a tail on the *current* slice (i.e.
    floater needed).  ``chunk_idx is None`` means no match was found.

    Mirrors the dual branch (numcfs>1 vs ==slice) at Fortran lines 1604–1769.
    """
    best_chunk = None
    best_occupied = False
    best_badness = 1e300
    for c_idx, ch in enumerate(chunks):
        tail = ch.orbits[-1]
        tail_slice = tail.slice_index
        n = len(ch)
        # Branch A: chunk already received an orbit on this slice (need to compare to second-last)
        if tail_slice == query_slice and n > 1:
            prev = ch.orbits[-2]
            if not (
                _match_conditions(query, prev, query_no_bif, ch.no_bif[-2])
                or _relaxed_match_conditions(query, prev, query_no_bif, ch.no_bif[-2])
            ):
                continue
            bad = _badness(prev, query)
            # Floater rule: only displace if our badness beats both the
            # previous match's badness AND the current best.
            if bad < prior_badnesses[c_idx][-1] and bad < best_badness:
                best_badness = bad
                best_chunk = c_idx
                best_occupied = True
        # Branch B: chunk's tail is on a recent previous slice -- normal extension.
        elif 1 <= query_slice - tail_slice <= _MAX_MATCH_SLICE_GAP:
            if not (
                _match_conditions(query, tail, query_no_bif, ch.no_bif[-1])
                or _relaxed_match_conditions(query, tail, query_no_bif, ch.no_bif[-1])
            ):
                continue
            bad = _badness(tail, query)
            if bad < best_badness:
                best_badness = bad
                best_chunk = c_idx
                best_occupied = False
    return best_chunk, best_occupied, best_badness


def _append_to_chunk(
    chunk: Chunk, orbit: SliceOrbit, prior_badnesses: List[float], badness: float
) -> None:
    """Extend ``chunk`` with a new orbit and record its match badness."""
    chunk.orbits.append(orbit)
    chunk.no_bif.append(True)
    prior_badnesses.append(badness)


def _start_new_chunk(
    chunks: List[Chunk], prior_badnesses: List[List[float]], orbit: SliceOrbit
) -> None:
    chunks.append(Chunk(orbits=[orbit], no_bif=[True]))
    prior_badnesses.append([0.0])


def match_chunks(per_slice_orbits: List[List[SliceOrbit]]) -> List[Chunk]:
    """Build chunks from a list of per-slice orbit lists.

    ``per_slice_orbits[s-1]`` (1-based slice ``s``) is the output of
    :func:`find_closed_orbits_in_slice` for slice ``s``.  Returns the list
    of chunks (in order of creation).

    Implements the full matching logic of skeaf lines 1410–1953 including
    forward/reverse bifurcation detection and the floater displacement loop
    (capped at 500 iterations per slice-orbit, matching Fortran).
    """
    chunks: List[Chunk] = []
    # Parallel array of per-orbit "prior badness" — needed for the floater rule.
    prior_badnesses: List[List[float]] = []

    for slice_idx_zero, orbits_this_slice in enumerate(per_slice_orbits):
        slice_idx = slice_idx_zero + 1  # 1-based to match Fortran

        # Reset slfsnobif default (True).
        slice_no_bif = [True] * len(orbits_this_slice)

        if chunks:
            tails_prev = [
                (i, ch.orbits[-1])
                for i, ch in enumerate(chunks)
                if ch.orbits[-1].slice_index == slice_idx - 1
            ]
            tails_prev_only = [t for _, t in tails_prev]

            # Forward bifurcation: mark chunk tails that branch into 2+ slice orbits.
            for c_idx, tail in tails_prev:
                others = [t for j, t in tails_prev if j != c_idx]
                if _detect_forward_bifurcation(tail, orbits_this_slice, others):
                    chunks[c_idx].no_bif[-1] = False

            # Reverse bifurcation: mark slice orbits that 2+ chunks point to.
            for fs_idx, fs in enumerate(orbits_this_slice):
                others = [o for j, o in enumerate(orbits_this_slice) if j != fs_idx]
                if _detect_reverse_bifurcation(fs, tails_prev_only, others):
                    slice_no_bif[fs_idx] = False

        # Now do the actual matching, in the order Fortran does (orbit by orbit).
        for fs_idx, fs in enumerate(orbits_this_slice):
            if not chunks:
                _start_new_chunk(chunks, prior_badnesses, fs)
                continue

            best_chunk, occupied, badness = _find_best_match(
                fs,
                slice_no_bif[fs_idx],
                slice_idx,
                chunks,
                prior_badnesses,
            )
            if best_chunk is None:
                _start_new_chunk(chunks, prior_badnesses, fs)
                continue
            if not occupied:
                _append_to_chunk(chunks[best_chunk], fs, prior_badnesses[best_chunk], badness)
                continue

            # Floater branch: displace the existing tail of best_chunk.
            displaced_orbit = chunks[best_chunk].orbits[-1]
            chunks[best_chunk].orbits[-1] = fs
            prior_badnesses[best_chunk][-1] = badness
            float_orbit = displaced_orbit
            for _ in range(500):
                # The displaced orbit was placed on `slice_idx` originally.
                # Try to relocate it; the Fortran loop searches against the
                # SAME slice it came from.
                f_chunk, f_occupied, f_bad = _find_best_match(
                    float_orbit,
                    True,
                    slice_idx,
                    chunks,
                    prior_badnesses,
                )
                if f_chunk is None:
                    _start_new_chunk(chunks, prior_badnesses, float_orbit)
                    break
                if not f_occupied:
                    _append_to_chunk(chunks[f_chunk], float_orbit, prior_badnesses[f_chunk], f_bad)
                    break
                # Still occupied — displace again, swap and retry.
                next_displaced = chunks[f_chunk].orbits[-1]
                chunks[f_chunk].orbits[-1] = float_orbit
                prior_badnesses[f_chunk][-1] = f_bad
                float_orbit = next_displaced
            else:
                # Loop did not break — Fortran prints the same warning.
                import warnings as _warnings

                _warnings.warn('Float loop got stuck after 500 iterations.', RuntimeWarning)

    return chunks


# ----------------------------------------------------------------------------
# 4b. Extremum detection per chunk + supercell→RUC centroid mapping
# ----------------------------------------------------------------------------


def _supercell_centroid_to_ruc(
    geom: SliceGeometry, avg_xy_frac: np.ndarray, slice_idx: int, n_slices: int
) -> np.ndarray:
    """Map a supercell centroid (slicing-frame x', y') and slice index to RUC fractional.

    Mirrors Fortran lines 1990–2009: builds the BZ-frame Cartesian centroid
    from the supercell-fractional coordinates, applies ``plr_inv``, and wraps
    by ``floor`` to land inside [0, 1).
    """
    M = geom.maxlreciplat
    z_prime_frac = (slice_idx - 1) / (n_slices - 1)  # 1-based slice → [0,1]
    Xp = (4.0 * avg_xy_frac[0] - 1.0) * M
    Yp = (4.0 * avg_xy_frac[1] - 1.0) * M
    Zp = (4.0 * z_prime_frac - 1.0) * M
    R = geom.rotation
    p1 = R[0, 0] * Xp + R[0, 1] * Yp + R[0, 2] * Zp
    p2 = R[1, 0] * Xp + R[1, 1] * Yp + R[1, 2] * Zp
    p3 = R[2, 0] * Xp + R[2, 1] * Yp + R[2, 2] * Zp
    invp = geom.plr_inverse
    fx = invp[0, 0] * p1 + invp[0, 1] * p2 + invp[0, 2] * p3
    fy = invp[1, 0] * p1 + invp[1, 1] * p2 + invp[1, 2] * p3
    fz = invp[2, 0] * p1 + invp[2, 1] * p2 + invp[2, 2] * p3
    return np.array([fx - np.floor(fx), fy - np.floor(fy), fz - np.floor(fz)])


def find_extremal(
    chunks: List[Chunk],
    geom: SliceGeometry,
    *,
    min_freq_kT: float = 0.0,
    allow_near_walls: bool = False,
) -> List[ExtremalOrbit]:
    """Find extremal orbits within each chunk (Fortran lines 1965–2050).

    For each interior position in a chunk (i.e. not at chunk endpoints),
    flag the orbit as extremal if its area is a local min, local max, or
    equals its predecessor's.  Compute curvature from the 3-point neighbourhood
    and convert centroid to RUC fractional coords.  Apply ``minextfreq`` cutoff
    and (optionally) reject orbits whose bbox lies within 2·std of a supercell
    wall (matching Fortran's ``allowextnearwalls`` default-no behaviour).
    """
    n_slices = geom.numx
    dk = geom.zkseparation
    dk2 = dk * dk
    out: List[ExtremalOrbit] = []

    for c_idx, ch in enumerate(chunks):
        n = len(ch)
        if n < 3:
            continue
        for pos in range(1, n - 1):
            o = ch.orbits[pos]
            o_prev = ch.orbits[pos - 1]
            o_next = ch.orbits[pos + 1]

            # minextfreq cutoff
            if o.frequency_kT <= min_freq_kT:
                continue

            # Reject orbits within 2·std of supercell walls (unless allowed).
            if not allow_near_walls:
                sx, sy = o.std_xy_frac
                if not (
                    o.max_xy_frac[0] < 1.0 - 2 * sx
                    and o.max_xy_frac[1] < 1.0 - 2 * sy
                    and o.min_xy_frac[0] > 0.0 + 2 * sx
                    and o.min_xy_frac[1] > 0.0 + 2 * sy
                ):
                    continue

            a_prev = o_prev.area
            a = o.area
            a_next = o_next.area
            is_min = (a_prev > a) and (a_next > a)
            is_max = (a_prev < a) and (a_next < a)
            is_flat = a_prev == a
            if not (is_min or is_max or is_flat):
                continue

            h_prev = (o.slice_index - o_prev.slice_index) * dk
            h_next = (o_next.slice_index - o.slice_index) * dk
            curvature = CONV_FSAREA_TO_KT * 2.0 * (
                ((a_next - a) / h_next) - ((a - a_prev) / h_prev)
            ) / (h_prev + h_next)
            ruc = _supercell_centroid_to_ruc(geom, o.avg_xy_frac, o.slice_index, n_slices)
            out.append(
                ExtremalOrbit(
                    slice_orbit=o,
                    chunk_index=c_idx,
                    cfs_index=pos,
                    curvature_kT_A2=curvature,
                    avg_xyz_ruc=ruc,
                )
            )
    return out


# ----------------------------------------------------------------------------
# 4c. Same-centre grouping + frequency-similarity averaging
# ----------------------------------------------------------------------------


def _group_by_centre(
    extrema: List[ExtremalOrbit], avg_same_frac: float
) -> List[List[ExtremalOrbit]]:
    """Greedy grouping of extrema by similar RUC centroid (with periodic ±1 wrap).

    Matches Fortran lines 2188–2266: at each pass the seed is element 0 of the
    remaining list; everything within ``avg_same_frac`` of the seed (allowing
    independent ±1 wrap on each axis) is pulled into the group.  Repeat until
    empty.  Returns one list per group.
    """
    remaining = list(extrema)
    groups: List[List[ExtremalOrbit]] = []
    while remaining:
        seed = remaining[0]
        same: List[ExtremalOrbit] = [seed]
        rest: List[ExtremalOrbit] = []
        for cand in remaining[1:]:
            in_same = True
            for axis in range(3):
                dx = abs(seed.avg_xyz_ruc[axis] - cand.avg_xyz_ruc[axis])
                dx_neg = abs(seed.avg_xyz_ruc[axis] - (cand.avg_xyz_ruc[axis] - 1.0))
                dx_pos = abs(seed.avg_xyz_ruc[axis] - (cand.avg_xyz_ruc[axis] + 1.0))
                if not (dx < avg_same_frac or dx_neg < avg_same_frac or dx_pos < avg_same_frac):
                    in_same = False
                    break
            if in_same:
                same.append(cand)
            else:
                rest.append(cand)
        groups.append(same)
        remaining = rest
    return groups


def _average_one_centre_group(
    group: List[ExtremalOrbit], freq_same_frac: float
) -> List[AveragedOrbit]:
    """Sort one centre-group by frequency and merge consecutive entries within ``freq_same_frac``.

    Mirrors lines 2284–2453.  Uses a *running tail* comparison: orbit i joins
    the current cluster iff its frequency is within ``(1+freq_same_frac)`` of
    the largest frequency in the cluster so far.
    """
    sorted_by_f = sorted(group, key=lambda e: e.slice_orbit.frequency_kT)
    out: List[AveragedOrbit] = []
    cluster: List[ExtremalOrbit] = []

    def _unwrap_rucs(rucs: np.ndarray) -> np.ndarray:
        if rucs.shape[0] <= 1:
            return rucs.copy()
        seed = rucs[0]
        out = rucs.copy()
        for i in range(1, out.shape[0]):
            delta = out[i] - seed
            out[i, delta > 0.5] -= 1.0
            out[i, delta < -0.5] += 1.0
        return out

    def flush(cluster: List[ExtremalOrbit]) -> None:
        if not cluster:
            return
        freqs = np.array([e.slice_orbit.frequency_kT for e in cluster])
        masses = np.array([e.slice_orbit.effective_mass for e in cluster])
        curvs = np.array([e.curvature_kT_A2 for e in cluster])
        types = np.array([float(e.slice_orbit.orbit_type) for e in cluster])
        rucs = _unwrap_rucs(np.array([e.avg_xyz_ruc for e in cluster]))  # (n, 3)
        n = len(cluster)
        ddof = 1 if n > 1 else 0
        mean_xyz = rucs.mean(axis=0)
        std_xyz = rucs.std(axis=0, ddof=ddof) if n > 1 else np.zeros(3)
        # The "representative" is the largest-frequency orbit in the cluster
        # (Fortran picks ``temporbnumarray(numtemp)`` — the last in sort order).
        rep = cluster[-1]
        out.append(
            AveragedOrbit(
                frequency_kT=float(freqs.mean()),
                frequency_std_kT=float(freqs.std(ddof=ddof) if n > 1 else 0.0),
                effective_mass=float(masses.mean()),
                effective_mass_std=float(masses.std(ddof=ddof) if n > 1 else 0.0),
                curvature_kT_A2=float(curvs.mean()),
                curvature_std_kT_A2=float(curvs.std(ddof=ddof) if n > 1 else 0.0),
                orbit_type=float(types.mean()),
                orbit_type_std=float(types.std(ddof=ddof) if n > 1 else 0.0),
                avg_xyz_ruc=mean_xyz,
                avg_xyz_ruc_std=std_xyz,
                num_copies=n,
                representative=rep,
            )
        )

    for ext in sorted_by_f:
        if not cluster:
            cluster = [ext]
            continue
        tail_freq = cluster[-1].slice_orbit.frequency_kT
        if ext.slice_orbit.frequency_kT <= (1.0 + freq_same_frac) * tail_freq:
            cluster.append(ext)
        else:
            flush(cluster)
            cluster = [ext]
    flush(cluster)
    return out


def average_orbits(
    extrema: List[ExtremalOrbit], *, freq_same_frac: float = 0.01, avg_same_frac: float = 0.05
) -> List[AveragedOrbit]:
    """Group supercell-copy extrema, average each group, and return sorted-by-frequency.

    Combines the same-centre grouping (lines 2188–2266) with the per-group
    frequency clustering (lines 2284–2453) and the final global frequency sort
    (lines 2474–2538).
    """
    groups = _group_by_centre(extrema, avg_same_frac)
    av: List[AveragedOrbit] = []
    for g in groups:
        av.extend(_average_one_centre_group(g, freq_same_frac))
    av.sort(key=lambda a: a.frequency_kT)
    return av
