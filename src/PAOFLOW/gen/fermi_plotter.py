"""``fermi-plotter`` — interactive 3-D Fermi-surface viewer for PAOFLOW BXSF output.

Reads a Fermi-surface BXSF file written by PAOFLOW (``FermiSurf_{ispin}.bxsf``,
produced by :func:`PAOFLOW.writers.write2bxsf.write2bxsf`) and renders the
Fermi sheets as interactive iso-surfaces with Mayavi.  The surfaces are
extracted with marching cubes, folded into the first Brillouin zone and
coloured by the Fermi velocity :math:`|\\nabla_{\\mathbf k} E|`.  The Mayavi
window supports rotation, zoom and pan out of the box.

This module backs the ``fermi-plotter`` console script (see ``[project.scripts]``
in ``pyproject.toml``).

Usage examples
--------------
Composite image (all bands crossing the Fermi window)::

    fermi-plotter FermiSurf_0.bxsf

A single band (by its BXSF band label)::

    fermi-plotter FermiSurf_0.bxsf --band 58

A subset of bands, upsampled 2x for smoother sheets, saved to PNG::

    fermi-plotter FermiSurf_0.bxsf --band 57,58 --interp 2 --save fermi.png

A 2x2x2 block of reciprocal cells with a Gamma point at the centre::

    fermi-plotter FermiSurf_0.bxsf --supercell 2 --center

The raw reciprocal parallelepiped instead of the true zone::

    fermi-plotter FermiSurf_0.bxsf --bz cell

The SKEAF magnetic-field axis and its slice plane, using the same field
selector and angles as :meth:`PAOFLOW.PAOFLOW.pyskeaf`::

    fermi-plotter FermiSurf_0.bxsf --b-field non_principal --azimuthal 30 \\
        --polar 45 --field-plane --field-label --opacity 0.6

A field rotation, drawn as start/end arrows joined by the swept arc::

    fermi-plotter FermiSurf_0.bxsf --b-field rotation --azimuthal 0,90 \\
        --polar 45,45 --num-angles 7

Only one spin channel is handled per run; point the tool at the desired
``FermiSurf_{ispin}.bxsf`` file directly.

Requires the ``fermisurface`` extra (``pip install "PAOFLOW[fermisurface]"`` or
``pip install mayavi scikit-image``).
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

# --------------------------------------------------------------------------- #
# BXSF parsing
# --------------------------------------------------------------------------- #

_FLOAT_RE = re.compile(r'[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?')
_FERMI_RE = re.compile(r'fermi\s*energy', re.IGNORECASE)
_BANDGRID_RE = re.compile(r'BANDGRID_3D_BANDS', re.IGNORECASE)
_BAND_RE = re.compile(r'BAND\s*:\s*(\d+)', re.IGNORECASE)
_END_RE = re.compile(r'END_BANDGRID_3D', re.IGNORECASE)


def _to_float(token: str) -> float:
    return float(token.replace('D', 'E').replace('d', 'e'))


@dataclass
class FermiSurfData:
    """Parsed contents of a (possibly multi-band) PAOFLOW BXSF file.

    Attributes
    ----------
    fermi_energy : float
        Fermi energy in eV, as written in the ``BEGIN_INFO`` block.
    dims : tuple[int, int, int]
        Grid dimensions ``(Nx, Ny, Nz)`` as stored in the file.  PAOFLOW writes
        the periodic wrap-around grid, so ``Nx = nk1 + 1`` etc.
    recip : np.ndarray, shape (3, 3)
        Reciprocal-lattice (spanning) vectors as stored in the file; row ``i``
        is the ``i``-th vector (units of :math:`2\\pi/a_{\\rm lat}`).
    bands : dict[int, np.ndarray]
        Mapping of BXSF band label -> energy grid of shape ``dims`` (eV).
    """

    fermi_energy: float
    dims: tuple[int, int, int]
    recip: np.ndarray
    bands: dict[int, np.ndarray]


def read_fermi_bxsf(path: str | Path) -> FermiSurfData:
    """Read a PAOFLOW Fermi-surface BXSF file (single or multi-band).

    Unlike ``PAOFLOW.pyskeaf.io_bxsf.read_bxsf`` (which targets single-band,
    non-periodic SKEAF grids), this reader accepts the multi-band,
    periodic-wrapped grid that ``write2bxsf`` emits.

    Parameters
    ----------
    path : str or Path
        Path to the ``.bxsf`` file.

    Returns
    -------
    FermiSurfData
        Parsed Fermi energy, grid dimensions, reciprocal vectors and per-band
        energy grids.
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    # 1. Fermi energy.
    fermi_energy = None
    for line in lines:
        if _FERMI_RE.search(line):
            toks = _FLOAT_RE.findall(line)
            if toks:
                fermi_energy = _to_float(toks[-1])
            break
    if fermi_energy is None:
        raise ValueError(f'{path}: no "Fermi Energy" line found.')

    # 2. Locate the BANDGRID_3D_BANDS keyword and read the header block.
    i = 0
    n = len(lines)
    while i < n and not _BANDGRID_RE.search(lines[i]):
        i += 1
    if i >= n:
        raise ValueError(f'{path}: no BANDGRID_3D_BANDS block found.')
    i += 1

    def _next_nonblank(idx: int) -> int:
        while idx < n and not lines[idx].strip():
            idx += 1
        return idx

    i = _next_nonblank(i)
    nbnd = int(_FLOAT_RE.findall(lines[i])[0])
    i = _next_nonblank(i + 1)
    dims = tuple(int(v) for v in _FLOAT_RE.findall(lines[i])[:3])
    i += 1  # origin
    origin = np.array([_to_float(v) for v in _FLOAT_RE.findall(lines[i])[:3]])
    recip = np.empty((3, 3), dtype=float)
    for r in range(3):
        i += 1
        recip[r] = [_to_float(v) for v in _FLOAT_RE.findall(lines[i])[:3]]

    npts = int(np.prod(dims))

    # 3. Read each BAND block.
    bands: dict[int, np.ndarray] = {}
    i += 1
    while i < n:
        line = lines[i]
        m = _BAND_RE.search(line)
        if m:
            label = int(m.group(1))
            vals: list[float] = []
            i += 1
            while i < n and not _BAND_RE.search(lines[i]) and not _END_RE.search(lines[i]):
                vals.extend(_to_float(t) for t in _FLOAT_RE.findall(lines[i]))
                i += 1
            if len(vals) < npts:
                raise ValueError(f'{path}: band {label} has {len(vals)} values, expected {npts}.')
            bands[label] = np.asarray(vals[:npts], dtype=float).reshape(dims)
            continue
        if _END_RE.search(line):
            break
        i += 1

    if len(bands) != nbnd:
        # Not fatal, just informational.
        print(
            f'warning: header declares {nbnd} bands, parsed {len(bands)}.',
            file=sys.stderr,
        )
    if not np.allclose(origin, 0.0):
        print(f'warning: non-zero grid origin {origin} ignored.', file=sys.stderr)

    return FermiSurfData(fermi_energy, dims, recip, bands)


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #


def fft_upsample(grid: np.ndarray, factor: int) -> np.ndarray:
    """Fourier zero-padding upsampling of a periodic energy grid.

    The PAOFLOW grid is periodic with a duplicated endpoint plane in each
    direction.  This routine strips the wrap plane, upsamples the base
    (period) grid by ``factor`` via Fourier zero-padding (exact for periodic
    band energies), and re-appends the wrap plane so the result stays closed
    for marching cubes.

    Parameters
    ----------
    grid : np.ndarray, shape (Nx, Ny, Nz)
        Wrap-closed energy grid (``N_i = n_i + 1``).
    factor : int
        Integer upsampling factor (``>= 1``).  ``1`` returns the input.

    Returns
    -------
    np.ndarray
        Upsampled, wrap-closed grid of shape
        ``(factor*n_x + 1, factor*n_y + 1, factor*n_z + 1)`` along axes with
        ``n_i > 1`` (axes of length 1 are left unchanged).
    """
    if factor <= 1:
        return grid

    active = tuple(s > 1 for s in grid.shape)  # axes carrying a periodic wrap
    base = grid[
        tuple(slice(0, s - 1) if a else slice(None) for s, a in zip(grid.shape, active))
    ]  # strip periodic wrap -> period grid
    ft = np.fft.fftn(base)
    old = base.shape
    # Only Fourier-upsample axes whose period has >1 sample; a constant axis
    # (period length 1, e.g. a 2-D k-grid) cannot be refined.
    new = tuple(s * factor if s > 1 else s for s in old)

    padded = np.zeros(new, dtype=complex)

    # Copy the frequency components into the enlarged spectrum, splitting the
    # Nyquist-symmetric halves so the inverse transform stays real.
    def _halves(o: int):
        pos = (o + 1) // 2  # number of non-negative freqs (incl. DC)
        neg = o - pos
        return pos, neg

    # General N-D placement of frequency blocks.
    def _place(src, dst):
        for combo in itertools.product(range(2), repeat=3):
            src_sl, dst_sl = [], []
            ok = True
            for ax in range(3):
                o = old[ax]
                ns = new[ax]
                pos, neg = _halves(o)
                if combo[ax] == 0:  # positive freqs
                    src_sl.append(slice(0, pos))
                    dst_sl.append(slice(0, pos))
                else:  # negative freqs
                    if neg == 0:
                        ok = False
                        break
                    src_sl.append(slice(o - neg, o))
                    dst_sl.append(slice(ns - neg, ns))
            if ok:
                dst[tuple(dst_sl)] = src[tuple(src_sl)]

    _place(ft, padded)

    scale = np.prod([ns / o for o, ns in zip(old, new)])
    up = np.fft.ifftn(padded).real * scale

    # Re-append the periodic wrap plane on every originally-active axis so the
    # grid stays closed for marching cubes (restores Nz=2 for 2-D k-grids too).
    out_shape = tuple(s + 1 if a else s for s, a in zip(up.shape, active))
    out = np.empty(out_shape, dtype=float)
    sx, sy, sz = up.shape
    out[:sx, :sy, :sz] = up
    if active[0]:
        out[sx, :sy, :sz] = up[0, :, :]
    if active[1]:
        out[:, sy, :sz] = out[:, 0, :sz]
    if active[2]:
        out[:, :, sz] = out[:, :, 0]
    return out


# --------------------------------------------------------------------------- #
# Geometry / velocity
# --------------------------------------------------------------------------- #


def fermi_velocity_field(energy: np.ndarray, recip: np.ndarray) -> np.ndarray:
    """Return the Fermi-velocity magnitude ``|grad_k E|`` on the grid.

    Parameters
    ----------
    energy : np.ndarray, shape (Nx, Ny, Nz)
        Wrap-closed energy grid (eV).
    recip : np.ndarray, shape (3, 3)
        Reciprocal spanning vectors (rows).

    Returns
    -------
    np.ndarray, shape (Nx, Ny, Nz)
        ``|grad_k E|`` at every grid node (eV per reciprocal-length unit).
    """
    dims = np.array(energy.shape)
    # Fractional spacing between nodes (period spans indices 0..N-1).
    span = np.where(dims > 1, dims - 1, 1)
    dfrac = 1.0 / span
    # np.gradient needs >=3 points for edge_order=2 and >=2 for edge_order=1;
    # constant axes (length 1) contribute a zero derivative.
    g_frac = np.zeros(energy.shape + (3,), dtype=float)
    for axis in range(3):
        if energy.shape[axis] < 2:
            continue
        eo = 2 if energy.shape[axis] >= 3 else 1
        g_frac[..., axis] = np.gradient(energy, dfrac[axis], axis=axis, edge_order=eo)
    binv = np.linalg.inv(recip)  # frac_i = k_j * binv[j, i]
    # grad_k_j = sum_i binv[j, i] * dE/dfrac_i
    grad_k = np.einsum('ji,...i->...j', binv, g_frac)
    return np.linalg.norm(grad_k, axis=-1)


def _box_edges(origin: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
    """Return the 12 line segments (list of ``(p0, p1)``) of a parallelepiped.

    Parameters
    ----------
    origin : np.ndarray, shape (3,)
        Cartesian position of the ``(0, 0, 0)`` corner.
    v0, v1, v2 : np.ndarray, shape (3,)
        Spanning vectors of the box.

    Returns
    -------
    list of tuple[np.ndarray, np.ndarray]
        Edge endpoint pairs.
    """
    vecs = (np.asarray(v0, dtype=float), np.asarray(v1, dtype=float), np.asarray(v2, dtype=float))
    corners = {
        key: origin + key[0] * vecs[0] + key[1] * vecs[1] + key[2] * vecs[2]
        for key in itertools.product((0, 1), repeat=3)
    }
    edges = []
    for key, p0 in corners.items():
        for axis in range(3):
            if key[axis] == 0:
                nb = list(key)
                nb[axis] = 1
                edges.append((p0, corners[tuple(nb)]))
    return edges


def _cell_edges(recip: np.ndarray):
    """Return line segments (list of (p0, p1)) for the reciprocal cell box."""
    return _box_edges(np.zeros(3), recip[0], recip[1], recip[2])


# --------------------------------------------------------------------------- #
# First Brillouin zone
# --------------------------------------------------------------------------- #


def reciprocal_neighbours(recip: np.ndarray, order: int = 2) -> np.ndarray:
    """Reciprocal-lattice vectors around the origin, excluding :math:`\\Gamma`.

    Parameters
    ----------
    recip : np.ndarray, shape (3, 3)
        Reciprocal spanning vectors (rows).
    order : int, optional
        Largest ``|n_i|`` in :math:`\\mathbf G = n_1\\mathbf b_1 + n_2\\mathbf b_2
        + n_3\\mathbf b_3` (default 2).

    Returns
    -------
    np.ndarray, shape ((2*order+1)**3 - 1, 3)
        Cartesian neighbour vectors.
    """
    rng = np.arange(-order, order + 1)
    idx = np.stack(np.meshgrid(rng, rng, rng, indexing='ij'), axis=-1).reshape(-1, 3)
    return idx[np.any(idx != 0, axis=1)].astype(float) @ recip


def reduce_basis(basis: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Greedily shorten a lattice basis without changing the lattice.

    Parameters
    ----------
    basis : np.ndarray, shape (3, 3)
        Spanning vectors (rows).
    iterations : int, optional
        Maximum reduction sweeps (default 50).

    Returns
    -------
    np.ndarray, shape (3, 3)
        A near-Minkowski-reduced basis of the same lattice.

    Notes
    -----
    Only integer multiples of one row are subtracted from another and rows are
    permuted, both unimodular operations, so the lattice — and hence its
    Wigner-Seitz cell — is unchanged.  Reduction matters because the number of
    neighbour shells needed to close the Brillouin zone depends on how skewed
    the *basis* is, not on the lattice itself.
    """
    reduced = np.array(basis, dtype=float)
    for _ in range(iterations):
        reduced = reduced[np.argsort(np.linalg.norm(reduced, axis=1))]
        changed = False
        for i in range(3):
            for j in range(i + 1, 3):
                factor = np.dot(reduced[j], reduced[i]) / np.dot(reduced[i], reduced[i])
                step = int(np.round(factor))
                if step != 0:
                    reduced[j] -= step * reduced[i]
                    changed = True
        if not changed:
            break
    return reduced


def _facet_polygons(verts: np.ndarray, hull, tol: int = 6) -> list[np.ndarray]:
    """Merge the coplanar hull simplices into cyclically ordered polygons."""
    groups: dict[tuple, set[int]] = {}
    for simplex, equation in zip(hull.simplices, hull.equations):
        groups.setdefault(tuple(np.round(equation, tol)), set()).update(int(i) for i in simplex)

    polygons = []
    for equation, members in groups.items():
        idx = np.array(sorted(members))
        pts = verts[idx]
        axis_u, axis_v = plane_basis(np.array(equation[:3]))
        rel = pts - pts.mean(axis=0)
        polygons.append(idx[np.argsort(np.arctan2(rel @ axis_v, rel @ axis_u))])
    return polygons


def brillouin_zone(recip: np.ndarray, max_order: int = 4) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build the first Brillouin zone of an arbitrary reciprocal lattice.

    Parameters
    ----------
    recip : np.ndarray, shape (3, 3)
        Reciprocal spanning vectors (rows).
    max_order : int, optional
        Largest neighbour shell tried before giving up (default 4).

    Returns
    -------
    vertices : np.ndarray, shape (nv, 3)
        BZ corners, centred on :math:`\\Gamma`.
    faces : list[np.ndarray]
        One array of vertex indices per facet, ordered cyclically.

    Raises
    ------
    RuntimeError
        If no neighbour shell up to ``max_order`` reproduces the primitive
        cell volume.

    Notes
    -----
    The first BZ is the Wigner-Seitz cell of the reciprocal lattice: the set
    of k closer to :math:`\\Gamma` than to any other reciprocal-lattice point.
    It is obtained here as the Voronoi region of the origin, which makes the
    construction purely geometric and therefore identical for every Bravais
    lattice — no per-crystal-system tabulation is involved.

    The basis is reduced first (see :func:`reduce_basis`) and the neighbour
    shell is then grown until the polyhedron volume matches
    :math:`|\\det(\\mathbf b)|`.  Because the Wigner-Seitz cell tiles space,
    that equality is a sufficient check that no bounding plane was missed.
    """
    from scipy.spatial import ConvexHull, Voronoi

    target = abs(float(np.linalg.det(recip)))
    if target <= 0.0:
        raise RuntimeError('reciprocal vectors are degenerate; cannot build a Brillouin zone.')

    reduced = reduce_basis(recip)
    for order in range(1, max_order + 1):
        points = np.vstack([np.zeros((1, 3)), reciprocal_neighbours(reduced, order)])
        voronoi = Voronoi(points)
        region = voronoi.regions[voronoi.point_region[0]]
        if -1 in region or len(region) < 4:
            continue
        verts = voronoi.vertices[region]
        hull = ConvexHull(verts)
        if abs(hull.volume - target) <= 1.0e-6 * target:
            return verts, _facet_polygons(verts, hull)

    raise RuntimeError(
        f'could not close the Brillouin zone within neighbour order {max_order}; '
        'the reciprocal basis is unusually skewed.'
    )


def bz_edges(verts: np.ndarray, faces: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the unique polyhedron edges as ``(p0, p1)`` endpoint pairs."""
    seen = set()
    edges = []
    for poly in faces:
        for a, b in zip(poly, np.roll(poly, -1)):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key not in seen:
                seen.add(key)
                edges.append((verts[a], verts[b]))
    return edges


def bz_planes(verts: np.ndarray, faces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Outward half-space description of the Brillouin zone.

    Parameters
    ----------
    verts : np.ndarray, shape (nv, 3)
        BZ corners from :func:`brillouin_zone`.
    faces : list[np.ndarray]
        Facet polygons from :func:`brillouin_zone`.

    Returns
    -------
    normals : np.ndarray, shape (nf, 3)
        Outward unit normals.
    offsets : np.ndarray, shape (nf,)
        Plane distances from :math:`\\Gamma`, so the zone interior is
        ``k @ normals[i] <= offsets[i]`` for every facet.
    """
    normals = np.empty((len(faces), 3))
    offsets = np.empty(len(faces))
    for i, poly in enumerate(faces):
        pts = verts[poly]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        normal /= np.linalg.norm(normal)
        centre = pts.mean(axis=0)
        if np.dot(normal, centre) < 0.0:
            normal = -normal
        normals[i] = normal
        offsets[i] = np.dot(normal, centre)
    return normals, offsets


def select_near_bz(
    cart: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    tol: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop triangles that lie wholly outside the Brillouin zone.

    Parameters
    ----------
    cart : np.ndarray, shape (nv, 3)
        Vertex positions.
    faces : np.ndarray, shape (nf, 3)
        Triangle vertex indices.
    scalars : np.ndarray, shape (nv,)
        Per-vertex scalars.
    normals, offsets : np.ndarray
        Half-space description from :func:`bz_planes`.
    tol : float, optional
        Slack on the outside test, in reciprocal-length units.

    Returns
    -------
    tuple of np.ndarray
        ``(vertices, faces, scalars)`` keeping only triangles that touch the
        zone.  Vertices are unshared (three per surviving triangle).

    Notes
    -----
    This is the trivial-reject half of the clip: a triangle whose three
    corners all violate the *same* facet plane cannot intersect the zone.
    Triangles straddling a face are kept whole here and cut exactly later by
    the renderer, so no geometry is lost.
    """
    tri = cart[faces]
    # (nf, 3 corners, nplanes) signed distances outside each facet plane
    outside = np.einsum('fck,pk->fcp', tri, normals) - offsets[None, None, :] > tol
    keep = ~outside.all(axis=1).any(axis=1)

    tri = tri[keep]
    values = scalars[faces][keep]
    nface = tri.shape[0]
    return (
        tri.reshape(-1, 3),
        np.arange(3 * nface).reshape(nface, 3),
        values.reshape(-1),
    )


def _clip_one_plane(tri, values, normal, offset):
    """Sutherland-Hodgman clip of triangles against a single half-space."""
    dist = tri @ normal - offset
    inside = dist <= 0.0
    count = inside.sum(axis=1)

    def rotate(sel, pivot):
        """Reorder each triangle so the odd-one-out vertex comes first."""
        order = (pivot[:, None] + np.arange(3)[None, :]) % 3
        return (
            np.take_along_axis(tri[sel], order[..., None], axis=1),
            np.take_along_axis(values[sel], order, axis=1),
            np.take_along_axis(dist[sel], order, axis=1),
        )

    def cut(v0, v1, s0, s1, d0, d1):
        frac = (d0 / (d0 - d1))[..., None]
        return v0 + frac * (v1 - v0), s0 + frac[..., 0] * (s1 - s0)

    pieces, weights = [], []

    whole = count == 3
    if whole.any():
        pieces.append(tri[whole])
        weights.append(values[whole])

    one = count == 1
    if one.any():
        v, s, d = rotate(one, np.argmax(inside[one], axis=1))
        a, sa = cut(v[:, 0], v[:, 1], s[:, 0], s[:, 1], d[:, 0], d[:, 1])
        b, sb = cut(v[:, 0], v[:, 2], s[:, 0], s[:, 2], d[:, 0], d[:, 2])
        pieces.append(np.stack([v[:, 0], a, b], axis=1))
        weights.append(np.stack([s[:, 0], sa, sb], axis=1))

    two = count == 2
    if two.any():
        v, s, d = rotate(two, np.argmin(inside[two], axis=1))
        a, sa = cut(v[:, 0], v[:, 1], s[:, 0], s[:, 1], d[:, 0], d[:, 1])
        b, sb = cut(v[:, 0], v[:, 2], s[:, 0], s[:, 2], d[:, 0], d[:, 2])
        pieces.append(np.stack([a, v[:, 1], v[:, 2]], axis=1))
        weights.append(np.stack([sa, s[:, 1], s[:, 2]], axis=1))
        pieces.append(np.stack([a, v[:, 2], b], axis=1))
        weights.append(np.stack([sa, s[:, 2], sb], axis=1))

    if not pieces:
        return np.empty((0, 3, 3)), np.empty((0, 3))
    return np.concatenate(pieces), np.concatenate(weights)


def clip_to_planes(
    tri: np.ndarray, values: np.ndarray, normals: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Clip triangles to the convex region ``k @ normals[i] <= offsets[i]``.

    Parameters
    ----------
    tri : np.ndarray, shape (nf, 3, 3)
        Triangle corner positions.
    values : np.ndarray, shape (nf, 3)
        Per-corner scalars, interpolated linearly onto any new corner.
    normals : np.ndarray, shape (np, 3)
        Outward unit normals of the bounding half-spaces.
    offsets : np.ndarray, shape (np,)
        Plane distances from the origin.

    Returns
    -------
    tuple of np.ndarray
        Clipped ``(tri, values)``.  Triangles crossing a plane are cut and
        re-triangulated, so the trimmed surface follows the boundary exactly
        instead of overhanging it.
    """
    for normal, offset in zip(normals, offsets):
        if tri.shape[0] == 0:
            break
        tri, values = _clip_one_plane(tri, values, normal, offset)
    return tri, values


def fold_into_bz(
    cart: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    recip: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather the periodic images of a sheet that reach into the first BZ.

    Parameters
    ----------
    cart : np.ndarray, shape (nv, 3)
        Vertex positions, spanning the reciprocal parallelepiped.
    faces : np.ndarray, shape (nf, 3)
        Triangle vertex indices.
    scalars : np.ndarray, shape (nv,)
        Per-vertex scalars.
    recip : np.ndarray, shape (3, 3)
        Reciprocal spanning vectors (rows).
    normals, offsets : np.ndarray
        Half-space description from :func:`bz_planes`.

    Returns
    -------
    tuple of np.ndarray
        ``(vertices, faces, scalars)`` of the sheet trimmed to the zone.
        Vertices are unshared (three per triangle).

    Notes
    -----
    The iso-surface is periodic, so every part of the zone is covered by some
    lattice translation of the parallelepiped.  All 27 translations of the
    reduced basis are laid down, trivially rejected against the zone, and then
    clipped exactly.  Because the Wigner-Seitz cell and the primitive cell
    have equal volume, the result holds about one cell's worth of triangles.
    """
    images = np.vstack([np.zeros((1, 3)), reciprocal_neighbours(reduce_basis(recip), 1)])
    cart, faces, scalars = select_near_bz(
        *tile_mesh(cart, faces, scalars, images), normals, offsets
    )

    tri, values = clip_to_planes(cart[faces], scalars[faces], normals, offsets)
    nface = tri.shape[0]
    return (
        tri.reshape(-1, 3),
        np.arange(3 * nface).reshape(nface, 3),
        values.reshape(-1),
    )


# --------------------------------------------------------------------------- #
# Supercell replication
# --------------------------------------------------------------------------- #


def central_replica(ncell: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return the index of the replica closest to the centre of the block."""
    cx, cy, cz = (int(n) // 2 for n in ncell)
    return cx, cy, cz


def supercell_translations(
    recip: np.ndarray,
    ncell: tuple[int, int, int] = (1, 1, 1),
    center: bool = False,
) -> np.ndarray:
    """Cartesian offsets replicating the reciprocal cell over an ``ncell`` block.

    Parameters
    ----------
    recip : np.ndarray, shape (3, 3)
        Reciprocal spanning vectors (rows).
    ncell : tuple[int, int, int], optional
        Number of repetitions along each reciprocal vector (default no
        replication).  Replicas grow in the ``+b_i`` directions, so element
        ``0`` is always the original cell.
    center : bool, optional
        Shift the block so the origin of the central replica sits at the
        Cartesian origin.

    Returns
    -------
    np.ndarray, shape (nx*ny*nz, 3)
        Translation :math:`i\\mathbf b_0 + j\\mathbf b_1 + k\\mathbf b_2` for each
        replica, in C order over ``(i, j, k)``.

    Notes
    -----
    The centring shift is an *integer* combination of reciprocal vectors, so
    the Cartesian origin always lands on a :math:`\\Gamma` point and no vertex
    needs to be wrapped (wrapping fractional coordinates into
    :math:`[-1/2, 1/2)` would tear every triangle crossing a cell face).  A
    single cell has no interior replica, so ``center`` is then a no-op.
    """
    nx, ny, nz = (int(n) for n in ncell)
    if min(nx, ny, nz) < 1:
        raise ValueError(f'supercell repetitions must be >= 1, got {ncell}')

    idx = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'),
        axis=-1,
    ).reshape(-1, 3)
    trans = idx.astype(float) @ recip
    if center:
        trans = trans - np.asarray(central_replica(ncell), dtype=float) @ recip
    return trans


def tile_mesh(
    cart: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate a triangular mesh by a set of translations.

    Parameters
    ----------
    cart : np.ndarray, shape (nv, 3)
        Vertex positions.
    faces : np.ndarray, shape (nf, 3)
        Triangle vertex indices.
    scalars : np.ndarray, shape (nv,)
        Per-vertex scalars (the Fermi speed used for colouring).
    translations : np.ndarray, shape (nc, 3)
        Offsets from :func:`supercell_translations`.

    Returns
    -------
    tuple of np.ndarray
        Concatenated ``(vertices, faces, scalars)`` for all replicas, so the
        whole block can be drawn as a single Mayavi actor.

    Notes
    -----
    The BXSF grid carries the periodic wrap plane, so the iso-surface is
    continuous across cell faces and plain translation tiles it seamlessly —
    no second marching-cubes pass is needed.
    """
    ncopy = translations.shape[0]
    nvert = cart.shape[0]
    cart_t = (cart[None, :, :] + translations[:, None, :]).reshape(-1, 3)
    faces_t = (faces[None, :, :] + (np.arange(ncopy) * nvert)[:, None, None]).reshape(
        -1, faces.shape[1]
    )
    return cart_t, faces_t, np.tile(scalars, ncopy)


# --------------------------------------------------------------------------- #
# SKEAF field geometry
# --------------------------------------------------------------------------- #


@dataclass
class FieldSpec:
    """Magnetic-field geometry to overlay, in SKEAF conventions.

    Attributes
    ----------
    hvd : str
        SKEAF field selector: ``'a'``, ``'b'``, ``'c'`` align with reciprocal
        vector 0, 1, 2; ``'n'`` uses ``theta``/``phi``; ``'r'`` sweeps from
        ``(theta, phi)`` to ``(theta_end, phi_end)``.
    theta, phi : float
        Azimuth and polar angle in **radians** (``theta`` is PAOFLOW's
        ``azimuthal``, ``phi`` its ``polar``).
    theta_end, phi_end : float
        Sweep endpoint in radians; used only for ``hvd='r'``.
    num_angles : int
        Number of orientations in the sweep.
    show_plane : bool
        Draw a translucent plane normal to the field, i.e. the plane SKEAF
        slices the Fermi surface with.
    show_label : bool
        Annotate the arrow with its angles.
    """

    hvd: str = 'n'
    theta: float = 0.0
    phi: float = 0.0
    theta_end: float = 0.0
    phi_end: float = 0.0
    num_angles: int = 1
    show_plane: bool = False
    show_label: bool = False


def resolve_field_angles(
    recip: np.ndarray, hvd: str, theta: float = 0.0, phi: float = 0.0
) -> tuple[float, float]:
    """Resolve a SKEAF ``hvd`` selector to ``(theta, phi)`` in radians.

    Delegates to :func:`PAOFLOW.pyskeaf.geometry.set_field_angle` so the arrow
    can never drift from the geometry SKEAF actually uses.  Only the direction
    of ``recip`` matters, so its scaling (BXSF stores :math:`2\\pi/a` units,
    pyskeaf documents |AA|:sup:`-1`) is irrelevant.
    """
    # Deferred: importing PAOFLOW.pyskeaf pulls in its runner (joblib/numba).
    from PAOFLOW.pyskeaf.geometry import set_field_angle

    return set_field_angle(np.asarray(recip, dtype=float), cast(Any, hvd), theta, phi)


def field_direction(theta: float, phi: float) -> np.ndarray:
    """Cartesian unit vector of the field axis for angles in radians.

    Notes
    -----
    Inverts the SKEAF convention ``theta = atan2(v_y, v_x)``,
    ``phi = arccos(v_z / |v|)``:

    .. math::

        \\hat{\\mathbf H} = (\\sin\\phi\\cos\\theta,\\;
                            \\sin\\phi\\sin\\theta,\\;
                            \\cos\\phi)
    """
    sin_phi = math.sin(phi)
    return np.array([sin_phi * math.cos(theta), sin_phi * math.sin(theta), math.cos(phi)])


def field_sweep_arc(theta0: float, phi0: float, theta1: float, phi1: float, num: int) -> np.ndarray:
    """Unit vectors traced by a SKEAF rotation sweep.

    Parameters
    ----------
    theta0, phi0, theta1, phi1 : float
        Sweep endpoints in radians.
    num : int
        Number of orientations (clamped to at least 2).

    Returns
    -------
    np.ndarray, shape (num, 3)
        Field directions at each step of the sweep.

    Notes
    -----
    ``theta`` and ``phi`` are stepped *independently and linearly*, matching
    :func:`PAOFLOW.pyskeaf.runner.run_angle_sweep`; the path is therefore not
    a great circle unless the endpoints happen to lie on one.
    """
    npts = max(2, int(num))
    thetas = np.linspace(theta0, theta1, npts)
    phis = np.linspace(phi0, phi1, npts)
    sin_phi = np.sin(phis)
    return np.stack([sin_phi * np.cos(thetas), sin_phi * np.sin(thetas), np.cos(phis)], axis=1)


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane normal to ``normal``."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _draw_field(mlab, fig, recip, field: FieldSpec, anchor: np.ndarray, length: float) -> None:
    """Overlay the SKEAF field axis (and optionally its slice plane) on ``fig``."""
    arrow_color = (0.05, 0.25, 0.85)

    theta, phi = resolve_field_angles(recip, field.hvd, field.theta, field.phi)
    directions = [(theta, phi, field_direction(theta, phi))]

    if field.hvd == 'r':
        theta_end, phi_end = field.theta_end, field.phi_end
        arc = field_sweep_arc(theta, phi, theta_end, phi_end, field.num_angles)
        directions.append((theta_end, phi_end, arc[-1]))
        path = anchor + 0.75 * length * arc
        mlab.plot3d(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=arrow_color,
            tube_radius=0.008 * length,
            figure=fig,
        )

    for ang_t, ang_p, vec in directions:
        quiver = mlab.quiver3d(
            anchor[0],
            anchor[1],
            anchor[2],
            vec[0],
            vec[1],
            vec[2],
            mode='arrow',
            scale_factor=length,
            color=arrow_color,
            resolution=24,
            figure=fig,
        )
        # VTK's default arrow glyph is far too stubby at these scales.
        quiver.glyph.glyph_source.glyph_position = 'tail'
        source = quiver.glyph.glyph_source.glyph_source
        source.shaft_radius = 0.012
        source.tip_radius = 0.045
        source.tip_length = 0.18
        if field.show_label:
            tip = anchor + 1.05 * length * vec
            mlab.text3d(
                tip[0],
                tip[1],
                tip[2],
                f'H ({math.degrees(ang_t):.0f}, {math.degrees(ang_p):.0f})',
                color=arrow_color,
                scale=0.07 * length,
                figure=fig,
            )

    if field.show_plane:
        # SKEAF searches for extremal orbits on planes normal to H.
        normal = directions[0][2]
        u, v = plane_basis(normal)
        half = 0.9 * length
        corners = np.array(
            [[anchor + su * half * u + sv * half * v for sv in (-1.0, 1.0)] for su in (-1.0, 1.0)]
        )
        mlab.mesh(
            corners[..., 0],
            corners[..., 1],
            corners[..., 2],
            color=(0.4, 0.55, 0.95),
            opacity=0.25,
            figure=fig,
        )


def _draw_edges(mlab, fig, edges, color, line_width) -> None:
    """Draw a set of disjoint line segments as one actor."""
    pts = np.asarray(edges, dtype=float).reshape(-1, 3)
    src = mlab.pipeline.scalar_scatter(pts[:, 0], pts[:, 1], pts[:, 2], figure=fig)
    src.mlab_source.dataset.lines = np.arange(pts.shape[0]).reshape(-1, 2)
    src.update()
    mlab.pipeline.surface(
        mlab.pipeline.stripper(src, figure=fig),
        color=color,
        line_width=line_width,
        figure=fig,
    )


def plot_fermi_surface(
    data: FermiSurfData,
    band_labels: list[int],
    interp: int = 1,
    fermi_shift: float = 0.0,
    cmap: str = 'jet',
    opacity: float = 1.0,
    bz: str = 'ws',
    supercell: tuple[int, int, int] = (1, 1, 1),
    center: bool = False,
    field: FieldSpec | None = None,
    figsize: tuple[int, int] = (900, 700),
    save: str | None = None,
) -> None:
    """Render the selected Fermi sheets with Mayavi.

    Parameters
    ----------
    data : FermiSurfData
        Parsed BXSF contents.
    band_labels : list[int]
        BXSF band labels to render.  A single element gives a one-band image;
        multiple elements give the composite image.
    interp : int, optional
        Integer Fourier upsampling factor for smoother sheets (default 1).
    fermi_shift : float, optional
        Energy offset (eV) added to the Fermi level defining the iso-surface.
    cmap : str, optional
        Mayavi/VTK colormap name for the velocity colouring.
    opacity : float, optional
        Surface opacity in ``[0, 1]``.
    bz : {'ws', 'cell', 'none'}, optional
        Zone drawn and used to place the sheets.  ``'ws'`` (default) folds the
        sheets into the true first Brillouin zone and outlines that
        polyhedron; ``'cell'`` keeps the raw reciprocal parallelepiped;
        ``'none'`` draws no outline (sheets stay in the parallelepiped).
    supercell : tuple[int, int, int], optional
        Number of cell repetitions along each reciprocal vector (default
        ``(1, 1, 1)``).  The sheets are replicated by translation.
    center : bool, optional
        Put the central replica's origin (a :math:`\\Gamma` point) at the
        Cartesian origin.  No-op for a single cell, and redundant with
        ``bz='ws'``, whose zone is centred on :math:`\\Gamma` already.
    field : FieldSpec, optional
        SKEAF magnetic-field geometry to overlay.  ``None`` draws nothing.
    figsize : tuple[int, int], optional
        Render window size in pixels.
    save : str, optional
        If given, render and save a PNG to this path instead of opening an
        interactive window.  (VTK on macOS has no OSMesa off-screen backend, so
        this still renders through a normal on-screen window.)
    """
    from mayavi import mlab
    from scipy.ndimage import map_coordinates
    from skimage import measure

    if bz not in ('ws', 'cell', 'none'):
        raise ValueError(f"bz must be 'ws', 'cell' or 'none', got {bz!r}")

    level = data.fermi_energy + fermi_shift
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=figsize)

    trans = supercell_translations(data.recip, supercell, center)
    ncopy = trans.shape[0]
    if ncopy > 64:
        print(
            f'warning: {ncopy} replicas requested; rendering may be slow.',
            file=sys.stderr,
        )

    zone = brillouin_zone(data.recip) if bz == 'ws' else None
    planes = bz_planes(*zone) if zone is not None else None

    # Global velocity range across selected bands for a shared colour scale.
    speeds_min, speeds_max = np.inf, -np.inf
    prepared = []
    for label in band_labels:
        energy = data.bands[label]
        if interp > 1:
            energy = fft_upsample(energy, interp)
        emin, emax = energy.min(), energy.max()
        if not (emin <= level <= emax):
            print(
                f'band {label}: does not cross E={level:.4f} eV '
                f'(range [{emin:.3f}, {emax:.3f}]); skipping.',
                file=sys.stderr,
            )
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(energy, level=level)
        except (ValueError, RuntimeError) as exc:
            print(f'band {label}: marching cubes failed ({exc}); skipping.', file=sys.stderr)
            continue
        speed = fermi_velocity_field(energy, data.recip)
        prepared.append((label, energy, verts, faces, speed))
        speeds_min = min(speeds_min, speed.min())
        speeds_max = max(speeds_max, speed.max())

    if not prepared:
        print('No band crosses the Fermi level; nothing to plot.', file=sys.stderr)
        return

    for label, energy, verts, faces, speed in prepared:
        dims = np.array(energy.shape)
        span = np.where(dims > 1, dims - 1, 1)
        frac = verts / span
        cart = frac @ data.recip
        vspeed = map_coordinates(speed, verts.T, order=1, mode='nearest')
        if planes is not None:
            cart, faces, vspeed = fold_into_bz(cart, faces, vspeed, data.recip, *planes)
        cart, faces, vspeed = tile_mesh(cart, faces, vspeed, trans)
        mesh = mlab.triangular_mesh(
            cart[:, 0],
            cart[:, 1],
            cart[:, 2],
            faces,
            scalars=vspeed,
            colormap=cmap,
            opacity=opacity,
            vmin=speeds_min,
            vmax=speeds_max,
            figure=fig,
        )
        mesh.name = f'band_{label}'

    block_origin = trans[0]  # supercell_translations orders the (0,0,0) replica first
    block_span = np.asarray(supercell, dtype=float)[:, None] * data.recip
    central_offset = block_origin + np.asarray(central_replica(supercell), float) @ data.recip

    if zone is not None:
        edges_bz = bz_edges(*zone)
        outer = [(p0 + t, p1 + t) for t in trans for p0, p1 in edges_bz]
        _draw_edges(mlab, fig, outer, (0.3, 0.3, 0.3), 1.5)
        if ncopy > 1:
            inner = [(p0 + central_offset, p1 + central_offset) for p0, p1 in edges_bz]
            _draw_edges(mlab, fig, inner, (0.85, 0.15, 0.15), 3.0)
    elif bz == 'cell':
        _draw_edges(mlab, fig, _box_edges(block_origin, *block_span), (0.3, 0.3, 0.3), 1.5)
        if ncopy > 1:
            _draw_edges(mlab, fig, _box_edges(central_offset, *data.recip), (0.85, 0.15, 0.15), 3.0)

    if field is not None:
        anchor = central_offset if zone is not None else block_origin + 0.5 * block_span.sum(axis=0)
        length = 0.6 * np.linalg.norm(block_span, axis=1).max()
        _draw_field(mlab, fig, data.recip, field, anchor, length)

    mlab.colorbar(title='|grad E|  (Fermi velocity)', orientation='vertical', nb_labels=5)
    mlab.orientation_axes()

    if save is not None:
        mlab.savefig(save, size=figsize)
        mlab.close(fig)
        print(f'saved {save}')
    else:
        mlab.show()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_bands(spec: str | None, available: list[int]) -> list[int]:
    if spec is None or spec.lower() in ('all', 'composite'):
        return sorted(available)
    labels = [int(x) for x in re.split(r'[,\s]+', spec.strip()) if x]
    missing = [b for b in labels if b not in available]
    if missing:
        raise SystemExit(f'band(s) {missing} not in file; available: {sorted(available)}')
    return labels


def _parse_supercell(spec: str) -> tuple[int, int, int]:
    """Parse ``"2"`` or ``"2,2,3"`` into a repetition triple."""
    parts = [p for p in re.split(r'[,\s x]+', spec.strip().lower()) if p]
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        raise SystemExit(f'bad --supercell "{spec}", expected N or NX,NY,NZ.') from None
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3 or min(vals) < 1:
        raise SystemExit(f'bad --supercell "{spec}", expected N or NX,NY,NZ with each >= 1.')
    return (vals[0], vals[1], vals[2])


def _parse_angles(spec: str, name: str) -> tuple[float, ...]:
    """Parse a scalar or a ``start,end`` pair of angles in degrees."""
    parts = [p for p in re.split(r'[,\s]+', spec.strip()) if p]
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit(f'bad {name} "{spec}", expected a number or "start,end".') from None
    if len(vals) not in (1, 2):
        raise SystemExit(f'bad {name} "{spec}", expected a number or "start,end".')
    return vals


def _build_field(args) -> FieldSpec | None:
    """Translate the pyskeaf-style CLI flags into a :class:`FieldSpec`."""
    if args.b_field is None:
        return None

    field_map = {'b1': 'a', 'b2': 'b', 'b3': 'c', 'non_principal': 'n', 'rotation': 'r'}
    hvd = field_map[args.b_field]

    azimuthal = _parse_angles(args.azimuthal, '--azimuthal')
    polar = _parse_angles(args.polar, '--polar')
    if len(azimuthal) != len(polar):
        raise SystemExit('--azimuthal and --polar must both be scalars or both be pairs.')

    # Same rule as PAOFLOW.pyskeaf(): two-element angle pairs mean a rotation.
    rotating = len(azimuthal) == 2
    if rotating:
        hvd = 'r'
        if args.num_angles < 2:
            raise SystemExit('--num-angles must be at least 2 for a rotation.')
    elif hvd == 'r':
        raise SystemExit("--b-field rotation requires 'start,end' --azimuthal and --polar.")

    return FieldSpec(
        hvd=hvd,
        theta=math.radians(azimuthal[0]),
        phi=math.radians(polar[0]),
        theta_end=math.radians(azimuthal[-1]),
        phi_end=math.radians(polar[-1]),
        num_angles=args.num_angles if rotating else 1,
        show_plane=args.field_plane,
        show_label=args.field_label,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='fermi-plotter',
        description='Interactive Mayavi Fermi-surface viewer for PAOFLOW BXSF files.',
    )
    p.add_argument('bxsf', help='Path to a FermiSurf_{ispin}.bxsf file.')
    p.add_argument(
        '--band',
        default=None,
        help='Band label(s) to plot: e.g. "58" (single) or "57,58" (subset). '
        'Default: composite (all bands in the file).',
    )
    p.add_argument(
        '--interp',
        type=int,
        default=1,
        metavar='N',
        help='Integer Fourier upsampling factor for smoother sheets (default 1).',
    )
    p.add_argument(
        '--fermi-shift',
        type=float,
        default=0.0,
        metavar='dE',
        help='Energy offset (eV) added to the Fermi level (default 0).',
    )
    p.add_argument('--cmap', default='jet', help='Colormap for velocity colouring (default jet).')
    p.add_argument('--opacity', type=float, default=1.0, help='Surface opacity 0..1 (default 1).')
    p.add_argument(
        '--bz',
        choices=('ws', 'cell', 'none'),
        default='ws',
        help='Zone to draw: "ws" (default) folds the sheets into the true first '
        'Brillouin zone, as XCrySDen does; "cell" keeps the raw reciprocal '
        'parallelepiped; "none" draws no outline.',
    )
    p.add_argument(
        '--no-bz', action='store_true', help='Hide the zone outline (same as --bz none).'
    )
    p.add_argument(
        '--supercell',
        default='1',
        metavar='N|NX,NY,NZ',
        help='Replicate the Fermi sheets over this many reciprocal cells (default 1).',
    )
    p.add_argument(
        '--center',
        action='store_true',
        help='Put a Gamma point at the centre of the displayed block. Needs a '
        'supercell to have any effect (e.g. --supercell 2 --center).',
    )
    p.add_argument(
        '--b-field',
        choices=('b1', 'b2', 'b3', 'non_principal', 'rotation'),
        default=None,
        help='Draw the SKEAF magnetic-field axis, using the same selector as '
        'PAOFLOW.pyskeaf(). Omit to draw no field.',
    )
    p.add_argument(
        '--azimuthal',
        default='0',
        metavar='DEG|D0,D1',
        help='Field azimuth theta in degrees; a "start,end" pair implies a rotation.',
    )
    p.add_argument(
        '--polar',
        default='0',
        metavar='DEG|D0,D1',
        help='Field polar angle phi in degrees; a "start,end" pair implies a rotation.',
    )
    p.add_argument(
        '--num-angles',
        type=int,
        default=1,
        metavar='N',
        help='Number of orientations sampled along a field rotation (default 1).',
    )
    p.add_argument(
        '--field-plane',
        action='store_true',
        help='Also draw the translucent plane normal to the field, i.e. the plane '
        'SKEAF slices the Fermi surface with. Pair with --opacity < 1 to see it.',
    )
    p.add_argument(
        '--field-label',
        action='store_true',
        help='Annotate the field arrow with its (theta, phi) in degrees.',
    )
    p.add_argument(
        '--size', default='900x700', metavar='WxH', help='Window size in pixels (default 900x700).'
    )
    p.add_argument(
        '--save',
        default=None,
        metavar='PNG',
        help='Render and save this PNG instead of opening an interactive window.',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``fermi-plotter``."""
    args = _build_parser().parse_args(argv)
    path = Path(args.bxsf)
    if not path.exists():
        print(f'error: file not found: {path}', file=sys.stderr)
        return 2

    data = read_fermi_bxsf(path)
    labels = _parse_bands(args.band, list(data.bands.keys()))
    supercell = _parse_supercell(args.supercell)
    field = _build_field(args)
    try:
        w, h = (int(v) for v in args.size.lower().split('x'))
    except ValueError:
        print(f'error: bad --size "{args.size}", expected WxH.', file=sys.stderr)
        return 2

    print(
        f'{path.name}: Efermi={data.fermi_energy:.4f} eV, grid={data.dims}, '
        f'bands={sorted(data.bands.keys())}; plotting {labels}.'
    )
    try:
        plot_fermi_surface(
            data,
            labels,
            interp=args.interp,
            fermi_shift=args.fermi_shift,
            cmap=args.cmap,
            opacity=args.opacity,
            bz='none' if args.no_bz else args.bz,
            supercell=supercell,
            center=args.center,
            field=field,
            figsize=(w, h),
            save=args.save,
        )
    except ImportError as exc:
        print(
            f'Missing dependency: {exc}.\n'
            'Install the viewer requirements with:\n'
            '    pip install "PAOFLOW[fermisurface]"   # or: pip install mayavi scikit-image',
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
