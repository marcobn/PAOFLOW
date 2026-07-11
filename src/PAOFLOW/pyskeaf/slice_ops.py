"""Construction of 2D energy slices perpendicular to the magnetic-field axis.

Reproduces the per-slice grid construction at the top of ``sliceext``
(skeaf_v1p3p0_r149.F90 lines 3853–3886) and the slice-frame setup at
lines 1249–1264 of the main program.

Geometry
--------
The slicing frame has axes (x', y', z') with **z' = magnetic-field
direction**.  In terms of (theta, phi) — both in radians, where ``phi`` is
the polar angle from +z and ``theta`` the azimuth in the xy-plane — the
H-vector in the BZ frame is

    Ĥ = (cos θ sin φ,  sin θ sin φ,  cos φ).

The Fortran builds the rotation matrix from slicing → BZ frame using

    p = sin θ,  q = cos θ,  s = sin φ,  c = cos φ,  u = 1 − cos φ

via the explicit form

.. code-block:: none

         | p²u + c    −p q u    q s |
    R =  | −p q u     q²u + c   p s |
         | −q s       −p s      c   |

so that ``(p1, p2, p3) = R · (X', Y', Z')`` with the columns being the
BZ-frame images of the slicing x', y', z' axes.

Supercell sampling
------------------
The orbit-detection supercell has

    numx = numy = numslices = 4·(numint − 1) + 1

points along each side, with spacing

    Δk = maxlreciplat / (numint − 1)         (Å^-1)

where ``maxlreciplat = max(|b_1|, |b_2|, |b_3|)``.  The slicing-frame
coordinates of grid index ``i ∈ [1, numx]`` are

    X'_i = ((i − 1) / (numint − 1) − 1) · maxlreciplat

i.e. the supercell spans ``[−1, +3] · maxlreciplat`` in each slicing
direction (asymmetric: matches Fortran).  The total supercell side length is
``xlength = ylength = 4 · maxlreciplat``.

After rotating each (X', Y', Z') sample into the BZ frame, the result is
mapped to BXSF fractional grid indices and wrapped periodically with
``mod (n − 1)`` (the same trick as ``floor(fintpoint/(n-1))*(n-1)`` in
``sliceext``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PAOFLOW.pyskeaf.geometry import k_axis_lengths
from PAOFLOW.pyskeaf.io_bxsf import BXSFData


@dataclass
class SliceGeometry:
    """Geometric metadata for one slicing configuration (independent of slice index)."""

    theta: float  # rad
    phi: float  # rad
    numint: int
    numx: int  # = numy = numslices = 4*(numint-1) + 1
    xkseparation: float  # Å^-1, spacing along slicing-x
    ykseparation: float  # Å^-1, spacing along slicing-y
    zkseparation: float  # Å^-1, spacing along slicing-z (between slices)
    xlength: float  # 4 * maxlreciplat, Å^-1
    ylength: float  # 4 * maxlreciplat, Å^-1
    maxlreciplat: float  # max |b_i|
    rotation: np.ndarray  # (3, 3): slicing-frame → BZ-frame
    h_vector: np.ndarray  # (3,) unit H-vector in BZ frame
    plr_inverse: np.ndarray  # (3, 3): BZ-frame Cartesian → fractional


@dataclass
class Slice2D:
    """Energies sampled on a 2D (numx × numy) slice perpendicular to H."""

    geometry: SliceGeometry
    slice_index: int  # 1-based, in [1, numx]
    z_prime: float  # slicing-frame z' coordinate (Å^-1)
    energies: np.ndarray  # shape (numx, numy), Ryd


def rotation_matrix(theta: float, phi: float) -> np.ndarray:
    """Return the 3×3 rotation matrix from slicing-frame to BZ-frame.

    Reproduces the parameterisation at lines 1260–1264 of the Fortran source
    (``p, q, s, c, u``) exactly; see module docstring for the formula.
    """
    p = float(np.sin(theta))
    q = float(np.cos(theta))
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    u = 1.0 - c
    return np.array(
        [
            [p * p * u + c, -p * q * u, q * s],
            [-p * q * u, q * q * u + c, p * s],
            [-q * s, -p * s, c],
        ]
    )


def make_slice_geometry(bxsf: BXSFData, numint: int, theta: float, phi: float) -> SliceGeometry:
    """Pre-compute the slice geometry for given (theta, phi) and ``numint``.

    Mirrors the bookkeeping at Fortran lines 669–688 plus 1249–1264.
    """
    if numint < 2:
        raise ValueError(f'numint must be >= 2, got {numint}')

    lens = k_axis_lengths(bxsf.recip_ang)
    maxlreciplat = float(lens.max())
    intkpointspacing = maxlreciplat / (numint - 1)
    numx = 4 * (numint - 1) + 1

    R = rotation_matrix(theta, phi)
    h_vec = R[:, 2].copy()  # (qs, ps, c)
    plr_inv = np.linalg.inv(bxsf.recip_ang)  # (BZ Cartesian) → fractional

    return SliceGeometry(
        theta=float(theta),
        phi=float(phi),
        numint=numint,
        numx=numx,
        xkseparation=intkpointspacing,
        ykseparation=intkpointspacing,
        zkseparation=intkpointspacing,
        xlength=4.0 * maxlreciplat,
        ylength=4.0 * maxlreciplat,
        maxlreciplat=maxlreciplat,
        rotation=R,
        h_vector=h_vec,
        plr_inverse=plr_inv,
    )


def _bxsf_indices_from_slice(
    geom: SliceGeometry,
    slice_indices: np.ndarray,  # (numx,)  1-based slice-x grid
    other_indices: np.ndarray,  # (numy,)  1-based slice-y grid
    z_index: int,  # 1-based slice-z grid
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a 2D slice of (slice_x, slice_y) coords to wrapped 0-based BXSF coords.

    Returns three (numx, numy) arrays of fractional indices ``ux, uy, uz`` in
    ``[0, n-1]``, ready to feed to a per-point interpolator.
    """
    M = geom.maxlreciplat
    invd = 1.0 / (geom.numint - 1)

    # Slicing-frame coordinates (broadcast to 2D mesh).
    Xp = ((slice_indices - 1) * invd - 1.0) * M  # (numx,)
    Yp = ((other_indices - 1) * invd - 1.0) * M  # (numy,)
    Zp = ((z_index - 1) * invd - 1.0) * M  # scalar

    # 2D mesh: shape (numx, numy)
    Xp2, Yp2 = np.meshgrid(Xp, Yp, indexing='ij')

    # Rotate into BZ frame: (p1, p2, p3) = R · (X', Y', Z')
    R = geom.rotation
    p1 = R[0, 0] * Xp2 + R[0, 1] * Yp2 + R[0, 2] * Zp
    p2 = R[1, 0] * Xp2 + R[1, 1] * Yp2 + R[1, 2] * Zp
    p3 = R[2, 0] * Xp2 + R[2, 1] * Yp2 + R[2, 2] * Zp

    # Convert BZ-frame Cartesian to fractional reciprocal coords.
    # Fortran uses cofactor form ai,aii,aiii / bigd; equivalently (plr^-1)·p.
    invp = geom.plr_inverse
    fx = (invp[0, 0] * p1 + invp[0, 1] * p2 + invp[0, 2] * p3) * (nx - 1)
    fy = (invp[1, 0] * p1 + invp[1, 1] * p2 + invp[1, 2] * p3) * (ny - 1)
    fz = (invp[2, 0] * p1 + invp[2, 1] * p2 + invp[2, 2] * p3) * (nz - 1)

    # Periodic wrap into [0, n-1).  Equivalent to Fortran's
    #   subtractor = floor(f / (n-1)) * (n-1);  d = f - subtractor + 1   (1-based)
    # in 0-based form:
    ux = fx - np.floor(fx / (nx - 1)) * (nx - 1)
    uy = fy - np.floor(fy / (ny - 1)) * (ny - 1)
    uz = fz - np.floor(fz / (nz - 1)) * (nz - 1)
    return ux, uy, uz


def build_slice(
    bxsf: BXSFData,
    geom: SliceGeometry,
    slice_index: int,
) -> Slice2D:
    """Build the 2D energy slice at index ``slice_index`` (1-based, in [1, numx]).

    Each of the ``numx · numy`` sample points is independently rotated into the
    BZ frame and Lagrange-interpolated.  Memory usage: O(numx²) scratch for
    fractional indices.
    """
    if not (1 <= slice_index <= geom.numx):
        raise ValueError(f'slice_index {slice_index} outside [1, {geom.numx}]')

    nx, ny, nz = bxsf.energies.shape
    ti = np.arange(1, geom.numx + 1, dtype=float)
    tj = np.arange(1, geom.numx + 1, dtype=float)
    ux, uy, uz = _bxsf_indices_from_slice(geom, ti, tj, slice_index, nx, ny, nz)

    energies = _interpolate_per_point(bxsf.energies, ux, uy, uz)

    z_prime = ((slice_index - 1) / (geom.numint - 1) - 1.0) * geom.maxlreciplat
    return Slice2D(
        geometry=geom,
        slice_index=slice_index,
        z_prime=z_prime,
        energies=energies,
    )


def _interpolate_per_point(
    energies: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
) -> np.ndarray:
    """Vectorised pointwise 4-point Lagrange interpolation.

    Inputs ``(ux, uy, uz)`` have a common shape; the output mirrors that
    shape.  Each point uses its own 4×4×4 stencil with the same periodic-wrap
    convention as :mod:`PAOFLOW.pyskeaf.interp`.
    """
    from PAOFLOW.pyskeaf.interp import _lagrange4_axis_weights
    from PAOFLOW.pyskeaf._numba_kernels import lagrange4_eval

    nx, ny, nz = energies.shape
    shape = ux.shape
    flat_x = np.ascontiguousarray(ux).reshape(-1)
    flat_y = np.ascontiguousarray(uy).reshape(-1)
    flat_z = np.ascontiguousarray(uz).reshape(-1)

    ix, wx = _lagrange4_axis_weights(flat_x, nx)  # (M, 4)
    iy, wy = _lagrange4_axis_weights(flat_y, ny)
    iz, wz = _lagrange4_axis_weights(flat_z, nz)

    # Fused gather + 4×4×4 Lagrange contraction.  Uses a Numba @njit kernel
    # when available (~25× faster than the einsum path) and falls back to the
    # pure-NumPy einsum implementation otherwise.  See PAOFLOW.pyskeaf._numba_kernels.
    out = lagrange4_eval(energies, ix, iy, iz, wx, wy, wz)
    return out.reshape(shape)
