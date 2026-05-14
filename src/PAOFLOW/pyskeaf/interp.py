"""4-point Lagrange tricubic interpolation of band energies.

Faithful port of the Fortran ``pinterpolation`` function (lines 3300–3764 of
``skeaf_v1p3p0_r149.F90``) plus a vectorised batch interface.

Conventions
-----------
The Fortran routine uses **1-based** continuous coordinates ``intpointx`` ∈
[1, nx], with the convention that the BXSF General Grid is *periodic*:
energies at index ``1`` and ``nx`` correspond to the same physical k-point
(opposite faces of one BZ).  Out-of-range stencil nodes wrap as

    if x2 == 1  then x1 -> nx - 1
    if x3 == nx then x4 -> 2

Internally we work in **0-based** coordinates ``u`` ∈ [0, nx - 1]; the
equivalent wrap is

    if i2 == 0       then i1 -> nx - 2
    if i3 == nx - 1  then i4 -> 1

Stencil
-------
For a fractional position ``u`` between integer grid nodes, we use the
4-node stencil ``(i2 - 1, i2, i2 + 1, i2 + 2)`` where ``i2 = floor(u)`` and
``i3 = i2 + 1``.  When ``u`` lies exactly on an integer node we collapse to
that node (mirrors the Fortran integer-coordinate fast path and avoids the
``x2 == x3`` zero-divide that the general formula would otherwise hit).

Performance
-----------
* :func:`interpolate_point` — single ``(x, y, z)`` evaluation, kept for
  unit-test parity with the Fortran scalar function.
* :func:`interpolate_grid` — vectorised tensor-product evaluation over
  separable axis grids ``xs``, ``ys``, ``zs`` (typical for SKEAF: a full
  4·numint × 4·numint × 4·numint supercell sample, or a 2D slice).  Builds
  the four Lagrange weights per axis once and contracts via :func:`numpy.einsum`.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Single-axis Lagrange-4 weights (with Fortran-style periodic wrap).
# ---------------------------------------------------------------------------

def _wrap_indices(i1: int, i4: int, n: int) -> tuple[int, int]:
    """Apply the Fortran periodic-wrap rule to the outer stencil nodes.

    In 0-based form: if ``i1 < 0`` use ``n - 2``; if ``i4 > n - 1`` use ``1``.
    """
    i1r = i1 if i1 >= 0 else (n - 2)
    i4r = i4 if i4 <= n - 1 else 1
    return i1r, i4r


def _lagrange4_weights_1d(u: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices_4, weights_4)`` for 4-point Lagrange interpolation.

    Parameters
    ----------
    u : float
        Continuous 0-based grid coordinate, ``0 <= u <= n - 1``.
    n : int
        Number of grid points along this axis.

    Notes
    -----
    If ``u`` is (within a small tolerance of) an integer node, returns the
    1-element trivial weight ``([i], [1.0])`` to bypass the singular Lagrange
    formula.  Callers MUST handle the variable length (use
    :func:`_lagrange4_weights_padded` for fixed-length output).
    """
    if u < -1e-12 or u > n - 1 + 1e-12:
        raise ValueError(f"u={u} out of range [0, {n - 1}]")

    iu = int(round(u))
    if abs(u - iu) < 1e-12:
        return np.array([iu], dtype=np.intp), np.array([1.0])

    i2 = int(np.floor(u))
    i3 = i2 + 1
    i1 = i2 - 1
    i4 = i3 + 1
    i1r, i4r = _wrap_indices(i1, i4, n)

    t = u - i2  # in (0, 1)
    # Lagrange basis on uniformly spaced nodes (i1, i2, i3, i4) = (-1, 0, 1, 2):
    #   L_1(t) = -t (t-1)(t-2) / 6
    #   L_2(t) =  (t+1)(t-1)(t-2) / 2
    #   L_3(t) = -(t+1) t (t-2) / 2
    #   L_4(t) =  (t+1) t (t-1) / 6
    w = np.array([
        -t * (t - 1.0) * (t - 2.0) / 6.0,
        (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
        -(t + 1.0) * t * (t - 2.0) / 2.0,
        (t + 1.0) * t * (t - 1.0) / 6.0,
    ])
    idx = np.array([i1r, i2, i3, i4r], dtype=np.intp)
    return idx, w


def _lagrange4_axis_weights(us: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised 4-point Lagrange weights for an array of coordinates.

    Always returns 4-wide stencils (no singular-case collapse): for points
    that land exactly on a node, all four weights remain finite because the
    stencil is offset by ±1, ±2 from the floor.  The ``u == n - 1`` case is
    snapped to ``u = n - 1 - eps`` to keep ``floor(u) <= n - 2``.

    Parameters
    ----------
    us : ndarray, shape (M,)
        0-based continuous coordinates in ``[0, n - 1]``.
    n : int
        Grid length.

    Returns
    -------
    idx : ndarray, shape (M, 4), dtype intp
        Wrapped grid indices.
    w : ndarray, shape (M, 4), dtype float
        Lagrange weights summing to 1.
    """
    us = np.asarray(us, dtype=float)
    if us.ndim != 1:
        raise ValueError(f"us must be 1-D, got shape {us.shape}")
    if us.size == 0:
        return np.zeros((0, 4), dtype=np.intp), np.zeros((0, 4), dtype=float)

    if (us < -1e-9).any() or (us > n - 1 + 1e-9).any():
        raise ValueError(f"some us outside [0, {n - 1}]: min={us.min()}, max={us.max()}")

    # Snap u == n - 1 inwards so floor(u) <= n - 2.
    us = np.clip(us, 0.0, n - 1 - 1e-12)

    i2 = np.floor(us).astype(np.intp)
    i3 = i2 + 1
    i1 = i2 - 1
    i4 = i3 + 1
    i1r = np.where(i1 >= 0, i1, n - 2)
    i4r = np.where(i4 <= n - 1, i4, 1)
    idx = np.stack([i1r, i2, i3, i4r], axis=-1)

    t = us - i2
    w = np.stack([
        -t * (t - 1.0) * (t - 2.0) / 6.0,
        (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
        -(t + 1.0) * t * (t - 2.0) / 2.0,
        (t + 1.0) * t * (t - 1.0) / 6.0,
    ], axis=-1)
    return idx, w


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def interpolate_point(
    energies: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> float:
    """Interpolate ``energies`` at a single ``(x, y, z)`` point (0-based coords).

    Mirrors the Fortran ``pinterpolation`` scalar function exactly.
    """
    nx, ny, nz = energies.shape
    ix, wx = _lagrange4_weights_1d(float(x), nx)
    iy, wy = _lagrange4_weights_1d(float(y), ny)
    iz, wz = _lagrange4_weights_1d(float(z), nz)

    sub = energies[np.ix_(ix, iy, iz)]
    return float(np.einsum("a,b,c,abc->", wx, wy, wz, sub))


def interpolate_grid(
    energies: np.ndarray,
    xs: ArrayLike,
    ys: ArrayLike,
    zs: ArrayLike,
) -> np.ndarray:
    """Interpolate ``energies`` on the tensor-product grid ``(xs, ys, zs)``.

    Parameters
    ----------
    energies : ndarray, shape (nx, ny, nz)
        Source energies in Rydbergs (or any scalar field).
    xs, ys, zs : 1-D array-like
        0-based continuous coordinates along each axis.  Coordinates must lie
        in ``[0, nx - 1]``, ``[0, ny - 1]``, ``[0, nz - 1]`` respectively.

    Returns
    -------
    out : ndarray, shape ``(len(xs), len(ys), len(zs))``
        Interpolated values, with axes in ``(x, y, z)`` order.
    """
    nx, ny, nz = energies.shape
    ix, wx = _lagrange4_axis_weights(np.asarray(xs, dtype=float), nx)
    iy, wy = _lagrange4_axis_weights(np.asarray(ys, dtype=float), ny)
    iz, wz = _lagrange4_axis_weights(np.asarray(zs, dtype=float), nz)

    # Gather the (M, N, P, 4, 4, 4) sub-cube of energies.  Use advanced
    # indexing one axis at a time to avoid materialising the full outer
    # product before reduction.
    #   step 1: gather along x  -> shape (M, 4, ny, nz)
    sub_x = energies[ix]                          # (M, 4, ny, nz)
    #   step 2: gather along y  -> shape (M, 4, N, 4, nz)
    sub_xy = sub_x[:, :, iy, :]                   # (M, 4, N, 4, nz)
    #   step 3: gather along z  -> shape (M, 4, N, 4, P, 4)
    sub_xyz = sub_xy[:, :, :, :, iz]              # (M, 4, N, 4, P, 4)

    # Contract weights: out[m, n, p] = sum_{a,b,c} wx[m,a] wy[n,b] wz[p,c] sub
    out = np.einsum(
        "ma,nb,pc,manbpc->mnp",
        wx, wy, wz, sub_xyz,
        optimize=True,
    )
    return out


def supercell_axis_coords(numint: int, n_axis: int) -> np.ndarray:
    """Return the 0-based supercell sample coordinates along one axis.

    Reproduces the Fortran formula

        dintpoint = ((ti - 1) / (4*numint - 1)) * (n - 1) + 1     (1-based)

    in 0-based form

        u_i = i / (4*numint - 1) * (n - 1)                        for i in [0, 4*numint - 1]

    Parameters
    ----------
    numint : int
        User-requested interpolated points per single-side; the supercell is
        sampled at ``4 * numint`` points along each axis.
    n_axis : int
        Original BXSF grid size along this axis.
    """
    if numint < 1:
        raise ValueError(f"numint must be >= 1, got {numint}")
    n_super = 4 * numint
    return np.arange(n_super, dtype=float) * (n_axis - 1) / (n_super - 1)

