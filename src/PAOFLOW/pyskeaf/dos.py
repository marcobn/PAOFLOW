"""Tetrahedron-method density of states (DOS) on the SKEAF supercell.

Reproduces ``slicedos`` (skeaf_v1p3p0_r149.F90 lines 3133–3291) and the
top-level DOS driver loop (lines 1043–1095).

Algorithm summary
-----------------
1. Sample the band energies on a ``(4·numint)³`` supercell grid via 4-point
   Lagrange tricubic interpolation (see :mod:`PAOFLOW.pyskeaf.interp`).
2. For each of the ``(4·numint - 1)³`` microcells (8 corner points), split
   into 6 tetrahedra using the Fortran's exact corner-index pattern.
3. For each tetrahedron whose energy span brackets ``E_F``, accumulate the
   Blöchl-style DOS contribution::

       e1 ≤ EF ≤ e2:  3V (EF - e1)² / [(e2 - e1)(e3 - e1)(e4 - e1)]
       e3 ≤ EF ≤ e4:  3V (e4 - EF)² / [(e4 - e1)(e4 - e2)(e4 - e3)]
       else        :  (3V / [(e3 - e1)(e4 - e1)]) · (
                          e2 - e1 + 2·EF - 2·e2
                          - (e3 - e1 + e4 - e2)(EF - e2)² / [(e3 - e2)(e4 - e2)]
                      )

   where ``V`` is the per-tetrahedron volume,
   ``V = unit_cell_volume / numtetrahedra`` and
   ``numtetrahedra = 6·(4·numint - 1)³``.

4. Also count "occupied" microcells (corner-(1,1,1) below ``E_F``) and
   "empty" microcells (above) to estimate the band electron / hole volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PAOFLOW.pyskeaf.geometry import unit_cell_volume
from PAOFLOW.pyskeaf.interp import interpolate_grid, supercell_axis_coords
from PAOFLOW.pyskeaf.io_bxsf import BXSFData


# Microcell corner indexing: each microcell at integer (i, j, k) has 8
# corners; the Fortran convention (slicedos lines 3198–3205) labels them:
#
#   c1 = (i  , j  , k  )    c5 = (i  , j  , k+1)
#   c2 = (i+1, j  , k  )    c6 = (i+1, j  , k+1)
#   c3 = (i  , j+1, k  )    c7 = (i  , j+1, k+1)
#   c4 = (i+1, j+1, k  )    c8 = (i+1, j+1, k+1)
#
# The 6 tetrahedra (1-based corner labels in the Fortran):
#   T1 = (1, 2, 3, 6)
#   T2 = (2, 3, 4, 6)
#   T3 = (3, 4, 6, 8)
#   T4 = (3, 6, 7, 8)
#   T5 = (3, 5, 6, 7)
#   T6 = (1, 3, 5, 6)
_TETRA_CORNERS = np.array(
    [
        [0, 1, 2, 5],
        [1, 2, 3, 5],
        [2, 3, 5, 7],
        [2, 5, 6, 7],
        [2, 4, 5, 6],
        [0, 2, 4, 5],
    ],
    dtype=np.intp,
)


@dataclass
class DosResult:
    """Output of :func:`compute_dos`."""

    fermi_energy: float  # Ryd
    unit_cell_volume: float  # Å^-3
    one_kpoint_volume: float  # Å^-3
    num_occupied_states: int  # microcells with c1 ≤ EF
    num_empty_states: int  # microcells with c1 > EF
    band_electron_volume: float  # Å^-3
    band_hole_volume: float  # Å^-3
    dos_at_ef: float  # Å^-3 Ryd^-1 per spin direction
    band_min: float  # Ryd
    band_max: float  # Ryd
    fermi_fraction: float  # (EF - band_min) / (band_max - band_min)


def _slice_dos_pair(
    slice_e: np.ndarray,
    slice_e_next: np.ndarray,
    fermi_energy: float,
    tetrahedron_volume: float,
) -> tuple[float, int, int]:
    """DOS contribution from all microcells between two adjacent z-slices.

    Vectorised port of ``slicedos`` (lines 3208–3290).

    Parameters
    ----------
    slice_e, slice_e_next : ndarray, shape (M, M)
        Interpolated energies on the supercell at z = k and z = k + 1.
    fermi_energy : float
        Fermi energy in Rydbergs.
    tetrahedron_volume : float
        Per-tetrahedron volume ``V`` in Å^3.

    Returns
    -------
    dos_contrib : float
        Sum of Blöchl DOS contributions from all tetrahedra in this slab,
        in units of Å^-3 Ryd^-1 (per spin) once divided by ``unit_cell_volume``
        — but the Fortran returns the *un-normalised* sum that is divided
        only at the very end.  We follow the Fortran convention.
    n_occ : int
        Number of microcells with corner ``c1 ≤ EF``.
    n_empty : int
        Number of microcells with corner ``c1 > EF``.
    """
    M = slice_e.shape[0]
    # --- 8 corner energies of every microcell, shape (M-1, M-1, 8) ---------
    c = np.empty((M - 1, M - 1, 8), dtype=float)
    c[..., 0] = slice_e[:-1, :-1]
    c[..., 1] = slice_e[1:, :-1]
    c[..., 2] = slice_e[:-1, 1:]
    c[..., 3] = slice_e[1:, 1:]
    c[..., 4] = slice_e_next[:-1, :-1]
    c[..., 5] = slice_e_next[1:, :-1]
    c[..., 6] = slice_e_next[:-1, 1:]
    c[..., 7] = slice_e_next[1:, 1:]

    # Occupied / empty count: microcell "occupied" iff its (i, j, k) corner
    # (== c[..., 0]) is below the Fermi energy.  Mirrors the Fortran check
    #   if (kslice(ti, tj) <= fermienergy) tmpocc = tmpocc + 1
    n_occ = int(np.sum(c[..., 0] <= fermi_energy))
    n_empty = int((M - 1) * (M - 1) - n_occ)

    # --- per-tetrahedron sorted energies, shape (6, M-1, M-1, 4) ----------
    tet = c[..., _TETRA_CORNERS]  # (M-1, M-1, 6, 4)
    tet = np.moveaxis(tet, -2, 0)  # (6, M-1, M-1, 4)

    e_min = tet.min(axis=-1)
    e_max = tet.max(axis=-1)
    crosses = (e_min <= fermi_energy) & (e_max >= fermi_energy)
    if not crosses.any():
        return 0.0, n_occ, n_empty

    sel = np.argwhere(crosses)  # (Nsel, 3) — (tet, i, j)
    if sel.size == 0:
        return 0.0, n_occ, n_empty

    e_sorted = np.sort(tet[crosses], axis=-1)  # (Nsel, 4)
    e1, e2, e3, e4 = e_sorted[:, 0], e_sorted[:, 1], e_sorted[:, 2], e_sorted[:, 3]
    EF = fermi_energy
    V = tetrahedron_volume

    # Three Blöchl branches.  Use np.where to combine; protect against the
    # zero-denominator cases the Fortran guards with explicit equality checks.
    contrib = np.zeros_like(e1)

    # Branch A: e1 <= EF <= e2
    A = (EF >= e1) & (EF <= e2)
    if A.any():
        denom = (e2 - e1) * (e3 - e1) * (e4 - e1)
        safe = denom != 0.0
        m = A & safe
        contrib = np.where(
            m,
            3.0 * V * (EF - e1) ** 2 / np.where(safe, denom, 1.0),
            contrib,
        )

    # Branch C: e3 <= EF <= e4
    C = (EF >= e3) & (EF <= e4) & (~A)
    if C.any():
        denom = (e4 - e1) * (e4 - e2) * (e4 - e3)
        safe = denom != 0.0
        m = C & safe
        contrib = np.where(
            m,
            3.0 * V * (e4 - EF) ** 2 / np.where(safe, denom, 1.0),
            contrib,
        )

    # Branch B: e2 < EF < e3 (the remainder)
    B = ~(A | C)
    if B.any():
        d1 = (e3 - e1) * (e4 - e1)
        d2 = (e3 - e2) * (e4 - e2)
        safe = (d1 != 0.0) & (d2 != 0.0)
        m = B & safe
        bracket = (
            (e2 - e1)
            + 2.0 * EF
            - 2.0 * e2
            - ((e3 - e1 + e4 - e2) * (EF - e2) ** 2) / np.where(safe, d2, 1.0)
        )
        contrib = np.where(
            m,
            (3.0 * V / np.where(safe, d1, 1.0)) * bracket,
            contrib,
        )

    return float(contrib.sum()), n_occ, n_empty


def compute_dos(
    bxsf: BXSFData,
    numint: int,
    fermi_energy: float | None = None,
    *,
    progress: bool = False,
) -> DosResult:
    """Tetrahedron DOS at the Fermi energy on a 4·numint supercell.

    Parameters
    ----------
    bxsf : BXSFData
        Loaded BXSF data (``recip_ang`` is used for volumes).
    numint : int
        Interpolated points per single-side (matches Fortran ``numint``).
        The supercell has ``4 * numint`` points along each axis.
    fermi_energy : float, optional
        Override Fermi energy in Rydbergs.  Defaults to the value stored in
        ``bxsf.fermi_energy``.
    progress : bool
        If True, print a one-line progress indicator (slice index / total).

    Returns
    -------
    :class:`DosResult`
    """
    if numint < 1:
        raise ValueError(f'numint must be >= 1, got {numint}')

    EF = bxsf.fermi_energy if fermi_energy is None else float(fermi_energy)

    ucv = unit_cell_volume(bxsf.recip_ang)
    M = 4 * numint
    n_microcells_per_axis = M - 1
    one_kpoint_volume = ucv / (n_microcells_per_axis**3)
    num_tetrahedra = 6 * n_microcells_per_axis**3
    tet_volume = ucv / num_tetrahedra

    # Pre-compute supercell sample coordinates along each axis.
    xs = supercell_axis_coords(numint, bxsf.nx)
    ys = supercell_axis_coords(numint, bxsf.ny)
    zs = supercell_axis_coords(numint, bxsf.nz)

    # Stream over z-slices to keep memory bounded: at any time we hold two
    # 2D slices of shape (M, M).  Compute slice k via a single 2D
    # interpolate_grid call at z = zs[k].
    prev_slice: np.ndarray | None = None
    total_dos = 0.0
    total_occ = 0
    total_empty = 0
    band_min = np.inf
    band_max = -np.inf

    for k in range(M):
        # interpolate_grid expects 1-D arrays; for one z value, pass length-1.
        cur_slice = interpolate_grid(bxsf.energies, xs, ys, zs[k : k + 1])[:, :, 0]
        band_min = min(band_min, float(cur_slice.min()))
        band_max = max(band_max, float(cur_slice.max()))

        if prev_slice is not None:
            dc, no, ne = _slice_dos_pair(prev_slice, cur_slice, EF, tet_volume)
            total_dos += dc
            total_occ += no
            total_empty += ne
            if progress:
                pct = 100.0 * (k) / (M - 1)
                print(f'\r DOS: {pct:5.1f} %', end='', flush=True)

        prev_slice = cur_slice

    if progress:
        print()

    band_electron_volume = total_occ * one_kpoint_volume
    band_hole_volume = total_empty * one_kpoint_volume
    bandwidth = band_max - band_min
    fermi_fraction = (EF - band_min) / bandwidth if bandwidth > 0 else float('nan')

    # The Fortran prints tetradoscontrib directly in Å^-3 Ryd^-1 per spin
    # direction (line 1160).  The per-tetrahedron volume already encodes the
    # per-BZ-volume normalisation (V_tet = ucv / N_tet, with V_tet in Å^-3),
    # so no extra division is required.
    dos_at_ef = total_dos

    return DosResult(
        fermi_energy=EF,
        unit_cell_volume=ucv,
        one_kpoint_volume=one_kpoint_volume,
        num_occupied_states=total_occ,
        num_empty_states=total_empty,
        band_electron_volume=band_electron_volume,
        band_hole_volume=band_hole_volume,
        dos_at_ef=dos_at_ef,
        band_min=band_min,
        band_max=band_max,
        fermi_fraction=fermi_fraction,
    )
