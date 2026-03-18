"""
surface_project.py — Projected bulk band structure onto arbitrary surface planes.

Given a bulk tight-binding model (via PAOFLOW model dictionary) and a
surface normal direction, this module:

  1. Identifies the shortest reciprocal-lattice vector  G₀  parallel to
     the surface normal — this sets the k⊥ periodicity.
  2. Finds the two shortest in-plane real-space lattice vectors, defining
     the 2D surface lattice and its reciprocal (surface Brillouin zone).
  3. Auto-detects the surface-BZ type (square, hexagonal, rectangular,
     oblique) and provides default high-symmetry paths.
  4. For each  k∥  along the surface-BZ path, sweeps  k⊥  over one full
     period and collects all bulk eigenvalues.
  5. Computes per-band envelopes (min / max over k⊥) and identifies
     projected gaps and lens-shaped features.
  6. Produces a shaded figure of the projected band structure.

Public API
----------
    project_bulk_bands      — main computation
    project_from_model      — convenience wrapper for EDTBModel objects
    find_absolute_gaps      — locate energy gaps persistent at all k∥
    plot_projected          — shaded band-projection figure
    SurfaceProjectionResult — dataclass container

Pre-defined constants
---------------------
    SURFACE_001, SURFACE_110, SURFACE_111  — common surface normals
    SURFACE_SYM_SQUARE, SURFACE_SYM_HEX, SURFACE_SYM_RECT — surface BZ points

References
----------
  F. J. Himpsel and D. E. Eastman, J. Vac. Sci. Technol. 16, 1297 (1979).
  M. C. Desjonquères and D. Spanjaard, *Concepts in Surface Physics*
  (Springer, 1996).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import eigh, inv, norm

# Reuse Hamiltonian extraction from band_unfold
from .band_unfold import _extract_hamiltonian

# ═══════════════════════════════════════════════════════════════════════
#  Data containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SurfaceProjectionResult:
    """Container for projected bulk band structure results.

    Attributes
    ----------
    kdist         : (nk_par,) cumulative k∥ distance along the surface path.
    sym_ticks     : list of (distance, label) for surface-BZ symmetry ticks.
    band_min      : (nk_par, nawf) minimum eigenvalue of band *n* over k⊥.
    band_max      : (nk_par, nawf) maximum eigenvalue of band *n* over k⊥.
    all_evals     : (nk_par, nk_perp, nawf) full eigenvalue array.
    nawf          : number of Wannier functions (bands).
    nk_perp       : number of k⊥ sample points.
    surface_normal: (3,) unit surface-normal direction (Cartesian / alat).
    G_perp        : (3,) shortest reciprocal vector along normal (1/alat).
    surf_vectors  : (2, 3) primitive surface real-space lattice vectors.
    surf_recip    : (2, 3) surface reciprocal-lattice vectors.
    a_bulk        : (3, 3) bulk lattice vectors (rows).
    lattice_2d    : str — detected 2D lattice type.
    """

    kdist: np.ndarray
    sym_ticks: list
    band_min: np.ndarray
    band_max: np.ndarray
    all_evals: np.ndarray
    nawf: int
    nk_perp: int
    surface_normal: np.ndarray
    G_perp: np.ndarray
    surf_vectors: np.ndarray
    surf_recip: np.ndarray
    a_bulk: np.ndarray = field(repr=False)
    lattice_2d: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  Surface geometry helpers
# ═══════════════════════════════════════════════════════════════════════


def _find_shortest_parallel_G(
    b: np.ndarray, g_hat: np.ndarray, max_int: int = 6, tol: float = 1e-4
) -> np.ndarray:
    """Find the shortest reciprocal-lattice vector parallel to *g_hat*.

    Parameters
    ----------
    b      : (3, 3) reciprocal-lattice vectors (rows), units 1/alat.
    g_hat  : (3,) unit vector along the desired direction.
    max_int : search range for integer coefficients.
    tol    : angular tolerance (|sin θ| < tol ⇒ parallel).

    Returns
    -------
    G0 : (3,) shortest reciprocal vector parallel to *g_hat*,
         oriented so that  G0 · g_hat > 0.
    """
    best = None
    best_len = np.inf
    for m1 in range(-max_int, max_int + 1):
        for m2 in range(-max_int, max_int + 1):
            for m3 in range(-max_int, max_int + 1):
                if m1 == 0 and m2 == 0 and m3 == 0:
                    continue
                G = m1 * b[0] + m2 * b[1] + m3 * b[2]
                G_len = norm(G)
                if G_len < 1e-12:
                    continue
                # Check parallelism:  |G × g_hat| / |G| < tol
                cross = norm(np.cross(G, g_hat))
                if cross / G_len < tol and G_len < best_len:
                    # Ensure same orientation as g_hat
                    if np.dot(G, g_hat) > 0:
                        best = G.copy()
                        best_len = G_len
    if best is None:
        raise RuntimeError(
            "Could not find a reciprocal-lattice vector parallel to the "
            "surface normal.  Try increasing max_int."
        )
    return best


def _find_surface_lattice_vectors(
    a: np.ndarray, g_hat: np.ndarray, max_int: int = 6, tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the two shortest in-plane real-space lattice vectors.

    A lattice vector  v = n₁ a₁ + n₂ a₂ + n₃ a₃  lies in the surface
    plane iff  v · ĝ = 0  (perpendicular to the surface normal).

    Parameters
    ----------
    a     : (3, 3) bulk lattice vectors (rows), units alat.
    g_hat : (3,) unit surface-normal direction.
    max_int : search range for integer coefficients.
    tol   : tolerance for perpendicularity check.

    Returns
    -------
    v1, v2 : (3,) the two shortest linearly independent in-plane vectors.
    """
    candidates = []
    for n1 in range(-max_int, max_int + 1):
        for n2 in range(-max_int, max_int + 1):
            for n3 in range(-max_int, max_int + 1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                v = n1 * a[0] + n2 * a[1] + n3 * a[2]
                v_len = norm(v)
                if v_len < 1e-12:
                    continue
                # Perpendicularity: |v · g_hat| / |v| < tol
                if abs(np.dot(v, g_hat)) / v_len < tol:
                    candidates.append(v.copy())

    if len(candidates) < 2:
        raise RuntimeError(
            "Could not find two in-plane lattice vectors.  Try increasing max_int."
        )

    # Sort by length
    candidates.sort(key=lambda v: norm(v))

    v1 = candidates[0]
    # Find next shortest linearly independent of v1
    v1_hat = v1 / norm(v1)
    for v in candidates[1:]:
        # Linearly independent iff cross product is nonzero
        cross = norm(np.cross(v1_hat, v / norm(v)))
        if cross > 0.01:  # sin(θ) > 0.01 ⇒ angle > ~0.6°
            v2 = v
            # Ensure right-handed system:  (v1 × v2) · g_hat > 0
            if np.dot(np.cross(v1, v2), g_hat) < 0:
                v2 = -v2
            # ── Enforce obtuse angle for hexagonal lattices ──────
            # If v1 and v2 have equal length and an acute angle
            # (~60°), K = (1/3, 1/3) is NOT the correct BZ corner.
            # Replace v2 → v2 − v1 to obtain the conventional 120°
            # angle; the cross product (right-handedness) is
            # invariant: v1 × (v2−v1) = v1 × v2.
            l1, l2 = norm(v1), norm(v2)
            cos_angle = np.dot(v1, v2) / (l1 * l2)
            lengths_equal = abs(l1 - l2) / max(l1, l2) < 0.05
            if cos_angle > 0.1 and lengths_equal:
                v2 = v2 - v1
            return v1, v2

    raise RuntimeError(
        "All in-plane lattice vectors are collinear.  This should not "
        "happen for a 3D lattice — please verify lattice vectors."
    )


def _compute_surface_reciprocal(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the 2D surface reciprocal-lattice vectors.

    Given surface primitive vectors  v₁, v₂  (3D Cartesian), returns
    vectors  b₁ˢ, b₂ˢ  satisfying  vᵢ · bⱼˢ = δᵢⱼ.

    Uses the Moore–Penrose pseudoinverse:  B = (V Vᵀ)⁻¹ V.

    Returns
    -------
    b_s : (2, 3) surface reciprocal vectors (rows).
    """
    V = np.array([v1, v2])  # (2, 3)
    G = V @ V.T  # (2, 2) Gram matrix
    b_s = inv(G) @ V  # (2, 3)
    return b_s


def _detect_lattice_type_2d(v1: np.ndarray, v2: np.ndarray, tol: float = 0.05) -> str:
    """Classify the 2D surface lattice.

    Returns one of: ``'square'``, ``'hexagonal'``, ``'rectangular'``,
    ``'oblique'``.
    """
    l1, l2 = norm(v1), norm(v2)
    cos_angle = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    equal_lengths = abs(l1 - l2) / max(l1, l2) < tol

    if equal_lengths and abs(angle_deg - 90) < 5.0:
        return "square"
    if equal_lengths and (abs(angle_deg - 60) < 5.0 or abs(angle_deg - 120) < 5.0):
        return "hexagonal"
    if abs(angle_deg - 90) < 5.0:
        return "rectangular"
    return "oblique"


# ═══════════════════════════════════════════════════════════════════════
#  Pre-defined surface BZ symmetry points  (2D fractional coordinates)
# ═══════════════════════════════════════════════════════════════════════

SURFACE_SYM_SQUARE = {
    "Γ": [0.0, 0.0],
    "X": [0.5, 0.0],
    "M": [0.5, 0.5],
}
SURFACE_PATH_SQUARE = "Γ-X-M-Γ"

SURFACE_SYM_HEX = {
    "Γ": [0.0, 0.0],
    "M": [0.5, 0.0],
    "K": [1 / 3, 1 / 3],
}
SURFACE_PATH_HEX = "Γ-M-K-Γ"

SURFACE_SYM_RECT = {
    "Γ": [0.0, 0.0],
    "X": [0.5, 0.0],
    "S": [0.5, 0.5],
    "Y": [0.0, 0.5],
}
SURFACE_PATH_RECT = "Γ-X-S-Y-Γ"

SURFACE_SYM_OBLIQUE = {
    "Γ": [0.0, 0.0],
    "X": [0.5, 0.0],
    "M": [0.5, 0.5],
    "Y": [0.0, 0.5],
}
SURFACE_PATH_OBLIQUE = "Γ-X-M-Y-Γ"


def _default_surface_bz(
    lattice_type: str,
) -> Tuple[dict, str]:
    """Return default surface-BZ sym points and path for a lattice type."""
    table = {
        "square": (SURFACE_SYM_SQUARE, SURFACE_PATH_SQUARE),
        "hexagonal": (SURFACE_SYM_HEX, SURFACE_PATH_HEX),
        "rectangular": (SURFACE_SYM_RECT, SURFACE_PATH_RECT),
        "oblique": (SURFACE_SYM_OBLIQUE, SURFACE_PATH_OBLIQUE),
    }
    return table[lattice_type]


# ═══════════════════════════════════════════════════════════════════════
#  Common surface normals (Cartesian, for cubic crystals)
# ═══════════════════════════════════════════════════════════════════════

SURFACE_001 = np.array([0.0, 0.0, 1.0])
SURFACE_110 = np.array([1.0, 1.0, 0.0])
SURFACE_111 = np.array([1.0, 1.0, 1.0])


# ═══════════════════════════════════════════════════════════════════════
#  Surface k-path generation
# ═══════════════════════════════════════════════════════════════════════


def _make_surface_kpath(
    sym_points_2d: dict,
    path_str: str,
    b_s: np.ndarray,
    nk_per_seg: int = 100,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Generate a k∥ path in the surface BZ.

    Parameters
    ----------
    sym_points_2d : dict mapping label → (f1, f2) fractional coordinates
                    in the surface reciprocal basis.
    path_str      : e.g. ``'Γ-X-M-Γ'``.  ``'-'`` connects, ``'|'`` breaks.
    b_s           : (2, 3) surface reciprocal vectors (rows).
    nk_per_seg    : k-points per linear segment.

    Returns
    -------
    kpath : (nk, 3)  Cartesian k∥ points (units 1/alat).
    kdist : (nk,)    cumulative k-distance.
    sym_ticks : list of (distance, label).
    """
    pipe_segs = path_str.split("|")
    kpts, dists, ticks = [], [], []
    d = 0.0

    for iseg, seg_str in enumerate(pipe_segs):
        labels = seg_str.split("-")
        if iseg > 0:
            old_d, old_lbl = ticks[-1]
            ticks[-1] = (old_d, old_lbl + "|" + labels[0])
        else:
            ticks.append((d, labels[0]))

        for i in range(len(labels) - 1):
            f0 = np.array(sym_points_2d[labels[i]], dtype=float)
            f1 = np.array(sym_points_2d[labels[i + 1]], dtype=float)
            k0 = f0 @ b_s  # Cartesian 3D
            k1 = f1 @ b_s
            seg_len = norm(k1 - k0)
            for ik in range(nk_per_seg):
                t = ik / nk_per_seg
                kpts.append(k0 + t * (k1 - k0))
                dists.append(d + t * seg_len)
            d += seg_len
            ticks.append((d, labels[i + 1]))

        # Endpoint of this pipe segment
        f_end = np.array(sym_points_2d[labels[-1]], dtype=float)
        kpts.append(f_end @ b_s)
        dists.append(d)

    return np.array(kpts), np.array(dists), ticks


# ═══════════════════════════════════════════════════════════════════════
#  Main computation
# ═══════════════════════════════════════════════════════════════════════


def project_bulk_bands(
    model_dict: dict,
    surface_normal,
    *,
    surface_sym_points: Optional[dict] = None,
    surface_path: Optional[str] = None,
    nk_par: int = 100,
    nk_perp: int = 100,
    verbose: bool = True,
) -> SurfaceProjectionResult:
    """Compute the projected bulk band structure onto a surface plane.

    For each  k∥  along a high-symmetry path in the 2D surface Brillouin
    zone, the function sweeps  k⊥  over one full period (the 1D BZ
    perpendicular to the surface) and collects all bulk eigenvalues.  The
    per-band envelopes (min / max over k⊥) define the projected band
    structure.

    Parameters
    ----------
    model_dict : PAOFLOW-compatible model dictionary for the **bulk**
        crystal.  Must contain ``'model'`` key with ``'a_vectors'`` and
        ``'atoms'``.
    surface_normal : (3,) array-like — Cartesian direction perpendicular
        to the surface.  Common choices for cubic crystals:
        ``[0,0,1]`` for (001),  ``[1,1,0]`` for (110),
        ``[1,1,1]`` for (111).  Need not be a unit vector.
    surface_sym_points : dict, optional — high-symmetry points in the
        surface BZ as ``{label: [f1, f2]}`` fractional coordinates in
        the surface reciprocal basis.  If *None*, auto-detected from
        the 2D lattice type.
    surface_path : str, optional — path string, e.g. ``'Γ-X-M-Γ'``.
        If *None*, auto-detected.
    nk_par  : int — number of k∥ points per path segment.
    nk_perp : int — number of k⊥ sample points.
    verbose : bool — print progress information.

    Returns
    -------
    SurfaceProjectionResult  dataclass with band envelopes, eigenvalues,
    and surface geometry metadata.
    """
    surface_normal = np.asarray(surface_normal, dtype=float)
    g_hat = surface_normal / norm(surface_normal)

    # ── 1.  Bulk lattice ─────────────────────────────────────────
    a = np.array(model_dict["model"]["a_vectors"], dtype=float)
    b = inv(a).T  # reciprocal lattice vectors (rows), 1/alat

    if verbose:
        print("Bulk lattice vectors (rows, in alat):")
        for row in a:
            print(f"  {row}")
        print(f"Surface normal direction: {g_hat}")

    # ── 2.  Surface-normal reciprocal vector G₀ ──────────────────
    G_perp = _find_shortest_parallel_G(b, g_hat)
    if verbose:
        print(f"\nShortest G ∥ n̂:  G₀ = {G_perp}  (|G₀| = {norm(G_perp):.4f} / alat)")
        print(f"k⊥ BZ extent = {norm(G_perp):.4f} / alat")

    # ── 3.  Surface lattice vectors ──────────────────────────────
    v1, v2 = _find_surface_lattice_vectors(a, g_hat)
    b_s = _compute_surface_reciprocal(v1, v2)
    lattice_2d = _detect_lattice_type_2d(v1, v2)

    if verbose:
        print("\nSurface lattice vectors:")
        print(f"  v₁ = {v1}  (|v₁| = {norm(v1):.4f} alat)")
        print(f"  v₂ = {v2}  (|v₂| = {norm(v2):.4f} alat)")
        angle = np.degrees(
            np.arccos(np.clip(np.dot(v1, v2) / (norm(v1) * norm(v2)), -1, 1))
        )
        print(f"  Angle = {angle:.1f}°  →  {lattice_2d} lattice")
        print("Surface reciprocal vectors:")
        print(f"  b₁ˢ = {b_s[0]}")
        print(f"  b₂ˢ = {b_s[1]}")

    # ── 4.  Surface BZ path ──────────────────────────────────────
    if surface_sym_points is None or surface_path is None:
        auto_pts, auto_path = _default_surface_bz(lattice_2d)
        if surface_sym_points is None:
            surface_sym_points = auto_pts
        if surface_path is None:
            surface_path = auto_path
        if verbose:
            print(f"\nAuto-detected surface BZ path: {surface_path}")

    kpath, kdist, sym_ticks = _make_surface_kpath(
        surface_sym_points, surface_path, b_s, nk_per_seg=nk_par
    )
    nk_total = len(kpath)
    if verbose:
        print(f"Surface k-path: {nk_total} points, {len(sym_ticks)} ticks")

    # ── 5.  Extract bulk Hamiltonian ─────────────────────────────
    if verbose:
        print("\nBuilding bulk Hamiltonian...")
    HRs, R, nawf, nspin, _ = _extract_hamiltonian(
        model_dict, "_surf_proj_tmp", verbose=False
    )
    nR = HRs.shape[2]

    if verbose:
        print(f"  nawf = {nawf},  nR = {nR},  nspin = {nspin}")

    # ── 6.  Eigenvalue sweep: k = k∥ + t·G₀,  t ∈ [0, 1) ───────
    t_vals = np.linspace(0.0, 1.0, nk_perp, endpoint=False)
    all_evals = np.zeros((nk_total, nk_perp, nawf))

    # Pre-flatten Hamiltonian for fast matrix multiply
    H_flat = HRs[:, :, :, 0].reshape(nawf * nawf, nR)  # (nawf², nR)

    # Pre-compute R · G₀ for the k⊥ phase
    R_dot_G = R @ G_perp  # (nR,)

    if verbose:
        print(
            f"\nComputing eigenvalues:  {nk_total} k∥ × {nk_perp} k⊥ "
            f"= {nk_total * nk_perp} points ..."
        )

    for ik in range(nk_total):
        k_par = kpath[ik]
        # Phases from k∥:  exp(2πi R·k∥)
        phase_par = np.exp(2j * np.pi * (R @ k_par))  # (nR,)
        # Phases from k⊥:  exp(2πi t R·G₀)
        phase_perp = np.exp(2j * np.pi * np.outer(t_vals, R_dot_G))  # (nk_perp, nR)
        # Combined phase:  shape (nk_perp, nR)
        phases = phase_par[np.newaxis, :] * phase_perp

        # Build all Hk:  (nawf², nR) @ (nR, nk_perp)  →  (nawf², nk_perp)
        Hk_flat = H_flat @ phases.T
        # Reshape:  (nawf, nawf, nk_perp) → (nk_perp, nawf, nawf)
        Hk_all = Hk_flat.reshape(nawf, nawf, nk_perp).transpose(2, 0, 1)
        # Symmetrise
        Hk_all = 0.5 * (Hk_all + Hk_all.conj().transpose(0, 2, 1))
        # Eigenvalues (sorted)
        all_evals[ik] = eigh(Hk_all)[0]  # (nk_perp, nawf)

    # ── 7.  Band envelopes ───────────────────────────────────────
    band_min = all_evals.min(axis=1)  # (nk_total, nawf)
    band_max = all_evals.max(axis=1)  # (nk_total, nawf)

    if verbose:
        emin_glob = band_min.min()
        emax_glob = band_max.max()
        print(f"Energy range: [{emin_glob:.2f}, {emax_glob:.2f}] eV")
        print("Done.")

    return SurfaceProjectionResult(
        kdist=kdist,
        sym_ticks=sym_ticks,
        band_min=band_min,
        band_max=band_max,
        all_evals=all_evals,
        nawf=nawf,
        nk_perp=nk_perp,
        surface_normal=g_hat,
        G_perp=G_perp,
        surf_vectors=np.array([v1, v2]),
        surf_recip=b_s,
        a_bulk=a,
        lattice_2d=lattice_2d,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: project directly from EDTBModel
# ═══════════════════════════════════════════════════════════════════════


def project_from_model(
    model,  # EDTBModel
    surface_normal,
    **kwargs,
) -> SurfaceProjectionResult:
    """Project bulk bands using an EDTBModel object directly.

    Parameters
    ----------
    model : EDTBModel for the bulk crystal.
    surface_normal : (3,) Cartesian surface-normal direction.
    **kwargs : forwarded to :func:`project_bulk_bands`.

    Returns
    -------
    SurfaceProjectionResult
    """
    return project_bulk_bands(model.to_model_dict(), surface_normal, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  Gap detection
# ═══════════════════════════════════════════════════════════════════════


def _merge_intervals(intervals: list, tol: float = 0.01) -> list:
    """Merge overlapping or nearly-touching energy intervals.

    Parameters
    ----------
    intervals : list of (lo, hi) pairs.
    tol : merge intervals closer than this (eV).

    Returns
    -------
    merged : list of (lo, hi), sorted by lo.
    """
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_iv[0])]
    for lo, hi in sorted_iv[1:]:
        if lo <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [tuple(iv) for iv in merged]


def find_absolute_gaps(
    result: SurfaceProjectionResult,
    y_lim: tuple = (-12, 6),
    tol: float = 0.01,
) -> List[Tuple[float, float]]:
    """Find energy gaps that persist at every k∥ point.

    A gap is a contiguous energy interval in which **no** bulk state
    exists at **any** k∥.  These are the "absolute" or "true" projected
    band gaps where surface states may reside.

    Parameters
    ----------
    result : SurfaceProjectionResult from :func:`project_bulk_bands`.
    y_lim  : energy window to consider, (E_min, E_max) in eV.
    tol    : merge tolerance for near-touching intervals (eV).

    Returns
    -------
    gaps : list of ``(E_lo, E_hi)`` for each absolute gap, sorted by energy.
    """
    nk = len(result.kdist)

    # At each k∥, compute the merged band intervals
    # Then the absolute gaps are the intersections of per-k∥ gaps.
    # Equivalently: the complement of the union of all band intervals.

    # Union of all band intervals across all k∥
    all_intervals = []
    for ik in range(nk):
        intervals_ik = [
            (result.band_min[ik, n], result.band_max[ik, n]) for n in range(result.nawf)
        ]
        all_intervals.extend(intervals_ik)

    # Merge all intervals globally
    merged = _merge_intervals(all_intervals, tol=tol)

    # Clip to y_lim
    clipped = []
    for lo, hi in merged:
        lo_c = max(lo, y_lim[0])
        hi_c = min(hi, y_lim[1])
        if lo_c < hi_c:
            clipped.append((lo_c, hi_c))

    # Gaps = complement of merged bands within y_lim
    gaps = []
    prev_hi = y_lim[0]
    for lo, hi in clipped:
        if lo > prev_hi + tol:
            gaps.append((prev_hi, lo))
        prev_hi = max(prev_hi, hi)
    if prev_hi < y_lim[1] - tol:
        gaps.append((prev_hi, y_lim[1]))

    return gaps


def find_gaps_at_kpar(
    result: SurfaceProjectionResult,
    y_lim: tuple = (-12, 6),
    tol: float = 0.01,
) -> List[List[Tuple[float, float]]]:
    """Find projected band gaps at each individual k∥ point.

    Returns
    -------
    gaps_per_k : list of length nk_par.  Each element is a list of
        ``(E_lo, E_hi)`` gap intervals at that k∥.
    """
    nk = len(result.kdist)
    gaps_per_k = []
    for ik in range(nk):
        intervals = [
            (result.band_min[ik, n], result.band_max[ik, n]) for n in range(result.nawf)
        ]
        merged = _merge_intervals(intervals, tol=tol)
        # Clip
        clipped = []
        for lo, hi in merged:
            lo_c = max(lo, y_lim[0])
            hi_c = min(hi, y_lim[1])
            if lo_c < hi_c:
                clipped.append((lo_c, hi_c))
        # Gaps
        gaps = []
        prev_hi = y_lim[0]
        for lo, hi in clipped:
            if lo > prev_hi + tol:
                gaps.append((prev_hi, lo))
            prev_hi = max(prev_hi, hi)
        if prev_hi < y_lim[1] - tol:
            gaps.append((prev_hi, y_lim[1]))
        gaps_per_k.append(gaps)
    return gaps_per_k


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════


def plot_projected(
    result: SurfaceProjectionResult,
    *,
    y_lim: tuple = (-12, 6),
    color: str = "steelblue",
    alpha: float = 0.45,
    gap_color: str = "lightyellow",
    show_gaps: bool = True,
    mode: str = "fill",
    ne: int = 500,
    surface_bands: Optional[np.ndarray] = None,
    surface_kdist: Optional[np.ndarray] = None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    show: bool = True,
    ax=None,
):
    """Plot the projected bulk band structure as a shaded E-vs-k∥ figure.

    Parameters
    ----------
    result  : SurfaceProjectionResult from :func:`project_bulk_bands`.
    y_lim   : energy window (eV).
    color   : fill colour for bulk-band regions.
    alpha   : opacity (used in ``'fill'`` mode).
    gap_color : background colour for gap regions (``'image'`` mode).
    show_gaps : if True, highlight absolute gaps.
    mode    : ``'fill'`` — ``fill_between`` per band (overlapping bands
        appear darker, giving a qualitative DOS indication);
        ``'image'`` — binary contour-fill (uniform colour inside bulk
        regions, clean edges).
    ne      : energy resolution for ``'image'`` mode.
    surface_bands : (nk_par, n_surf_bands) array of surface-state
        eigenvalues to overlay (optional).
    surface_kdist : (nk_par,) k-distances for surface bands (optional;
        defaults to ``result.kdist``).
    figsize : figure size for a new figure.
    title   : plot title.
    show    : call ``plt.show()``.
    ax      : existing matplotlib Axes (optional).

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    kdist = result.kdist

    # ── Projected bulk bands ─────────────────────────────────────
    if mode == "fill":
        for n in range(result.nawf):
            # Only plot bands that overlap with y_lim
            if result.band_max[:, n].max() < y_lim[0]:
                continue
            if result.band_min[:, n].min() > y_lim[1]:
                continue
            ax.fill_between(
                kdist,
                result.band_min[:, n],
                result.band_max[:, n],
                color=color,
                alpha=alpha,
                edgecolor="none",
                linewidth=0,
                rasterized=True,
            )

    elif mode == "image":
        E_grid = np.linspace(y_lim[0], y_lim[1], ne)
        image = np.zeros((len(kdist), ne))
        for ik in range(len(kdist)):
            for n in range(result.nawf):
                lo = result.band_min[ik, n]
                hi = result.band_max[ik, n]
                mask = (E_grid >= lo) & (E_grid <= hi)
                image[ik, mask] = 1.0
        K_mesh, E_mesh = np.meshgrid(kdist, E_grid)
        ax.contourf(
            K_mesh,
            E_mesh,
            image.T,
            levels=[0.5, 1.5],
            colors=[color],
            alpha=alpha,
        )
    else:
        raise ValueError(f"Unknown plot mode '{mode}'.  Use 'fill' or 'image'.")

    # ── Highlight absolute gaps ──────────────────────────────────
    if show_gaps:
        abs_gaps = find_absolute_gaps(result, y_lim=y_lim)
        for g_lo, g_hi in abs_gaps:
            if g_hi - g_lo > 0.05:  # skip tiny gaps
                ax.axhspan(
                    g_lo,
                    g_hi,
                    color=gap_color,
                    alpha=0.6,
                    zorder=0,
                    label=f"gap [{g_lo:.2f}, {g_hi:.2f}] eV"
                    if g_lo == abs_gaps[0][0]
                    else None,
                )

    # ── Surface bands overlay ────────────────────────────────────
    if surface_bands is not None:
        sk = surface_kdist if surface_kdist is not None else kdist
        for n in range(surface_bands.shape[1]):
            ax.plot(
                sk,
                surface_bands[:, n],
                "r-",
                lw=1.5,
                label="surface states" if n == 0 else None,
            )

    # ── Symmetry ticks ───────────────────────────────────────────
    tick_pos = [t[0] for t in result.sym_ticks]
    tick_lbl = []
    for t in result.sym_ticks:
        lbl = t[1]
        # Pretty-format: add overline via mathtext
        parts = lbl.split("|")
        formatted = []
        for p in parts:
            if p == "Γ":
                formatted.append(r"$\bar{\Gamma}$")
            else:
                formatted.append(r"$\overline{\mathrm{" + p + r"}}$")
        tick_lbl.append("|".join(formatted))

    for x in tick_pos:
        ax.axvline(x, color="gray", lw=0.5, alpha=0.5)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=12)

    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylim(*y_lim)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_xlabel(r"$k_\parallel$", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
#  Utility: lens detection
# ═══════════════════════════════════════════════════════════════════════


def find_band_lenses(
    result: SurfaceProjectionResult,
    y_lim: tuple = (-12, 6),
    tol: float = 0.01,
) -> List[dict]:
    """Identify "lens" features where a gap opens and closes along k∥.

    A band lens occurs when two adjacent projected bands have a gap at
    some k∥ values but overlap at others, producing a lens-shaped gap
    region in the (k∥, E) plane.

    Parameters
    ----------
    result : SurfaceProjectionResult
    y_lim  : energy window.
    tol    : energy tolerance for gap detection (eV).

    Returns
    -------
    lenses : list of dicts, each with keys:
        ``'gap_center'``  — (float) mean energy of the lens midpoint,
        ``'max_gap'``     — (float) maximum gap width (eV),
        ``'k_range'``     — (float, float) k-distance range where gap exists,
        ``'k_indices'``   — (int, int) index range in kdist.
    """
    gaps_per_k = find_gaps_at_kpar(result, y_lim=y_lim, tol=tol)
    nk = len(result.kdist)

    # Collect all unique gap "bands":
    # At each k, a gap is (lo, hi).  Track connected gap components.
    # Simple approach: for each k, record all gap midpoints.
    # Cluster across k to find lenses (gaps that exist at some k but not all).

    abs_gaps = find_absolute_gaps(result, y_lim=y_lim, tol=tol)
    abs_gap_set = set(abs_gaps)

    lenses = []
    # Track "active gaps" across k using gap midpoints
    active = {}  # midpoint_bin → {start_k, energies, max_width}
    bin_size = 0.2  # eV

    for ik in range(nk):
        current_mids = set()
        for g_lo, g_hi in gaps_per_k[ik]:
            mid = 0.5 * (g_lo + g_hi)
            width = g_hi - g_lo
            # Check if this is part of an absolute gap (skip those)
            is_abs = False
            for a_lo, a_hi in abs_gaps:
                if g_lo >= a_lo - tol and g_hi <= a_hi + tol:
                    is_abs = True
                    break
            if is_abs:
                continue

            mid_bin = round(mid / bin_size) * bin_size
            current_mids.add(mid_bin)

            if mid_bin not in active:
                active[mid_bin] = {
                    "start_k": ik,
                    "end_k": ik,
                    "energies": [(g_lo, g_hi)],
                    "max_width": width,
                }
            else:
                active[mid_bin]["end_k"] = ik
                active[mid_bin]["energies"].append((g_lo, g_hi))
                active[mid_bin]["max_width"] = max(active[mid_bin]["max_width"], width)

        # Check for gaps that closed (no longer active)
        closed = [
            m for m in active if m not in current_mids and active[m]["end_k"] == ik - 1
        ]
        for m in closed:
            info = active.pop(m)
            # Only report if it opened and closed (not at boundary)
            if info["start_k"] > 0 or info["end_k"] < nk - 1:
                all_lo = [e[0] for e in info["energies"]]
                all_hi = [e[1] for e in info["energies"]]
                lenses.append(
                    {
                        "gap_center": 0.5 * (np.mean(all_lo) + np.mean(all_hi)),
                        "max_gap": info["max_width"],
                        "k_range": (
                            result.kdist[info["start_k"]],
                            result.kdist[info["end_k"]],
                        ),
                        "k_indices": (info["start_k"], info["end_k"]),
                    }
                )

    # Flush remaining active gaps
    for m, info in active.items():
        if info["start_k"] > 0 or info["end_k"] < nk - 1:
            all_lo = [e[0] for e in info["energies"]]
            all_hi = [e[1] for e in info["energies"]]
            lenses.append(
                {
                    "gap_center": 0.5 * (np.mean(all_lo) + np.mean(all_hi)),
                    "max_gap": info["max_width"],
                    "k_range": (
                        result.kdist[info["start_k"]],
                        result.kdist[info["end_k"]],
                    ),
                    "k_indices": (info["start_k"], info["end_k"]),
                }
            )

    # Sort by energy
    lenses.sort(key=lambda x: x["gap_center"])
    return lenses
