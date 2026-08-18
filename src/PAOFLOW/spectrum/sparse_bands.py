"""sparse_bands.py — Sparse on-the-fly Lanczos band structure for EDTB models.

Bypasses dense H(R) storage.  For each k-point the Hamiltonian H(k) is
assembled as a sparse CSR matrix from a precomputed bond list, then
``scipy.sparse.linalg.eigsh`` extracts the requested eigenvalues via
implicitly-restarted Lanczos.

Usage
-----
    from sparse_bands import SparseEDTB
    ham = SparseEDTB(model_dict)
    result = ham.compute_bands("K-G-M-K'", high_sym_pts, nk=100, n_eigs=50)
"""

import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# ── Constants ───────────────────────────────────────────────────────
_SQRT3 = np.sqrt(3.0)
_HSQRT3 = _SQRT3 / 2.0

_P_INDEX = {"px": 0, "py": 1, "pz": 2}
_D_SET = {"dxy", "dyz", "dzx", "dx2-y2", "dz2"}


# ── Slater-Koster two-center integrals ──────────────────────────────


def _sd_value(d_orb, lx, ly, lz, h):
    l2, m2, n2 = lx * lx, ly * ly, lz * lz
    sds = h["sds"]
    if d_orb == "dxy":
        return _SQRT3 * lx * ly * sds
    if d_orb == "dyz":
        return _SQRT3 * ly * lz * sds
    if d_orb == "dzx":
        return _SQRT3 * lz * lx * sds
    if d_orb == "dx2-y2":
        return _HSQRT3 * (l2 - m2) * sds
    if d_orb == "dz2":
        return (n2 - 0.5 * (l2 + m2)) * sds
    return 0.0


def _pd_value(p_orb, d_orb, lx, ly, lz, h):
    l2, m2, n2 = lx * lx, ly * ly, lz * lz
    pds, pdp = h["pds"], h["pdp"]
    if p_orb == "px":
        if d_orb == "dxy":
            return _SQRT3 * l2 * ly * pds + ly * (1.0 - 2.0 * l2) * pdp
        if d_orb == "dyz":
            return _SQRT3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
        if d_orb == "dzx":
            return _SQRT3 * l2 * lz * pds + lz * (1.0 - 2.0 * l2) * pdp
        if d_orb == "dx2-y2":
            return _HSQRT3 * lx * (l2 - m2) * pds + lx * (1.0 - l2 + m2) * pdp
        if d_orb == "dz2":
            return lx * (n2 - 0.5 * (l2 + m2)) * pds - _SQRT3 * lx * n2 * pdp
    elif p_orb == "py":
        if d_orb == "dxy":
            return _SQRT3 * m2 * lx * pds + lx * (1.0 - 2.0 * m2) * pdp
        if d_orb == "dyz":
            return _SQRT3 * m2 * lz * pds + lz * (1.0 - 2.0 * m2) * pdp
        if d_orb == "dzx":
            return _SQRT3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
        if d_orb == "dx2-y2":
            return _HSQRT3 * ly * (l2 - m2) * pds - ly * (1.0 + l2 - m2) * pdp
        if d_orb == "dz2":
            return ly * (n2 - 0.5 * (l2 + m2)) * pds - _SQRT3 * ly * n2 * pdp
    elif p_orb == "pz":
        if d_orb == "dxy":
            return _SQRT3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
        if d_orb == "dyz":
            return _SQRT3 * n2 * ly * pds + ly * (1.0 - 2.0 * n2) * pdp
        if d_orb == "dzx":
            return _SQRT3 * n2 * lx * pds + lx * (1.0 - 2.0 * n2) * pdp
        if d_orb == "dx2-y2":
            return _HSQRT3 * lz * (l2 - m2) * pds - lz * (l2 - m2) * pdp
        if d_orb == "dz2":
            return lz * (n2 - 0.5 * (l2 + m2)) * pds + _SQRT3 * lz * (l2 + m2) * pdp
    return 0.0


def _dd_value(da, db, lx, ly, lz, h):
    l2, m2, n2 = lx * lx, ly * ly, lz * lz
    lm, ln, mn = lx * ly, lx * lz, ly * lz
    l2m2, l2n2, m2n2 = l2 * m2, l2 * n2, m2 * n2
    diff = l2 - m2
    dds, ddp, ddd = h["dds"], h["ddp"], h["ddd"]

    # Diagonal
    if da == db == "dxy":
        return 3.0 * l2m2 * dds + (l2 + m2 - 4.0 * l2m2) * ddp + (n2 + l2m2) * ddd
    if da == db == "dyz":
        return 3.0 * m2n2 * dds + (m2 + n2 - 4.0 * m2n2) * ddp + (l2 + m2n2) * ddd
    if da == db == "dzx":
        return 3.0 * l2n2 * dds + (l2 + n2 - 4.0 * l2n2) * ddp + (m2 + l2n2) * ddd
    if da == db == "dx2-y2":
        return (
            0.75 * diff**2 * dds
            + (l2 + m2 - diff**2) * ddp
            + (n2 + 0.25 * diff**2) * ddd
        )
    if da == db == "dz2":
        t = n2 - 0.5 * (l2 + m2)
        return t**2 * dds + 3.0 * n2 * (l2 + m2) * ddp + 0.75 * (l2 + m2) ** 2 * ddd

    # Off-diagonal (symmetric under swap)
    pair = frozenset((da, db))
    if pair == frozenset(("dxy", "dyz")):
        return (
            3.0 * lx * m2 * lz * dds
            + ln * (1.0 - 4.0 * m2) * ddp
            + ln * (m2 - 1.0) * ddd
        )
    if pair == frozenset(("dxy", "dzx")):
        return (
            3.0 * l2 * ly * lz * dds
            + mn * (1.0 - 4.0 * l2) * ddp
            + mn * (l2 - 1.0) * ddd
        )
    if pair == frozenset(("dyz", "dzx")):
        return (
            3.0 * ly * n2 * lx * dds
            + lm * (1.0 - 4.0 * n2) * ddp
            + lm * (n2 - 1.0) * ddd
        )
    if pair == frozenset(("dxy", "dx2-y2")):
        return (
            1.5 * lm * diff * dds + 2.0 * lm * (m2 - l2) * ddp + 0.5 * lm * diff * ddd
        )
    if pair == frozenset(("dyz", "dx2-y2")):
        return (
            1.5 * mn * diff * dds
            - mn * (1.0 + 2.0 * diff) * ddp
            + mn * (1.0 + 0.5 * diff) * ddd
        )
    if pair == frozenset(("dzx", "dx2-y2")):
        return (
            1.5 * ln * diff * dds
            + ln * (1.0 - 2.0 * diff) * ddp
            - ln * (1.0 - 0.5 * diff) * ddd
        )
    if pair == frozenset(("dxy", "dz2")):
        t = n2 - 0.5 * (l2 + m2)
        return _SQRT3 * (
            lm * t * dds - 2.0 * lm * n2 * ddp + 0.5 * lm * (1.0 + n2) * ddd
        )
    if pair == frozenset(("dyz", "dz2")):
        t = n2 - 0.5 * (l2 + m2)
        return _SQRT3 * (
            mn * t * dds + mn * (l2 + m2 - n2) * ddp - 0.5 * mn * (l2 + m2) * ddd
        )
    if pair == frozenset(("dzx", "dz2")):
        t = n2 - 0.5 * (l2 + m2)
        return _SQRT3 * (
            ln * t * dds + ln * (l2 + m2 - n2) * ddp - 0.5 * ln * (l2 + m2) * ddd
        )
    if pair == frozenset(("dx2-y2", "dz2")):
        t = n2 - 0.5 * (l2 + m2)
        return _SQRT3 * (
            0.5 * diff * t * dds + n2 * (m2 - l2) * ddp + 0.25 * (1.0 + n2) * diff * ddd
        )

    return 0.0


def _sk_element(orb_a, orb_b, lx, ly, lz, h):
    """Single SK two-center integral.  Matches PAOFLOW's _sk_sp_value."""
    # s-d / d-s
    if orb_a == "s" and orb_b in _D_SET:
        return _sd_value(orb_b, lx, ly, lz, h)
    if orb_b == "s" and orb_a in _D_SET:
        return _sd_value(orb_a, lx, ly, lz, h)
    # p-d
    if orb_a in _P_INDEX and orb_b in _D_SET:
        return _pd_value(orb_a, orb_b, lx, ly, lz, h)
    # d-p (sign flip)
    if orb_b in _P_INDEX and orb_a in _D_SET:
        v = _pd_value(orb_b, orb_a, lx, ly, lz, h)
        return -v
    # d-d
    if orb_a in _D_SET and orb_b in _D_SET:
        return _dd_value(orb_a, orb_b, lx, ly, lz, h)
    # s-s
    if orb_a == "s" and orb_b == "s":
        return h.get("sss", 0.0)
    # s-p
    if orb_a == "s" and orb_b in _P_INDEX:
        return (lx, ly, lz)[_P_INDEX[orb_b]] * h.get("sps", 0.0)
    # p-s (sign flip)
    if orb_b == "s" and orb_a in _P_INDEX:
        return -(lx, ly, lz)[_P_INDEX[orb_a]] * h.get("sps", 0.0)
    # p-p
    if orb_a in _P_INDEX and orb_b in _P_INDEX:
        pps = h.get("pps", 0.0)
        ppp = h.get("ppp", 0.0)
        if orb_a == orb_b:
            ll = (lx, ly, lz)[_P_INDEX[orb_a]]
            return ll**2 * pps + (1.0 - ll**2) * ppp
        return (
            (lx, ly, lz)[_P_INDEX[orb_a]] * (lx, ly, lz)[_P_INDEX[orb_b]] * (pps - ppp)
        )
    return 0.0


# ── Smooth cutoff (vectorized) ──────────────────────────────────────


def _f_cutoff_vec(r, r_taper, r_cut):
    """Vectorized smooth cutoff: 1 for r ≤ r_taper, cosine taper to 0 at r_cut."""
    fc = np.where(
        r <= r_taper,
        1.0,
        np.where(
            r >= r_cut,
            0.0,
            0.5 * (1.0 + np.cos(np.pi * (r - r_taper) / (r_cut - r_taper))),
        ),
    )
    return fc


# ── Goodwin distance-dependent hopping ──────────────────────────────


def _goodwin_all_channels(dist, channels, r_0, r_c, n_c):
    """Evaluate Goodwin V(r) for all SK channels at a given distance.

    Returns dict {channel_name: V_value}.
    """
    ratio = r_0 / dist
    exp_arg = -((dist / r_c) ** n_c) + (r_0 / r_c) ** n_c
    hop = {}
    for ch_name, ch_p in channels.items():
        V0 = ch_p["V0"]
        n_ch = ch_p["n"]
        hop[ch_name] = V0 * ratio**n_ch * np.exp(n_ch * exp_arg)
    return hop


def _goodwin_all_channels_vec(dists, channels, r_0, r_c, n_c):
    """Vectorised Goodwin V(r) for all SK channels over an array of distances.

    Parameters are the same as ``_goodwin_all_channels``; *dists* is
    an ndarray of shape (N,). Returns dict {channel: ndarray (N,)}.
    """
    ratio = r_0 / dists
    exp_arg = -((dists / r_c) ** n_c) + (r_0 / r_c) ** n_c
    hop = {}
    for ch_name, ch_p in channels.items():
        V0 = ch_p["V0"]
        n_ch = ch_p["n"]
        hop[ch_name] = V0 * ratio**n_ch * np.exp(n_ch * exp_arg)
    return hop


# Canonical spd orbital ordering used by the vectorised SK path
_SPD_ORBS = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2-y2", "dz2")


def _sk_block_spd_vec(lx, ly, lz, hop):
    """Vectorised full 9x9 Slater-Koster block for the canonical spd basis.

    Parameters
    ----------
    lx, ly, lz : ndarray, shape (N,)
        Direction cosines from atom *i* to atom *j*.
    hop : dict of ndarray, shape (N,)
        Screened SK hopping integrals keyed by channel name
        (sss, sps, pps, ppp, sds, pds, pdp, dds, ddp, ddd).

    Returns
    -------
    H : ndarray, shape (N, 9, 9)
        SK matrix blocks.  Orbital order:
        [s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2].
    """
    N = len(lx)
    H = np.zeros((N, 9, 9))

    sss = hop["sss"]
    sps = hop["sps"]
    pps = hop["pps"]
    ppp = hop["ppp"]
    sds = hop["sds"]
    pds = hop["pds"]
    pdp = hop["pdp"]
    dds = hop["dds"]
    ddp = hop["ddp"]
    ddd = hop["ddd"]

    l2 = lx * lx
    m2 = ly * ly
    n2 = lz * lz
    lm = lx * ly
    ln = lx * lz
    mn = ly * lz
    l2m2 = l2 * m2
    l2n2 = l2 * n2
    m2n2 = m2 * n2
    diff = l2 - m2
    t_dz2 = n2 - 0.5 * (l2 + m2)

    # ── s-s ──
    H[:, 0, 0] = sss

    # ── s-p / p-s ──
    H[:, 0, 1] = lx * sps
    H[:, 0, 2] = ly * sps
    H[:, 0, 3] = lz * sps
    H[:, 1, 0] = -H[:, 0, 1]
    H[:, 2, 0] = -H[:, 0, 2]
    H[:, 3, 0] = -H[:, 0, 3]

    # ── p-p ──
    pp_diff = pps - ppp
    H[:, 1, 1] = l2 * pps + (1.0 - l2) * ppp
    H[:, 2, 2] = m2 * pps + (1.0 - m2) * ppp
    H[:, 3, 3] = n2 * pps + (1.0 - n2) * ppp
    H[:, 1, 2] = lm * pp_diff
    H[:, 2, 1] = H[:, 1, 2]
    H[:, 1, 3] = ln * pp_diff
    H[:, 3, 1] = H[:, 1, 3]
    H[:, 2, 3] = mn * pp_diff
    H[:, 3, 2] = H[:, 2, 3]

    # ── s-d / d-s (no sign flip) ──
    H[:, 0, 4] = _SQRT3 * lm * sds
    H[:, 0, 5] = _SQRT3 * mn * sds
    H[:, 0, 6] = _SQRT3 * ln * sds
    H[:, 0, 7] = _HSQRT3 * diff * sds
    H[:, 0, 8] = t_dz2 * sds
    H[:, 4, 0] = H[:, 0, 4]
    H[:, 5, 0] = H[:, 0, 5]
    H[:, 6, 0] = H[:, 0, 6]
    H[:, 7, 0] = H[:, 0, 7]
    H[:, 8, 0] = H[:, 0, 8]

    # ── p-d ── (px row)
    H[:, 1, 4] = _SQRT3 * l2 * ly * pds + ly * (1.0 - 2.0 * l2) * pdp
    H[:, 1, 5] = _SQRT3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
    H[:, 1, 6] = _SQRT3 * l2 * lz * pds + lz * (1.0 - 2.0 * l2) * pdp
    H[:, 1, 7] = _HSQRT3 * lx * diff * pds + lx * (1.0 - l2 + m2) * pdp
    H[:, 1, 8] = lx * t_dz2 * pds - _SQRT3 * lx * n2 * pdp
    # (py row)
    H[:, 2, 4] = _SQRT3 * m2 * lx * pds + lx * (1.0 - 2.0 * m2) * pdp
    H[:, 2, 5] = _SQRT3 * m2 * lz * pds + lz * (1.0 - 2.0 * m2) * pdp
    H[:, 2, 6] = _SQRT3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
    H[:, 2, 7] = _HSQRT3 * ly * diff * pds - ly * (1.0 + l2 - m2) * pdp
    H[:, 2, 8] = ly * t_dz2 * pds - _SQRT3 * ly * n2 * pdp
    # (pz row)
    H[:, 3, 4] = _SQRT3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
    H[:, 3, 5] = _SQRT3 * n2 * ly * pds + ly * (1.0 - 2.0 * n2) * pdp
    H[:, 3, 6] = _SQRT3 * n2 * lx * pds + lx * (1.0 - 2.0 * n2) * pdp
    H[:, 3, 7] = _HSQRT3 * lz * diff * pds - lz * diff * pdp
    H[:, 3, 8] = lz * t_dz2 * pds + _SQRT3 * lz * (l2 + m2) * pdp

    # ── d-p = -(p-d)^T ──
    H[:, 4:9, 1:4] = -np.swapaxes(H[:, 1:4, 4:9], 1, 2)

    # ── d-d diagonal ──
    H[:, 4, 4] = 3.0 * l2m2 * dds + (l2 + m2 - 4.0 * l2m2) * ddp + (n2 + l2m2) * ddd
    H[:, 5, 5] = 3.0 * m2n2 * dds + (m2 + n2 - 4.0 * m2n2) * ddp + (l2 + m2n2) * ddd
    H[:, 6, 6] = 3.0 * l2n2 * dds + (l2 + n2 - 4.0 * l2n2) * ddp + (m2 + l2n2) * ddd
    H[:, 7, 7] = (
        0.75 * diff**2 * dds + (l2 + m2 - diff**2) * ddp + (n2 + 0.25 * diff**2) * ddd
    )
    H[:, 8, 8] = (
        t_dz2**2 * dds + 3.0 * n2 * (l2 + m2) * ddp + 0.75 * (l2 + m2) ** 2 * ddd
    )

    # ── d-d off-diagonal (upper triangle, then symmetrise) ──
    H[:, 4, 5] = (
        3.0 * lx * m2 * lz * dds + ln * (1.0 - 4.0 * m2) * ddp + ln * (m2 - 1.0) * ddd
    )
    H[:, 4, 6] = (
        3.0 * l2 * ly * lz * dds + mn * (1.0 - 4.0 * l2) * ddp + mn * (l2 - 1.0) * ddd
    )
    H[:, 5, 6] = (
        3.0 * ly * n2 * lx * dds + lm * (1.0 - 4.0 * n2) * ddp + lm * (n2 - 1.0) * ddd
    )
    H[:, 4, 7] = (
        1.5 * lm * diff * dds + 2.0 * lm * (m2 - l2) * ddp + 0.5 * lm * diff * ddd
    )
    H[:, 5, 7] = (
        1.5 * mn * diff * dds
        - mn * (1.0 + 2.0 * diff) * ddp
        + mn * (1.0 + 0.5 * diff) * ddd
    )
    H[:, 6, 7] = (
        1.5 * ln * diff * dds
        + ln * (1.0 - 2.0 * diff) * ddp
        - ln * (1.0 - 0.5 * diff) * ddd
    )
    H[:, 4, 8] = _SQRT3 * (
        lm * t_dz2 * dds - 2.0 * lm * n2 * ddp + 0.5 * lm * (1.0 + n2) * ddd
    )
    H[:, 5, 8] = _SQRT3 * (
        mn * t_dz2 * dds + mn * (l2 + m2 - n2) * ddp - 0.5 * mn * (l2 + m2) * ddd
    )
    H[:, 6, 8] = _SQRT3 * (
        ln * t_dz2 * dds + ln * (l2 + m2 - n2) * ddp - 0.5 * ln * (l2 + m2) * ddd
    )
    H[:, 7, 8] = _SQRT3 * (
        0.5 * diff * t_dz2 * dds + n2 * (m2 - l2) * ddp + 0.25 * (1.0 + n2) * diff * ddd
    )

    # Symmetrise d-d lower triangle
    H[:, 5, 4] = H[:, 4, 5]
    H[:, 6, 4] = H[:, 4, 6]
    H[:, 6, 5] = H[:, 5, 6]
    H[:, 7, 4] = H[:, 4, 7]
    H[:, 7, 5] = H[:, 5, 7]
    H[:, 7, 6] = H[:, 6, 7]
    H[:, 8, 4] = H[:, 4, 8]
    H[:, 8, 5] = H[:, 5, 8]
    H[:, 8, 6] = H[:, 6, 8]
    H[:, 8, 7] = H[:, 7, 8]

    return H


# ── K-path utilities ────────────────────────────────────────────────


def _generate_kpath(path_str, high_sym_points, nk, b_vectors):
    """Generate k-points along a high-symmetry path.

    Returns
    -------
    kpoints : ndarray (nk_total, 3)  fractional coordinates
    k_dist  : ndarray (nk_total,)    cumulative distance
    tick_pos : list of float         tick positions for labels
    tick_labels : list of str        high-symmetry labels
    """
    labels = path_str.split("-")
    segments = []
    for i in range(len(labels) - 1):
        k_start = np.array(high_sym_points[labels[i]])
        k_end = np.array(high_sym_points[labels[i + 1]])
        # Cartesian distance for uniform spacing
        dk_cart = (k_end - k_start) @ b_vectors
        seg_len = np.linalg.norm(dk_cart)
        segments.append((k_start, k_end, seg_len))

    total_len = sum(s[2] for s in segments)
    kpoints = []
    k_dist = []
    tick_pos = [0.0]
    tick_labels = [labels[0]]
    offset = 0.0

    for iseg, (k_start, k_end, seg_len) in enumerate(segments):
        nk_seg = max(2, int(round(nk * seg_len / total_len)))
        for j in range(nk_seg):
            t = j / max(1, nk_seg - 1)
            k = k_start + t * (k_end - k_start)
            dk_cart = (k - k_start) @ b_vectors
            d = offset + np.linalg.norm(dk_cart)
            kpoints.append(k)
            k_dist.append(d)
        offset += seg_len
        tick_pos.append(offset)
        tick_labels.append(labels[iseg + 1])

    return np.array(kpoints), np.array(k_dist), tick_pos, tick_labels


# ═══════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════


class SparseEDTB:
    """Sparse EDTB Hamiltonian for large supercells.

    Parameters
    ----------
    model_dict : dict
        PAOFLOW-format model dictionary (as produced by
        ``EDTBModel.to_model_dict()``).
    verbose : bool
        Print progress messages.
    """

    def __init__(self, model_dict, verbose=True):
        self.verbose = verbose
        self._parse_model(model_dict)
        self._build_bonds()

    # ── Parse model dict ────────────────────────────────────────

    def _parse_model(self, md):
        model = md["model"]
        self.alat = md.get("alat", 1.0)

        # Lattice vectors (in alat units)
        self.a_vectors = np.array(model["a_vectors"], dtype=float)
        # Reciprocal vectors (physics convention, with 2π)
        vol = np.dot(np.cross(self.a_vectors[0], self.a_vectors[1]), self.a_vectors[2])
        self.b_vectors = np.empty((3, 3), dtype=float)
        self.b_vectors[0] = (
            2.0 * np.pi * np.cross(self.a_vectors[1], self.a_vectors[2]) / vol
        )
        self.b_vectors[1] = (
            2.0 * np.pi * np.cross(self.a_vectors[2], self.a_vectors[0]) / vol
        )
        self.b_vectors[2] = (
            2.0 * np.pi * np.cross(self.a_vectors[0], self.a_vectors[1]) / vol
        )

        # Atoms
        atoms_dict = model["atoms"]
        self.natoms = len(atoms_dict)
        self.tau = np.zeros((self.natoms, 3), dtype=float)
        self.species = []
        self.orbitals = []
        self.norbitals = np.zeros(self.natoms, dtype=int)
        for ia in range(self.natoms):
            ad = atoms_dict[str(ia)]
            self.tau[ia] = np.array(ad["tau"])
            self.species.append(ad["name"])
            self.orbitals.append(list(ad["orbitals"]))
            self.norbitals[ia] = len(ad["orbitals"])

        self.nawf = int(self.norbitals.sum())
        self.atom_block_start = np.zeros(self.natoms, dtype=int)
        for ia in range(1, self.natoms):
            self.atom_block_start[ia] = (
                self.atom_block_start[ia - 1] + self.norbitals[ia - 1]
            )

        # On-site energies
        self.onsite = np.zeros(self.nawf, dtype=float)
        for ia in range(self.natoms):
            ad = atoms_dict[str(ia)]
            start = self.atom_block_start[ia]
            # Multi-shell (configuration-based) on-site
            if "configuration" in ad:
                _cfg_l = {"S": 0, "P": 1, "D": 2}
                idx = 0
                for cfg_label in ad["configuration"]:
                    l_val = _cfg_l[cfg_label[-1].upper()]
                    norb_l = 2 * l_val + 1
                    if l_val <= 1:
                        e = ad[cfg_label]
                        for io in range(norb_l):
                            self.onsite[start + idx + io] = e
                    elif l_val == 2:
                        key_t2g = f"{cfg_label}_t2g"
                        key_eg = f"{cfg_label}_eg"
                        if key_t2g in ad:
                            e_t2g, e_eg = ad[key_t2g], ad[key_eg]
                        else:
                            e_t2g = e_eg = ad[cfg_label]
                        for io in range(3):
                            self.onsite[start + idx + io] = e_t2g
                        for io in range(3, 5):
                            self.onsite[start + idx + io] = e_eg
                    idx += norb_l
            else:
                for io, orb_label in enumerate(ad["orbitals"]):
                    self.onsite[start + io] = ad[orb_label]

        # DD hopping parameters
        hoppings = model["hoppings"]
        pair_key = next(iter(hoppings))
        dd_spec = hoppings[pair_key]
        self.dd_r_0 = dd_spec["r_0"] / self.alat  # in alat units
        self.dd_r_c = dd_spec["r_c"] / self.alat
        self.dd_n_c = dd_spec["n_c"]
        self.dd_channels = dd_spec["channels"]

        # Per-axis cell ranges (detect vacuum gaps to skip unnecessary images)
        a_norms = np.linalg.norm(self.a_vectors, axis=1)
        frac_coords = self.tau @ np.linalg.inv(self.a_vectors)
        self.cell_range = np.zeros(3, dtype=int)
        for dim in range(3):
            a_len = a_norms[dim]
            if a_len < 1e-10:
                continue
            frac_span = frac_coords[:, dim].max() - frac_coords[:, dim].min()
            gap = (1.0 - frac_span) * a_len
            if gap > self.dd_r_c:
                self.cell_range[dim] = 0  # vacuum — no images needed
            else:
                self.cell_range[dim] = max(1, int(np.ceil(self.dd_r_c / a_len)))

        # Screening
        screening = model.get("screening", {})
        r_cut_phys = screening.get("r_cut", dd_spec["r_c"])
        self.r_cut = r_cut_phys / self.alat
        self.r_taper = 0.8 * self.r_cut
        gamma_spec = screening.get("gamma", 0.0)

        # Unwrap species-pair-keyed gamma (e.g. {"C-C": {"ss": ..}})
        if isinstance(gamma_spec, dict) and len(gamma_spec) > 0:
            first_key = next(iter(gamma_spec))
            if isinstance(gamma_spec[first_key], dict) and "-" in first_key:
                gamma_spec = gamma_spec[first_key]

        # Resolve gamma for each SK channel
        _sk_to_lpair = {
            "sss": "ss",
            "sps": "sp",
            "pps": "pp",
            "ppp": "pp",
            "sds": "sd",
            "pds": "pd",
            "pdp": "pd",
            "dds": "dd",
            "ddp": "dd",
            "ddd": "dd",
        }
        self.gamma_map = {}
        for sk_key in self.dd_channels:
            if isinstance(gamma_spec, (int, float)):
                self.gamma_map[sk_key] = float(gamma_spec)
            elif sk_key in gamma_spec:
                self.gamma_map[sk_key] = gamma_spec[sk_key]
            else:
                lp = _sk_to_lpair.get(sk_key, "")
                self.gamma_map[sk_key] = gamma_spec.get(lp, 0.0)

        # On-site shift
        self.onsite_shift = screening.get("onsite_shift", None)

    # ── Build sparse bond list ──────────────────────────────────

    def _build_bonds(self):
        """Enumerate all bonds within r_c, compute SK integrals + screening.

        Stores flat COO arrays (rows, cols, vals, R_int) for efficient
        CSR assembly at each k-point.
        """
        natoms = self.natoms
        tau = self.tau
        cr = self.cell_range  # (3,) per-axis
        nk1, nk2, nk3 = 2 * cr + 1

        if self.verbose:
            print(
                f"SparseEDTB: {natoms} atoms, nawf={self.nawf}, "
                f"cell_range={cr.tolist()} (nR={int(nk1 * nk2 * nk3)})"
            )

        # Supercell positions (flat)
        R_ints = []
        sctau_list = []
        for i in range(-cr[0], cr[0] + 1):
            for j in range(-cr[1], cr[1] + 1):
                for k in range(-cr[2], cr[2] + 1):
                    R_vec = (
                        i * self.a_vectors[0]
                        + j * self.a_vectors[1]
                        + k * self.a_vectors[2]
                    )
                    for ia in range(natoms):
                        sctau_list.append(tau[ia] + R_vec)
                        R_ints.append([i, j, k])
        sctau_flat = np.array(sctau_list)  # (n_sc, 3)
        # R_ints_flat = np.array(R_ints, dtype=int)  # (n_sc, 3)
        n_sc = len(sctau_flat)

        if self.verbose:
            print(f"  Precomputing screening (n_sc={n_sc})...")

        # ── Sparse f_c via KDTree (memory-efficient) ────────────
        # For large supercells, cdist(tau, sctau_flat) is O(natoms * n_sc)
        # dense, but f_c is zero for distances > r_cut.  Use a KDTree to
        # only compute distances within r_cut — keeps memory O(nnz).
        _use_kdtree = (natoms * n_sc * 8) > 500_000_000  # >500 MB dense

        if _use_kdtree:
            if self.verbose:
                print("    Using KDTree-sparse screening")
            tree_sc = cKDTree(sctau_flat)
            # tree_home = cKDTree(tau)

            # f_c_table as sparse CSR: (natoms, n_sc)
            fc_rows, fc_cols, fc_vals = [], [], []
            for ia in range(natoms):
                idxs = tree_sc.query_ball_point(tau[ia], self.r_cut)
                if len(idxs) == 0:
                    continue
                idxs = np.array(idxs, dtype=int)
                dists_ia = np.linalg.norm(sctau_flat[idxs] - tau[ia], axis=1)
                fc_ia = _f_cutoff_vec(dists_ia, self.r_taper, self.r_cut)
                # Zero out self-distances
                fc_ia[dists_ia < 1e-10] = 0.0
                nz = fc_ia > 0.0
                if nz.any():
                    fc_rows.append(np.full(nz.sum(), ia, dtype=np.int32))
                    fc_cols.append(idxs[nz])
                    fc_vals.append(fc_ia[nz])

            if fc_rows:
                fc_rows = np.concatenate(fc_rows)
                fc_cols = np.concatenate(fc_cols)
                fc_vals = np.concatenate(fc_vals)
            else:
                fc_rows = np.empty(0, dtype=np.int32)
                fc_cols = np.empty(0, dtype=np.int32)
                fc_vals = np.empty(0, dtype=float)

            f_c_sparse = csr_matrix((fc_vals, (fc_rows, fc_cols)), shape=(natoms, n_sc))
            del fc_rows, fc_cols, fc_vals

            # Coordination for on-site shift
            if self.onsite_shift is not None:
                coord_i = np.asarray(f_c_sparse.sum(axis=1)).ravel()
                for ia in range(natoms):
                    start = self.atom_block_start[ia]
                    for io, orb_label in enumerate(self.orbitals[ia]):
                        if orb_label == "s":
                            orb_type = "s"
                        elif orb_label in _P_INDEX:
                            orb_type = "p"
                        else:
                            orb_type = "d"
                        eta = self.onsite_shift.get(orb_type, 0.0)
                        self.onsite[start + io] += eta * coord_i[ia]

            # S_all[ia, jsc] = Σ_k f_c(d_ik) · f_c(d_jk)
            # = f_c_sparse[ia, :] @ f_c_full_sc[:, jsc]
            # We need f_c_full_sc as sparse (n_sc, n_sc): f_c(|sctau[i]-sctau[j]|)
            # But that's still too large.  Instead compute S_all on-the-fly
            # per R-vector during the bond loop, since we only need S_all[ia, jsc]
            # for bonds that exist.
            #
            # Strategy: build sparse f_c_sc (n_sc, n_sc) in chunks using KDTree,
            # then S_all = f_c_sparse @ f_c_sc.T  (sparse × sparse = sparse-friendly).
            if self.verbose:
                print("    Building sparse f_c_sc via KDTree ...")
            fc_sc_rows, fc_sc_cols, fc_sc_vals = [], [], []
            _BATCH = max(1, min(2000, natoms))  # process supercell atoms in batches
            for s in range(0, n_sc, _BATCH):
                e = min(s + _BATCH, n_sc)
                batch_pts = sctau_flat[s:e]
                # Query all neighbors within r_cut for this batch
                neigh_lists = tree_sc.query_ball_point(batch_pts, self.r_cut)
                for local_i, nlist in enumerate(neigh_lists):
                    if len(nlist) == 0:
                        continue
                    global_i = s + local_i
                    nlist = np.array(nlist, dtype=int)
                    dists_i = np.linalg.norm(
                        sctau_flat[nlist] - sctau_flat[global_i], axis=1
                    )
                    fc_i = _f_cutoff_vec(dists_i, self.r_taper, self.r_cut)
                    fc_i[dists_i < 1e-10] = 0.0
                    nz = fc_i > 0.0
                    if nz.any():
                        nn = nz.sum()
                        fc_sc_rows.append(np.full(nn, global_i, dtype=np.int32))
                        fc_sc_cols.append(nlist[nz])
                        fc_sc_vals.append(fc_i[nz])

            if fc_sc_rows:
                fc_sc_rows = np.concatenate(fc_sc_rows)
                fc_sc_cols = np.concatenate(fc_sc_cols)
                fc_sc_vals = np.concatenate(fc_sc_vals)
            else:
                fc_sc_rows = np.empty(0, dtype=np.int32)
                fc_sc_cols = np.empty(0, dtype=np.int32)
                fc_sc_vals = np.empty(0, dtype=float)

            f_c_sc_sparse = csr_matrix(
                (fc_sc_vals, (fc_sc_rows, fc_sc_cols)), shape=(n_sc, n_sc)
            )
            del fc_sc_rows, fc_sc_cols, fc_sc_vals, tree_sc

            # S_all = f_c_sparse @ f_c_sc_sparse.T  →  (natoms, n_sc) dense
            # This is sparse × sparse → dense, but the result is only accessed
            # at bond locations.  For memory, compute it row-by-row.
            if self.verbose:
                est_mb = natoms * n_sc * 8 / 1e6
                print(f"    Computing S_all ({est_mb:.0f} MB) via sparse matmul ...")
            S_all = np.zeros((natoms, n_sc), dtype=float)
            _ROW_BATCH = max(1, min(natoms, 500_000_000 // (8 * n_sc)))
            for s in range(0, natoms, _ROW_BATCH):
                e = min(s + _ROW_BATCH, natoms)
                S_all[s:e] = (f_c_sparse[s:e] @ f_c_sc_sparse.T).toarray()

            del f_c_sparse, f_c_sc_sparse

        else:
            # ── Dense path (small systems — original code) ──────
            # f_c_table[ia, n] = f_c(|tau[ia] - sctau_flat[n]|)
            dist_tau_sc = cdist(tau, sctau_flat)
            f_c_table = _f_cutoff_vec(dist_tau_sc, self.r_taper, self.r_cut)
            f_c_table[dist_tau_sc < 1e-10] = 0.0
            del dist_tau_sc

            # Coordination for on-site shift
            if self.onsite_shift is not None:
                coord_i = f_c_table.sum(axis=1)
                for ia in range(natoms):
                    start = self.atom_block_start[ia]
                    for io, orb_label in enumerate(self.orbitals[ia]):
                        if orb_label == "s":
                            orb_type = "s"
                        elif orb_label in _P_INDEX:
                            orb_type = "p"
                        else:
                            orb_type = "d"
                        eta = self.onsite_shift.get(orb_type, 0.0)
                        self.onsite[start + io] += eta * coord_i[ia]

            # S_all[ia, jsc] = Σ_k f_c(d_ik) · f_c(d_jk)
            # Chunked computation to avoid materializing the full (n_sc × n_sc) matrix.
            if self.verbose:
                est_mb = n_sc * n_sc * 8 / 1e6
                print(f"  Computing S_all (chunked, full would be {est_mb:.0f} MB)...")
            S_all = np.zeros((natoms, n_sc), dtype=float)
            _CHUNK = min(n_sc, max(256, 100_000_000 // (8 * n_sc)))  # ~100 MB per chunk
            for start in range(0, n_sc, _CHUNK):
                end = min(start + _CHUNK, n_sc)
                d_chunk = cdist(sctau_flat[start:end], sctau_flat)  # (cs, n_sc)
                fc_chunk = _f_cutoff_vec(d_chunk, self.r_taper, self.r_cut)
                fc_chunk[d_chunk < 1e-10] = 0.0
                del d_chunk
                S_all[:, start:end] = f_c_table @ fc_chunk.T  # (natoms, cs)
                del fc_chunk
            del f_c_table

        if self.verbose:
            print("  Building bond list...")

        # ── Enumerate bonds (vectorised when possible) ──────────
        _ref_orbs = tuple(self.orbitals[0])
        _norb = len(_ref_orbs)
        _use_vec = (
            all(tuple(self.orbitals[ia]) == _ref_orbs for ia in range(natoms))
            and _ref_orbs == _SPD_ORBS
        )

        if _use_vec:
            # ── Fully vectorised path (homogeneous spd) ─────────
            if self.verbose:
                print("    (vectorised spd path)")
            # oa_range = np.arange(_norb, dtype=np.int32)
            rows_chunks = []
            cols_chunks = []
            vals_chunks = []
            R_chunks = []

            for ii in range(-cr[0], cr[0] + 1):
                for jj in range(-cr[1], cr[1] + 1):
                    for kk in range(-cr[2], cr[2] + 1):
                        R_vec = (
                            ii * self.a_vectors[0]
                            + jj * self.a_vectors[1]
                            + kk * self.a_vectors[2]
                        )
                        tau_shifted = tau + R_vec
                        D = cdist(tau, tau_shifted)
                        mask = (D > 1e-10) & (D <= self.dd_r_c)
                        ia_arr, ib_arr = np.nonzero(mask)
                        if len(ia_arr) == 0:
                            continue
                        dists = D[ia_arr, ib_arr]
                        del D

                        # Direction cosines
                        delta = tau_shifted[ib_arr] - tau[ia_arr]
                        dc = delta / dists[:, None]

                        # Goodwin hoppings (vectorised)
                        hop = _goodwin_all_channels_vec(
                            dists,
                            self.dd_channels,
                            self.dd_r_0,
                            self.dd_r_c,
                            self.dd_n_c,
                        )

                        # Screening
                        R_offset = ((ii + cr[0]) * int(nk2) + (jj + cr[1])) * int(
                            nk3
                        ) + (kk + cr[2])
                        jsc = R_offset * natoms + ib_arr
                        S_bonds = S_all[ia_arr, jsc]
                        for ch in hop:
                            g = self.gamma_map.get(ch, 0.0)
                            if g != 0.0:
                                hop[ch] = hop[ch] * np.exp(-g * S_bonds)

                        # SK blocks (vectorised)
                        blocks = _sk_block_spd_vec(
                            dc[:, 0],
                            dc[:, 1],
                            dc[:, 2],
                            hop,
                        )

                        # Scatter into COO arrays
                        nz_b, nz_oa, nz_ob = np.nonzero(np.abs(blocks) > 1e-15)
                        if len(nz_b) == 0:
                            continue
                        sa = self.atom_block_start[ia_arr[nz_b]]
                        sb = self.atom_block_start[ib_arr[nz_b]]
                        rows_chunks.append((sa + nz_oa).astype(np.int32))
                        cols_chunks.append((sb + nz_ob).astype(np.int32))
                        vals_chunks.append(blocks[nz_b, nz_oa, nz_ob])
                        R_chunks.append(
                            np.broadcast_to(
                                np.array([[ii, jj, kk]], dtype=np.int32),
                                (len(nz_b), 3),
                            ).copy()
                        )
                        del blocks

            del S_all
            if rows_chunks:
                self._bond_rows = np.concatenate(rows_chunks)
                self._bond_cols = np.concatenate(cols_chunks)
                self._bond_vals = np.concatenate(vals_chunks)
                self._bond_R = np.concatenate(R_chunks)
            else:
                self._bond_rows = np.empty(0, dtype=np.int32)
                self._bond_cols = np.empty(0, dtype=np.int32)
                self._bond_vals = np.empty(0, dtype=np.float64)
                self._bond_R = np.empty((0, 3), dtype=np.int32)

        else:
            # ── Original scalar path (heterogeneous / non-spd) ──
            if self.verbose:
                print("    (scalar fallback)")
            rows_list = []
            cols_list = []
            vals_list = []
            R_bond_list = []

            for ii in range(-cr[0], cr[0] + 1):
                for jj in range(-cr[1], cr[1] + 1):
                    for kk in range(-cr[2], cr[2] + 1):
                        for ia in range(natoms):
                            for ib in range(natoms):
                                i_wrap = ii + cr[0]
                                j_wrap = jj + cr[1]
                                k_wrap = kk + cr[2]
                                jsc = (
                                    (i_wrap * nk2 + j_wrap) * nk3 + k_wrap
                                ) * natoms + ib

                                pos_j = sctau_flat[jsc]
                                dist_val = np.sqrt(np.sum((tau[ia] - pos_j) ** 2))
                                if dist_val < 1e-10 or dist_val > self.dd_r_c:
                                    continue

                                dc = (pos_j - tau[ia]) / dist_val
                                lx, ly, lz = dc[0], dc[1], dc[2]

                                hop = _goodwin_all_channels(
                                    dist_val,
                                    self.dd_channels,
                                    self.dd_r_0,
                                    self.dd_r_c,
                                    self.dd_n_c,
                                )

                                S_ij = S_all[ia, jsc]
                                screened = {}
                                for ch, val in hop.items():
                                    g = self.gamma_map.get(ch, 0.0)
                                    screened[ch] = val * np.exp(-g * S_ij)

                                orbs_a = self.orbitals[ia]
                                orbs_b = self.orbitals[ib]
                                sa = self.atom_block_start[ia]
                                sb = self.atom_block_start[ib]

                                for noa, oa in enumerate(orbs_a):
                                    for nob, ob in enumerate(orbs_b):
                                        v = _sk_element(oa, ob, lx, ly, lz, screened)
                                        if abs(v) > 1e-15:
                                            rows_list.append(sa + noa)
                                            cols_list.append(sb + nob)
                                            vals_list.append(v)
                                            R_bond_list.append([ii, jj, kk])

            del S_all
            self._bond_rows = np.array(rows_list, dtype=np.int32)
            self._bond_cols = np.array(cols_list, dtype=np.int32)
            self._bond_vals = np.array(vals_list, dtype=np.float64)
            self._bond_R = np.array(R_bond_list, dtype=np.int32)

        n_bonds = len(self._bond_vals)
        if self.verbose:
            mem_mb = (
                self._bond_vals.nbytes
                + self._bond_rows.nbytes
                + self._bond_cols.nbytes
                + self._bond_R.nbytes
            ) / 1e6
            print(f"  {n_bonds} non-zero bond entries, {mem_mb:.1f} MB")

    # ── Assemble sparse H(k) ───────────────────────────────────

    def build_hk(self, k_frac):
        """Build sparse H(k) as a CSR matrix.

        Parameters
        ----------
        k_frac : array_like, shape (3,)
            k-point in fractional coordinates.

        Returns
        -------
        Hk : csr_matrix, shape (nawf, nawf), complex128
        """
        k_frac = np.asarray(k_frac, dtype=float)
        # Phase: exp(2πi k · R_int)
        phases = np.exp(2j * np.pi * (self._bond_R @ k_frac))
        vals_k = self._bond_vals * phases

        # Add on-site (R=0 diagonal)
        n_onsite = self.nawf
        all_rows = np.concatenate(
            [self._bond_rows, np.arange(n_onsite, dtype=np.int32)]
        )
        all_cols = np.concatenate(
            [self._bond_cols, np.arange(n_onsite, dtype=np.int32)]
        )
        all_vals = np.concatenate([vals_k, self.onsite.astype(complex)])

        Hk = csr_matrix(
            (all_vals, (all_rows, all_cols)),
            shape=(self.nawf, self.nawf),
        )
        return Hk

    # ── Eigenvalues at a single k-point ─────────────────────────

    def eigvals_at_k(self, k_frac, n_eigs=50, sigma=None, **kwargs):
        """Compute selected eigenvalues at one k-point via Lanczos.

        Parameters
        ----------
        k_frac : array_like, shape (3,)
        n_eigs : int
            Number of eigenvalues to compute.
        sigma : float or None
            Shift-invert target.  If provided, eigsh computes eigenvalues
            nearest to sigma.  Much faster convergence for interior eigenvalues
            but requires a sparse LU factorization.
            If None, computes the smallest eigenvalues (which='SM').
        **kwargs
            Extra arguments passed to scipy.sparse.linalg.eigsh.

        Returns
        -------
        eigenvalues : ndarray, shape (n_eigs,)  sorted ascending
        """
        Hk = self.build_hk(k_frac)
        n_eigs = min(n_eigs, self.nawf - 2)

        if sigma is not None:
            evals = eigsh(
                Hk,
                k=n_eigs,
                sigma=sigma,
                which="LM",
                return_eigenvectors=False,
                **kwargs,
            )
        else:
            evals = eigsh(Hk, k=n_eigs, which="SA", return_eigenvectors=False, **kwargs)
        return np.sort(evals)

    # ── Eigenpairs at a single k-point (for unfolding) ──────────

    def eigh_at_k(self, k_frac, n_eigs=50, sigma=None, **kwargs):
        """Compute selected eigenvalues *and* eigenvectors at one k-point.

        Same solver as :meth:`eigvals_at_k` but also returns the Lanczos
        eigenvectors, needed to build spectral weights when unfolding.

        Parameters
        ----------
        k_frac : array_like, shape (3,)
        n_eigs : int
            Number of eigenpairs to compute.
        sigma : float or None
            Shift-invert target (see :meth:`eigvals_at_k`).
        **kwargs
            Extra arguments passed to scipy.sparse.linalg.eigsh.

        Returns
        -------
        evals : ndarray, shape (n_eigs,)          sorted ascending
        evecs : ndarray, shape (nawf, n_eigs)     columns match ``evals``
        """
        Hk = self.build_hk(k_frac)
        n_eigs = min(n_eigs, self.nawf - 2)

        if sigma is not None:
            evals, evecs = eigsh(
                Hk,
                k=n_eigs,
                sigma=sigma,
                which="LM",
                return_eigenvectors=True,
                **kwargs,
            )
        else:
            evals, evecs = eigsh(
                Hk, k=n_eigs, which="SA", return_eigenvectors=True, **kwargs
            )
        order = np.argsort(evals)
        return evals[order], evecs[:, order]

    # ── Full band structure ─────────────────────────────────────

    def compute_bands(
        self,
        band_path,
        high_sym_points,
        nk=100,
        n_eigs=50,
        sigma=None,
        outputdir=None,
        return_eigenvectors=False,
        n_workers=1,
        backend="loky",
        **kwargs,
    ):
        """Compute band structure (and optionally eigenvectors) along a k-path.

        Parameters
        ----------
        band_path : str
            Path specification, e.g. "K-G-M-K'".
        high_sym_points : dict
            {label: [k1, k2, k3]} in fractional coordinates.
        nk : int
            Total number of k-points along the path.
        n_eigs : int
            Number of eigenvalues at each k-point.
        sigma : float or None
            Shift-invert target energy (eV).
        outputdir : str or None
            If set, write bands_0.dat and kpath_points.txt.
        n_workers : int
            Number of parallel workers for k-point loop.
            1 = serial (default).  -1 = all available cores.
            Requires joblib; falls back to serial if unavailable.
        backend : str
            joblib backend for the k-point loop.  Default 'loky' (separate
            processes).  ARPACK's shift-invert solve runs a Python-level
            reverse-communication loop that holds the GIL, so a 'threading'
            backend does not parallelize it — use processes for real speedup.
        **kwargs
            Extra arguments for eigsh (e.g. tol, maxiter).

        Returns
        -------
        result : dict
            'eigenvalues' : ndarray (nk, n_eigs)
            'k_dist'      : ndarray (nk,)
            'tick_pos'     : list
            'tick_labels'  : list
            'bands_file'   : str or None
        """
        kpoints, k_dist, tick_pos, tick_labels = _generate_kpath(
            band_path, high_sym_points, nk, self.b_vectors
        )

        nk_actual = len(kpoints)
        eigenvalues = np.zeros((nk_actual, n_eigs))

        if self.verbose:
            print(f"Computing {n_eigs} eigenvalues at {nk_actual} k-points...")

        _use_parallel = (n_workers != 1) and _HAS_JOBLIB

        # Auto-scale workers: for large Hamiltonians the per-k-point LU
        # factorization is memory-bandwidth-limited.  Running many workers
        # simultaneously thrashes the cache and multiplies peak memory
        # (each LU creates GB-scale fill-in).  Better to run fewer workers
        # and let BLAS/SuperLU use wider vectors internally.
        if _use_parallel and n_workers == -1:
            import multiprocessing

            n_cpus = multiprocessing.cpu_count()
            if self.nawf > 20_000:
                # Very large: serial is fastest (let BLAS use all cores)
                n_workers = 1
                _use_parallel = False
                if self.verbose:
                    print(
                        f"  nawf={self.nawf} > 20k: switching to serial "
                        f"(LU memory per k-point too large for parallel)"
                    )
            elif self.nawf > 5_000:
                # Medium: limit to 2-4 workers
                n_workers = min(4, max(1, n_cpus // 2))
                if self.verbose:
                    print(
                        f"  nawf={self.nawf} > 5k: limiting to {n_workers} "
                        f"workers (of {n_cpus} cores)"
                    )
            # else: small system — keep n_workers=-1 (all cores)

        if _use_parallel:
            if self.verbose:
                print(f"  Parallel: {n_workers} workers (joblib/{backend})")

            # threadpoolctl pins BLAS/LAPACK threads at runtime (env vars are
            # ignored once the library is loaded) so process workers don't
            # oversubscribe cores.
            try:
                from threadpoolctl import threadpool_limits

                _ctx = threadpool_limits(limits=1, user_api="blas")
            except ImportError:
                from contextlib import nullcontext

                _ctx = nullcontext()

            with _ctx:
                all_evals = Parallel(n_jobs=n_workers, backend=backend)(
                    delayed(self.eigvals_at_k)(k, n_eigs=n_eigs, sigma=sigma, **kwargs)
                    for k in kpoints
                )

            for ik, evals in enumerate(all_evals):
                eigenvalues[ik, : len(evals)] = evals
        else:
            if n_workers != 1 and not _HAS_JOBLIB:
                print("  Warning: joblib not installed, falling back to serial.")
            for ik, k in enumerate(kpoints):
                evals = self.eigvals_at_k(k, n_eigs=n_eigs, sigma=sigma, **kwargs)
                eigenvalues[ik, : len(evals)] = evals
                if self.verbose and (ik + 1) % 10 == 0:
                    print(f"  k-point {ik + 1}/{nk_actual}")

        # Model parameters (V0, onsite) are already in eV from the fitter,
        # so eigenvalues come out in eV — no unit conversion needed.

        # Write output files
        bands_file = None
        if outputdir is not None:
            os.makedirs(outputdir, exist_ok=True)
            bands_file = os.path.join(outputdir, "bands_0.dat")
            data = np.column_stack([k_dist, eigenvalues])
            np.savetxt(bands_file, data, fmt="%.10e")

            kpath_file = os.path.join(outputdir, "kpath_points.txt")
            with open(kpath_file, "w") as f:
                f.writelines(
                    f"{pos:.10e}  {lab}\n" for pos, lab in zip(tick_pos, tick_labels)
                )
            if self.verbose:
                print(f"Bands saved to {bands_file}")

        return {
            "eigenvalues": eigenvalues,
            "k_dist": k_dist,
            "kpoints": kpoints,
            "tick_pos": tick_pos,
            "tick_labels": tick_labels,
            "bands_file": bands_file,
        }
