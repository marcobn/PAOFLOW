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

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist

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
        R_ints_flat = np.array(R_ints, dtype=int)  # (n_sc, 3)
        n_sc = len(sctau_flat)

        if self.verbose:
            print(f"  Precomputing screening (n_sc={n_sc})...")

        # ── Vectorized f_c tables ───────────────────────────────
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

        # ── Enumerate bonds ─────────────────────────────────────
        rows_list = []
        cols_list = []
        vals_list = []
        R_bond_list = []  # integer R for each bond entry

        n_bonds = 0
        distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
        cosines = lambda x, y: (y - x) / np.sqrt(np.sum((x - y) ** 2))

        for i in range(-cr[0], cr[0] + 1):
            for j in range(-cr[1], cr[1] + 1):
                for k in range(-cr[2], cr[2] + 1):
                    for ia in range(natoms):
                        for ib in range(natoms):
                            # Index into sctau_flat
                            i_wrap = i + cr[0]
                            j_wrap = j + cr[1]
                            k_wrap = k + cr[2]
                            jsc = ((i_wrap * nk2 + j_wrap) * nk3 + k_wrap) * natoms + ib

                            pos_j = sctau_flat[jsc]
                            dist_val = distance(tau[ia], pos_j)
                            if dist_val < 1e-10 or dist_val > self.dd_r_c:
                                continue

                            # Direction cosines
                            dc = cosines(tau[ia], pos_j)
                            lx, ly, lz = dc[0], dc[1], dc[2]

                            # Goodwin hoppings
                            hop = _goodwin_all_channels(
                                dist_val,
                                self.dd_channels,
                                self.dd_r_0,
                                self.dd_r_c,
                                self.dd_n_c,
                            )

                            # Apply screening
                            S_ij = S_all[ia, jsc]
                            screened = {}
                            for ch, val in hop.items():
                                g = self.gamma_map.get(ch, 0.0)
                                screened[ch] = val * np.exp(-g * S_ij)

                            # SK block
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
                                        R_bond_list.append([i, j, k])
                                        n_bonds += 1

        del S_all

        # Store as NumPy arrays
        self._bond_rows = np.array(rows_list, dtype=np.int32)
        self._bond_cols = np.array(cols_list, dtype=np.int32)
        self._bond_vals = np.array(vals_list, dtype=np.float64)
        self._bond_R = np.array(R_bond_list, dtype=np.int32)  # (n_bonds, 3)

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

    # ── Full band structure ─────────────────────────────────────

    def compute_bands(
        self,
        band_path,
        high_sym_points,
        nk=100,
        n_eigs=50,
        sigma=None,
        outputdir=None,
        **kwargs,
    ):
        """Compute band structure along a k-path.

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
            import os

            os.makedirs(outputdir, exist_ok=True)
            bands_file = os.path.join(outputdir, "bands_0.dat")
            data = np.column_stack([k_dist, eigenvalues])
            np.savetxt(bands_file, data, fmt="%.10e")

            kpath_file = os.path.join(outputdir, "kpath_points.txt")
            with open(kpath_file, "w") as f:
                for pos, lab in zip(tick_pos, tick_labels):
                    f.write(f"{pos:.10e}  {lab}\n")
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
