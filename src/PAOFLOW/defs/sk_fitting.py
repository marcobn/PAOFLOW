# sk_fitting.py
#
# Slater-Koster two-center tight-binding fitting engine.
#
# Extracts SK parameters from a PAOFLOW PAO Hamiltonian (HRs) via
# eigenvalue-based least-squares fitting with analytic (Hellmann-Feynman)
# Jacobian.  Supports single-shell and multi-shell configurations.
#
# Usage
# -----
#   from sk_fitting import SKFitter
#
#   fitter = SKFitter(arryp, attrp, n_shells=2, nkfit=6)
#   result = fitter.fit(n_trials=20, seed=123)
#
#   # Optimized parameter vector & RMSE
#   p_opt  = result['p_opt']
#   rmse   = result['rmse']
#
#   # Ready-to-use PAOFLOW model dict
#   model  = fitter.build_model_dict(p_opt)
#
# References
# ----------
# J.C. Slater & G.F. Koster, Phys. Rev. 94, 1498 (1954), Table I.

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

# ═══════════════════════════════════════════════════════════════
#  1. Slater-Koster two-center integrals (standard √3 convention)
# ═══════════════════════════════════════════════════════════════

ORBITAL_NAMES = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2-y2", "dz2")
SK_PARAM_NAMES = ["sss", "sps", "pps", "ppp", "sds", "pds", "pdp", "dds", "ddp", "ddd"]
SK_LABELS = [
    "Vssσ",
    "Vspσ",
    "Vppσ",
    "Vppπ",
    "Vsdσ",
    "Vpdσ",
    "Vpdπ",
    "Vddσ",
    "Vddπ",
    "Vddδ",
]

_SQ3 = np.sqrt(3.0)
_HSQ3 = _SQ3 / 2.0

SHELL_TO_ORBITALS = {
    0: ["s"],
    1: ["px", "py", "pz"],
    2: ["dxy", "dyz", "dzx", "dx2-y2", "dz2"],
}

# Active SK parameter names for each angular-momentum pair (l_lo, l_hi)
LPAIR_ACTIVE_NAMES = {
    (0, 0): ["sss"],
    (0, 1): ["sps"],
    (0, 2): ["sds"],
    (1, 1): ["pps", "ppp"],
    (1, 2): ["pds", "pdp"],
    (2, 2): ["dds", "ddp", "ddd"],
}
LPAIR_ACTIVE_INDICES = {
    lpair: [SK_PARAM_NAMES.index(n) for n in names]
    for lpair, names in LPAIR_ACTIVE_NAMES.items()
}
CHANNEL_LABELS = {
    (0, 0): ["σ"],
    (0, 1): ["σ"],
    (0, 2): ["σ"],
    (1, 1): ["σ", "π"],
    (1, 2): ["σ", "π"],
    (2, 2): ["σ", "π", "δ"],
}


def sk_element(
    orb_a: str, orb_b: str, lx: float, ly: float, lz: float, sh: dict
) -> float:
    """Slater-Koster two-center hopping matrix element H(orb_a, orb_b).

    Parameters
    ----------
    orb_a, orb_b : str
        Orbital names from ``ORBITAL_NAMES``.
    lx, ly, lz : float
        Direction cosines of the bond vector.
    sh : dict
        SK parameter dict with keys from ``SK_PARAM_NAMES``.

    Returns
    -------
    float
        The matrix element value.
    """
    sq3, hsq3 = _SQ3, _HSQ3
    l2, m2, n2 = lx * lx, ly * ly, lz * lz
    lm, ln, mn = lx * ly, lx * lz, ly * lz
    pd = {"px": lx, "py": ly, "pz": lz}
    do = {"dxy", "dyz", "dzx", "dx2-y2", "dz2"}

    # s-s
    if orb_a == "s" and orb_b == "s":
        return sh["sss"]

    # s-p / p-s
    if orb_a == "s" and orb_b in pd:
        return pd[orb_b] * sh["sps"]
    if orb_b == "s" and orb_a in pd:
        return -pd[orb_a] * sh["sps"]

    # p-p
    if orb_a in pd and orb_b in pd:
        if orb_a == orb_b:
            ll = pd[orb_a]
            return ll**2 * sh["pps"] + (1 - ll**2) * sh["ppp"]
        return pd[orb_a] * pd[orb_b] * (sh["pps"] - sh["ppp"])

    # s-d / d-s
    def _sd(d):
        if d == "dxy":
            return sq3 * lm * sh["sds"]
        if d == "dyz":
            return sq3 * mn * sh["sds"]
        if d == "dzx":
            return sq3 * ln * sh["sds"]
        if d == "dx2-y2":
            return hsq3 * (l2 - m2) * sh["sds"]
        if d == "dz2":
            return (n2 - 0.5 * (l2 + m2)) * sh["sds"]

    if orb_a == "s" and orb_b in do:
        return _sd(orb_b)
    if orb_b == "s" and orb_a in do:
        return _sd(orb_a)

    # p-d / d-p
    def _pd(p, d):
        S, P = sh["pds"], sh["pdp"]
        if p == "px":
            if d == "dxy":
                return sq3 * l2 * ly * S + ly * (1 - 2 * l2) * P
            if d == "dyz":
                return sq3 * lm * lz * S - 2 * lm * lz * P
            if d == "dzx":
                return sq3 * l2 * lz * S + lz * (1 - 2 * l2) * P
            if d == "dx2-y2":
                return hsq3 * lx * (l2 - m2) * S + lx * (1 - l2 + m2) * P
            if d == "dz2":
                return lx * (n2 - 0.5 * (l2 + m2)) * S - sq3 * lx * n2 * P
        if p == "py":
            if d == "dxy":
                return sq3 * m2 * lx * S + lx * (1 - 2 * m2) * P
            if d == "dyz":
                return sq3 * m2 * lz * S + lz * (1 - 2 * m2) * P
            if d == "dzx":
                return sq3 * lm * lz * S - 2 * lm * lz * P
            if d == "dx2-y2":
                return hsq3 * ly * (l2 - m2) * S - ly * (1 + l2 - m2) * P
            if d == "dz2":
                return ly * (n2 - 0.5 * (l2 + m2)) * S - sq3 * ly * n2 * P
        if p == "pz":
            if d == "dxy":
                return sq3 * lm * lz * S - 2 * lm * lz * P
            if d == "dyz":
                return sq3 * n2 * ly * S + ly * (1 - 2 * n2) * P
            if d == "dzx":
                return sq3 * n2 * lx * S + lx * (1 - 2 * n2) * P
            if d == "dx2-y2":
                return hsq3 * lz * (l2 - m2) * S - lz * (l2 - m2) * P
            if d == "dz2":
                return lz * (n2 - 0.5 * (l2 + m2)) * S + sq3 * lz * (l2 + m2) * P
        return 0.0

    if orb_a in pd and orb_b in do:
        return _pd(orb_a, orb_b)
    if orb_b in pd and orb_a in do:
        return -_pd(orb_b, orb_a)

    # d-d
    if orb_a in do and orb_b in do:
        S, P, D = sh["dds"], sh["ddp"], sh["ddd"]
        l2m2, l2n2, m2n2 = l2 * m2, l2 * n2, m2 * n2
        df = l2 - m2
        da, db = orb_a, orb_b

        # diagonal
        if da == db == "dxy":
            return 3 * l2m2 * S + (l2 + m2 - 4 * l2m2) * P + (n2 + l2m2) * D
        if da == db == "dyz":
            return 3 * m2n2 * S + (m2 + n2 - 4 * m2n2) * P + (l2 + m2n2) * D
        if da == db == "dzx":
            return 3 * l2n2 * S + (l2 + n2 - 4 * l2n2) * P + (m2 + l2n2) * D
        if da == db == "dx2-y2":
            return 0.75 * df**2 * S + (l2 + m2 - df**2) * P + (n2 + 0.25 * df**2) * D
        if da == db == "dz2":
            t = n2 - 0.5 * (l2 + m2)
            return t**2 * S + 3 * n2 * (l2 + m2) * P + 0.75 * (l2 + m2) ** 2 * D

        # off-diagonal
        p = frozenset([da, db])
        if p == frozenset(["dxy", "dyz"]):
            return 3 * lx * m2 * lz * S + ln * (1 - 4 * m2) * P + ln * (m2 - 1) * D
        if p == frozenset(["dxy", "dzx"]):
            return 3 * l2 * ly * lz * S + mn * (1 - 4 * l2) * P + mn * (l2 - 1) * D
        if p == frozenset(["dyz", "dzx"]):
            return 3 * ly * n2 * lx * S + lm * (1 - 4 * n2) * P + lm * (n2 - 1) * D
        if p == frozenset(["dxy", "dx2-y2"]):
            return 1.5 * lm * df * S + 2 * lm * (m2 - l2) * P + 0.5 * lm * df * D
        if p == frozenset(["dyz", "dx2-y2"]):
            return 1.5 * mn * df * S - mn * (1 + 2 * df) * P + mn * (1 + 0.5 * df) * D
        if p == frozenset(["dzx", "dx2-y2"]):
            return 1.5 * ln * df * S + ln * (1 - 2 * df) * P - ln * (1 - 0.5 * df) * D
        # couplings to dz² (√3 factors)
        t = n2 - 0.5 * (l2 + m2)
        if p == frozenset(["dxy", "dz2"]):
            return sq3 * (lm * t * S - 2 * lm * n2 * P + 0.5 * lm * (1 + n2) * D)
        if p == frozenset(["dyz", "dz2"]):
            return sq3 * (
                mn * t * S + mn * (l2 + m2 - n2) * P - 0.5 * mn * (l2 + m2) * D
            )
        if p == frozenset(["dzx", "dz2"]):
            return sq3 * (
                ln * t * S + ln * (l2 + m2 - n2) * P - 0.5 * ln * (l2 + m2) * D
            )
        if p == frozenset(["dx2-y2", "dz2"]):
            return sq3 * (
                0.5 * df * t * S + n2 * (m2 - l2) * P + 0.25 * (1 + n2) * df * D
            )
    return 0.0


def sk_design_row(
    orb_a: str, orb_b: str, lx: float, ly: float, lz: float
) -> np.ndarray:
    """Coefficient of each SK parameter for matrix element H(orb_a, orb_b).

    Returns
    -------
    np.ndarray
        Length-10 array: entry *k* is the coefficient of ``SK_PARAM_NAMES[k]``.
    """
    row = np.zeros(len(SK_PARAM_NAMES))
    for k, name in enumerate(SK_PARAM_NAMES):
        unit = {n: (1.0 if n == name else 0.0) for n in SK_PARAM_NAMES}
        row[k] = sk_element(orb_a, orb_b, lx, ly, lz, unit)
    return row


# ═══════════════════════════════════════════════════════════════
#  2. SKFitter — main fitting engine
# ═══════════════════════════════════════════════════════════════


class SKFitter:
    """Slater-Koster eigenvalue-based fitter.

    Constructs design tensors from a PAO Hamiltonian and fits SK parameters
    by minimising the eigenvalue RMSE on a uniform k-mesh.

    Parameters
    ----------
    arryp : dict
        PAOFLOW ``arrays`` dict (needs ``a_vectors``, ``b_vectors``, ``tau``,
        ``atoms``, ``shells``, ``HRs``; optionally ``configuration``).
    attrp : dict
        PAOFLOW ``attributes`` dict (needs ``alat``, ``natoms``).
    n_shells : int
        Number of neighbor shells to include (default 2 → NN + NNN).
    nkfit : int
        Subdivisions along each reciprocal axis for the fitting k-mesh
        (total k-points = ``nkfit**3``).
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        arryp: dict,
        attrp: dict,
        *,
        n_shells: int = 2,
        nkfit: int = 6,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self._setup_system(arryp, attrp)
        self._enumerate_bonds()
        self._select_shells(n_shells)
        self._build_parameter_registry()
        self._build_reference_eigenvalues(nkfit)
        self._build_design_tensors()
        self._build_onsite_map()
        self._precompute_dHk()
        self._build_regularization_weights()

    # ── 2a. System setup ──────────────────────────────────────

    def _setup_system(self, arryp, attrp):
        HRs = arryp["HRs"]
        self.a_vecs = arryp["a_vectors"]
        self.b_vecs = arryp["b_vectors"]
        self.alat = attrp["alat"]
        self.nat = attrp["natoms"]
        self.nawf = HRs.shape[0]
        self.nk1 = HRs.shape[2]
        self.nk2 = HRs.shape[3]
        self.nk3 = HRs.shape[4]
        self.HRs = HRs

        self.atoms_list = arryp["atoms"]
        self.tau_bohr = arryp["tau"]
        self.tau_alat = self.tau_bohr / self.alat
        self.shells_dict = arryp["shells"]
        self.config_dict = arryp.get("configuration", None)
        self.unique_species = list(dict.fromkeys(self.atoms_list))

        # Per-atom orbital structure
        self.atom_orbitals = []
        self.atom_orbital_group = []
        self.atom_block_start = []
        self.norb_per_atom = []

        idx = 0
        for iat in range(self.nat):
            sp = self.atoms_list[iat]
            orb_list, grp_list = [], []
            for ig, shell_l in enumerate(self.shells_dict[sp]):
                for orb_name in SHELL_TO_ORBITALS[shell_l]:
                    orb_list.append(orb_name)
                    grp_list.append(ig)
            self.atom_orbitals.append(orb_list)
            self.atom_orbital_group.append(grp_list)
            self.atom_block_start.append(idx)
            self.norb_per_atom.append(len(orb_list))
            idx += len(orb_list)
        assert idx == self.nawf, f"Total orbitals ({idx}) != nawf ({self.nawf})"

        sp0 = self.unique_species[0]
        self.cfg_names = (
            list(self.config_dict[sp0])
            if self.config_dict
            else [
                f"g{i}(l={self.shells_dict[sp0][i]})"
                for i in range(len(self.shells_dict[sp0]))
            ]
        )
        self.n_groups = len(self.shells_dict[sp0])
        self.group_l = list(self.shells_dict[sp0])

        # R-vectors and H(R) blocks
        R_list, HR_list = [], []
        for i1 in range(self.nk1):
            for i2 in range(self.nk2):
                for i3 in range(self.nk3):
                    r1 = i1 if 2 * i1 <= self.nk1 else i1 - self.nk1
                    r2 = i2 if 2 * i2 <= self.nk2 else i2 - self.nk2
                    r3 = i3 if 2 * i3 <= self.nk3 else i3 - self.nk3
                    R_cart = (
                        r1 * self.a_vecs[0] + r2 * self.a_vecs[1] + r3 * self.a_vecs[2]
                    )
                    R_list.append(R_cart)
                    HR_list.append(HRs[:, :, i1, i2, i3, 0])
        self.R_arr = np.array(R_list)
        self.HR_arr = np.array(HR_list)

        if self.verbose:
            print(
                f"SKFitter: {self.nat} atoms, nawf={self.nawf}, "
                f"alat={self.alat:.4f} Bohr"
            )
            print(f"  Shell groups: {self.cfg_names} (l = {self.group_l})")

    # ── 2b. Bond enumeration ──────────────────────────────────

    def _enumerate_bonds(self):
        shell_bonds = defaultdict(list)
        for i1 in range(self.nk1):
            for i2 in range(self.nk2):
                for i3 in range(self.nk3):
                    r1 = i1 if 2 * i1 <= self.nk1 else i1 - self.nk1
                    r2 = i2 if 2 * i2 <= self.nk2 else i2 - self.nk2
                    r3 = i3 if 2 * i3 <= self.nk3 else i3 - self.nk3
                    R_cart = (
                        r1 * self.a_vecs[0] + r2 * self.a_vecs[1] + r3 * self.a_vecs[2]
                    )
                    for iat in range(self.nat):
                        for jat in range(self.nat):
                            d_vec = R_cart + self.tau_alat[jat] - self.tau_alat[iat]
                            d_norm = np.linalg.norm(d_vec)
                            if d_norm < 1e-8:
                                continue
                            shell_bonds[round(d_norm, 5)].append(
                                (R_cart, iat, jat, d_vec, d_norm, i1, i2, i3)
                            )
        self._shell_bonds = shell_bonds
        self.shell_dists = sorted(shell_bonds.keys())

        if self.verbose:
            n_show = min(5, len(self.shell_dists))
            print(f"  Neighbor shells (first {n_show} of {len(self.shell_dists)}):")
            for s, d in enumerate(self.shell_dists[:n_show]):
                bonds = shell_bonds[d]
                pairs = set((b[1], b[2]) for b in bonds)
                pair_str = ", ".join(
                    f"{self.atoms_list[i]}({i})→{self.atoms_list[j]}({j})"
                    for i, j in sorted(pairs)
                )
                print(
                    f"    Shell {s + 1}: d={d:.5f}, {len(bonds):>3d} bonds  ({pair_str})"
                )

    def _select_shells(self, n_shells: int):
        """Pick the neighbor shells to include in the fit."""
        self.n_shells = min(n_shells, len(self.shell_dists))
        self.shell_bonds_list = [
            self._shell_bonds[self.shell_dists[i]] for i in range(self.n_shells)
        ]
        shell_tags = ["nn", "nnn", "nnnn", "nnnnn"]
        self.shell_tags = shell_tags[: self.n_shells]
        if self.verbose:
            for i, tag in enumerate(self.shell_tags):
                print(f"  Included: {tag} ({len(self.shell_bonds_list[i])} bonds)")

    # ── 2c. Parameter registry ────────────────────────────────

    def _build_parameter_registry(self):
        """Register hopping parameters for each shell-group pair."""
        self.hop_pair_list = []
        self.hop_pair_start = {}
        self.hop_pair_active = {}
        self.n_hop = 0
        self.hop_param_labels = []

        for ga in range(self.n_groups):
            for gb in range(ga, self.n_groups):
                la, lb = self.group_l[ga], self.group_l[gb]
                lpair = (min(la, lb), max(la, lb))
                active = LPAIR_ACTIVE_INDICES[lpair]
                labels = CHANNEL_LABELS[lpair]

                self.hop_pair_list.append((ga, gb))
                self.hop_pair_start[(ga, gb)] = self.n_hop
                self.hop_pair_active[(ga, gb)] = active

                pair_tag = f"{self.cfg_names[ga]}-{self.cfg_names[gb]}"
                for lab in labels:
                    self.hop_param_labels.append(f"V({pair_tag}){lab}")
                self.n_hop += len(active)

        # On-site parameters
        self.species_onsite_groups = {}
        self.species_param_start = {}
        self.n_onsite = 0
        self.onsite_param_names = []

        for sp in self.unique_species:
            self.species_param_start[sp] = self.n_onsite
            cfg = (
                list(self.config_dict[sp])
                if self.config_dict
                else [f"g{i}" for i in range(len(self.shells_dict[sp]))]
            )
            groups = self._get_onsite_groups(self.shells_dict[sp], cfg)
            self.species_onsite_groups[sp] = groups
            for pname, _ in groups:
                self.onsite_param_names.append(pname)
            self.n_onsite += len(groups)

        self.n_params = self.n_onsite + self.n_shells * self.n_hop
        self.param_labels = list(self.onsite_param_names)
        for i, tag in enumerate(self.shell_tags):
            self.param_labels += [f"{tag.upper()}_{l}" for l in self.hop_param_labels]

        if self.verbose:
            print(
                f"  Parameters: {self.n_params} total "
                f"({self.n_onsite} on-site + "
                f"{self.n_shells}×{self.n_hop} hopping)"
            )

    @staticmethod
    def _get_onsite_groups(shell_l_list, cfg_labels):
        """Map each configuration group to its orbital indices, splitting d into t2g/eg."""
        groups = []
        idx = 0
        for ig, (l_val, cfg) in enumerate(zip(shell_l_list, cfg_labels)):
            norb_l = 2 * l_val + 1
            if l_val <= 1:
                groups.append((f"ε({cfg})", list(range(idx, idx + norb_l))))
            elif l_val == 2:
                groups.append((f"ε({cfg}_t2g)", [idx, idx + 1, idx + 2]))
                groups.append((f"ε({cfg}_eg)", [idx + 3, idx + 4]))
            idx += norb_l
        return groups

    # ── 2d. Reference eigenvalues ─────────────────────────────

    def _build_reference_eigenvalues(self, nkfit):
        kpts = []
        for ik1 in range(nkfit):
            for ik2 in range(nkfit):
                for ik3 in range(nkfit):
                    kfrac = np.array([ik1 / nkfit, ik2 / nkfit, ik3 / nkfit])
                    kpts.append(kfrac @ self.b_vecs)
        self.kpts = np.array(kpts)
        self.Nk = len(kpts)

        phases_all = np.exp(2j * np.pi * (self.kpts @ self.R_arr.T))
        self.E_pao = np.zeros((self.Nk, self.nawf))
        for ik in range(self.Nk):
            Hk = np.einsum("r,rij->ij", phases_all[ik], self.HR_arr)
            self.E_pao[ik] = np.sort(np.linalg.eigvalsh(Hk).real)

        if self.verbose:
            print(
                f"  Reference: {self.Nk} k-points, {self.nawf} bands, "
                f"E ∈ [{self.E_pao.min():.3f}, {self.E_pao.max():.3f}] eV"
            )

    # ── 2e. Design tensors ────────────────────────────────────

    def _precompute_design_tensor(self, bonds):
        nbonds = len(bonds)
        M = np.zeros((nbonds, self.n_hop, self.nawf, self.nawf))
        R_bond = np.zeros((nbonds, 3))

        for ib, (R_cart, iat, jat, d_vec, d_norm, *_) in enumerate(bonds):
            lx, ly, lz = d_vec / d_norm
            R_bond[ib] = R_cart

            bi = self.atom_block_start[iat]
            bj = self.atom_block_start[jat]
            oi = self.atom_orbitals[iat]
            oj = self.atom_orbitals[jat]
            gi = self.atom_orbital_group[iat]
            gj = self.atom_orbital_group[jat]

            for ai, orb_a in enumerate(oi):
                for aj, orb_b in enumerate(oj):
                    ga_loc, gb_loc = gi[ai], gj[aj]
                    canonical = (min(ga_loc, gb_loc), max(ga_loc, gb_loc))
                    start = self.hop_pair_start[canonical]
                    active = self.hop_pair_active[canonical]
                    design = sk_design_row(orb_a, orb_b, lx, ly, lz)
                    for local_k, sk_k in enumerate(active):
                        M[ib, start + local_k, bi + ai, bj + aj] = design[sk_k]
        return M, R_bond

    def _build_design_tensors(self):
        self._M_shells = []
        self._R_shells = []
        for i, bonds in enumerate(self.shell_bonds_list):
            M, R = self._precompute_design_tensor(bonds)
            self._M_shells.append(M)
            self._R_shells.append(R)
            if self.verbose:
                print(f"  Design tensor ({self.shell_tags[i]}): {M.shape}")

    # ── 2f. On-site map ──────────────────────────────────────

    def _build_onsite_map(self):
        self._onsite_map = np.zeros((self.n_onsite, self.nawf, self.nawf))
        for iat in range(self.nat):
            sp = self.atoms_list[iat]
            bi = self.atom_block_start[iat]
            pstart = self.species_param_start[sp]
            for ig, (_, local_indices) in enumerate(self.species_onsite_groups[sp]):
                for li in local_indices:
                    self._onsite_map[pstart + ig, bi + li, bi + li] = 1.0
        self._onsite_diag = np.array(
            [np.diag(self._onsite_map[p]) for p in range(self.n_onsite)]
        )

    # ── 2g. Precompute dH/dk arrays ──────────────────────────

    def _precompute_dHk(self):
        self._phases_shells = []
        self._dHk_shells = []
        for M, R in zip(self._M_shells, self._R_shells):
            phases = np.exp(2j * np.pi * (self.kpts @ R.T))
            dHk = np.einsum("kb,bpij->kpij", phases, M)
            self._phases_shells.append(phases)
            self._dHk_shells.append(dHk)

        mem_MB = sum(d.nbytes for d in self._dHk_shells) / 1e6
        if self.verbose:
            print(f"  Precomputed dHk arrays: {mem_MB:.1f} MB")

    # ── 2g′. Regularization weights ──────────────────────────

    def _build_regularization_weights(self):
        """Build per-parameter Tikhonov weights.

        On-site parameters get weight 0 (not penalized).
        Hopping parameters get weight proportional to the shell distance
        (farther neighbors are penalized more strongly).
        """
        w = np.zeros(self.n_params)
        for s in range(self.n_shells):
            d_s = self.shell_dists[s]  # distance in alat units
            i0 = self.n_onsite + s * self.n_hop
            w[i0 : i0 + self.n_hop] = d_s
        # Normalise so the mean hopping weight is 1
        hop_w = w[self.n_onsite :]
        if hop_w.sum() > 0:
            w[self.n_onsite :] = hop_w / hop_w.mean()
        self._reg_weights = w

    # ── 2h. Forward model (eigenvalues + Jacobian) ────────────

    def _eigenvalues_and_jacobian(self, p):
        """Compute SK eigenvalues and Hellmann-Feynman Jacobian for all k-points."""
        Nk, nawf = self.Nk, self.nawf
        n_onsite, n_hop = self.n_onsite, self.n_hop

        # On-site contribution
        H_onsite = np.einsum("p,pij->ij", p[:n_onsite], self._onsite_map)

        # Build H(k) for all k-points
        Hk_all = np.broadcast_to(H_onsite, (Nk, nawf, nawf)).astype(complex).copy()

        for s in range(self.n_shells):
            i0 = n_onsite + s * n_hop
            i1 = i0 + n_hop
            HR_s = np.einsum("i,bija->bja", p[i0:i1], self._M_shells[s])
            Hk_all += np.einsum("kb,bij->kij", self._phases_shells[s], HR_s)

        # Batch eigendecomposition
        evals_all, evecs_all = np.linalg.eigh(Hk_all)
        E_sk = evals_all.real

        # Hellmann-Feynman Jacobian
        dE_dp = np.zeros((Nk, nawf, self.n_params))
        psi_sq = np.abs(evecs_all) ** 2
        dE_dp[:, :, :n_onsite] = np.einsum("kin,pi->knp", psi_sq, self._onsite_diag)

        evecs_bcast = evecs_all[:, np.newaxis, :, :]
        for s in range(self.n_shells):
            i0 = n_onsite + s * n_hop
            i1 = i0 + n_hop
            tmp = np.matmul(self._dHk_shells[s], evecs_bcast)
            dE_dp[:, :, i0:i1] = np.real(
                np.einsum("kin,kpin->knp", evecs_all.conj(), tmp)
            )

        return E_sk, dE_dp

    # ── 2i. Initial parameter guess ───────────────────────────

    def extract_onsite_from_HR0(self) -> np.ndarray:
        """Extract initial on-site energies from H(R=0) diagonal blocks.

        Uses QE orbital ordering (m = 0, +1, -1, …) for t2g/eg splitting.

        Returns
        -------
        np.ndarray
            On-site parameter vector of length ``n_onsite``.
        """
        H_R0 = self.HRs[:, :, 0, 0, 0, 0]

        species_onsites_avg = {}
        for sp in self.unique_species:
            cfg = (
                list(self.config_dict[sp])
                if self.config_dict
                else [f"g{i}" for i in range(len(self.shells_dict[sp]))]
            )
            sp_onsites = defaultdict(list)
            for iat in range(self.nat):
                if self.atoms_list[iat] != sp:
                    continue
                bi = self.atom_block_start[iat]
                ni = self.norb_per_atom[iat]
                diag = np.real(np.diag(H_R0)[bi : bi + ni])
                onsites = self._extract_onsite_block(diag, self.shells_dict[sp], cfg)
                for k, v in onsites.items():
                    sp_onsites[k].append(v)
            species_onsites_avg[sp] = {k: np.mean(v) for k, v in sp_onsites.items()}

        p0 = np.zeros(self.n_onsite)
        for sp in self.unique_species:
            pstart = self.species_param_start[sp]
            for ig, (pname, _) in enumerate(self.species_onsite_groups[sp]):
                if pname in species_onsites_avg[sp]:
                    p0[pstart + ig] = species_onsites_avg[sp][pname]
        return p0

    @staticmethod
    def _extract_onsite_block(diag_block, shell_l_list, cfg_labels):
        """Extract on-site energies from one atom's H(R=0) diagonal (QE ordering)."""
        onsites = {}
        idx = 0
        for ig, (l_val, cfg) in enumerate(zip(shell_l_list, cfg_labels)):
            norb_l = 2 * l_val + 1
            sub = diag_block[idx : idx + norb_l]
            if l_val == 0:
                onsites[f"ε({cfg})"] = sub[0]
            elif l_val == 1:
                onsites[f"ε({cfg})"] = np.mean(sub)
            elif l_val == 2:
                t2g_local = [1, 2, 4]  # dzx, dyz, dxy in QE ordering
                eg_local = [0, 3]  # dz2, dx2-y2
                onsites[f"ε({cfg}_t2g)"] = np.mean(sub[t2g_local])
                onsites[f"ε({cfg}_eg)"] = np.mean(sub[eg_local])
            idx += norb_l
        return onsites

    # ── 2j. Fitting ───────────────────────────────────────────

    def fit(
        self,
        n_trials: int = 10,
        seed: int | None = 123,
        max_nfev: int = 1000,
        ftol: float = 1e-12,
        xtol: float = 1e-12,
        gtol: float = 1e-12,
        alpha: float = 0.0,
    ) -> dict:
        """Run multi-start least-squares optimisation.

        Parameters
        ----------
        n_trials : int
            Number of random restarts.
        seed : int or None
            Random seed for reproducibility.
        max_nfev : int
            Max function evaluations per trial.
        ftol, xtol, gtol : float
            Tolerances for ``scipy.optimize.least_squares``.
        alpha : float
            Tikhonov regularization strength (default 0 = no penalty).
            Adds ``alpha * w_i * p_i`` penalty rows to the residual, where
            ``w_i`` is proportional to the neighbor-shell distance (farther
            shells are penalized more).  On-site parameters are not penalized.
            Typical values: 0.01–1.0 (start small, increase if far-neighbor
            hoppings blow up).

        Returns
        -------
        dict
            ``p_opt`` : best parameter vector.
            ``rmse`` : best RMSE (eV) (data-only, excluding penalty).
            ``max_err`` : max absolute error (eV).
            ``all_results`` : list of ``(rmse, p, OptimizeResult)`` sorted by RMSE.
            ``param_labels`` : parameter names.
        """
        Nk, nawf = self.Nk, self.nawf

        # Hopping scale for initialisation
        E_half = 0.5 * (self.E_pao.max() - self.E_pao.min())
        hop_scales = [E_half / np.sqrt(len(bonds)) for bonds in self.shell_bonds_list]
        p0_onsite = self.extract_onsite_from_HR0()

        if seed is not None:
            np.random.seed(seed)

        # Regularization
        use_reg = alpha > 0.0
        reg_w = self._reg_weights  # shape (n_params,)
        n_data = Nk * nawf

        # Functor with cached Jacobian
        last_jac = [None]

        def fun(p):
            E_sk, dE_dp = self._eigenvalues_and_jacobian(p)
            res_data = (E_sk - self.E_pao).ravel()
            J_data = dE_dp.reshape(n_data, self.n_params)
            if use_reg:
                res_reg = alpha * reg_w * p
                J_reg = np.diag(alpha * reg_w)
                last_jac[0] = np.vstack([J_data, J_reg])
                return np.concatenate([res_data, res_reg])
            else:
                last_jac[0] = J_data
                return res_data

        def jac(p):
            return last_jac[0]

        best_rmse = np.inf
        best_p = None
        best_res = None
        all_results = []

        if self.verbose:
            print(f"\n{'=' * 65}")
            print(f"Multi-start optimisation: {n_trials} trials")
            print(
                f"{'Trial':>5s}  {'Init RMSE (meV)':>15s}  "
                f"{'Final RMSE (meV)':>16s}  {'nfev':>5s}"
            )
            print("-" * 50)

        for trial in range(n_trials):
            p_init = np.zeros(self.n_params)
            p_init[: self.n_onsite] = p0_onsite
            for s, sc in enumerate(hop_scales):
                i0 = self.n_onsite + s * self.n_hop
                p_init[i0 : i0 + self.n_hop] = np.random.uniform(-sc, sc, self.n_hop)

            r_init = fun(p_init)
            rmse_init = np.sqrt(np.mean(r_init**2))

            res = least_squares(
                fun,
                p_init,
                jac=jac,
                method="lm",
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                max_nfev=max_nfev,
            )
            # RMSE on data residuals only (exclude penalty rows)
            res_data = res.fun[:n_data]
            rmse_final = np.sqrt(np.mean(res_data**2))
            all_results.append((rmse_final, res.x.copy(), res))

            tag = " *" if rmse_final < best_rmse else ""
            if self.verbose:
                print(
                    f"{trial + 1:5d}  {rmse_init * 1000:15.2f}  "
                    f"{rmse_final * 1000:16.2f}  {res.nfev:5d}{tag}"
                )

            if rmse_final < best_rmse:
                best_rmse = rmse_final
                best_p = res.x.copy()
                best_res = res

        all_results.sort(key=lambda x: x[0])

        best_data_res = best_res.fun[:n_data]

        if self.verbose:
            print(f"{'=' * 65}")
            msg = (
                f"Best RMSE = {best_rmse * 1000:.2f} meV, "
                f"max|δ| = {np.max(np.abs(best_data_res)) * 1000:.2f} meV"
            )
            if use_reg:
                msg += f"  (α = {alpha:.4g})"
            print(msg)
            print(f"\n{'Parameter':<30s}  {'Value':>10s}")
            print("-" * 43)
            for i, name in enumerate(self.param_labels):
                print(f"{name:<30s}  {best_p[i]: .5f}")

        return {
            "p_opt": best_p,
            "rmse": best_rmse,
            "max_err": np.max(np.abs(best_data_res)),
            "all_results": all_results,
            "param_labels": list(self.param_labels),
        }

    # ── 2k. Build PAOFLOW model dict ─────────────────────────

    def build_model_dict(self, p: np.ndarray) -> dict:
        """Convert a fitted parameter vector into a PAOFLOW-compatible model dict.

        Parameters
        ----------
        p : np.ndarray
            Parameter vector (length ``n_params``).

        Returns
        -------
        dict
            Model dict suitable for ``PAOFLOW.PAOFLOW(savedir=None, model=...)``.
        """

        # ── Hopping sub-dicts ──
        def _build_hop(p_hop):
            if self.config_dict:
                d = {}
                for ga, gb in self.hop_pair_list:
                    start = self.hop_pair_start[(ga, gb)]
                    active = self.hop_pair_active[(ga, gb)]
                    pair_key = f"{self.cfg_names[ga]}-{self.cfg_names[gb]}"
                    d[pair_key] = {
                        SK_PARAM_NAMES[sk_k]: float(p_hop[start + lk])
                        for lk, sk_k in enumerate(active)
                    }
                return d
            else:
                d = {}
                for ga, gb in self.hop_pair_list:
                    start = self.hop_pair_start[(ga, gb)]
                    active = self.hop_pair_active[(ga, gb)]
                    for lk, sk_k in enumerate(active):
                        d[SK_PARAM_NAMES[sk_k]] = float(p_hop[start + lk])
                return d

        hoppings = {}
        for s, tag in enumerate(self.shell_tags):
            i0 = self.n_onsite + s * self.n_hop
            hoppings[tag] = _build_hop(p[i0 : i0 + self.n_hop])

        # ── Atom dicts ──
        model_atoms = {}
        for iat in range(self.nat):
            sp = self.atoms_list[iat]
            pstart = self.species_param_start[sp]
            groups = self.species_onsite_groups[sp]
            atom_d = {"name": sp, "tau": self.tau_alat[iat].tolist()}

            if self.config_dict:
                atom_d["configuration"] = list(self.config_dict[sp])
                for ig, (pname, _) in enumerate(groups):
                    key = pname[2:-1]  # 'ε(3S)' → '3S'
                    atom_d[key] = float(p[pstart + ig])
            else:
                orb_list = []
                for shell_l in self.shells_dict[sp]:
                    orb_list.extend(SHELL_TO_ORBITALS[shell_l])
                atom_d["orbitals"] = list(self.atom_orbitals[iat])
                for ig, (_, local_indices) in enumerate(groups):
                    val = float(p[pstart + ig])
                    for li in local_indices:
                        atom_d[orb_list[li]] = val

            model_atoms[str(iat)] = atom_d

        return {
            "label": "Slater_Koster",
            "alat": float(self.alat),
            "model": {
                "a_vectors": self.a_vecs.tolist(),
                "atoms": model_atoms,
                "hoppings": hoppings,
            },
        }

    # ── 2l. Convenience: eigenvalues for given p ──────────────

    def eigenvalues(self, p: np.ndarray) -> np.ndarray:
        """Compute SK eigenvalues for parameter vector *p* on the fitting k-mesh.

        Returns
        -------
        np.ndarray
            Shape ``(Nk, nawf)`` eigenvalues in eV.
        """
        E_sk, _ = self._eigenvalues_and_jacobian(p)
        return E_sk
