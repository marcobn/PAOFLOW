# dual_params.py
#
# Site-labeled dual-parameter tight-binding model.
#
# Assigns "bulk" or "surface" labels to atoms and uses
# environment-appropriate SK parameters for each bond,
# with configurable mixing rules for interface bonds.
#
# Strategy
# --------
# Given two parameter sets (P_bulk and P_surf) and a slab geometry:
#
# 1. Label each atom as "bulk" or "surface" (coordination,
#    geometric, or manual).
# 2. For each bond (ia, ib), select the SK hopping parameters
#    based on the pair of labels:
#      bulk–bulk   → P_bulk
#      surf–surf   → P_surf
#      mixed       → element-wise mix of P_bulk and P_surf
# 3. Build H(R) using environment-dependent screening and
#    full spd Slater-Koster angular integrals.
#
# Mixing rules (for interface bonds):
#   'geometric'  : sign(a+b)·√|a·b|   (signed geometric mean, default)
#   'arithmetic' : (a + b) / 2
#   'bulk'       : use bulk parameters only
#
# Atom labeling methods:
#   'coordination' : Z_i = Σ f_c(d_ik), bulk if Z ≥ threshold (default)
#   'geometric'    : distance from vacuum along slab normal
#   'manual'       : explicit list of surface atom indices
#
# Usage
# -----
#   model = DualParamModel.from_files(bulk, surf, geom)
#   HRs, meta = model.build_hamiltonian()
#   pao = PAOFLOW(model=model.to_model_dict())
#   arry, attr = pao.data_controller.data_dicts()
#   arry['HRs'] = HRs
#   pao.bands(...)

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from edtb_params import (
    EDTBModel,
    read_geometry,
    read_params,
    to_model_dict,
)

ANGSTROM_AU = 1.0 / 0.52917720859  # Bohr → Å

# ═══════════════════════════════════════════════════════════════
#  Mixing utilities
# ═══════════════════════════════════════════════════════════════


def _signed_geom_mean_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise signed geometric mean for complex arrays.

    sign(Re(a) + Re(b)) · √|Re(a) · Re(b)|

    Handles zeros gracefully.  Imaginary parts are discarded
    (SK models without SOC are real-valued).
    """
    ar = np.real(a)
    br = np.real(b)
    s = np.sign(ar + br)
    result = s * np.sqrt(np.abs(ar * br))
    return result.astype(complex)


def _mix_block(
    block_bulk: np.ndarray,
    block_surf: np.ndarray,
    rule: str,
) -> np.ndarray:
    """Mix two H(R) blocks according to the specified mixing rule.

    Parameters
    ----------
    block_bulk, block_surf : ndarray
        Same-shape sub-blocks of H(R).
    rule : str
        'geometric', 'arithmetic', or 'bulk'.

    Returns
    -------
    ndarray
        Mixed block.
    """
    if rule == "bulk":
        return block_bulk.copy()
    elif rule == "arithmetic":
        return 0.5 * (block_bulk + block_surf)
    elif rule == "geometric":
        return _signed_geom_mean_array(block_bulk, block_surf)
    else:
        raise ValueError(f"Unknown mixing rule: {rule!r}")


# ═══════════════════════════════════════════════════════════════
#  Atom labeling
# ═══════════════════════════════════════════════════════════════


def label_atoms_coordination(
    geometry: dict,
    r_cut_bohr: float = 8.0,
    threshold: Optional[float] = None,
    r_taper_frac: float = 0.8,
) -> Tuple[List[str], np.ndarray]:
    """Label atoms as 'bulk' or 'surface' based on coordination number.

    The smooth coordination number is defined as:

        Z_i = Σ_{k ≠ i} f_c(d_{ik})

    where f_c is a cosine-tapered cutoff function.  Atoms with
    Z_i ≥ threshold are labeled 'bulk'; others 'surface'.

    Parameters
    ----------
    geometry : dict
        Geometry dict (alat, a_vectors, atoms).
    r_cut_bohr : float
        Cutoff radius in Bohr for the smooth cutoff.
    threshold : float, optional
        Coordination threshold. If None, uses (max + min) / 2.
    r_taper_frac : float
        Fraction of r_cut at which tapering begins (default 0.8).

    Returns
    -------
    labels : list of str
        'bulk' or 'surface' for each atom.
    coord : ndarray
        Smooth coordination numbers.
    """
    alat = geometry["alat"]
    a_vecs = np.array(geometry["a_vectors"])
    atoms = geometry["atoms"]
    natoms = len(atoms)
    tau = np.array([atom["tau"] for atom in atoms])

    # Convert r_cut from Bohr to alat units
    r_cut = r_cut_bohr / alat
    r_taper = r_taper_frac * r_cut

    def f_cutoff(r):
        if r <= r_taper:
            return 1.0
        if r >= r_cut:
            return 0.0
        return 0.5 * (1.0 + np.cos(np.pi * (r - r_taper) / (r_cut - r_taper)))

    # Build supercell positions for neighbor search
    cell_range = 2
    sctau_list = []
    for ia in range(natoms):
        for i in range(-cell_range, cell_range + 1):
            for j in range(-cell_range, cell_range + 1):
                for k in range(-cell_range, cell_range + 1):
                    pos = tau[ia] + i * a_vecs[0] + j * a_vecs[1] + k * a_vecs[2]
                    sctau_list.append(pos)
    sctau = np.array(sctau_list)
    n_sc = len(sctau)

    # Compute coordination for each atom
    coord = np.zeros(natoms)
    for ia in range(natoms):
        for n in range(n_sc):
            d = np.sqrt(np.sum((tau[ia] - sctau[n]) ** 2))
            if d > 1e-10:
                coord[ia] += f_cutoff(d)

    # Determine threshold
    if threshold is None:
        threshold = 0.5 * (np.max(coord) + np.min(coord))

    labels = ["bulk" if z >= threshold else "surface" for z in coord]
    return labels, coord


def label_atoms_geometric(
    geometry: dict,
    n_surface_layers: int = 2,
    surface_normal: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray]:
    """Label atoms based on distance from the vacuum surface.

    For slab geometries, atoms near the top/bottom surfaces are
    labeled 'surface'.  The surface normal is identified from the
    longest lattice vector (slab direction).

    Parameters
    ----------
    geometry : dict
        Geometry dict.
    n_surface_layers : int
        Number of atomic layers from each surface to label 'surface'.
    surface_normal : ndarray, optional
        Explicit surface normal. If None, uses the longest a_vector.

    Returns
    -------
    labels : list of str
    projections : ndarray
        Signed distances along the surface normal.
    """
    a_vecs = np.array(geometry["a_vectors"])
    atoms = geometry["atoms"]
    natoms = len(atoms)
    tau = np.array([atom["tau"] for atom in atoms])

    # Find slab normal: longest lattice vector
    if surface_normal is None:
        norms = np.linalg.norm(a_vecs, axis=1)
        slab_idx = np.argmax(norms)
        surface_normal = a_vecs[slab_idx] / norms[slab_idx]

    # Project positions onto the surface normal
    projections = tau @ surface_normal

    # Find unique layers by clustering projections
    sorted_proj = np.sort(projections)
    # Group atoms into layers (tolerance: 0.1 alat units)
    layers = []
    current_layer = [sorted_proj[0]]
    for p in sorted_proj[1:]:
        if p - current_layer[-1] < 0.1:
            current_layer.append(p)
        else:
            layers.append(np.mean(current_layer))
            current_layer = [p]
    layers.append(np.mean(current_layer))
    n_layers = len(layers)

    # Assign each atom to a layer
    atom_layer = np.zeros(natoms, dtype=int)
    for ia in range(natoms):
        dists = [abs(projections[ia] - lz) for lz in layers]
        atom_layer[ia] = np.argmin(dists)

    # Label: bottom n_surface_layers and top n_surface_layers are 'surface'
    labels = []
    for ia in range(natoms):
        lay = atom_layer[ia]
        if lay < n_surface_layers or lay >= n_layers - n_surface_layers:
            labels.append("surface")
        else:
            labels.append("bulk")

    return labels, projections


def label_atoms_manual(
    n_atoms: int,
    surface_indices: List[int],
) -> List[str]:
    """Label atoms from an explicit list of surface atom indices.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms.
    surface_indices : list of int
        Indices of atoms to label as 'surface'.

    Returns
    -------
    labels : list of str
    """
    labels = ["bulk"] * n_atoms
    for i in surface_indices:
        if i < 0 or i >= n_atoms:
            raise IndexError(f"Surface index {i} out of range [0, {n_atoms})")
        labels[i] = "surface"
    return labels


# ═══════════════════════════════════════════════════════════════
#  Standalone vectorized H(R) builder
# ═══════════════════════════════════════════════════════════════

# -- SK angular-integral tables ---------------------------------

_sq3 = np.sqrt(3.0)
_hsq3 = _sq3 / 2.0
_p_index_map = {"px": 0, "py": 1, "pz": 2}
_d_orbitals = {"dxy", "dyz", "dzx", "dx2-y2", "dz2"}


def _sd_value(d_orb, lx, ly, lz, sh):
    l2, m2, n2 = lx * lx, ly * ly, lz * lz
    if d_orb == "dxy":
        return _sq3 * lx * ly * sh["sds"]
    if d_orb == "dyz":
        return _sq3 * ly * lz * sh["sds"]
    if d_orb == "dzx":
        return _sq3 * lz * lx * sh["sds"]
    if d_orb == "dx2-y2":
        return _hsq3 * (l2 - m2) * sh["sds"]
    if d_orb == "dz2":
        return (n2 - 0.5 * (l2 + m2)) * sh["sds"]
    return 0.0


def _pd_value(p_orb, d_orb, lx, ly, lz, sh):
    l2, m2, n2 = lx * lx, ly * ly, lz * lz
    pds, pdp = sh["pds"], sh["pdp"]
    if p_orb == "px":
        if d_orb == "dxy":
            return _sq3 * l2 * ly * pds + ly * (1.0 - 2.0 * l2) * pdp
        if d_orb == "dyz":
            return _sq3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
        if d_orb == "dzx":
            return _sq3 * l2 * lz * pds + lz * (1.0 - 2.0 * l2) * pdp
        if d_orb == "dx2-y2":
            return _hsq3 * lx * (l2 - m2) * pds + lx * (1.0 - l2 + m2) * pdp
        if d_orb == "dz2":
            return lx * (n2 - 0.5 * (l2 + m2)) * pds - _sq3 * lx * n2 * pdp
    elif p_orb == "py":
        if d_orb == "dxy":
            return _sq3 * m2 * lx * pds + lx * (1.0 - 2.0 * m2) * pdp
        if d_orb == "dyz":
            return _sq3 * m2 * lz * pds + lz * (1.0 - 2.0 * m2) * pdp
        if d_orb == "dzx":
            return _sq3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
        if d_orb == "dx2-y2":
            return _hsq3 * ly * (l2 - m2) * pds - ly * (1.0 + l2 - m2) * pdp
        if d_orb == "dz2":
            return ly * (n2 - 0.5 * (l2 + m2)) * pds - _sq3 * ly * n2 * pdp
    elif p_orb == "pz":
        if d_orb == "dxy":
            return _sq3 * lx * ly * lz * pds - 2.0 * lx * ly * lz * pdp
        if d_orb == "dyz":
            return _sq3 * n2 * ly * pds + ly * (1.0 - 2.0 * n2) * pdp
        if d_orb == "dzx":
            return _sq3 * n2 * lx * pds + lx * (1.0 - 2.0 * n2) * pdp
        if d_orb == "dx2-y2":
            return _hsq3 * lz * (l2 - m2) * pds - lz * (l2 - m2) * pdp
        if d_orb == "dz2":
            return lz * (n2 - 0.5 * (l2 + m2)) * pds + _sq3 * lz * (l2 + m2) * pdp
    return 0.0


def _dd_value(da, db, lx, ly, lz, sh):
    l2, m2, n2 = lx * lx, ly * ly, lz * lz
    lm, ln, mn = lx * ly, lx * lz, ly * lz
    dds, ddp, ddd = sh["dds"], sh["ddp"], sh["ddd"]
    diff_lm = l2 - m2

    # Diagonal
    if da == db:
        if da == "dxy":
            return (
                3 * l2 * m2 * dds + (l2 + m2 - 4 * l2 * m2) * ddp + (n2 + l2 * m2) * ddd
            )
        if da == "dyz":
            return (
                3 * m2 * n2 * dds + (m2 + n2 - 4 * m2 * n2) * ddp + (l2 + m2 * n2) * ddd
            )
        if da == "dzx":
            return (
                3 * l2 * n2 * dds + (l2 + n2 - 4 * l2 * n2) * ddp + (m2 + l2 * n2) * ddd
            )
        if da == "dx2-y2":
            return (
                0.75 * diff_lm**2 * dds
                + (l2 + m2 - diff_lm**2) * ddp
                + (n2 + 0.25 * diff_lm**2) * ddd
            )
        if da == "dz2":
            t = n2 - 0.5 * (l2 + m2)
            return t**2 * dds + 3 * n2 * (l2 + m2) * ddp + 0.75 * (l2 + m2) ** 2 * ddd

    pair = tuple(sorted([da, db]))
    # Off-diagonal (sorted pair)
    if pair == ("dxy", "dyz"):
        return 3 * lx * m2 * lz * dds + ln * (1 - 4 * m2) * ddp + ln * (m2 - 1) * ddd
    if pair == ("dxy", "dzx"):
        return 3 * l2 * ly * lz * dds + mn * (1 - 4 * l2) * ddp + mn * (l2 - 1) * ddd
    if pair == ("dyz", "dzx"):
        return 3 * ly * n2 * lx * dds + lm * (1 - 4 * n2) * ddp + lm * (n2 - 1) * ddd
    if pair == ("dxy", "dx2-y2"):
        return (
            1.5 * lm * diff_lm * dds
            + 2 * lm * (m2 - l2) * ddp
            + 0.5 * lm * diff_lm * ddd
        )
    if pair == ("dx2-y2", "dyz"):
        return (
            1.5 * mn * diff_lm * dds
            - mn * (1 + 2 * diff_lm) * ddp
            + mn * (1 + 0.5 * diff_lm) * ddd
        )
    if pair == ("dx2-y2", "dzx"):
        return (
            1.5 * ln * diff_lm * dds
            + ln * (1 - 2 * diff_lm) * ddp
            - ln * (1 - 0.5 * diff_lm) * ddd
        )
    if pair == ("dxy", "dz2"):
        t = n2 - 0.5 * (l2 + m2)
        return _sq3 * (lm * t * dds - 2 * lm * n2 * ddp + 0.5 * lm * (1 + n2) * ddd)
    if pair == ("dyz", "dz2"):
        t = n2 - 0.5 * (l2 + m2)
        return _sq3 * (
            mn * t * dds + mn * (l2 + m2 - n2) * ddp - 0.5 * mn * (l2 + m2) * ddd
        )
    if pair == ("dz2", "dzx"):
        t = n2 - 0.5 * (l2 + m2)
        return _sq3 * (
            ln * t * dds + ln * (l2 + m2 - n2) * ddp - 0.5 * ln * (l2 + m2) * ddd
        )
    if pair == ("dx2-y2", "dz2"):
        t = n2 - 0.5 * (l2 + m2)
        return _sq3 * (
            0.5 * diff_lm * t * dds
            + n2 * (m2 - l2) * ddp
            + 0.25 * (1 + n2) * diff_lm * ddd
        )
    return 0.0


def _sk_value(orb_a, orb_b, lx, ly, lz, sh):
    """Compute a single SK matrix element for orbital pair."""
    # s-s
    if orb_a == "s" and orb_b == "s":
        return sh["sss"]
    # s-p
    if orb_a == "s" and orb_b in _p_index_map:
        return (lx, ly, lz)[_p_index_map[orb_b]] * sh["sps"]
    if orb_b == "s" and orb_a in _p_index_map:
        return -(lx, ly, lz)[_p_index_map[orb_a]] * sh["sps"]
    # p-p
    if orb_a in _p_index_map and orb_b in _p_index_map:
        if orb_a == orb_b:
            ll = (lx, ly, lz)[_p_index_map[orb_a]]
            return ll**2 * sh["pps"] + (1.0 - ll**2) * sh["ppp"]
        ia = _p_index_map[orb_a]
        ib = _p_index_map[orb_b]
        return (lx, ly, lz)[ia] * (lx, ly, lz)[ib] * (sh["pps"] - sh["ppp"])
    # s-d
    if orb_a == "s" and orb_b in _d_orbitals:
        return _sd_value(orb_b, lx, ly, lz, sh)
    if orb_b == "s" and orb_a in _d_orbitals:
        return _sd_value(orb_a, lx, ly, lz, sh)
    # p-d
    if orb_a in _p_index_map and orb_b in _d_orbitals:
        return _pd_value(orb_a, orb_b, lx, ly, lz, sh)
    if orb_b in _p_index_map and orb_a in _d_orbitals:
        v = _pd_value(orb_b, orb_a, lx, ly, lz, sh)
        return -v if v != 0.0 else 0.0
    # d-d
    if orb_a in _d_orbitals and orb_b in _d_orbitals:
        return _dd_value(orb_a, orb_b, lx, ly, lz, sh)
    return 0.0


# -- Vectorized cutoff function ---------------------------------


def _f_cutoff_vec(r, r_taper, r_cut):
    """Vectorized smooth cutoff function."""
    result = np.zeros_like(r)
    mask_inner = (r > 1e-10) & (r <= r_taper)
    mask_taper = (r > r_taper) & (r < r_cut)
    result[mask_inner] = 1.0
    result[mask_taper] = 0.5 * (
        1.0 + np.cos(np.pi * (r[mask_taper] - r_taper) / (r_cut - r_taper))
    )
    return result


# -- SK parameter extraction helpers ----------------------------

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
_sk_param_names = ["sss", "sps", "pps", "ppp", "sds", "pds", "pdp", "dds", "ddp", "ddd"]

_onsite_group = {
    "s": "s",
    "px": "p",
    "py": "p",
    "pz": "p",
    "dxy": "t2g",
    "dyz": "t2g",
    "dzx": "t2g",
    "dx2-y2": "eg",
    "dz2": "eg",
}


def _get_gamma_map(gamma_spec):
    """Build per-SK-channel gamma map from the gamma specification."""
    if isinstance(gamma_spec, (int, float)):
        return {k: float(gamma_spec) for k in _sk_param_names}
    gmap = {}
    for k in _sk_param_names:
        if k in gamma_spec:
            gmap[k] = gamma_spec[k]
        else:
            lp = _sk_to_lpair.get(k)
            gmap[k] = gamma_spec.get(lp, 0.0) if lp else 0.0
    return gmap


def _screened_hoppings(shell_params, S_ij, gamma_map):
    """Apply screening modulation to shell hopping parameters."""
    return {
        k: v * np.exp(-gamma_map.get(k, 0.0) * S_ij) for k, v in shell_params.items()
    }


def _mix_sk_params(params_bulk, params_surf, rule):
    """Mix two sets of SK hopping parameters."""
    if rule == "bulk":
        return dict(params_bulk)
    elif rule == "arithmetic":
        return {
            k: 0.5 * (params_bulk[k] + params_surf.get(k, 0.0)) for k in params_bulk
        }
    elif rule == "geometric":
        mixed = {}
        for k in params_bulk:
            a, b = params_bulk[k], params_surf.get(k, 0.0)
            mixed[k] = np.sign(a + b) * np.sqrt(abs(a * b))
        return mixed
    else:
        raise ValueError(f"Unknown mixing rule: {rule!r}")


def _mix_gamma_maps(gmap_bulk, gmap_surf, rule):
    """Mix two gamma maps."""
    if rule == "bulk":
        return dict(gmap_bulk)
    elif rule == "arithmetic":
        return {k: 0.5 * (gmap_bulk[k] + gmap_surf.get(k, 0.0)) for k in gmap_bulk}
    elif rule == "geometric":
        mixed = {}
        for k in gmap_bulk:
            a, b = gmap_bulk[k], gmap_surf.get(k, 0.0)
            mixed[k] = np.sign(a + b) * np.sqrt(abs(a * b))
        return mixed
    else:
        raise ValueError(f"Unknown mixing rule: {rule!r}")


# -- Standalone dual-parameter H(R) builder ---------------------


def _build_dual_HRs(
    params_bulk: dict,
    params_surf: dict,
    geometry: dict,
    labels: List[str],
    mixing: str = "geometric",
) -> Tuple[np.ndarray, dict]:
    """Build dual-parameter H(R) with vectorized screening.

    This standalone builder mirrors PAOFLOW's SK_EDTB but:
    - Computes screening sums once (geometry-dependent only)
    - Selects parameters per-bond based on atom labels
    - Uses numpy vectorization for distance/screening computations

    Parameters
    ----------
    params_bulk, params_surf : dict
        EDTB parameter dicts (new format).
    geometry : dict
        Geometry dict.
    labels : list of str
        'bulk' or 'surface' for each atom.
    mixing : str
        Mixing rule for interface bonds.

    Returns
    -------
    HRs : ndarray, shape (nawf, nawf, nk1, nk2, nk3, 1)
    metadata : dict
        Keys: nawf, natoms, tau, a_vectors, b_vectors, norbitals,
        atom_block_start, nk1, nk2, nk3, cutoffs, atoms, shells.
    """
    alat = geometry["alat"]
    a_vecs = np.array(geometry["a_vectors"])
    atoms_list = geometry["atoms"]
    natoms = len(atoms_list)

    # Atom positions (Cartesian / alat units)
    tau = np.array([a["tau"] for a in atoms_list])

    # Orbital info from bulk params (both must have same basis)
    basis = params_bulk["basis"]
    orbitals_per_atom = []
    norbitals = np.zeros(natoms, dtype=int)
    for ia, atom in enumerate(atoms_list):
        sp = atom["species"]
        orbs = list(basis[sp]["orbitals"])
        orbitals_per_atom.append(orbs)
        norbitals[ia] = len(orbs)

    nawf = int(np.sum(norbitals))
    atom_block_start = np.zeros(natoms, dtype=int)
    for ia in range(1, natoms):
        atom_block_start[ia] = atom_block_start[ia - 1] + norbitals[ia - 1]

    # ── Extract hopping parameters ──
    # Get species pair key (only single-species supported here)
    species_list = sorted(basis.keys())
    if len(species_list) != 1:
        raise NotImplementedError(
            "Dual-param builder currently supports single-species systems."
        )
    sp_key = f"{species_list[0]}-{species_list[0]}"

    hop_shells_bulk = sorted(params_bulk["hoppings"][sp_key], key=lambda s: s["r_ref"])
    hop_shells_surf = sorted(params_surf["hoppings"][sp_key], key=lambda s: s["r_ref"])
    n_shells = len(hop_shells_bulk)

    # Screening
    scr_bulk = params_bulk.get("screening", {})
    scr_surf = params_surf.get("screening", {})
    r_cut_bohr = scr_bulk.get("r_cut", 8.0)
    r_cut = r_cut_bohr / alat
    r_taper = 0.8 * r_cut

    gamma_raw_bulk = scr_bulk.get("gamma", {}).get(sp_key, 0.0)
    gamma_raw_surf = scr_surf.get("gamma", {}).get(sp_key, 0.0)
    gmap_bulk = _get_gamma_map(gamma_raw_bulk)
    gmap_surf = _get_gamma_map(gamma_raw_surf)
    gmap_mixed = _mix_gamma_maps(gmap_bulk, gmap_surf, mixing)

    # On-site energies
    onsite_bulk = params_bulk["onsite"][species_list[0]]
    onsite_surf = params_surf["onsite"][species_list[0]]

    # ── Supercell ──
    cell_range = min(n_shells, 3)
    nk = 2 * cell_range + 1
    nk1 = nk2 = nk3 = nk

    # Build flat supercell positions
    sc_offsets = []
    for i in range(-cell_range, cell_range + 1):
        for j in range(-cell_range, cell_range + 1):
            for k in range(-cell_range, cell_range + 1):
                sc_offsets.append(i * a_vecs[0] + j * a_vecs[1] + k * a_vecs[2])
    sc_offsets = np.array(sc_offsets)  # (nk³, 3)

    sctau_flat = (tau[:, None, :] + sc_offsets[None, :, :]).reshape(
        -1, 3
    )  # (natoms*nk³, 3)
    n_sc = len(sctau_flat)

    # ── Determine shell cutoffs from geometry ──
    # Compute all unique distances
    dists_all = []
    for ia in range(natoms):
        d_vec = sctau_flat - tau[ia]
        d = np.sqrt(np.sum(d_vec**2, axis=1))
        dists_all.extend(d[d > 1e-10].tolist())
    unique_dist = np.unique(np.round(dists_all, decimals=8))

    dist_shells = unique_dist[: n_shells + 1]  # first n_shells+1 unique distances
    cutoffs = []
    for s in range(n_shells):
        if s + 1 < len(dist_shells):
            c = dist_shells[s] + (dist_shells[s + 1] - dist_shells[s]) / 2.0
        else:
            c = dist_shells[s] * 1.2
        cutoffs.append(c)

    # ── Precompute f_c table: f_c(d(tau[ia], sctau_flat[n])) ──
    fc_table = np.zeros((natoms, n_sc))
    for ia in range(natoms):
        d_vec = sctau_flat - tau[ia]
        d = np.sqrt(np.sum(d_vec**2, axis=1))
        fc_table[ia] = _f_cutoff_vec(d, r_taper, r_cut)

    # ── Build H(R) ──
    HRs = np.zeros((nawf, nawf, nk1, nk2, nk3, 1), dtype=complex)

    # On-site energies
    for ia in range(natoms):
        on = onsite_bulk if labels[ia] == "bulk" else onsite_surf
        bs = atom_block_start[ia]
        for no, orb in enumerate(orbitals_per_atom[ia]):
            grp = _onsite_group[orb]
            HRs[bs + no, bs + no, 0, 0, 0, 0] = on[grp]

    # Pre-mix hopping parameters for the three label combinations
    hop_mixed = []
    for s in range(n_shells):
        hop_mixed.append(
            _mix_sk_params(
                hop_shells_bulk[s]["params"], hop_shells_surf[s]["params"], mixing
            )
        )

    # Hopping loop
    n_bonds = 0
    for i in range(-cell_range, cell_range + 1):
        for j in range(-cell_range, cell_range + 1):
            for k in range(-cell_range, cell_range + 1):
                R = i * a_vecs[0] + j * a_vecs[1] + k * a_vecs[2]
                for ia in range(natoms):
                    for ib in range(natoms):
                        pos_j = tau[ib] + R
                        d_vec = pos_j - tau[ia]
                        dist_val = np.sqrt(np.sum(d_vec**2))

                        if dist_val < 1e-10:
                            continue

                        # Determine shell
                        shell_idx = -1
                        for s in range(n_shells):
                            if dist_val < cutoffs[s]:
                                shell_idx = s
                                break
                        if shell_idx < 0:
                            continue

                        # Direction cosines
                        inv_d = 1.0 / dist_val
                        lx = d_vec[0] * inv_d
                        ly = d_vec[1] * inv_d
                        lz = d_vec[2] * inv_d

                        # Screening sum S_ij (vectorized)
                        d_jk = np.sqrt(np.sum((sctau_flat - pos_j) ** 2, axis=1))
                        fc_jk = _f_cutoff_vec(d_jk, r_taper, r_cut)
                        S_ij = np.dot(fc_table[ia], fc_jk)

                        # Select parameters based on labels
                        la = labels[ia]
                        lb = labels[ib]
                        if la == "bulk" and lb == "bulk":
                            sh = _screened_hoppings(
                                hop_shells_bulk[shell_idx]["params"], S_ij, gmap_bulk
                            )
                        elif la == "surface" and lb == "surface":
                            sh = _screened_hoppings(
                                hop_shells_surf[shell_idx]["params"], S_ij, gmap_surf
                            )
                        else:
                            sh = _screened_hoppings(
                                hop_mixed[shell_idx], S_ij, gmap_mixed
                            )

                        # Fill orbital block
                        orbs_a = orbitals_per_atom[ia]
                        orbs_b = orbitals_per_atom[ib]
                        bs_a = atom_block_start[ia]
                        bs_b = atom_block_start[ib]
                        for noa, oa in enumerate(orbs_a):
                            for nob, ob in enumerate(orbs_b):
                                v = _sk_value(oa, ob, lx, ly, lz, sh)
                                if abs(v) > 1e-30:
                                    HRs[bs_a + noa, bs_b + nob, i, j, k, 0] = v
                        n_bonds += 1

    # Reciprocal lattice
    b_vecs = np.zeros((3, 3))
    vol = np.dot(np.cross(a_vecs[0], a_vecs[1]), a_vecs[2])
    b_vecs[0] = np.cross(a_vecs[1], a_vecs[2]) / vol
    b_vecs[1] = np.cross(a_vecs[2], a_vecs[0]) / vol
    b_vecs[2] = np.cross(a_vecs[0], a_vecs[1]) / vol

    # R-grid (same convention as PAOFLOW get_R_grid_fft)
    nrtot = nk1 * nk2 * nk3
    R_grid = np.zeros((nrtot, 3), dtype=float)
    R_fft = np.zeros((nk1, nk2, nk3, 3), dtype=float)
    R_idx = np.zeros((nk1, nk2, nk3), dtype=int)
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j * nk3 + i * nk2 * nk3
                Rx = float(i) / float(nk1)
                Ry = float(j) / float(nk2)
                Rz = float(k) / float(nk3)
                if Rx >= 0.5:
                    Rx -= 1.0
                if Ry >= 0.5:
                    Ry -= 1.0
                if Rz >= 0.5:
                    Rz -= 1.0
                R_grid[n, :] = (
                    Rx * nk1 * a_vecs[0, :]
                    + Ry * nk2 * a_vecs[1, :]
                    + Rz * nk3 * a_vecs[2, :]
                )
                R_fft[i, j, k, :] = R_grid[n, :]
                R_idx[i, j, k] = n

    metadata = {
        "nawf": nawf,
        "natoms": natoms,
        "tau": tau,
        "a_vectors": a_vecs,
        "b_vectors": b_vecs,
        "norbitals": norbitals,
        "orbitals_per_atom": orbitals_per_atom,
        "atom_block_start": atom_block_start,
        "nk1": nk1,
        "nk2": nk2,
        "nk3": nk3,
        "R": R_grid,  # (nrtot, 3) lattice vectors in alat units
        "Rfft": R_fft,  # (nk1, nk2, nk3, 3)
        "Ridx": R_idx,  # (nk1, nk2, nk3) → flat index
        "cutoffs": cutoffs,
        "n_bonds": n_bonds,
        "species": [a["species"] for a in atoms_list],
        "alat": alat,
        "alat_ang": alat / ANGSTROM_AU,
        "volume": vol,
    }
    print(f"  Built H(R): {nawf}x{nawf}, grid {nk1}x{nk2}x{nk3}, {n_bonds} bonds")
    return HRs, metadata


# ═══════════════════════════════════════════════════════════════
#  DualParamModel — user-facing class
# ═══════════════════════════════════════════════════════════════


class DualParamModel:
    """Dual-parameter tight-binding model with site-labeled atoms.

    Uses two independently fitted parameter sets (P_bulk and P_surf)
    with atoms labeled as "bulk" or "surface".  Interface bonds are
    handled with configurable mixing rules.

    Construction
    ------------
    From files::

        model = DualParamModel.from_files(
            params_bulk="Si_SK_params.json",
            params_surf="Si_surface_EDTB_params.json",
            geometry="Surface_Si/Si_111_slab_geom.json",
        )

    From EDTBModel objects::

        model = DualParamModel(
            params_bulk=bulk_params_dict,
            params_surf=surf_params_dict,
            geometry=geometry_dict,
        )

    Labeling is applied automatically on construction (default:
    coordination-based).  Override with ``labeling='geometric'``,
    ``labeling='manual'``, or pass explicit ``labels=[...]``.

    Hamiltonian
    -----------
    Build H(R) and get the PAOFLOW model dict::

        HRs, meta = model.build_hamiltonian()
        model_dict = model.to_model_dict()

    Then feed into PAOFLOW and replace HRs::

        pao = PAOFLOW(model=model_dict)
        arry, attr = pao.data_controller.data_dicts()
        arry['HRs'] = HRs
        pao.bands(ibrav=0, band_path='G-M-K-G', nk=500)
    """

    def __init__(
        self,
        params_bulk: dict,
        params_surf: dict,
        geometry: dict,
        *,
        labels: Optional[List[str]] = None,
        mixing: str = "geometric",
        labeling: str = "coordination",
        label_kwargs: Optional[dict] = None,
    ):
        """Create a dual-parameter model.

        Parameters
        ----------
        params_bulk : dict
            EDTB parameter dict for bulk-like atoms.
        params_surf : dict
            EDTB parameter dict for surface atoms.
        geometry : dict
            Geometry dict (alat, a_vectors, atoms).
        labels : list of str, optional
            Explicit 'bulk'/'surface' labels per atom. If None,
            computed from ``labeling`` method.
        mixing : str
            Mixing rule: 'geometric' (default), 'arithmetic', or 'bulk'.
        labeling : str
            Labeling method if ``labels`` is None:
            'coordination' (default), 'geometric', or 'manual'.
        label_kwargs : dict, optional
            Extra kwargs passed to the labeling function.
        """
        self._params_bulk = copy.deepcopy(params_bulk)
        self._params_surf = copy.deepcopy(params_surf)
        self._geometry = copy.deepcopy(geometry)
        self._mixing = mixing

        if label_kwargs is None:
            label_kwargs = {}

        natoms = len(geometry["atoms"])

        # Compute labels if not provided
        if labels is not None:
            if len(labels) != natoms:
                raise ValueError(
                    f"labels has {len(labels)} entries, but geometry has {natoms} atoms."
                )
            self._labels = list(labels)
            self._coord = None
        elif labeling == "coordination":
            self._labels, self._coord = label_atoms_coordination(
                geometry, **label_kwargs
            )
        elif labeling == "geometric":
            self._labels, self._coord = label_atoms_geometric(geometry, **label_kwargs)
        elif labeling == "manual":
            if "surface_indices" not in label_kwargs:
                raise ValueError(
                    "labeling='manual' requires surface_indices in label_kwargs."
                )
            self._labels = label_atoms_manual(natoms, label_kwargs["surface_indices"])
            self._coord = None
        else:
            raise ValueError(f"Unknown labeling method: {labeling!r}")

        # Count
        n_bulk = self._labels.count("bulk")
        n_surf = self._labels.count("surface")
        print(
            f"DualParamModel: {natoms} atoms, "
            f"{n_bulk} bulk + {n_surf} surface, "
            f"mixing={mixing!r}"
        )

    # ── Class-method constructors ─────────────────────────────

    @classmethod
    def from_files(
        cls,
        params_bulk: Union[str, Path],
        params_surf: Union[str, Path],
        geometry: Union[str, Path],
        **kwargs,
    ) -> "DualParamModel":
        """Load from JSON files.

        Parameters
        ----------
        params_bulk : str or Path
            Path to bulk parameter file.
        params_surf : str or Path
            Path to surface parameter file.
        geometry : str or Path
            Path to geometry file.
        **kwargs
            Passed to constructor (labels, mixing, labeling, label_kwargs).

        Returns
        -------
        DualParamModel
        """
        p_bulk = read_params(params_bulk)
        p_surf = read_params(params_surf)
        geom = read_geometry(geometry)
        return cls(p_bulk, p_surf, geom, **kwargs)

    @classmethod
    def from_edtb_models(
        cls,
        model_bulk: EDTBModel,
        model_surf: EDTBModel,
        geometry: dict,
        **kwargs,
    ) -> "DualParamModel":
        """Create from two EDTBModel objects and a new geometry.

        Parameters
        ----------
        model_bulk : EDTBModel
            Bulk-fitted model.
        model_surf : EDTBModel
            Surface-fitted model.
        geometry : dict
            Slab geometry to apply the model to.
        **kwargs
            Passed to constructor.

        Returns
        -------
        DualParamModel
        """
        return cls(model_bulk.params, model_surf.params, geometry, **kwargs)

    # ── Model dict ────────────────────────────────────────────

    def to_model_dict(self) -> dict:
        """Return a PAOFLOW-compatible model dict.

        Uses the **bulk** parameters and the slab geometry.
        The returned dict can be passed directly to
        ``PAOFLOW.PAOFLOW(model=...)``, which will build its own
        H(R) internally and set up all metadata.  You then replace
        ``arry['HRs']`` with the dual-parameter H(R) from
        :meth:`build_hamiltonian` before calling ``pao.bands()``.

        Returns
        -------
        dict
            Model dict accepted by PAOFLOW.
        """
        return to_model_dict(self._params_bulk, self._geometry)

    # ── Hamiltonian construction ───────────────────────────────

    def build_hamiltonian(self, verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """Build the real-space Hamiltonian H(R).

        Returns
        -------
        HRs : ndarray, complex, shape (nawf, nawf, nk1, nk2, nk3, 1)
            Real-space Hamiltonian on the supercell R-grid.
        meta : dict
            Metadata needed for Fourier interpolation:

            - **nawf** : int — number of Wannier functions (= total orbitals)
            - **natoms** : int
            - **tau** : (natoms, 3) — atomic positions in alat units
            - **a_vectors** : (3, 3) — lattice vectors in alat units
            - **b_vectors** : (3, 3) — reciprocal lattice (2π/alat units)
            - **norbitals** : (natoms,) int — orbitals per atom
            - **orbitals_per_atom** : list of list of str — orbital names
            - **atom_block_start** : (natoms,) int — orbital offset per atom
            - **nk1, nk2, nk3** : int — R-grid dimensions
            - **R** : (nrtot, 3) — R-vectors in alat units (FFT order)
            - **Rfft** : (nk1, nk2, nk3, 3) — same, 3-D indexed
            - **Ridx** : (nk1, nk2, nk3) int — 3-D → flat index map
            - **alat** : float — lattice parameter in Bohr
            - **alat_ang** : float — lattice parameter in Å
            - **volume** : float — unit cell volume (alat³ units)
            - **species** : list of str — species name per atom
            - **cutoffs** : list of float — shell cutoffs in alat units
            - **n_bonds** : int — number of bonds processed
        """
        import time as _time

        t0 = _time.perf_counter()
        if verbose:
            print("Building dual-parameter H(R)...")
        HRs, meta = _build_dual_HRs(
            self._params_bulk,
            self._params_surf,
            self._geometry,
            self._labels,
            self._mixing,
        )
        if verbose:
            dt = _time.perf_counter() - t0
            print(f"  H(R) construction: {dt:.1f} s")
        return HRs, meta

    # ── Access / display ──────────────────────────────────────

    @property
    def labels(self) -> List[str]:
        """Atom labels ('bulk' or 'surface')."""
        return list(self._labels)

    @property
    def coord(self) -> Optional[np.ndarray]:
        """Coordination numbers (if coordination labeling was used)."""
        return self._coord.copy() if self._coord is not None else None

    @property
    def mixing(self) -> str:
        """Mixing rule for interface bonds."""
        return self._mixing

    @property
    def geometry(self) -> dict:
        """Geometry dict."""
        return copy.deepcopy(self._geometry)

    @property
    def n_atoms(self) -> int:
        """Total number of atoms."""
        return len(self._geometry["atoms"])

    @property
    def n_bulk(self) -> int:
        """Number of bulk-labeled atoms."""
        return self._labels.count("bulk")

    @property
    def n_surface(self) -> int:
        """Number of surface-labeled atoms."""
        return self._labels.count("surface")

    def label_summary(self) -> str:
        """Return a summary of atom labels with coordination numbers."""
        lines = [
            f"DualParamModel label summary: "
            f"{self.n_atoms} atoms, {self.n_bulk} bulk + {self.n_surface} surface",
            f"Mixing rule: {self._mixing}",
            "",
            f"{'Atom':>6s}  {'Species':>8s}  {'Label':>8s}",
            "-" * 40,
        ]
        atoms = self._geometry["atoms"]
        for ia, (atom, label) in enumerate(zip(atoms, self._labels)):
            coord_str = ""
            if self._coord is not None:
                coord_str = f"  Z={self._coord[ia]:.2f}"
            lines.append(f"{ia:6d}  {atom['species']:>8s}  {label:>8s}{coord_str}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DualParamModel("
            f"n_atoms={self.n_atoms}, "
            f"n_bulk={self.n_bulk}, "
            f"n_surface={self.n_surface}, "
            f"mixing={self._mixing!r})"
        )
