import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.linalg import eigh
from PAOFLOW import GPAO
from PAOFLOW.models.band_unfold import _extract_hamiltonian, plot_unfolded, unfold_bands
from PAOFLOW.models.edtb_params import EDTBModel

pplt = GPAO.GPAO()

Ry2eV = 13.60569193

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except ImportError:
    COMM = None
    RANK = 0


def barrier():
    """Block until all MPI ranks arrive (no-op without mpi4py)."""
    if COMM is not None:
        COMM.Barrier()


# user-defined TBG unit
n_tbg = int(sys.argv[1])
m_tbg = int(sys.argv[2])
unit = str(n_tbg) + "_" + str(m_tbg)
if RANK == 0:
    print("Running calculation for TBG (" + unit + ")")
barrier()


def build_tbg_supercell(n, m, a_cc=1.42, d_inter=3.35, vacuum=15.0):
    """Build a commensurate twisted bilayer graphene supercell.

    Parameters
    ----------
    n, m : int
        Commensurate indices (n > m >= 0).
    a_cc : float
        C-C nearest-neighbor distance (Å).
    d_inter : float
        Interlayer separation (Å).
    vacuum : float
        Vacuum thickness (Å).

    Returns
    -------
    geometry : dict
        Twisted geometry dict compatible with EDTBModel.with_geometry().
    geometry_untwisted : dict
        Untwisted (theta=0, AA-stacked) geometry in the same supercell.
    info : dict
        twist_angle (deg), n_atoms, n_cells, L_moire (Å/Bohr).
    """
    Ang2Bohr = 1.0 / 0.529177249

    a = a_cc * np.sqrt(3)  # graphene lattice constant (Å)

    # Primitive lattice vectors (Å)
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3) / 2])

    # Sublattice basis (Å): tau_B is the 1NN vector from A
    tau_A = np.array([0.0, 0.0])
    tau_B = (a1 + a2) / 3.0  # |tau_B| = a/sqrt(3) = a_cc

    # Supercell lattice vectors
    T1 = n * a1 + m * a2
    T2 = -m * a1 + (n + m) * a2  # 60° rotation of T1

    # Number of primitive cells per layer
    N_cells = n**2 + n * m + m**2

    # Twist angle
    cos_theta = (n**2 + m**2 + 4 * n * m) / (2.0 * N_cells)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Moiré period
    L_moire = np.linalg.norm(T1)

    # 2D rotation matrix (Layer 2 rotated by theta so that T1 = n*a1+m*a2
    # is a lattice vector of both layers: coefficients (n,m) unrotated, (m,n) rotated)
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, st], [-st, ct]])

    # Supercell matrix (rows = T1, T2) and its inverse for fractional coords
    T_mat = np.array([T1, T2])
    T_inv = np.linalg.inv(T_mat)

    # Search range for primitive-cell indices
    smax = n + m + 2
    tol = 1e-8

    def _collect_atoms(basis_vecs, sublattice_positions):
        """Find all atoms inside the supercell parallelogram (deduplicated)."""
        a1_l, a2_l = basis_vecs
        atoms = []
        fracs_seen = set()
        for i in range(-smax, smax + 1):
            for j in range(-smax, smax + 1):
                for tau in sublattice_positions:
                    pos = tau + i * a1_l + j * a2_l
                    # Fractional coords: pos = f @ T_mat  =>  f = pos @ T_inv
                    frac = pos @ T_inv
                    frac -= np.floor(frac + tol)
                    if np.all(frac >= -tol) and np.all(frac < 1.0 - tol):
                        key = (round(frac[0], 6), round(frac[1], 6))
                        if key not in fracs_seen:
                            fracs_seen.add(key)
                            # Store wrapped Cartesian position (inside parallelogram)
                            atoms.append((frac @ T_mat).copy())
        return atoms

    # Layer 1: unrotated
    atoms_L1 = _collect_atoms((a1, a2), [tau_A, tau_B])

    # Layer 2 (twisted): rotated by theta
    a1_rot, a2_rot = R @ a1, R @ a2
    tau_A_rot, tau_B_rot = R @ tau_A, R @ tau_B
    atoms_L2 = _collect_atoms((a1_rot, a2_rot), [tau_A_rot, tau_B_rot])

    # Layer 2 (untwisted): same basis as layer 1 -> AA stacking in the same cell
    atoms_L2_untw = atoms_L1

    n_per_layer = 2 * N_cells
    assert len(atoms_L1) == n_per_layer, (
        f"Layer 1: expected {n_per_layer}, got {len(atoms_L1)}"
    )
    assert len(atoms_L2) == n_per_layer, (
        f"Layer 2: expected {n_per_layer}, got {len(atoms_L2)}"
    )

    # Build 3D positions (Bohr)
    d_bohr = d_inter * Ang2Bohr
    vac_bohr = vacuum * Ang2Bohr
    c_bohr = d_bohr + vac_bohr

    # Use alat = graphene primitive constant (Bohr) so that ALL (n,m) cells
    # share a common length unit. This makes a supercell's a_vectors an exact
    # integer multiple of the primitive-cell a_vectors, as required by the
    # band-unfolding routine (which compares a_vectors directly, same alat).
    alat = a * Ang2Bohr

    # 3D supercell vectors (in units of alat)
    a_vectors = (
        np.array(
            [
                [T1[0], T1[1], 0.0],
                [T2[0], T2[1], 0.0],
                [0.0, 0.0, c_bohr / Ang2Bohr],
            ]
        )
        * Ang2Bohr
        / alat
    )

    def _make_geometry(layer2_atoms):
        positions = []
        for p2d in atoms_L1:
            positions.append(np.array([p2d[0], p2d[1], 0.0]) * Ang2Bohr)
        for p2d in layer2_atoms:
            positions.append(np.array([p2d[0], p2d[1], d_inter]) * Ang2Bohr)
        atoms = [
            {"species": "C", "tau": (pos / alat).tolist()}
            for pos in np.array(positions)
        ]
        return {
            "alat": float(alat),
            "a_vectors": a_vectors.tolist(),
            "atoms": atoms,
        }

    geometry = _make_geometry(atoms_L2)
    geometry_untwisted = _make_geometry(atoms_L2_untw)

    info = {
        "n": n,
        "m": m,
        "theta_deg": float(np.degrees(theta)),
        "n_atoms": len(geometry["atoms"]),
        "n_cells_per_layer": N_cells,
        "L_moire_ang": float(L_moire),
        "L_moire_bohr": float(L_moire * Ang2Bohr),
        "d_inter_ang": d_inter,
    }

    return geometry, geometry_untwisted, info


# ── Print table of commensurate structures ──
if RANK == 0:
    print(f"{'(n, m)':>8s}  {'θ (°)':>8s}  {'N_at':>6s}  {'L_moiré (Å)':>12s}")
    print("─" * 42)
    for n, m in [
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 5),
        (10, 9),
        (20, 19),
        (31, 30),
    ]:
        _, _, info = build_tbg_supercell(n, m)
        print(
            f"({n:2d},{m:2d})  {info['theta_deg']:8.2f}  {info['n_atoms']:6d}  {info['L_moire_ang']:12.2f}"
        )

# # Twisted bilayer

d_scale_all = 0.8  # scale factor for interlayer distance (1.0 = unchanged)

# 2D hexagonal high-symmetry path (in-plane only)
# NOTE: with a2 = (1/2, √3/2)·a (the convention used by build_tbg_supercell),
# the Dirac corner is K = (2/3, 1/3), NOT (1/3, 1/3). Using (1/3,1/3) samples an
# interior point and spuriously opens a large "gap" at the (mis)labelled K.
hex2d_path = "G-M-K-G"
hex2d_pts = {"G": [0.0, 0.0, 0.0], "M": [0.5, 0.0, 0.0], "K": [2.0 / 3, 1.0 / 3, 0.0]}

if RANK == 0:
    # ── Build (1,0) primitive cell ──

    n_tbg_0, m_tbg_0 = 1, 0
    geom_tbg, geom_tbg_untw, info_tbg = build_tbg_supercell(
        n_tbg_0, m_tbg_0, d_inter=3.35 * d_scale_all
    )

    geom_tbg_untw_file = f"./geom_tbg_{n_tbg_0}_{m_tbg_0}_untwisted.json"
    with open(geom_tbg_untw_file, "w") as f:
        json.dump(geom_tbg_untw, f, indent=2)
    print(f"Saved geom_tbg_untw to {geom_tbg_untw_file}")

    print(
        f"PC reference ({n_tbg_0},{m_tbg_0}): θ = {info_tbg['theta_deg']:.2f}°, "
        f"N_at = {info_tbg['n_atoms']}, L_moiré = {info_tbg['L_moire_ang']:.2f} Å"
    )

    # ── Save the TBG geometries to file ──
    geom_tbg_file = f"./geom_tbg_{n_tbg_0}_{m_tbg_0}.json"
    with open(geom_tbg_file, "w") as f:
        json.dump(geom_tbg, f, indent=2)
    print(f"Saved geom_tbg to {geom_tbg_file}")

    # ── Build and visualize a TBG supercell ──

    geom_tbg, geom_tbg_untw, info_tbg = build_tbg_supercell(
        n_tbg, m_tbg, d_inter=3.35 * d_scale_all
    )

    print(
        f"TBG ({n_tbg},{m_tbg}): θ = {info_tbg['theta_deg']:.2f}°, "
        f"N_at = {info_tbg['n_atoms']}, L_moiré = {info_tbg['L_moire_ang']:.2f} Å"
    )

    # ── Plot atom positions (top-down view) - twisted bilayer ──
    alat = geom_tbg["alat"]
    tau_all = np.array([a["tau"] for a in geom_tbg["atoms"]]) * alat  # Bohr
    Bohr2Ang = 0.529177249
    xy_ang = tau_all[:, :2] * Bohr2Ang
    z_ang = tau_all[:, 2] * Bohr2Ang
    n_half = info_tbg["n_atoms"] // 2

    # if RANK == 0:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xy_ang[:n_half, 0], xy_ang[:n_half, 1], s=8, c="C0", label="Layer 1")
    ax.scatter(xy_ang[n_half:, 0], xy_ang[n_half:, 1], s=8, c="C3", label="Layer 2")

    # Draw supercell outline
    a_vecs = np.array(geom_tbg["a_vectors"]) * alat * Bohr2Ang
    origin = np.array([0, 0])
    corners = np.array(
        [origin, a_vecs[0, :2], a_vecs[0, :2] + a_vecs[1, :2], a_vecs[1, :2], origin]
    )
    ax.plot(corners[:, 0], corners[:, 1], "k-", lw=1.5)

    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(
        f"TBG ({n_tbg},{m_tbg}): θ = {info_tbg['theta_deg']:.2f}°, {info_tbg['n_atoms']} atoms"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── Plot atom positions (top-down view) untwisted AA stacking ──
    alat = geom_tbg_untw["alat"]
    tau_all = np.array([a["tau"] for a in geom_tbg_untw["atoms"]]) * alat  # Bohr
    Bohr2Ang = 0.529177249
    xy_ang = tau_all[:, :2] * Bohr2Ang
    z_ang = tau_all[:, 2] * Bohr2Ang
    n_half = info_tbg["n_atoms"] // 2

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xy_ang[:n_half, 0], xy_ang[:n_half, 1], s=8, c="C0", label="Layer 1")
    ax.scatter(xy_ang[n_half:, 0], xy_ang[n_half:, 1], s=8, c="C3", label="Layer 2")

    # Draw supercell outline
    a_vecs = np.array(geom_tbg_untw["a_vectors"]) * alat * Bohr2Ang
    origin = np.array([0, 0])
    corners = np.array(
        [origin, a_vecs[0, :2], a_vecs[0, :2] + a_vecs[1, :2], a_vecs[1, :2], origin]
    )
    ax.plot(corners[:, 0], corners[:, 1], "k-", lw=1.5)

    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(f"AA ({n_tbg},{m_tbg}): θ = {0:.2f}°, {info_tbg['n_atoms']} atoms")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── Save the TBG geometries to file ──
    geom_tbg_file = f"./geom_tbg_{n_tbg}_{m_tbg}.json"
    with open(geom_tbg_file, "w") as f:
        json.dump(geom_tbg, f, indent=2)
    print(f"Saved geom_tbg to {geom_tbg_file}")

    geom_tbg_untw_file = f"./geom_tbg_{n_tbg}_{m_tbg}_untwisted.json"
    with open(geom_tbg_untw_file, "w") as f:
        json.dump(geom_tbg_untw, f, indent=2)
    print(f"Saved geom_tbg_untw to {geom_tbg_untw_file}")

# All ranks wait until rank 0 has written the geometry JSON files.
barrier()

# ═══════════════════════════════════════════════════════════
# Band unfolding
# ═══════════════════════════════════════════════════════════

# Load PC and SC models
model_pc_uf = EDTBModel.from_files(
    "./C_EDTB_DD_params.json", "./geom_tbg_1_0_untwisted.json"
)
model_sc_uf = EDTBModel.from_files(
    "./C_EDTB_DD_params.json", "./geom_tbg_" + unit + "_untwisted.json"
)

md_pc = model_pc_uf.to_model_dict()
md_sc = model_sc_uf.to_model_dict()

# Run unfolding (fully automatic: finds M, translations, atom mapping)

result = unfold_bands(
    md_pc,
    md_sc,
    sym_points=hex2d_pts,
    path_str=hex2d_path,
    nk_per_seg=80,
    verbose=True,
)
if RANK == 0:
    plot_unfolded(result, w_thresh=0.5)

# ── Calculate bands for twisted bilayer ──
model_tbg_unit = EDTBModel.from_files(
    "./C_EDTB_DD_params.json", "./geom_tbg_" + unit + ".json"
)

bands_tbg_unit = model_tbg_unit.compute_bands(
    ibrav=4,
    nk=500,
    band_path=hex2d_path,
    high_sym_points=hex2d_pts,
    outputdir="Carbon_tbg_" + unit,
)

# Twisted eigenvalues on the unfolding k-path. _extract_hamiltonian instantiates
# PAOFLOW, so it is collective and must run on every rank.
HRs_tw, R_tw, nawf_tw, _, _ = _extract_hamiltonian(
    model_tbg_unit.to_model_dict(), verbose=False
)
E_tw = np.empty((len(result.kpath_cart), nawf_tw))
for ik, k in enumerate(result.kpath_cart):
    Hk = HRs_tw[:, :, :, 0] @ np.exp(2j * np.pi * (R_tw @ k))
    E_tw[ik] = eigh(0.5 * (Hk + Hk.conj().T))[0]

barrier()

if RANK == 0:
    # ═══════════════════════════════════════════════════════════
    # Two-panel figure:
    #   (left)  (3,2) bands folded in the (3,2) supercell BZ
    #   (right) same bands unfolded onto the graphene (1,0) BZ,
    #           coloured by spectral weight
    # 3-colour scheme: deep blue → vermillion → amber
    # ═══════════════════════════════════════════════════════════

    BLUE, RED, AMBER, GREEN = "#004488", "#D55E00", "#FFC20A", "#009E73"
    w_cmap = LinearSegmentedColormap.from_list("BRA", [BLUE, RED, AMBER])

    E_F = -0.517  # Dirac reference (eV) -> 0
    y_lim = (-4.0, 4.0)

    # ── left panel: folded (3,2) bands ──
    fold = np.loadtxt("Carbon_tbg_" + unit + "/bands_0.dat")
    kf, Ef = fold[:, 0], fold[:, 1:] - E_F
    labs, cnts = [], []
    for line in open("Carbon_tbg_" + unit + "/kpath_points.txt"):
        p = line.split()
        if len(p) == 2 and p[1].lstrip("-").isdigit():
            labs.append(p[0])
            cnts.append(int(p[1]))
    rows = [sum(cnts[:i]) for i in range(len(labs))]
    rows[-1] = len(kf) - 1
    f_tickpos = kf[np.array(rows)]
    f_ticklab = [l.replace("G", r"$\Gamma$") for l in labs]

    # ── right panel: unfolded spectral weights ──
    kdist = result.kdist
    r_tickpos = [float(t[0]) for t in result.sym_ticks]
    r_ticklab = [str(t[1]).replace("G", r"$\Gamma$") for t in result.sym_ticks]

    with plt.rc_context(
        {
            "axes.linewidth": 1.1,
            "font.size": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    ):
        fig, (axL, axR) = plt.subplots(
            1,
            2,
            figsize=(11, 5.0),
            sharey=True,
            gridspec_kw={"width_ratios": [1, 1.15]},
        )

        axL.plot(kf, Ef, color=BLUE, lw=1.0)
        # axL.set_title(r'$(3,2)$ supercell BZ (folded)')
        axL.set_ylabel("Energy (eV)")
        axL.set_xticks(f_tickpos)
        axL.set_xticklabels(f_ticklab)
        axL.set_xlim(kf[0], kf[-1])

        kk = np.repeat(kdist[:, None], nawf_tw, axis=1).ravel()
        EE = (E_tw - E_F).ravel()
        ww = result.W.ravel()
        o = np.argsort(ww)  # strong weights drawn on top
        axR.plot(kdist, result.E_pc - E_F, color=GREEN, lw=1.0, zorder=0)
        sca = axR.scatter(
            kk[o],
            EE[o],
            c=ww[o],
            s=5 + 22 * ww[o],
            cmap=w_cmap,
            vmin=0.0,
            vmax=1.0,
            edgecolors="none",
            zorder=2,
        )
        # axR.set_title(r'unfolded onto graphene $1{\times}1$ BZ')
        axR.set_xticks(r_tickpos)
        axR.set_xticklabels(r_ticklab)
        axR.set_xlim(kdist[0], kdist[-1])
        cb = fig.colorbar(sca, ax=axR, pad=0.02)
        cb.set_label(r"spectral weight  $W_{n}(\mathbf{k})$")

        for ax, ticks in ((axL, f_tickpos), (axR, r_tickpos)):
            for xp in ticks:
                ax.axvline(xp, color="k", lw=0.8)
            ax.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.5)
            ax.set_ylim(*y_lim)

        fig.tight_layout()
        fig.savefig(
            "./tbg_" + unit + "_folded_vs_unfolded.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
