"""
do_orbital_chern.py
==================
Orbital Chern number C_L of a 2D material from the *feature-spectrum topology* of
the orbital angular momentum L_z, following Yao, Chu, Bansil, Lin & Chang,
"Topological Nature of Orbital Chern Insulators" [arXiv:2503.08138], and the
feature-spectrum framework of Wang, Hung, Lin et al.

Method (the orbital analogue of the spin Chern number)
------------------------------------------------------
1. Build the orbital angular momentum operator L_z in the lm PAO basis.
2. At each k, project L_z onto the occupied bands: the FEATURE operator
   P(k) L_z P(k).  Its eigenvalues -- the L_z-feature spectrum -- cluster into
   gapped SECTORS mu (for p orbitals: m-bar = +1, 0, -1).
3. Each sector is a sub-bundle P^mu(k); its Chern number C^mu is computed with the
   Fukui-Hatsugai-Suzuki (FHS) plaquette method (the L_z-resolved Wilson loop).
4. The orbital Chern number (Prodan-style, as for the spin Chern number) is

       C_L = ( C^{+} - C^{-} ) / 2

   over the highest (+) and lowest (-) feature sectors.  C_L != 0 signals an
   orbital Chern insulator with an orbital Hall plateau in the gap.

The feature spectrum must be GAPPED (sectors well separated) for C_L to be well
defined -- the orbital analogue of needing a spin gap.  Works spinless or spinful
(spin doubles C_L).  Only s/p/d shells (the j_to_lm limit); pure numpy + PAOFLOW
(no z2pack).

Spin Chern number (sanity check)
--------------------------------
The identical construction with S_z in place of L_z (``operator='S'``, see
``build_Sz``) is the original Prodan spin Chern number C_s; its parity
nu = C_s mod 2 is the Z2 index, so it cross-checks the z2pack / mirror-Chern Z2
for QSHIs (C_s = 1) versus trivial insulators (C_s = 0).

Basis dependence of L_z (IMPORTANT)
-----------------------------------
The feature spectrum is that of the MODEL operator P L_z P, and the atomic L_z is
basis dependent.  A minimal valence basis (e.g. QE projwfc s,p for blue-P) gives
the reference C_L.  An EXTENDED internal basis that adds polarization d shells to
sharpen the conduction bands leaks a small d-admixture into the valence Bloch
states; the atomic L_z assigns those a spurious +-1/+-2 moment, splitting off an
extra low-<L_z> sector whose Chern CANCELS the true one -> C_L collapses to 0 even
though the occupied bands are unchanged.  Verified on blue phosphorene: minimal
s,p -> C_L=1; extended (+3d,4d) full L_z -> C_L=0; extended with L_z restricted to
p -> C_L=1.  Fix: pass ``lz_channels=('p',)`` (or the physical valence character,
e.g. 'd' for a transition metal) to restrict L_z to the valence shell, so an
extended Hamiltonian can be used for the conduction bands while C_L stays the
basis-robust valence value.
"""

from __future__ import annotations

from os.path import join

import numpy as np

from .do_mirror_chern import _orbital, fractional_coords, read_hr


# --------------------------------------------------------------------------- #
#  L_z in the lm [up|down] basis
# --------------------------------------------------------------------------- #
def build_Lz(labels, channels=None):
    """Orbital angular momentum L_z (units of hbar) in the lm [up|down] basis.

    L_z couples the |m| = 1 pairs (p_x,p_y), (d_xz,d_yz) with +-i and the |m| = 2
    pair (d_x2-y2, d_xy) with +-2i (QE real-harmonic order), and is identity on
    spin (same block in the up and down spin blocks).  Sign convention matches
    L_z p_x = +i p_y (Eq. 1 of arXiv:2503.08138).

    channels: restrict which l-shells carry orbital moment, e.g. ('p',) keeps only
        the p contribution and zeroes d, ('p','d') keeps both, None keeps all.
        In an EXTENDED PAO basis the polarization d shells (added to sharpen the
        conduction bands) leak a small d-admixture into the valence Bloch states;
        the atomic L_z then assigns them spurious moment and can cancel the true
        orbital Chern.  Restricting L_z to the physical VALENCE character (e.g. p
        for a p-band material, d for a transition metal) removes that artifact and
        makes C_L basis-robust -- see the blue-phosphorene note in the module doc.
    """
    ell = {"s": "s", "pz": "p", "px": "p", "py": "p",
           "dz2": "d", "dzx": "d", "dzy": "d", "dyz": "d", "dx2-y2": "d", "dxy": "d"}
    keep = None if channels is None else set(channels)
    def inc(nm):
        return keep is None or ell.get(nm) in keep
    norb = len(labels) // 2
    names = [_orbital(l) for l in labels[:norb]]
    Lo = np.zeros((norb, norb), dtype=complex)
    i = 0
    while i < norb:
        nm = names[i]
        if nm in ("s", "pz", "dz2"):
            i += 1                                    # m = 0
        elif nm == "px":                              # (p_x, p_y): |m| = 1
            if inc(nm): Lo[i+1, i] = 1j; Lo[i, i+1] = -1j
            i += 2
        elif nm == "dzx":                             # (d_xz, d_yz): |m| = 1
            if inc(nm): Lo[i+1, i] = 1j; Lo[i, i+1] = -1j
            i += 2
        elif nm == "dx2-y2":                          # (d_x2-y2, d_xy): |m| = 2
            if inc(nm): Lo[i+1, i] = 2j; Lo[i, i+1] = -2j
            i += 2
        else:
            raise ValueError("unexpected orbital %r" % nm)
    Lz = np.zeros((2*norb, 2*norb), dtype=complex)
    Lz[:norb, :norb] = Lo
    Lz[norb:, norb:] = Lo
    return Lz


def build_Sz(labels):
    """Spin S_z (units of hbar) in the lm [up|down] basis.

    Diagonal: +1/2 on the up block (first norb) and -1/2 on the down block.
    Feeding this to the same feature-spectrum machinery gives the (Prodan) spin
    Chern number C_s -- the spin analogue of the orbital Chern number, and a
    Z2 sanity check:  nu = C_s mod 2  should equal the z2pack/mirror Z2 index.
    """
    norb = len(labels) // 2
    diag = np.concatenate([np.full(norb, 0.5), np.full(norb, -0.5)])
    return np.diag(diag).astype(complex)


def _Hk(H, R_list, kf, nawf):
    M = np.zeros((nawf, nawf), dtype=complex)
    for R in R_list:
        M += np.exp(2j*np.pi*(kf[0]*R[0] + kf[1]*R[1])) * H[R]
    return M


def _feature_states(Hm, Lz, nocc):
    """Occupied feature states: diagonalise P L_z P, return (feature eigs sorted,
    Psi with columns = feature Bloch states ordered by feature eigenvalue)."""
    w, V = np.linalg.eigh(Hm)
    Vo = V[:, :nocc]
    Lfeat = Vo.conj().T @ Lz @ Vo                     # nocc x nocc, Hermitian
    fw, fV = np.linalg.eigh(Lfeat)                    # ascending feature eigenvalues
    return fw, Vo @ fV                                # Psi = V_occ @ (feature vecs)


def _fhs_chern(psi_mesh, cols, nk):
    """Fukui-Hatsugai-Suzuki Chern number of the sub-bundle ``cols`` over the mesh."""
    Ux = np.empty((nk, nk), complex)
    Uy = np.empty((nk, nk), complex)
    for i in range(nk):
        for j in range(nk):
            A = psi_mesh[i][j][:, cols]
            dx = np.linalg.det(A.conj().T @ psi_mesh[(i+1) % nk][j][:, cols])
            dy = np.linalg.det(A.conj().T @ psi_mesh[i][(j+1) % nk][:, cols])
            Ux[i, j] = dx / abs(dx)
            Uy[i, j] = dy / abs(dy)
    C = 0.0
    for i in range(nk):
        for j in range(nk):
            C += np.angle(Ux[i, j] * Uy[(i+1) % nk, j]
                          * np.conj(Ux[i, (j+1) % nk]) * np.conj(Uy[i, j]))
    return C / (2*np.pi)


# --------------------------------------------------------------------------- #
def do_orbital_chern(data_controller, nbnd_occ="auto", nk=24, gap_tol=0.1,
                     n_sectors="auto", is_lm=False, verbose=True, operator="L",
                     lz_channels=None):
    """Feature-spectrum Chern number of P Op P (see module docstring).

    With ``operator='L'`` this is the orbital Chern number C_L; with
    ``operator='S'`` the same construction on S_z gives the (Prodan) spin Chern
    number C_s, whose parity nu = C_s mod 2 is a Z2 index (QSHI sanity check).

    Arguments:
        nbnd_occ: occupied bands ('auto' = nelec).
        nk: k-mesh (nk x nk) for the feature spectrum and FHS Chern.
        gap_tol: minimum feature-spectrum gap (units of hbar) to split sectors.
        n_sectors: 'auto' (split at every band-boundary whose min gap > gap_tol)
            or an int (split at the n_sectors-1 largest feature gaps).
        is_lm: True if 'HRs' is already the lm basis.
        operator: 'L' (orbital, L_z) or 'S' (spin, S_z).

    Returns dict: ``C_L`` (or ``C_s`` for operator='S'; the other is None),
    ``nu`` (C_s mod 2, only for operator='S'), ``sector_cherns`` (list,
    low->high feature eig), ``sector_means`` (feature-eig mean per sector),
    ``sector_sizes``, ``n_sectors``, ``feature_gap`` (min inter-sector gap over
    the mesh), ``feature_range``, ``C_total`` (sum of sector Cherns; ~0 sanity
    check), ``nocc``, ``nk``, ``gapped`` (bool), ``operator``.
    """
    from mpi4py import MPI
    from ..hamiltonian.do_j_to_lm import j_to_lm_hamiltonian, lm_basis_labels

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    op = str(operator).upper()
    if op not in ("L", "S"):
        raise ValueError("operator must be 'L' or 'S', got %r" % operator)
    oplab, clab, tag = (("L_z", "C_L", "orbital_chern") if op == "L"
                        else ("S_z", "C_s", "spin_chern"))
    arry, attr = data_controller.data_dicts()
    if not attr.get("dftSO", False):
        raise RuntimeError("orbital_chern_number requires a fully-relativistic (dftSO) run.")
    nelec = int(round(attr["nelec"]))
    nocc = nelec if nbnd_occ == "auto" else int(nbnd_occ)

    if not is_lm:
        stash = {k: np.copy(arry[k]) for k in ("HRs", "Hks") if k in arry}
        stash_basis = arry.get("basis")
        j_to_lm_hamiltonian(data_controller)
    labels = lm_basis_labels(data_controller)
    fname = "orbital_chern_lm_HRs.dat"
    data_controller.write_HRs(fname)
    if not is_lm:
        for k, v in stash.items():
            arry[k] = v
        if stash_basis is not None:
            arry["basis"] = stash_basis

    out = dict(C_L=None, C_s=None, nu=None, operator=op, C_plus=None, C_minus=None,
               sector_cherns=None, sector_means=None, sector_sizes=None, n_sectors=None,
               feature_gap=None, feature_range=None, C_total=None, nocc=nocc, nk=nk,
               gapped=None)
    if rank != 0:
        return comm.bcast(None, root=0)

    num_wann, R_list, degen, H = read_hr(join(attr["opath"], fname))
    nawf = num_wann
    Op = build_Lz(labels, channels=lz_channels) if op == "L" else build_Sz(labels)
    if verbose and op == "L" and lz_channels is not None:
        print("%s: L_z restricted to channels %s (valence-shell orbital moment)"
              % (tag, tuple(lz_channels)))

    # --- feature states on the k-mesh ----------------------------------------
    psi = [[None]*nk for _ in range(nk)]
    feig = np.empty((nk, nk, nocc))
    for i in range(nk):
        for j in range(nk):
            Hm = _Hk(H, R_list, [i/nk, j/nk], nawf)
            fw, P = _feature_states(Hm, Op, nocc)
            feig[i, j] = fw
            psi[i][j] = P
    fmin, fmax = feig.min(), feig.max()
    out["feature_range"] = (float(fmin), float(fmax))

    if op == "S":
        # --- spin Chern: Prodan split at feature eigenvalue 0 (spin up/down) ---
        # The S_z-feature spectrum sits near +-1/2; the two spin sectors are the
        # positive- and negative-eigenvalue halves, split at 0 (NOT at a fixed band
        # index).  It is gapped iff the number of positive eigenvalues is the same
        # at every k (clusters never cross 0); the spin gap is that eigenvalue-0 gap.
        npos_all = (feig > 0).sum(axis=2)
        npos = int(npos_all[0, 0])
        pos = np.where(feig > 0, feig, np.inf).min(axis=2)
        neg = np.where(feig < 0, feig, -np.inf).max(axis=2)
        spin_gap = float((pos - neg).min())
        const = bool(npos_all.min() == npos_all.max())
        gapped = bool(const and 0 < npos < nocc and spin_gap > gap_tol)
        ranges = [(0, nocc - npos), (nocc - npos, nocc)] if gapped else [(0, nocc)]
        out["feature_gap"] = spin_gap if const else 0.0
        out["gapped"] = gapped
    else:
        # --- orbital Chern: sectors from the (band-index) feature gaps ---------
        # gap at boundary b = min over the mesh of the LOCAL (same-k) gap
        # feat[b+1] - feat[b] (feature bands are sorted ascending per k).  A positive
        # min means band b/b+1 never cross, so the sector membership is k-independent.
        band_gap = np.array([(feig[:, :, b+1] - feig[:, :, b]).min() for b in range(nocc-1)])
        if n_sectors == "auto":
            bounds = [b for b in range(nocc-1) if band_gap[b] > gap_tol]
        else:
            bounds = list(np.argsort(band_gap)[-(int(n_sectors)-1):]) if int(n_sectors) > 1 else []
            bounds = sorted(bounds)
        edges = [0] + [b+1 for b in bounds] + [nocc]
        ranges = [(edges[s], edges[s+1]) for s in range(len(edges)-1)]
        out["gapped"] = bool(len(ranges) > 1 and (min([band_gap[b] for b in bounds]) > gap_tol if bounds else False))
        out["feature_gap"] = float(min([band_gap[b] for b in bounds])) if bounds else 0.0
    out["n_sectors"] = len(ranges)
    out["sector_sizes"] = [hi - lo for lo, hi in ranges]

    if verbose:
        print("%s: nocc=%d, %s-feature spectrum in [%.3f, %.3f], "
              "%d sector(s) sizes=%s (min inter-sector gap=%.3f)"
              % (tag, nocc, oplab, fmin, fmax, len(ranges), out["sector_sizes"], out["feature_gap"]))
    if len(ranges) < 2:
        if verbose:
            print("%s: feature spectrum not gapped (need >=2 sectors) -> "
                  "%s undefined; lower gap_tol or check the %s character."
                  % (tag, clab, oplab))
        return comm.bcast(out, root=0)

    # --- FHS Chern of each sector --------------------------------------------
    cherns, means = [], []
    for lo, hi in ranges:
        cols = np.arange(lo, hi)
        cherns.append(float(round(_fhs_chern(psi, cols, nk), 3)))
        means.append(float(feig[:, :, lo:hi].mean()))
    out["sector_cherns"] = cherns
    out["sector_means"] = [round(m, 3) for m in means]
    out["C_total"] = float(sum(cherns))
    # C = (C^+ - C^-)/2 with C^+ (C^-) the summed Chern of ALL sectors of positive
    # (negative) <Op> -- for a multi-|m| spectrum the topological sectors need not be
    # the outermost, so sum over the Op sign rather than the extreme sector.  For S_z
    # the two sectors sit near +-1/2 (spin up/down) and this is the spin Chern number.
    mtol = 0.1
    Cp = float(sum(c for c, m in zip(cherns, means) if m > mtol))
    Cm = float(sum(c for c, m in zip(cherns, means) if m < -mtol))
    out["C_plus"], out["C_minus"] = Cp, Cm
    C = 0.5 * (Cp - Cm)
    out[clab] = C
    if op == "S":
        out["nu"] = int(round(C)) % 2

    if verbose:
        for (lo, hi), c, m in zip(ranges, cherns, means):
            print("%s:   sector <%s>~%+.2f  (%d bands)  C^mu=%+.3f" % (tag, oplab, m, hi-lo, c))
        extra = " -> Z2 nu=%d" % out["nu"] if op == "S" else ""
        print("%s: C+=%.3f C-=%.3f (summed over %s sign),  "
              "%s=(C+ - C-)/2=%+.3f  [C_total=%.3f ~0]%s"
              % (tag, Cp, Cm, oplab, clab, C, out["C_total"], extra))
    return comm.bcast(out, root=0)
