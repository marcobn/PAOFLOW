"""
do_hoti_indicators.py
=====================
Rotation-eigenvalue (symmetry-indicator) characterization of 2D higher-order
topological insulators (HOTIs): the quantized fractional CORNER CHARGE of a
C_n-symmetric layer from the C_n rotation eigenvalues of the occupied bands at the
high-symmetry momenta, following Benalcazar, Li & Hughes [PRB 99, 245151 (2019)].

Indices  [Pi_p^(m)] = #Pi_p^(m) - #Gamma_p^(m)  count occupied bands with rotation
eigenvalue  Pi_p^(m) = e^{2 pi i (p-1)/m}  at the C_m-invariant momentum, relative
to Gamma.  The nominal electronic corner charges are (Eq. 11 of the reference)

    Q^(4) = (e/4)([X1^2] + 2[M1^4] + 3[M2^4])          (square, C4)
    Q^(2) = (e/4)(-[X1^2] - [Y1^2] + [M1^2])           (rectangular, C2)
    Q^(6) = (e/4)[M1^2] + (e/6)[K1^3]                  (hexagonal, C6)
    Q^(3) = (e/3)[K2^3]                                (hexagonal, C3)

all mod e, and are well defined only when the bulk polarization P (Eq. 8) vanishes.

Choice of rotation axis
-----------------------
A rotation of a PERIODIC crystal has no unique centre: the same operation is a C_n
rotation about |det(1-M)| inequivalent axes (3 for C3 in a hexagonal cell -- the
Wyckoff positions 1a/1b/1c; 1 for C6; 2 for C4; 4 for C2).  They differ by a global
phase on the C_n operator at each C_m-invariant momentum, so each axis carries its
OWN [Pi_p], polarization and corner charge: physically, the axis is the Wyckoff
position that sits at the CORNER of the flake the indicator describes.  Since Q^(n)
only means anything where P = 0, this module evaluates every axis and reports the
one with vanishing polarization (``centre`` / ``centre_atom``; all are kept in
``axes``).  For 2H-TMDs that is the chalcogen axis, NOT the cell origin.

Spin-orbit caveat
-----------------
Eq. 11 is derived for SPINLESS bands.  For a spin-orbit-coupled (spinful) system
the rotation eigenvalues are the m-th roots of -1, and mapping them onto the labels
{1, w, w^2, ...} needs an arbitrary global phase (which eigenvalue is p=1), so the
symmetry indicators DO NOT uniquely fix the corner charge (Sec. VIII of the
reference: "symmetry representations at high symmetry points do not suffice to
determine the Wannier centers in spinful systems").  For spinful input this module
therefore returns the eigenvalue multiplicities, the polarization, and the SET of
corner-charge values over the admissible relabelings (``corner_charge_set``),
flagging ``certified=False``.  A certified spinful corner charge needs the
real-space filling anomaly or a nested Wilson loop.

Reuses the lm-basis operator machinery of :mod:`do_mirror_chern`.  Requires the C_n
rotation to be symmorphic and s/p/d shells (the ``j_to_lm`` limit).
"""

from __future__ import annotations

from itertools import product
from os.path import join

import numpy as np

from .do_mirror_chern import (_orbital, _spin, _atom_index, fractional_coords, read_hr)

# 120 deg (C3) and 60 deg (C6) in the hexagonal direct basis a1=x, a2=(-1/2, sqrt3/2);
# 90 deg (C4) in the square basis.  All satisfy R_cart a_i = a_j (M acts on frac coords).
_M = {
    ("hex", 6): np.array([[1, -1], [1, 0]]),
    ("hex", 3): np.array([[0, -1], [1, -1]]),
    ("hex", 2): np.array([[-1, 0], [0, -1]]),
    ("sq", 4): np.array([[0, -1], [1, 0]]),
    ("sq", 2): np.array([[-1, 0], [0, -1]]),
}


def _rot2(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _orbital_block(orb_names, theta):
    """C_n orbital rotation for one atom's real-harmonic shell sequence (QE order)."""
    k = len(orb_names)
    O = np.zeros((k, k))
    i = 0
    while i < k:
        nm = orb_names[i]
        if nm in ("s", "pz", "dz2"):
            O[i, i] = 1.0; i += 1
        elif nm == "px":
            O[i:i+2, i:i+2] = _rot2(theta); i += 2
        elif nm == "dzx":
            O[i:i+2, i:i+2] = _rot2(theta); i += 2
        elif nm == "dx2-y2":
            O[i:i+2, i:i+2] = _rot2(2*theta); i += 2
        else:
            raise ValueError("unexpected orbital %r" % nm)
    return O


def _build_Ofull(names, atom_of, perm, theta):
    """Spinless orbital rotation on all N_orb orbitals: rotate + permute atoms."""
    n = len(names)
    atoms = sorted(set(atom_of))
    cols_of = {a: [j for j in range(n) if atom_of[j] == a] for a in atoms}
    O = np.zeros((n, n))
    for a in atoms:
        cols, rows = cols_of[a], cols_of[perm[a]]
        Oa = _orbital_block([names[j] for j in cols], theta)
        for ii, r in enumerate(rows):
            for jj, c in enumerate(cols):
                O[r, c] = Oa[ii, jj]
    return O


def rotation_map(frac, species, Mfrac, symprec=1e-2, t=None):
    """Atom permutation perm[a]=b induced by  f -> Mfrac f + t  (t=None: about the
    origin), or None if that map is not a symmetry of the layer."""
    n = len(frac)
    t = np.zeros(2) if t is None else np.asarray(t, float)
    perm = -np.ones(n, dtype=int)
    for a in range(n):
        img = (Mfrac @ frac[a, :2]) + t
        for b in range(n):
            if species[b] != species[a]:
                continue
            d = np.zeros(3)
            d[:2] = img - frac[b, :2]
            d[2] = frac[a, 2] - frac[b, 2]        # rotation keeps z
            d -= np.round(d)
            if np.max(np.abs(d)) < symprec:
                perm[a] = b
                break
        if perm[a] < 0:
            return None
    return perm


def rotation_axes(frac, species, Mfrac, symprec=1e-2):
    """All C_n axes of the layer: (list of in-plane centres, atom permutation).

    If  f -> M f + t  is a symmetry then the same operation is a rotation about
    c = (1-M)^-1 (t + R)  for every lattice vector R, i.e. about |det(1-M)|
    inequivalent axes (see the module docstring).  They share the atom permutation
    -- only the Bloch phases differ -- so one permutation is returned alongside the
    centres.  Returns ([], None) if the rotation is not a symmetry at all.
    """
    IM = np.eye(2) - np.asarray(Mfrac, float)
    nax = int(round(abs(np.linalg.det(IM))))          # number of distinct axes
    IMi = np.linalg.inv(IM)
    t0 = perm = None
    for b0 in range(len(frac)):                       # t is fixed by the image of atom 0
        if species[b0] != species[0] or abs(frac[b0, 2] - frac[0, 2]) > symprec:
            continue
        t = frac[b0, :2] - Mfrac @ frac[0, :2]
        p = rotation_map(frac, species, Mfrac, symprec, t)
        if p is not None:
            t0, perm = t, p
            break
    if t0 is None:
        return [], None
    cs = []
    for r1 in range(-nax, nax + 1):
        for r2 in range(-nax, nax + 1):
            c = IMi @ (t0 + np.array([r1, r2], float))
            c = c - np.floor(c + 1e-6)                # representative in [0, 1)
            if not any(np.max(np.abs((c - o + 0.5) % 1.0 - 0.5)) < 1e-4 for o in cs):
                cs.append(c)
    cs.sort(key=lambda v: (round(v[0], 6), round(v[1], 6)))
    return cs, perm


def _axis_label(c, frac, species, symprec=1e-2):
    """Species of the atom sitting on the axis through c, or 'empty'."""
    for a in range(len(frac)):
        d = frac[a, :2] - np.asarray(c, float)
        d -= np.round(d)
        if np.max(np.abs(d)) < symprec:
            return str(species[a])
    return "empty"


def rotation_operator(m, kf, names, atom_of, frac, perm, Mrec, spinful=True):
    """C_m rotation operator at momentum kf.  spinful=True -> lm [up|down] basis
    with the spin-1/2 rotation e^{-/+ i theta/2}; spinful=False -> orbital-only
    (scalar-relativistic run, single orbital block)."""
    theta = 2*np.pi/m
    norb = len(names)
    O = _build_Ofull(names, atom_of, perm, theta)
    Mk = Mrec @ np.asarray(kf, float)
    # phase per column j (atom a): exp(2 pi i [ (Mrec k).f_b - k.f_a ]), b = perm[a]
    ph = np.array([np.exp(2j*np.pi*(Mk @ frac[perm[atom_of[j]], :2]
                                    - np.asarray(kf) @ frac[atom_of[j], :2]))
                   for j in range(norb)])
    D = O * ph[None, :]
    if not spinful:
        return D                                    # orbital only, norb x norb
    su, sd = np.exp(-1j*theta/2), np.exp(+1j*theta/2)
    C = np.zeros((2*norb, 2*norb), complex)
    C[:norb, :norb] = su * D
    C[norb:, norb:] = sd * D
    return C


def _Hk(H, R_list, kf, nawf):
    M = np.zeros((nawf, nawf), complex)
    for R in R_list:
        M += np.exp(2j*np.pi*(kf[0]*R[0] + kf[1]*R[1])) * H[R]
    return M


def _occ_rot_eigs(Hm, C, nocc):
    """Eigenvalues of the C_m operator restricted to the occupied subspace."""
    w, V = np.linalg.eigh(Hm)
    Vo = V[:, :nocc]
    return np.linalg.eigvals(Vo.conj().T @ C @ Vo)


def _count(eigs, m, spinful=True):
    """Multiplicities of the C_m eigenvalues with spinless labels p=1..m
    (Pi_p = e^{2 pi i (p-1)/m}).  spinful=True: eigenvalues are the m-th roots of -1
    (factor the global phase e^{i pi/m}); spinful=False: roots of +1 directly, so
    the p-labels are canonical (no gauge freedom -> certified)."""
    if spinful:
        allowed = [np.exp(1j*np.pi*(2*j+1)/m) for j in range(m)]    # m-th roots of -1
    else:
        allowed = [np.exp(2j*np.pi*j/m) for j in range(m)]          # m-th roots of +1
    label = [((j) % m) + 1 for j in range(m)]                       # -> p=j+1
    mult = {p: 0 for p in range(1, m+1)}
    dev = 0.0
    for e in eigs:
        k = int(np.argmin([abs(e-a) for a in allowed]))
        dev = max(dev, abs(e-allowed[k]))
        mult[label[k]] += 1
    return mult, dev


# --------------------------------------------------------------------------- #
#  Inversion (parity) Z4 indicator -- for centrosymmetric materials (e.g. 1T)
# --------------------------------------------------------------------------- #
_PARITY = {"s": 1, "pz": -1, "px": -1, "py": -1,                   # (-1)^l
           "dz2": 1, "dzx": 1, "dzy": 1, "dyz": 1, "dx2-y2": 1, "dxy": 1}


def inversion_map(frac, species, symprec=1e-2):
    """Atom permutation under inversion (-f_a + t0 == f_b) and the center-doubling
    t0, or None if the structure is not centrosymmetric.  Unlike the in-plane
    rotations this flips z, so buckled top/bottom atoms are swapped.

    t0 is unique modulo the lattice, but its REPRESENTATIVE matters (see
    _z4_indicator): it is reduced to the one closest to 0, i.e. the inversion centre
    the structure is actually written about.  Do not wrap it into [0,1) -- a
    coordinate sitting at -1e-9 would then wrap to t0 ~ 1 and move the centre by
    half a lattice vector, which flips Z4."""
    n = len(frac)
    for b0 in range(n):
        if species[b0] != species[0]:
            continue
        t0 = frac[0] + frac[b0]                                   # 2 * inversion center
        t0 = t0 - np.round(t0)                                    # representative ~ 0
        perm = -np.ones(n, dtype=int); ok = True
        for a in range(n):
            img = -frac[a] + t0
            hit = -1
            for b in range(n):
                if species[b] != species[a]:
                    continue
                d = img - frac[b]; d -= np.round(d)
                if np.max(np.abs(d)) < symprec:
                    hit = b; break
            if hit < 0:
                ok = False; break
            perm[a] = hit
        if ok:
            return perm, t0
    return None


def parity_operator(kf, names, atom_of, frac, perm, spinful=True, t0=None):
    """Inversion operator P at momentum kf about the centre c = t0/2: parity (-1)^l
    per orbital, the atom permutation under inversion, and the Bloch phase;
    spin-diagonal (inversion commutes with spin).  spinful=False -> orbital block
    only (scalar run).

    The block a -> perm[a] carries exp(2 pi i k.(t0 - f_a - f_perm[a])) -- the
    lattice vector inversion must add to bring -f_a + t0 back onto f_perm[a].  The
    t0 term is a GLOBAL phase, so dropping it leaves [H,P] intact but silently moves
    the centre to the cell origin; at the TRIMs it is +-1 and flips Z4.  t0=None
    keeps that historical behaviour (correct only if the centre IS the origin)."""
    norb = len(names)
    atoms = sorted(set(atom_of))
    cols_of = {a: [j for j in range(norb) if atom_of[j] == a] for a in atoms}
    par = np.array([_PARITY[n] for n in names])
    kf = np.asarray(kf, float)
    t0 = np.zeros(2) if t0 is None else np.asarray(t0, float)[:2]
    D = np.zeros((norb, norb), complex)
    for a in atoms:
        cols, rows = cols_of[a], cols_of[perm[a]]
        ph = np.exp(2j*np.pi*(kf @ t0 - kf @ frac[perm[a], :2] - kf @ frac[a, :2]))
        for ii, c in enumerate(cols):
            D[rows[ii], c] = par[c] * ph                           # same species -> aligned order
    if not spinful:
        return D
    P = np.zeros((2*norb, 2*norb), complex)
    P[:norb, :norb] = D
    P[norb:, norb:] = D
    return P


def _z4_at(H, R_list, nawf, nocc, names, atom_of, frac, perm, spinful, t0):
    """Z4 about the single inversion centre c = t0/2.  Returns (z4, parities, res)."""
    trims = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]
    bw = max(np.abs(H[R]).max() for R in R_list)
    fac = 0.25 if spinful else 0.5
    s = 0; par = {}; res = 0.0
    for kf in trims:
        P = parity_operator(kf, names, atom_of, frac, perm, spinful, t0)
        Hm = _Hk(H, R_list, kf, nawf)
        res = max(res, np.abs(Hm @ P - P @ Hm).max() / bw)
        w, V = np.linalg.eigh(Hm); Vo = V[:, :nocc]
        pe = np.real(np.linalg.eigvals(Vo.conj().T @ P @ Vo))
        npos = int(np.sum(pe > 1e-6)); nneg = int(np.sum(pe < -1e-6))
        s += (nneg - npos); par["%.1f,%.1f" % (kf[0], kf[1])] = (npos, nneg)
    return int(round(fac * s)) % 4, par, float(res)


def _z4_indicator(H, R_list, nawf, nocc, names, atom_of, frac, perm, spinful, t0):
    """2D inversion index  Z4 = (1/4) sum_{TRIM} (n^- - n^+)  mod 4  (spinful; the
    Kramers factor is 1/2 for a scalar run), with Z2 = Z4 mod 2.  n^-/n^+ are the
    occupied-band counts of inversion eigenvalue -1/+1 at the four 2D TRIMs.

    The centre is fixed only to within HALF a lattice vector: t0 is unique mod 1,
    but exp(2 pi i k.R) = +-1 at the TRIMs, so each of the 4 representatives t0 + R
    -- the 4 inversion centres of the cell -- gives its own Z4, exactly as each C_n
    axis has its own corner charge.  [H,P] is identical on all of them (the centre
    enters as a global phase), so the commutator is NOT a guard.  The reduced t0,
    i.e. the centre the structure is written about, is the answer; the other three
    are returned so the ambiguity is visible.

    Returns (best, all) with best = {centre, z4, parities, residual}."""
    t0 = np.asarray(t0, float)
    out = []
    for R in ([0., 0.], [1., 0.], [0., 1.], [1., 1.]):
        tt = t0[:2] + np.asarray(R)
        z4, par, res = _z4_at(H, R_list, nawf, nocc, names, atom_of, frac, perm,
                              spinful, tt)
        out.append({"centre": [float(tt[0]/2), float(tt[1]/2)], "z4": int(z4),
                    "parities": par, "residual": res})
    return out[0], out


# --------------------------------------------------------------------------- #
#  high-symmetry momenta and BBH formulas per principal rotation n
# --------------------------------------------------------------------------- #
def _hsp_table(lat, n):
    """Return {name: (kf, m_rot)} of C_m-invariant momenta needed for Q^(n)."""
    if (lat, n) == ("hex", 3):
        return {"K": ([1/3, 1/3], 3), "Kp": ([2/3, 2/3], 3)}
    if (lat, n) == ("hex", 6):
        return {"M": ([0.5, 0.0], 2), "K": ([1/3, 1/3], 3)}
    if (lat, n) == ("sq", 4):
        return {"X": ([0.5, 0.0], 2), "M": ([0.5, 0.5], 4)}
    if (lat, n) == ("sq", 2):
        return {"X": ([0.5, 0.0], 2), "Y": ([0.0, 0.5], 2), "M": ([0.5, 0.5], 2)}
    raise ValueError("no HOTI table for (%s, C%d)" % (lat, n))


def _corner_charge(n, inv):
    """Nominal electronic corner charge Q^(n)/e (mod 1) from the invariants dict."""
    if n == 3:
        return (1/3) * inv["K"][2]
    if n == 6:
        return 0.25 * inv["M"][1] + (1/6) * inv["K"][1]
    if n == 4:
        return 0.25 * (inv["X"][1] + 2*inv["M"][1] + 3*inv["M"][2])
    if n == 2:
        return 0.25 * (-inv["X"][1] - inv["Y"][1] + inv["M"][1])
    raise ValueError(n)


def _polarization(n, inv):
    """Bulk polarization (p1, p2) in units of e (mod 1); must vanish for a HOTI."""
    if n == 3:
        # P = v (a1 + 2 a2), v in {0, 1/3, 2/3}: those are the only C3-invariant
        # polarizations -- (v, v) is NOT one of them (C3 (v,v) = (-2v, -v)).
        v = (2/3) * (inv["K"][1] + 2*inv["K"][2]); return (v, 2*v)
    if n == 6:
        return (0.0, 0.0)
    if n == 4:
        v = 0.5 * inv["X"][1]; return (v, v)
    if n == 2:
        return (0.5*(inv["Y"][1] + inv["M"][1]), 0.5*(inv["X"][1] + inv["M"][1]))
    raise ValueError(n)


# --------------------------------------------------------------------------- #
def do_hoti_indicators(data_controller, nbnd_occ="auto", is_lm=False,
                       symprec=1e-2, verbose=True):
    """C_n rotation-eigenvalue HOTI indicators (see module docstring).

    Returns a dict: ``n`` (principal rotation), ``lattice``, ``multiplicities``
    (per-HSP dict of {p: count}), ``invariants`` ([Pi_p^(m)] = #HSP_p - #Gamma_p),
    ``polarization`` (p1,p2 in e), ``polarization_vanishes`` (False -> Q^(n) is
    ill-defined on every axis), ``corner_charge`` (nominal, e; the natural spinful
    mapping), ``corner_charge_set`` (all values over the spinful relabeling),
    ``certified`` (False for spin-orbit input), ``residual`` (max [H,C_n] at Gamma /
    bandwidth), ``centre`` / ``centre_atom`` (the C_n axis the numbers refer to --
    the one with P = 0) and ``axes`` (centre, P and Q for every inequivalent axis).
    All the per-HSP quantities belong to the selected axis.
    """
    from mpi4py import MPI
    from ..hamiltonian.do_j_to_lm import j_to_lm_hamiltonian, lm_basis_labels

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    arry, attr = data_controller.data_dicts()
    # spinful (SOC): C_n eigenvalues are the m-th roots of -1, so the spinless->p
    # labeling has an unfixable global-phase gauge -> corner charge uncertified.
    # spinless (scalar-relativistic): roots of +1, canonical p-labels -> certified.
    spinful = bool(attr.get("dftSO", False))

    nelec = int(round(attr["nelec"]))
    nocc = (nelec if spinful else nelec // 2) if nbnd_occ == "auto" else int(nbnd_occ)

    # spinful HR is in the j basis -> rotate to lm; spinless HR is already real-lm.
    if not is_lm and spinful:
        stash = {k: np.copy(arry[k]) for k in ("HRs", "Hks") if k in arry}
        stash_basis = arry.get("basis")
        j_to_lm_hamiltonian(data_controller)
    labels = lm_basis_labels(data_controller)
    fname = "hoti_lm_HRs.dat"
    data_controller.write_HRs(fname)
    if not is_lm and spinful:
        for k, v in stash.items():
            arry[k] = v
        if stash_basis is not None:
            arry["basis"] = stash_basis

    out = dict(n=None, lattice=None, multiplicities={}, invariants={},
               polarization=None, polarization_vanishes=None, corner_charge=None,
               corner_charge_set=None, centre=None, centre_atom=None, axes=None,
               certified=False, residual=None, hsp_dev=None, hsp_gapped=None,
               nocc=nocc, spinful=spinful, z4=None, z2_inv=None, z4_centre=None,
               z4_centre_atom=None, z4_centres=None, z4_centre_ambiguous=None,
               error=None)
    if rank != 0:
        return comm.bcast(None, root=0)

    hr = join(attr["opath"], fname)
    num_wann, R_list, degen, H = read_hr(hr)
    nawf = num_wann
    # C_n operator dimension: spinful basis is spin-doubled (norb = nawf/2), the
    # spinless HR is a single orbital block (norb = nawf).  lm_basis_labels always
    # lists the up block first, so labels[:norb] are the orbital names either way.
    norb = nawf // 2 if spinful else nawf
    names = [_orbital(l) for l in labels[:norb]]
    atom_of = [_atom_index(l) for l in labels[:norb]]
    frac = fractional_coords(arry["tau"], arry["a_vectors"], attr["alat"])
    species = list(arry["atoms"])

    # --- inversion Z4 indicator (centrosymmetric materials, e.g. 1T-TMDs) -----
    # Independent of the rotation route below, so it is reported even for cells
    # with no in-plane C_n.  Z4 = 2 -> HOTI (Z2 = 0); Z4 odd -> QSHI.
    inv_op = inversion_map(frac, species, symprec)
    if inv_op is not None:
        perm_i, t0 = inv_op
        best, allz = _z4_indicator(H, R_list, nawf, nocc, names, atom_of,
                                   frac, perm_i, spinful, t0)
        out["z4"] = best["z4"]; out["z2_inv"] = best["z4"] % 2
        out["parities"] = best["parities"]
        out["z4_centre"] = best["centre"]
        out["z4_centre_atom"] = _axis_label(best["centre"], frac, species, symprec)
        out["z4_centres"] = [{"centre": c["centre"], "z4": c["z4"]} for c in allz]
        out["z4_centre_ambiguous"] = len({c["z4"] for c in allz}) > 1
        if verbose:
            print("hoti: inversion present -> Z4 = %d (Z2 = %d)%s  about the centre "
                  "(%.3f, %.3f) [%s]  [H,P]/W=%.1e"
                  % (best["z4"], best["z4"] % 2,
                     "  HOTI" if best["z4"] == 2 else
                     ("  QSHI" if best["z4"] % 2 else ""),
                     best["centre"][0], best["centre"][1], out["z4_centre_atom"],
                     best["residual"]))
            if out["z4_centre_ambiguous"]:
                print("hoti: NOTE Z4 is centre-dependent [%s] -- the value above is "
                      "for the centre the structure is written about"
                      % ", ".join("(%.2f,%.2f)->%d" % (c["centre"][0], c["centre"][1],
                                                       c["z4"]) for c in allz))

    # --- lattice type from the in-plane cell angle ---------------------------
    av = np.asarray(arry["a_vectors"], float)
    ang = np.degrees(np.arccos(np.dot(av[0, :2], av[1, :2]) /
                               (np.linalg.norm(av[0, :2]) * np.linalg.norm(av[1, :2]))))
    lat = "hex" if abs(ang - 120) < 5 or abs(ang - 60) < 5 else ("sq" if abs(ang - 90) < 5 else None)
    if lat is None:
        out["error"] = ("unsupported lattice angle %.1f deg (need hexagonal or "
                        "square/rectangular)" % ang)
        if verbose:
            print("hoti: %s -> corner charge not applicable" % out["error"])
        return comm.bcast(out, root=0)
    out["lattice"] = lat

    HG = _Hk(H, R_list, [0.0, 0.0], nawf)
    bandwidth = max(np.abs(H[R]).max() for R in R_list)

    # --- principal rotation n: highest C_n that commutes with H at Gamma ------
    # The axis does not enter here: moving the centre multiplies C by a global phase
    # at each k, so [H,C] is identical on every axis.
    n, axes = None, None
    for cand in ((6, 4, 3, 2) if lat == "hex" else (4, 2)):
        if (lat, cand) not in _M:
            continue
        Mf = _M[(lat, cand)]
        cs, perm = rotation_axes(frac, species, Mf, symprec)
        if perm is None:
            continue
        Mrec = np.linalg.inv(Mf).T
        fr = np.array(frac, float, copy=True); fr[:, :2] -= cs[0][None, :]
        C = rotation_operator(cand, [0.0, 0.0], names, atom_of, fr, perm, Mrec, spinful)
        res = np.abs(HG @ C - C @ HG).max() / bandwidth
        if res < 1e-3:
            n, axes = cand, cs
            out["residual"] = float(res)
            break
    if n is None:
        out["error"] = ("no out-of-plane C_n rotation (only in-plane C2 / mirrors) "
                        "-- rotation corner-charge indicator not applicable")
        if verbose:
            print("hoti: %s" % out["error"])
        return comm.bcast(out, root=0)
    out["n"] = n
    if verbose:
        print("hoti: lattice=%s, principal rotation C%d, [H,C%d]/W=%.2e at Gamma; "
              "%d inequivalent C%d axes"
              % (lat, n, n, out["residual"], len(axes), n))

    table = _hsp_table(lat, n)
    m_orders = sorted({m for _, m in table.values()})
    order_of = {name: m for name, (_, m) in table.items()}

    # --- eigenvalue multiplicities at Gamma and each HSP, for one C_n axis ----
    def analyse(centre):
        fr = np.array(frac, float, copy=True)
        fr[:, :2] -= np.asarray(centre, float)[None, :]

        def mult_at(kf, m):
            Mfm = _M[(lat, m)]; Mrm = np.linalg.inv(Mfm).T
            permm = rotation_map(fr, species, Mfm, symprec)
            C = rotation_operator(m, kf, names, atom_of, fr, permm, Mrm, spinful)
            Hm = _Hk(H, R_list, kf, nawf)
            r = np.abs(Hm @ C - C @ Hm).max() / bandwidth
            mlt, dev = _count(_occ_rot_eigs(Hm, C, nocc), m, spinful)
            return mlt, r, dev

        gamma, gdev = {}, 0.0
        for m in m_orders:
            g, _, dv = mult_at([0.0, 0.0], m)
            gamma[m] = g
            gdev = max(gdev, dv)
        mult = {"Gamma": {m: gamma[m] for m in m_orders}}
        inv, diag = {}, {}
        max_dev = gdev
        for name, (kf, m) in table.items():
            mlt, r, dev = mult_at(kf, m)
            mult[name] = mlt
            inv[name] = {p: mlt[p] - gamma[m][p] for p in mlt}     # [Pi_p^(m)]
            diag[name] = (r, dev)
            max_dev = max(max_dev, dev)
        p1, p2 = _polarization(n, inv)
        return (mult, inv, (float(p1 % 1.0), float(p2 % 1.0)),
                float(_corner_charge(n, inv) % 1.0), diag, float(max_dev))

    # --- select the axis on which the bulk polarization vanishes -------------
    # Q^(n) is defined only where P = 0, and P depends on the axis; the P = 0 axis is
    # the Wyckoff position that sits at the corner of the corresponding flake.
    results = [analyse(c) for c in axes]
    out["axes"] = [{"centre": [float(c[0]), float(c[1])],
                    "atom": _axis_label(c, frac, species, symprec),
                    "polarization": r[2], "corner_charge": r[3]}
                   for c, r in zip(axes, results)]
    pick = next((i for i, r in enumerate(results)
                 if abs(r[2][0]) < 1e-6 and abs(r[2][1]) < 1e-6), None)
    out["polarization_vanishes"] = pick is not None
    if pick is None:
        pick = 0                     # no P = 0 axis -> corner charge is ill-defined
    mult, inv, pol, Qc, diag, max_dev = results[pick]
    out["centre"] = [float(axes[pick][0]), float(axes[pick][1])]
    out["centre_atom"] = _axis_label(axes[pick], frac, species, symprec)
    out["multiplicities"] = mult
    out["invariants"] = inv
    out["polarization"] = pol
    out["corner_charge"] = Qc
    # The C_n eigenvalues of the occupied subspace must be clean m-th roots of
    # +-1; max_dev is the largest deviation.  A big value means the occupied
    # projector is NOT C_n-symmetric -- a band touches the Fermi level / a
    # degenerate multiplet at the HSP is split by the nocc cutoff (near-metallic
    # or an unopened crossing, e.g. a spinless run before SOC gaps it).  The
    # [Pi_p] counts, hence the corner charge, are then meaningless.
    out["hsp_dev"] = float(max_dev)
    out["hsp_gapped"] = bool(max_dev < 1e-1)

    if verbose:
        for i, (c, r) in enumerate(zip(axes, results)):
            print("hoti: C%d axis (%.3f, %.3f) [%-5s]  P=(%.3f, %.3f)  Q=%.3f%s"
                  % (n, c[0], c[1], _axis_label(c, frac, species, symprec),
                     r[2][0], r[2][1], r[3],
                     "   <- P=0, selected" if (i == pick and
                                               out["polarization_vanishes"]) else ""))
        for name in table:
            r, dev = diag[name]
            print("hoti: %-3s (C%d)  mult=%s  [Pi_p]=%s  ([H,C]/W=%.1e, dev=%.0e)%s"
                  % (name, order_of[name], dict(mult[name]), dict(inv[name]), r, dev,
                     "  <- NOT a clean C_n eigenspace" if dev > 1e-1 else ""))

    if spinful:
        # the spinless->p labeling has a global-phase gauge, one per rotation order;
        # report the set over the admissible relabelings (Q is one of these).
        qset = set()
        for shifts in product(*[range(m) for m in m_orders]):
            sh_of = dict(zip(m_orders, shifts))
            inv_sh = {name: {p: inv[name][((p - 1 + sh_of[order_of[name]])
                                           % order_of[name]) + 1] for p in inv[name]}
                      for name in inv}
            try:
                qset.add(round(_corner_charge(n, inv_sh) % 1.0, 6))
            except Exception:
                pass
        out["corner_charge_set"] = sorted(qset)
        out["certified"] = False    # spinful: Eq. 11 is a spinless formula
    else:
        # spinless: canonical p-labels -> the corner charge is unique... but only if
        # the occupied projector is actually C_n-symmetric (hsp_gapped).
        out["corner_charge_set"] = [round(float(Qc), 6)]
        out["certified"] = bool(out["hsp_gapped"])

    if verbose:
        if not out["hsp_gapped"]:
            print("hoti: WARNING occupied C%d eigenvalues deviate by %.2f from clean "
                  "roots of unity -- a band touches E_F / a degenerate multiplet is "
                  "cut by the occupation at an HSP; the corner charge is UNRELIABLE "
                  "(check the gap; if this is a spinless run, SOC may gap it)."
                  % (n, out["hsp_dev"]))
        if not out["polarization_vanishes"]:
            print("hoti: no C%d axis has P=0 (best P=(%.3f, %.3f) e) -- corner charge "
                  "ill-defined" % (n, pol[0], pol[1]))
        elif spinful:
            print("hoti: Q_corner = %.3f e on the (%.3f, %.3f) [%s] axis, P=0 ;  "
                  "spinful set = %s e  (NOT certified: Eq. 11 is spinless -- rerun "
                  "scalar-relativistically to confirm, 2 x Q_spinless == Q mod e)"
                  % (Qc, out["centre"][0], out["centre"][1], out["centre_atom"],
                     out["corner_charge_set"]))
        else:
            print("hoti: Q_corner = %.3f e on the (%.3f, %.3f) [%s] axis, P=0  (%s)"
                  % (Qc, out["centre"][0], out["centre"][1], out["centre_atom"],
                     "CERTIFIED: scalar-relativistic, canonical C_%d eigenvalue labels"
                     % n if out["hsp_gapped"] else
                     "UNCERTIFIED: occupied C_%d eigenvalues not clean (dev=%.2f)"
                     % (n, out["hsp_dev"])))
    return comm.bcast(out, root=0)
