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


def rotation_map(frac, species, Mfrac, symprec=1e-2):
    """Atom permutation perm[a]=b induced by the rotation Mfrac (about the origin)."""
    n = len(frac)
    perm = -np.ones(n, dtype=int)
    for a in range(n):
        img = (Mfrac @ frac[a, :2])
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


def rotation_operator(m, kf, names, atom_of, frac, perm, Mrec):
    """C_m rotation operator at momentum kf in the lm [up|down] basis."""
    theta = 2*np.pi/m
    norb = len(names)
    O = _build_Ofull(names, atom_of, perm, theta)
    Mk = Mrec @ np.asarray(kf, float)
    # phase per column j (atom a): exp(2 pi i [ (Mrec k).f_b - k.f_a ]), b = perm[a]
    ph = np.array([np.exp(2j*np.pi*(Mk @ frac[perm[atom_of[j]], :2]
                                    - np.asarray(kf) @ frac[atom_of[j], :2]))
                   for j in range(norb)])
    D = O * ph[None, :]
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


def _count(eigs, m):
    """Multiplicities of the C_m eigenvalues, snapped to the m-th roots of -1
    (spinful) with the spinless labels p=1..m (Pi_p = e^{2 pi i (p-1)/m}) recovered
    by factoring the global phase e^{i pi/m}."""
    # spinful allowed eigenvalues and their spinless label p (1-based)
    allowed = [np.exp(1j*np.pi*(2*j+1)/m) for j in range(m)]        # m-th roots of -1
    label = [((j) % m) + 1 for j in range(m)]                       # e^{i pi/m} * w^{j} -> p=j+1
    mult = {p: 0 for p in range(1, m+1)}
    dev = 0.0
    for e in eigs:
        k = int(np.argmin([abs(e-a) for a in allowed]))
        dev = max(dev, abs(e-allowed[k]))
        mult[label[k]] += 1
    return mult, dev


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
        v = (2/3) * (inv["K"][1] + 2*inv["K"][2]); return (v, v)
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
    ``polarization`` (p1,p2 in e; must be ~0), ``corner_charge`` (nominal, e; the
    natural spinful mapping), ``corner_charge_set`` (all values over the spinful
    relabeling), ``certified`` (False for spin-orbit input), ``residual`` (max
    [H,C_n] at Gamma / bandwidth).
    """
    from mpi4py import MPI
    from ..hamiltonian.do_j_to_lm import j_to_lm_hamiltonian, lm_basis_labels

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    arry, attr = data_controller.data_dicts()
    if not attr.get("dftSO", False):
        raise RuntimeError("hoti_characterize requires a fully-relativistic (dftSO) run.")

    nelec = int(round(attr["nelec"]))
    nocc = nelec if nbnd_occ == "auto" else int(nbnd_occ)

    if not is_lm:
        stash = {k: np.copy(arry[k]) for k in ("HRs", "Hks") if k in arry}
        stash_basis = arry.get("basis")
        j_to_lm_hamiltonian(data_controller)
    labels = lm_basis_labels(data_controller)
    fname = "hoti_lm_HRs.dat"
    data_controller.write_HRs(fname)
    if not is_lm:
        for k, v in stash.items():
            arry[k] = v
        if stash_basis is not None:
            arry["basis"] = stash_basis

    out = dict(n=None, lattice=None, multiplicities={}, invariants={},
               polarization=None, corner_charge=None, corner_charge_set=None,
               certified=False, residual=None, nocc=nocc, error=None)
    if rank != 0:
        return comm.bcast(None, root=0)

    hr = join(attr["opath"], fname)
    num_wann, R_list, degen, H = read_hr(hr)
    nawf = num_wann
    names = [_orbital(l) for l in labels[:nawf // 2]]
    atom_of = [_atom_index(l) for l in labels[:nawf // 2]]
    frac = fractional_coords(arry["tau"], arry["a_vectors"], attr["alat"])
    species = list(arry["atoms"])

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
    n = None
    for cand in ((6, 4, 3, 2) if lat == "hex" else (4, 2)):
        if (lat, cand) not in _M:
            continue
        Mf = _M[(lat, cand)]
        perm = rotation_map(frac, species, Mf, symprec)
        if perm is None:
            continue
        Mrec = np.linalg.inv(Mf).T
        C = rotation_operator(cand, [0.0, 0.0], names, atom_of, frac, perm, Mrec)
        res = np.abs(HG @ C - C @ HG).max() / bandwidth
        if res < 1e-3:
            n = cand; out["residual"] = float(res)
            break
    if n is None:
        out["error"] = ("no out-of-plane C_n rotation (only in-plane C2 / mirrors) "
                        "-- rotation corner-charge indicator not applicable")
        if verbose:
            print("hoti: %s" % out["error"])
        return comm.bcast(out, root=0)
    out["n"] = n
    Mf = _M[(lat, n)]; Mrec = np.linalg.inv(Mf).T
    perm_n = rotation_map(frac, species, Mf, symprec)
    if verbose:
        print("hoti: lattice=%s, principal rotation C%d, [H,C%d]/W=%.2e at Gamma"
              % (lat, n, n, out["residual"]))

    # --- eigenvalue multiplicities at Gamma and each HSP ----------------------
    def mult_at(kf, m):
        Mfm = _M[(lat, m)]; Mrm = np.linalg.inv(Mfm).T
        permm = rotation_map(frac, species, Mfm, symprec)
        C = rotation_operator(m, kf, names, atom_of, frac, permm, Mrm)
        Hm = _Hk(H, R_list, kf, nawf)
        r = np.abs(Hm @ C - C @ Hm).max() / bandwidth
        mlt, dev = _count(_occ_rot_eigs(Hm, C, nocc), m)
        return mlt, r, dev

    table = _hsp_table(lat, n)
    m_orders = sorted({m for _, m in table.values()})
    gamma = {m: mult_at([0.0, 0.0], m)[0] for m in m_orders}
    out["multiplicities"]["Gamma"] = {m: gamma[m] for m in m_orders}
    inv = {}
    for name, (kf, m) in table.items():
        mlt, r, dev = mult_at(kf, m)
        out["multiplicities"][name] = mlt
        inv[name] = {p: mlt[p] - gamma[m][p] for p in mlt}     # [Pi_p^(m)]
        if verbose:
            print("hoti: %-3s (C%d)  mult=%s  [Pi_p]=%s  ([H,C]/W=%.1e, dev=%.0e)"
                  % (name, m, dict(mlt), {p: inv[name][p] for p in inv[name]}, r, dev))
    out["invariants"] = inv

    # --- polarization (must vanish) and nominal corner charge -----------------
    p1, p2 = _polarization(n, inv)
    out["polarization"] = (float(p1 % 1.0), float(p2 % 1.0))
    Qc = _corner_charge(n, inv) % 1.0
    out["corner_charge"] = float(Qc)
    # spinful: report the set over the 3-(or m-)fold relabeling of p
    m_main = max(m_orders)
    shifts = range(m_main)
    qset = set()
    for sh in shifts:
        inv_sh = {name: {p: (inv[name][((p - 1 + sh) % len(inv[name])) + 1]) for p in inv[name]}
                  for name in inv}
        try:
            qset.add(round(_corner_charge(n, inv_sh) % 1.0, 6))
        except Exception:
            pass
    out["corner_charge_set"] = sorted(qset)
    out["certified"] = False    # spinful: HSP eigenvalues do not fix the WCs

    if verbose:
        pol_ok = abs(p1 % 1.0) < 1e-6 and abs(p2 % 1.0) < 1e-6
        print("hoti: polarization P=(%.3f, %.3f) e  %s"
              % (p1 % 1.0, p2 % 1.0, "(vanishes: corner charge well defined)"
                 if pol_ok else "(NONZERO: corner charge ill-defined until P removed)"))
        print("hoti: nominal Q_corner = %.3f e ;  spinful set = %s e  (NOT certified: "
              "HSP eigenvalues do not fix the spinful corner charge)"
              % (Qc, out["corner_charge_set"]))
    return comm.bcast(out, root=0)
