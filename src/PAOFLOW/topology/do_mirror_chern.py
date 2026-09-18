"""
do_mirror_chern.py
==================
Mirror Chern number C_M of a 2D material from the PAOFLOW Hamiltonian, using
Z2Pack for the sector Chern numbers.

For a 2D crystal the only mirror whose invariant plane fills the whole Brillouin
zone is the HORIZONTAL mirror sigma_h (M_z : z -> -z); a vertical mirror leaves
invariant only a 1D line.  When M_z commutes with H(k) it block-diagonalizes it
into the two eigen-sectors M_z = +-i (spin-1/2), each an ordinary Chern insulator
with Chern numbers C_{+i} = -C_{-i} (time reversal).  The mirror Chern number and
the Z2 index are

    C_M = (C_{+i} - C_{-i}) / 2 = C_{+i},     nu = C_M mod 2.

C_M != 0 with nu = 0 is a *pure* topological crystalline insulator (e.g. monolayer
SnTe, |C_M| = 2); C_M odd is a QSHI that is also mirror-nontrivial ("dual", e.g.
Ir2O2 with C_M = 3).

Method
------
After ``PAOFLOW.j_to_lm_hamiltonian`` the Hamiltonian is in the QE real
spherical-harmonic (lm) basis, ``[spin-up | spin-down]`` order, where M_z is the
CONSTANT operator

    M_z = P_site (x) diag(eta_z) (x) diag(-i_up, +i_down),   M_z^2 = -1,

with orbital z-parity eta_z = -1 for the z-odd harmonics {p_z, d_xz, d_yz} and +1
otherwise, and P_site the atom permutation induced by z -> -z (identity for a
planar layer; the top<->bottom half-layer swap for a buckled X-M-M-X layer).  We
auto-detect P_site from the relaxed coordinates, verify [H(k), M_z] = 0,
diagonalize M_z into its +-i eigenspaces, rotate each sector's Hamiltonian and
feed it to Z2Pack.

Fallbacks / limits
------------------
* No sigma_h (e.g. a centrosymmetric D3d layer): the search fails and, if
  requested, the Z2 index is computed over half the BZ instead (C_M undefined).
* A glide sigma_h (nonsymmorphic, with an in-plane fractional translation tau) is
  detected and reported; the sector Chern for a glide needs k-dependent
  eigenspaces and is not computed here (symmorphic sigma_h only).
* Only s, p, d shells (the ``j_to_lm`` limit).

Requires ``z2pack`` and ``tbmodels`` (imported lazily, only when the Chern step
runs).  Intended for serial execution (Z2Pack drives its own k-loop).
"""

from __future__ import annotations

import re
from os.path import join

import numpy as np

Z_ODD = {"pz", "dzx", "dzy"}          # z-odd real harmonics -> eta_z = -1
_SITE_RE = re.compile(r"(\d+)$")


# --------------------------------------------------------------------------- #
#  lm-label parsing
# --------------------------------------------------------------------------- #
def _orbital(label):      # 'Ti2_dzx#2_up' -> 'dzx'
    return label.split("_")[1].split("#")[0]


def _orbital_full(label):  # 'Ti2_dzx#2_up' -> 'dzx#2'
    return label.split("_")[1]


def _spin(label):          # -> 'up' / 'down'
    return label.split("_")[-1]


def _atom_index(label):    # 'Ti2_dzx#2_up' -> 2
    m = _SITE_RE.search(label.split("_")[0])
    if not m:
        raise ValueError("no atom index in label %r" % label)
    return int(m.group(1))


# --------------------------------------------------------------------------- #
#  geometry: sigma_h atom permutation (+ optional glide translation)
# --------------------------------------------------------------------------- #
def fractional_coords(tau, a_vectors, alat):
    a_cart = np.asarray(a_vectors, float) * float(alat)
    return np.asarray(tau, float) @ np.linalg.inv(a_cart)


def site_permutation(frac, species, symprec=1e-2, perp=2):
    """Atom permutation induced by sigma_h (z -> 2*z0 - z, + in-plane tau).

    Returns (perm, z0, tau) with perm[a] = image atom of a, z0 the mirror plane
    (fractional), tau the in-plane fractional translation (0 for a symmorphic
    sigma_h).  Raises if no sigma_h maps the atom set onto itself.
    """
    frac = np.asarray(frac, float)
    n = len(frac)
    inplane = [i for i in range(3) if i != perp]
    z0 = frac[:, perp].mean()

    def try_tau(tau):
        perm = -np.ones(n, dtype=int)
        for a in range(n):
            img = frac[a].copy()
            img[perp] = 2 * z0 - img[perp]
            img[inplane] = img[inplane] + tau
            for b in range(n):
                if species[b] != species[a]:
                    continue
                d = img - frac[b]
                d -= np.round(d)
                if np.max(np.abs(d)) < symprec:
                    perm[a] = b
                    break
            if perm[a] < 0:
                return None
        if not np.array_equal(perm[perm], np.arange(n)):
            return None
        return perm

    # candidate in-plane translations: 0 (symmorphic) first, then atom-0 pairings
    a0 = 0
    img0 = frac[a0].copy()
    img0[perp] = 2 * z0 - img0[perp]
    cand = [np.zeros(2)]
    for b0 in range(n):
        if species[b0] == species[a0]:
            t = frac[b0, inplane] - img0[inplane]
            t -= np.round(t)
            cand.append(t)
    for tau in cand:
        perm = try_tau(tau)
        if perm is not None:
            return perm, z0, np.asarray(tau, float)
    raise ValueError(
        "no horizontal mirror sigma_h maps the atom set onto itself "
        "(symprec=%g): the layer is not sigma_h-symmetric." % symprec
    )


# --------------------------------------------------------------------------- #
#  mirror operator and eigenbasis
# --------------------------------------------------------------------------- #
def mirror_operator(labels, perm):
    """Constant M_z (nawf x nawf) in the lm [up|down] basis (symmorphic sigma_h)."""
    n = len(labels)
    key = {(_atom_index(l), _orbital_full(l), _spin(l)): j for j, l in enumerate(labels)}
    Mz = np.zeros((n, n), dtype=complex)
    for j, l in enumerate(labels):
        eta = -1 if _orbital(l) in Z_ODD else +1
        spin = -1j if _spin(l) == "up" else +1j
        i = key[(perm[_atom_index(l)], _orbital_full(l), _spin(l))]
        Mz[i, j] = eta * spin
    return Mz


def mirror_eigenbasis(Mz, tol=1e-6):
    """Orthonormal bases (U_plus, U_minus) of the M_z = +i / -i eigenspaces.

    Diagonalizes the Hermitian A = i*M_z (eigenvalue -1 <-> M_z=+i, +1 <-> -i).
    """
    A = 1j * np.asarray(Mz)
    if np.max(np.abs(A - A.conj().T)) > 1e-8:
        raise ValueError("i*M_z is not Hermitian; check the mirror operator.")
    w, V = np.linalg.eigh(A)
    if np.max(np.abs(np.abs(w) - 1.0)) > tol:
        raise ValueError("M_z eigenvalues are not +-i.")
    plus, minus = w < 0, w > 0
    if plus.sum() != minus.sum():
        raise ValueError("mirror sectors are unbalanced (%d vs %d)." % (plus.sum(), minus.sum()))
    return V[:, plus], V[:, minus]


def check_commutator(R_list, H, Mz):
    """max |[H(R), M_z]| and its ratio to the bandwidth (feasibility gate)."""
    max_comm = max_all = 0.0
    for R in R_list:
        HR = H[R]
        max_all = max(max_all, np.abs(HR).max())
        max_comm = max(max_comm, np.abs(HR @ Mz - Mz @ HR).max())
    return {"max_comm": max_comm, "max_all": max_all,
            "ratio": max_comm / max_all if max_all else np.nan}


# --------------------------------------------------------------------------- #
#  Wannier90 / Z2Pack hr.dat I/O
# --------------------------------------------------------------------------- #
def read_hr(fname):
    """Parse a Wannier90-style hr.dat -> (num_wann, R_list, degen, H_dict)."""
    with open(fname) as f:
        f.readline()
        num_wann = int(f.readline().split()[0])
        nrpts = int(f.readline().split()[0])
        degen = []
        while len(degen) < nrpts:
            degen += [int(x) for x in f.readline().split()]
        R_list, H = [], {}
        ndata, read = nrpts * num_wann * num_wann, 0
        while read < ndata:
            line = f.readline()
            if line == "":
                raise EOFError("hr.dat ended early (%d/%d)" % (read, ndata))
            tok = line.split()
            if not tok:                       # PAOFLOW appends a blank line when nrpts%15==0
                continue
            Rx, Ry, Rz, m, nn = (int(t) for t in tok[:5])
            R = (Rx, Ry, Rz)
            if R not in H:
                H[R] = np.zeros((num_wann, num_wann), dtype=complex)
                R_list.append(R)
            H[R][m - 1, nn - 1] = float(tok[5]) + 1j * float(tok[6])
            read += 1
    return num_wann, R_list, degen, H


def rotate_sector_hr(R_list, degen, H, U, fname):
    """Write H_sec(R) = U^dag H(R) U as a stand-alone hr.dat."""
    Ud, nsec = U.conj().T, U.shape[1]
    with open(fname, "w") as f:
        f.write("PAOFLOW mirror-sector Hamiltonian\n")
        f.write("%5d\n%5d\n" % (nsec, len(R_list)))
        for i, d in enumerate(degen):
            f.write("%5d" % d)
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(degen) % 15 != 0:
            f.write("\n")
        for R in R_list:
            sub = Ud @ H[R] @ U
            for c in range(nsec):
                for r in range(nsec):
                    f.write("%3d %3d %3d %5d %5d %28.14f %28.14f\n"
                            % (R[0], R[1], R[2], r + 1, c + 1, sub[r, c].real, sub[r, c].imag))
    return fname


# --------------------------------------------------------------------------- #
#  Z2Pack helpers (lazy import)
# --------------------------------------------------------------------------- #
def _load_model(hr_file):
    import tbmodels
    return tbmodels.Model.from_wannier_files(hr_file=hr_file)


def _chern(hr_file, occ, skw):
    import z2pack
    model = _load_model(hr_file)
    res = z2pack.surface.run(system=z2pack.tb.System(model, bands=occ),
                             surface=lambda s, t: [s, t, 0.0], **skw)
    return z2pack.invariant.chern(res)


def _z2_half(hr_file, occ, skw):
    import z2pack
    model = _load_model(hr_file)
    res = z2pack.surface.run(system=z2pack.tb.System(model, bands=occ),
                             surface=lambda s, t: [s / 2, t, 0.0], **skw)
    return z2pack.invariant.z2(res)


def _min_gap(hr_file, occ, nk=24):
    model = _load_model(hr_file)
    ks = np.linspace(0, 1, nk, endpoint=False)
    dg, cbm, vbm = np.inf, np.inf, -np.inf
    for kx in ks:
        for ky in ks:
            ev = np.sort(np.linalg.eigvalsh(model.hamilton([kx, ky, 0.0])))
            dg = min(dg, ev[occ] - ev[occ - 1]); vbm = max(vbm, ev[occ - 1]); cbm = min(cbm, ev[occ])
    return dg, cbm - vbm


def _auto_tighten(gap, skw, user_keys):
    """Tighten the Z2Pack surface sampling for a small (min direct) gap.

    A small direct gap makes the Wannier charge centers swing between neighbouring
    loops; z2pack then tries to insert loops and is blocked by
    ``min_neighbour_dist`` (the ``'min_neighbour_dist reached'`` warnings and the
    failed Gap/Move checks).  These tiers lower ``min_neighbour_dist`` / ``move_tol``
    and densify the loops accordingly.  Only keys the user did NOT pass in
    ``surface_kwargs`` (``user_keys``) are set, so an explicit override always wins.
    Returns ``(skw, applied)``.
    """
    if gap is None or gap >= 0.20:
        return skw, {}
    if gap >= 0.10:
        p = dict(min_neighbour_dist=1e-3, move_tol=0.20, gap_tol=0.25,
                 iterator=range(15, 151, 6), num_lines=15)
    elif gap >= 0.05:
        p = dict(min_neighbour_dist=1e-4, move_tol=0.15, gap_tol=0.20, pos_tol=2e-3,
                 iterator=range(21, 201, 10), num_lines=21)
    else:
        p = dict(min_neighbour_dist=1e-5, move_tol=0.10, gap_tol=0.15, pos_tol=1e-3,
                 iterator=range(31, 301, 10), num_lines=31)
    applied = {k: v for k, v in p.items() if k not in user_keys}
    skw2 = dict(skw); skw2.update(applied)
    return skw2, applied


# --------------------------------------------------------------------------- #
#  main entry point
# --------------------------------------------------------------------------- #
def do_mirror_chern(data_controller, nbnd_occ="auto", z2pack=True, is_lm=False,
                    symprec=1e-2, surface_kwargs=None, gap_check=True,
                    auto_tighten=True, z2_fallback=True, verbose=True):
    """Compute the mirror Chern number C_M (see module docstring).

    When ``auto_tighten`` is set (default), the min direct gap is measured first
    and, if small (< 0.20 eV), the Z2Pack surface sampling is tightened by tiers
    (lower ``min_neighbour_dist`` / ``move_tol``, denser loops) -- only for keys
    the caller did not pass in ``surface_kwargs``, which always take precedence.

    Returns a dict with keys: ``sigma_h``, ``glide``, ``perm``, ``z0``, ``tau``,
    ``residual`` (max|[H,M_z]|/bandwidth), ``nawf``, ``nocc``, ``gap`` (indirect),
    ``gap_direct``, ``C_plus``, ``C_minus``, ``C_M``, ``nu`` (and ``nu_z2`` in the
    no-sigma_h fallback).  Values that were not computed are ``None``.
    """
    from mpi4py import MPI
    from ..hamiltonian.do_j_to_lm import j_to_lm_hamiltonian, lm_basis_labels

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    arry, attr = data_controller.data_dicts()

    if not attr.get("dftSO", False):
        raise RuntimeError("mirror_chern_number requires a fully-relativistic (dftSO) run.")

    nelec = int(round(attr["nelec"]))
    nocc = nelec if nbnd_occ == "auto" else int(nbnd_occ)
    skw = dict(pos_tol=1e-2, iterator=range(11, 61, 4), load=False)
    if surface_kwargs:
        skw.update(surface_kwargs)

    # ---- lm Hamiltonian on disk (non-destructive) ---------------------------
    if not is_lm:
        stash = {k: np.copy(arry[k]) for k in ("HRs", "Hks") if k in arry}
        stash_basis = arry.get("basis")
        j_to_lm_hamiltonian(data_controller)
    labels = lm_basis_labels(data_controller)
    fname = "mirror_chern_lm_HRs.dat"
    data_controller.write_HRs(fname)
    if not is_lm:
        for k, v in stash.items():
            arry[k] = v
        if stash_basis is not None:
            arry["basis"] = stash_basis

    out = dict(sigma_h=False, glide=False, perm=None, z0=None, tau=None,
               residual=None, nawf=len(labels), nocc=nocc, gap=None, gap_direct=None,
               C_plus=None, C_minus=None, C_M=None, nu=None, nu_z2=None)

    if rank != 0:
        return comm.bcast(None, root=0)

    opath = attr["opath"]
    hr_lm = join(opath, fname)

    # ---- gap (drives auto-tightening of the Z2Pack sampling) ----------------
    dg = None
    if gap_check or (auto_tighten and z2pack):
        try:
            dg, ig = _min_gap(hr_lm, nocc)
            out["gap"], out["gap_direct"] = float(ig), float(dg)
            if verbose:
                print("mirror_chern: gap (fill %d) direct=%.4f indirect=%.4f eV"
                      % (nocc, dg, ig))
        except Exception as e:
            if verbose:
                print("mirror_chern: gap not computed (%s)" % e)
    if auto_tighten and z2pack:
        skw, applied = _auto_tighten(dg, skw, set((surface_kwargs or {}).keys()))
        if applied and verbose:
            print("mirror_chern: small direct gap (%.3f eV) -> auto-tightened z2pack: %s"
                  % (dg, ", ".join("%s=%s" % (k, v) for k, v in applied.items())))

    # ---- geometry: sigma_h permutation --------------------------------------
    frac = fractional_coords(arry["tau"], arry["a_vectors"], attr["alat"])
    species = list(arry["atoms"])
    try:
        perm, z0, tau = site_permutation(frac, species, symprec=symprec)
        out.update(sigma_h=True, perm=perm.tolist(), z0=float(z0), tau=tau.tolist())
        out["glide"] = bool(np.max(np.abs(tau)) > symprec)
    except ValueError as e:
        if verbose:
            print("mirror_chern: %s" % e)
        perm = None

    # ---- no sigma_h: optional Z2 fallback -----------------------------------
    if perm is None:
        if verbose:
            print("mirror_chern: no sigma_h -> C_M undefined.")
        if z2pack and z2_fallback:
            out["nu_z2"] = int(_z2_half(hr_lm, nocc, skw))
            if verbose:
                print("mirror_chern: Z2 (half BZ) = %d" % out["nu_z2"])
        return comm.bcast(out, root=0)

    # ---- build M_z, feasibility gate ----------------------------------------
    num_wann, R_list, degen, H = read_hr(hr_lm)
    Mz = mirror_operator(labels, perm)
    diag = check_commutator(R_list, H, Mz)
    out["residual"] = float(diag["ratio"])
    if verbose:
        print("mirror_chern: sigma_h z0=%.4f perm=%s%s" %
              (z0, perm.tolist(), "  (GLIDE tau=%s)" % tau.tolist() if out["glide"] else ""))
        print("mirror_chern: [H,M_z] residual = %.2e (bandwidth %.1f eV)"
              % (diag["ratio"], diag["max_all"]))
    if diag["ratio"] > 1e-3:
        print("mirror_chern: WARNING large mirror-breaking residual; C_M unreliable.")

    if out["glide"]:
        print("mirror_chern: glide sigma_h (nonsymmorphic) -> sector Chern not "
              "implemented (needs k-dependent eigenspaces); returning diagnostics only.")
        return comm.bcast(out, root=0)

    # ---- split, Chern per sector --------------------------------------------
    U_plus, U_minus = mirror_eigenbasis(Mz)
    hr_p = join(opath, "mirror_chern_Mz+i.dat")
    hr_m = join(opath, "mirror_chern_Mz-i.dat")
    rotate_sector_hr(R_list, degen, H, U_plus, hr_p)
    rotate_sector_hr(R_list, degen, H, U_minus, hr_m)

    if z2pack:
        occ_sec = nocc // 2
        out["C_plus"] = float(_chern(hr_p, occ_sec, skw))
        out["C_minus"] = float(_chern(hr_m, occ_sec, skw))
        out["C_M"] = 0.5 * (out["C_plus"] - out["C_minus"])
        out["nu"] = int(round(out["C_M"])) % 2
        if verbose:
            print("mirror_chern: C(+i)=%+.3f C(-i)=%+.3f  ->  C_M=%+.3f  nu=%d"
                  % (out["C_plus"], out["C_minus"], out["C_M"], out["nu"]))
    elif verbose:
        print("mirror_chern: wrote sector Hamiltonians (z2pack=False); "
              "run z2pack.invariant.chern on %s / %s." % (hr_p, hr_m))

    return comm.bcast(out, root=0)
