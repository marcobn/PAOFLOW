"""Shared Bloch / PAO-gauge vertex machinery for the electron-phonon AO route.

Provides the geometry reader and the interpolation primitives used by
:mod:`PAOFLOW.elphon.do_ao_eph` to turn QE's coarse-grid Cartesian coupling into
the dense-grid Eliashberg properties:

* :func:`read_nscf` -- k-points, eigenvalues, Fermi level and lattice from the QE
  ``data-file-schema.xml``;
* :func:`kq_index_map` -- ``k -> (index of k+q on the grid, umklapp G0)``;
* :func:`vertex_pao_R` -- rotate the band-basis Cartesian coupling into the PAO
  gauge ``A_{k+q}^dagger d A_k`` and Fourier-transform it to the electron
  real-space cells ``g(R_e)``;
* :func:`lambda_q_dense_ws` -- Wigner-Seitz interpolate the electrons (``HRs``)
  and the vertex to a dense grid and evaluate the Fermi-surface double delta ->
  ``lambda_{q nu}``.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HARTREE_TO_RY = 2.0
RY_TO_THZ = 3289.842
RY_TO_EV = 13.605693122994
AMU_RY = 911.4442421


def read_nscf(save_dir):
    """Read k-points, eigenvalues, Fermi level and lattice from ``data-file-schema.xml``.

    Returns
    -------
    dict
        ``{'kpts_cart', 'kpts_cryst', 'eigs_ry', 'ef_ry', 'nbnd', 'nk',
        'bg', 'at', 'alat', 'omega', 'fft'}``.  ``kpts_cart`` in 2*pi/alat,
        ``kpts_cryst`` in reciprocal-lattice fractions, energies in Ry (referred
        to E_F), ``bg`` reciprocal vectors (rows, 2*pi/alat), ``at`` real vectors
        (rows, alat).
    """
    root = ET.parse(os.path.join(save_dir, 'data-file-schema.xml')).getroot()
    out = root.find('output')

    alat = float(out.find('atomic_structure').attrib['alat'])
    cell = out.find('atomic_structure/cell')
    at = np.array([[float(x) for x in cell.findtext(v).split()] for v in ('a1', 'a2', 'a3')]) / alat
    rl = out.find('basis_set/reciprocal_lattice')
    bg = np.array([[float(x) for x in rl.findtext(v).split()] for v in ('b1', 'b2', 'b3')])
    omega = abs(np.dot(at[0], np.cross(at[1], at[2]))) * alat**3

    # atomic positions (Bohr in the schema) -> crystal coordinates, and species.
    astruct = out.find('atomic_structure')
    atoms = astruct.findall('atomic_positions/atom')
    tau_cart = np.array([[float(x) for x in a.text.split()] for a in atoms]) / alat  # alat units
    tau_cryst = tau_cart @ np.linalg.inv(at)  # tau_alat = s @ at  ->  s = tau_alat at^{-1}
    atom_names = [a.attrib['name'] for a in atoms]
    species = {
        sp.attrib['name']: sp.findtext('pseudo_file').strip()
        for sp in out.findall('atomic_species/species')
    }

    fft_el = out.find('basis_set/fft_grid')
    fft = (int(fft_el.attrib['nr1']), int(fft_el.attrib['nr2']), int(fft_el.attrib['nr3']))

    ef_ha = out.find('band_structure/fermi_energy')
    ef_ry = float(ef_ha.text) * HARTREE_TO_RY

    ks = out.findall('band_structure/ks_energies')
    nk = len(ks)
    kpts_cart = np.array([[float(x) for x in k.findtext('k_point').split()] for k in ks])
    eigs = [np.array(k.findtext('eigenvalues').split(), dtype=float) for k in ks]
    nbnd = len(eigs[0])
    eigs_ry = np.array(eigs) * HARTREE_TO_RY - ef_ry  # (nk, nbnd), referred to E_F

    # crystal coords: k_cart = sum_i k_cryst_i bg_i  ->  k_cryst = (bg^T)^{-1} k_cart
    kpts_cryst = np.linalg.solve(bg.T, kpts_cart.T).T
    kpts_cryst = kpts_cryst - np.round(kpts_cryst - 0.5 + 1e-9)  # fold into [0,1)

    return {
        'kpts_cart': kpts_cart,
        'kpts_cryst': kpts_cryst,
        'eigs_ry': eigs_ry,
        'ef_ry': 0.0,
        'nbnd': nbnd,
        'nk': nk,
        'bg': bg,
        'at': at,
        'alat': alat,
        'omega': omega,
        'fft': fft,
        'tau_cryst': tau_cryst,
        'atom_names': atom_names,
        'species': species,
    }


def kq_index_map(kpts_cryst, q_cryst, nkgrid, tol=1.0e-5):
    """Map ``k -> (index of k+q on the grid, umklapp G0)``.

    Parameters
    ----------
    kpts_cryst : ndarray ``(nk, 3)``
        k-points in crystal coordinates.
    q_cryst : ndarray ``(3,)``
        q-point in crystal coordinates.
    nkgrid : tuple(int, int, int)
        Monkhorst-Pack grid dimensions (for snapping to integer grid indices).

    Returns
    -------
    ikq : ndarray ``(nk,)`` int
        Index of the grid point equal to ``k + q`` modulo a reciprocal vector.
    G0 : ndarray ``(nk, 3)`` int
        The umklapp vector ``G0 = (k + q) - k'`` in reciprocal-lattice units.
    """
    nk = len(kpts_cryst)
    ng = np.array(nkgrid)
    # integer grid labels of each k
    kint = np.round(kpts_cryst * ng).astype(int) % ng
    lookup = {tuple(kint[i]): i for i in range(nk)}
    ikq = np.empty(nk, dtype=int)
    G0 = np.empty((nk, 3), dtype=int)
    for i in range(nk):
        kq = kpts_cryst[i] + q_cryst
        kqint = np.round(kq * ng).astype(int) % ng
        j = lookup[tuple(kqint)]
        ikq[i] = j
        G0[i] = np.round(kq - kpts_cryst[j]).astype(int)
    return ikq, G0


# --------------------------------------------------------------------------- #
# EPW-style electron interpolation of the vertex (dense k, k+q).
# --------------------------------------------------------------------------- #


def vertex_pao_R(d, A, ikq, kgrid_idx, ng):
    """PAO-gauge vertex in real space ``g(R_e)`` for one q (EPW electron transform).

    Rotates the band-basis deformation potentials to the PAO gauge,
    ``g^{PAO}_{ij,c}(k) = (A_{k+q}^\\dagger d_c(k) A_k)_{ij}``, places them on the
    coarse k-grid and Fourier-transforms to the electron cells ``R_e``.  The
    transform matches PAOFLOW's ``HRs`` convention so the result can be
    interpolated with :func:`PAOFLOW.elphon.eph_kq.estates_on_grid`.

    Parameters
    ----------
    d : ndarray ``(nk, nbnd, nbnd, ncart)``
        Band-basis Cartesian deformation potentials ``d_{mn,c}(k)`` (QE's
        ``el_ph_mat`` / ``ahc_gkk`` on the coarse k-grid).
    A : ndarray ``(nbnd, nawf, nk)``
        PAO projections ``A_{n i}(k) = <psi_{nk}|phi_i>`` (``arry['U'][..., ispin]``).
    ikq : ndarray ``(nk,)``
        Index of ``k+q`` on the coarse grid.
    kgrid_idx : ndarray ``(nk, 3)`` int
        Integer grid labels ``round(k_cryst * ng) % ng`` of each k-point.
    ng : tuple(int, int, int)
        Coarse k-grid dimensions.

    Returns
    -------
    ndarray ``(nawf, nawf, ncart, n1, n2, n3)`` complex
        The PAO-gauge vertex in the electron real-space cells.
    """
    nk, nbnd, _, ncart = d.shape
    nawf = A.shape[1]
    # Batched PAO rotation g^{PAO}(k) = A_{k+q}^dagger d(k) A_k over all k at once
    # (BLAS-backed einsum) instead of a Python loop over k-points.
    Ak = np.transpose(A, (2, 0, 1))  # (nk, nbnd, nawf)
    Akq = np.transpose(A[:, :, ikq], (2, 0, 1))  # (nk, nbnd, nawf)
    gk_b = np.einsum('kmi,kmnc,knj->kijc', Akq.conj(), d, Ak, optimize=True)  # (nk,nawf,nawf,ncart)
    gk = np.zeros((nawf, nawf, ncart, ng[0], ng[1], ng[2]), dtype=complex)
    i1, i2, i3 = kgrid_idx[:, 0], kgrid_idx[:, 1], kgrid_idx[:, 2]
    # each k maps to a unique coarse-grid cell, so this scatter is unambiguous
    gk[:, :, :, i1, i2, i3] = np.moveaxis(gk_b, 0, -1)  # (nawf, nawf, ncart, nk)
    # k -> R (inverse of estates_on_grid's ifftn*N convention).
    gR = np.fft.fftn(gk, axes=(3, 4, 5)) / (ng[0] * ng[1] * ng[2])
    return gR


def _fermi_level(E_flat, sigma, nelec, spin_deg=2.0, niter=120):
    """Fermi level for ``nelec`` electrons on a uniform k-grid (gaussian smearing).

    ``E_flat`` are the ``(nk, nbnd)`` eigenvalues (Ry, any reference); returns the
    Fermi level in the same reference via bisection of the smeared occupation
    ``f = 0.5 erfc((E - E_F)/sigma)`` (matches QE ``efermig`` with ``ngauss=0``).
    """
    from scipy.special import erfc

    nk = E_flat.shape[0]
    lo, hi = float(E_flat.min()) - 1.0, float(E_flat.max()) + 1.0
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        occ = spin_deg / nk * np.sum(0.5 * erfc((E_flat - mid) / sigma))
        if occ > nelec:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _ws_lattice(ng, at, span=2):
    """Wigner-Seitz images of the electron real-space cells (single site at origin).

    Vectorised equivalent of QE ``wsinit``/``wsweight`` for the lattice: every
    integer cell ``n`` in ``[-span*ng, span*ng]`` is kept if it lies in (or on the
    boundary of) the Wigner-Seitz cell of the ``ng``-supercell, with degeneracy
    weight ``1/deg``.

    Returns
    -------
    Nint : ndarray ``(nws, 3)`` int
        Integer cells ``n`` (Bloch phase ``exp(2 pi i k.n)``).
    W : ndarray ``(nws,)``
        Wigner-Seitz degeneracy weights.
    Midx : ndarray ``(nws, 3)`` int
        Grid indices ``n mod ng`` into the real-space arrays.
    """
    ng = np.asarray(ng, dtype=int)
    # candidate integer cells
    axes = [np.arange(-span * ng[d], span * ng[d] + 1) for d in range(3)]
    N = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    R = N @ at  # cartesian (alat)
    # supercell lattice vectors defining the WS cell of the ng-supercell
    sup = [np.arange(-span, span + 1) for _ in range(3)]
    S = np.stack(np.meshgrid(*sup, indexing='ij'), axis=-1).reshape(-1, 3)
    S = S[np.any(S != 0, axis=1)]
    rws = (S * ng[None, :]) @ at  # (nsup, 3) cartesian supercell vectors
    half = 0.5 * np.einsum('ij,ij->i', rws, rws)
    proj = R @ rws.T - half[None, :]  # (ncand, nsup)
    tol = 1.0e-6
    inside = np.all(proj <= tol, axis=1)
    deg = 1 + np.count_nonzero(np.abs(proj) < tol, axis=1)
    keep = inside
    Nint = N[keep]
    W = 1.0 / deg[keep]
    Midx = Nint % ng
    return Nint, W, Midx.astype(int)


def lambda_q_dense_ws(
    gR,
    HRs,
    q_cryst,
    ng_coarse,
    at,
    zmass,
    freqs_thz,
    sigmas_ry,
    Nk,
    ispin=0,
    kblock=4096,
    nelec=None,
):
    """``lambda_{q nu}`` with a Wigner-Seitz (EPW-style) electron interpolation.

    Same as :func:`lambda_q_dense` but the vertex ``gR`` and the PAO Hamiltonian
    ``HRs`` are interpolated to the dense grid with explicit Wigner-Seitz real-space
    sums (proper zone-boundary degeneracy weights) instead of a plain zero-padding
    Fourier embed, and ``k+q`` is evaluated directly at ``(k+q)`` (no grid roll).

    Parameters
    ----------
    gR : ndarray ``(nawf, nawf, ncart, n1, n2, n3)``
        Vertex real-space cells; its grid may differ from ``HRs`` (e.g. a coarse
        6^3 coupling combined with a dense 18^3 electron Hamiltonian).
    HRs : ndarray ``(nawf, nawf, m1, m2, m3, nspin)``
    q_cryst : ndarray ``(3,)``
        q-point in crystal coordinates (exact, not restricted to the coarse grid).
    ng_coarse : ignored
        Kept for backward compatibility; the H and vertex grids are read from the
        array shapes so the two may differ.
    at : ndarray ``(3, 3)``
        Real-lattice vectors (rows, alat units) for the WS construction.
    zmass, freqs_thz, sigmas_ry, Nk : see :func:`lambda_q_dense`.

    Returns
    -------
    dict
        ``{'lambda_qnu', 'gamma_ghz', 'dos_ef', 'nk_dense'}``.
    """
    nawf = HRs.shape[0]
    ncart = gR.shape[2]
    nmode = zmass.shape[0]
    H = HRs[:, :, :, :, :, ispin]  # (nawf, nawf, m1, m2, m3), eV
    ng_H = H.shape[2:5]
    ng_g = gR.shape[3:6]

    # separate Wigner-Seitz lattices for the (possibly different) H and g grids
    NintH, WH, MidxH = _ws_lattice(ng_H, at)
    Nintg, Wg, Midxg = _ws_lattice(ng_g, at)
    Hn = np.transpose(H[:, :, MidxH[:, 0], MidxH[:, 1], MidxH[:, 2]], (2, 0, 1))  # (nwsH,nawf,nawf)
    Hn = Hn * WH[:, None, None]
    gn = np.transpose(
        gR[:, :, :, Midxg[:, 0], Midxg[:, 1], Midxg[:, 2]], (3, 0, 1, 2)
    )  # (nwsg,nawf,nawf,ncart)
    gn = gn * Wg[:, None, None, None]
    # flattened views for BLAS matmul interpolation (avoid non-BLAS einsum)
    Hn_flat = Hn.reshape(Hn.shape[0], -1)  # (nwsH, nawf*nawf)
    gn_flat = gn.reshape(gn.shape[0], -1)  # (nwsg, nawf*nawf*ncart)

    # dense k-grid in crystal coordinates
    ax = [np.arange(Nk) / Nk for _ in range(3)]
    K = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)  # (nkd,3)
    nkd = K.shape[0]

    sigmas = np.atleast_1d(sigmas_ry)
    omega_ry = np.abs(freqs_thz) / RY_TO_THZ
    safe = omega_ry > 1.0e-8
    ry_to_ghz = RY_TO_EV * 2.417989242e5

    num = np.zeros((sigmas.size, nmode))
    dsum = np.zeros(sigmas.size)
    # Optional: recompute the Fermi level on the dense interpolated grid (per QE
    # efermig) so the double-delta samples the Fermi surface at the grid-converged
    # E_F rather than the coarse-grid reference (HRs E_F = 0 is the coarse E_F).
    ef_sig = np.zeros(sigmas.size)
    if nelec is not None:
        Eall = np.empty((nkd, nawf))
        for s0 in range(0, nkd, kblock):
            Kb = K[s0 : s0 + kblock]
            phk = np.exp(2j * np.pi * (Kb @ NintH.T))
            Hk = (phk @ Hn_flat).reshape(Kb.shape[0], nawf, nawf)
            Hk = 0.5 * (Hk + np.conjugate(np.transpose(Hk, (0, 2, 1))))
            Eall[s0 : s0 + Kb.shape[0]] = np.linalg.eigvalsh(Hk) / RY_TO_EV
        for isig, sig in enumerate(sigmas):
            ef_sig[isig] = _fermi_level(Eall, sig, nelec)
    for s0 in range(0, nkd, kblock):
        Kb = K[s0 : s0 + kblock]
        phkH = np.exp(2j * np.pi * (Kb @ NintH.T))  # (nb, nwsH)
        phkqH = np.exp(2j * np.pi * ((Kb + q_cryst) @ NintH.T))
        phkg = np.exp(2j * np.pi * (Kb @ Nintg.T))  # (nb, nwsg)
        nb = Kb.shape[0]
        Hk = (phkH @ Hn_flat).reshape(nb, nawf, nawf)
        Hkq = (phkqH @ Hn_flat).reshape(nb, nawf, nawf)
        Hk = 0.5 * (Hk + np.conjugate(np.transpose(Hk, (0, 2, 1))))
        Hkq = 0.5 * (Hkq + np.conjugate(np.transpose(Hkq, (0, 2, 1))))
        Ek, Vk = np.linalg.eigh(Hk)
        Ekq, Vkq = np.linalg.eigh(Hkq)
        Ek = Ek / RY_TO_EV
        Ekq = Ekq / RY_TO_EV
        gk = (phkg @ gn_flat).reshape(nb, nawf, nawf, ncart)
        tmp = np.einsum('bim,bijc->bmjc', Vkq.conj(), gk)
        gband = np.einsum('bmjc,bjn->bmnc', tmp, Vk)
        gnu = np.einsum('bmnc,vc->bvmn', gband, zmass)
        absg2 = np.abs(gnu) ** 2
        for isig, sig in enumerate(sigmas):
            ef = ef_sig[isig]
            dk = np.exp(-(((Ek - ef) / sig) ** 2)) / (sig * np.sqrt(np.pi))
            dkq = np.exp(-(((Ekq - ef) / sig) ** 2)) / (sig * np.sqrt(np.pi))
            dsum[isig] += dk.sum()
            num[isig] += np.einsum('bvmn,bm,bn->v', absg2, dkq, dk)

    lam = np.zeros((sigmas.size, nmode))
    gam_ghz = np.zeros((sigmas.size, nmode))
    dos_out = dsum / nkd
    for isig in range(sigmas.size):
        gam_ghz[isig] = np.pi * num[isig] / nkd * ry_to_ghz
        lam[isig, safe] = num[isig, safe] / (nkd * dos_out[isig] * omega_ry[safe] ** 2)
    return {'lambda_qnu': lam, 'gamma_ghz': gam_ghz, 'dos_ef': dos_out, 'nk_dense': Nk}


def precompute_dense_electrons(
    HRs, at, Nk, sigmas_ry, nelec, ng_vertex, ispin=0, kblock=4096, fs_window=8.0
):
    """Dense-grid electron spectrum + Fermi-surface delta, shared by every q.

    Diagonalises ``H(k)`` **once** on the regular ``Nk^3`` grid (Wigner-Seitz
    interpolation of ``HRs``).  Because the electronic spectrum is q-independent,
    ``E(k+q)`` / ``V(k+q)`` for any commensurate q are index shifts of this array,
    so :func:`lambda_q_dense_ws_fast` needs no per-q re-diagonalisation (the
    dominant cost of :func:`lambda_q_dense_ws` when looping over many q).  The
    Wigner-Seitz phase of the coupling grid ``ng_vertex`` is cached too, so the
    per-q vertex interpolation reduces to a single BLAS matmul.

    Parameters
    ----------
    HRs : ndarray ``(nawf, nawf, m1, m2, m3, nspin)``
        PAO Hamiltonian (``E_F`` at 0).
    at : ndarray ``(3, 3)``
        Real-lattice vectors (rows, alat units) for the Wigner-Seitz sums.
    Nk : int
        Dense grid size.
    sigmas_ry : array_like
        Fermi-surface smearing(s) (Ry).
    nelec : float or None
        Valence electrons for the dense Fermi-level recompute (``None`` -> 0).
    ng_vertex : tuple(int, int, int)
        Real-space grid of the coupling vertex ``gR`` (its ``shape[3:6]``).
    ispin, kblock : int, optional
    fs_window : float, optional
        Fermi-surface window in smearings for the shell mask (default 8, exact to
        ``exp(-fs_window^2)``); lower it to prune wide-band metals more aggressively.

    Returns
    -------
    dict
        Cache consumed by :func:`lambda_q_dense_ws_fast` -- keys ``Nk``, ``nawf``,
        ``E`` ``(nkd, nawf)`` (Ry), ``V`` ``(nkd, nawf, nawf)``, ``dk``
        ``(nsigma, nkd, nawf)``, ``dos`` ``(nsigma,)``, plus the cached vertex
        Wigner-Seitz phase (``phkg``, ``Midxg``, ``Wg``, ``ng_vertex``).
    """
    nawf = HRs.shape[0]
    H = HRs[:, :, :, :, :, ispin]
    ng_H = H.shape[2:5]
    NintH, WH, MidxH = _ws_lattice(ng_H, at)
    Hn = np.transpose(H[:, :, MidxH[:, 0], MidxH[:, 1], MidxH[:, 2]], (2, 0, 1))
    Hn = Hn * WH[:, None, None]
    Hn_flat = Hn.reshape(Hn.shape[0], -1)

    ax = [np.arange(Nk) / Nk for _ in range(3)]
    K = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)
    nkd = K.shape[0]
    sigmas = np.atleast_1d(sigmas_ry)

    E = np.empty((nkd, nawf))
    V = np.empty((nkd, nawf, nawf), dtype=complex)
    for s0 in range(0, nkd, kblock):
        Kb = K[s0 : s0 + kblock]
        phk = np.exp(2j * np.pi * (Kb @ NintH.T))
        Hk = (phk @ Hn_flat).reshape(Kb.shape[0], nawf, nawf)
        Hk = 0.5 * (Hk + np.conjugate(np.transpose(Hk, (0, 2, 1))))
        e, v = np.linalg.eigh(Hk)
        E[s0 : s0 + Kb.shape[0]] = e / RY_TO_EV
        V[s0 : s0 + Kb.shape[0]] = v

    ef_sig = np.zeros(sigmas.size)
    dk = np.empty((sigmas.size, nkd, nawf))
    dos = np.zeros(sigmas.size)
    for isig, sig in enumerate(sigmas):
        ef = _fermi_level(E, sig, nelec) if nelec is not None else 0.0
        ef_sig[isig] = ef
        d = np.exp(-(((E - ef) / sig) ** 2)) / (sig * np.sqrt(np.pi))
        dk[isig] = d
        dos[isig] = d.sum() / nkd

    # Fermi-surface shell: k with a band within ``fs_window`` smearings of E_F.
    # The double delta is negligible (exp(-fs_window^2)) outside it, so it prunes
    # the per-q vertex contraction to the states that actually scatter.
    shell = np.empty((sigmas.size, nkd), dtype=bool)
    for isig, sig in enumerate(sigmas):
        shell[isig] = np.min(np.abs(E - ef_sig[isig]), axis=1) < fs_window * sig

    Nintg, Wg, Midxg = _ws_lattice(tuple(ng_vertex), at)
    phkg = np.exp(2j * np.pi * (K @ Nintg.T))  # (nkd, nwsg)
    return {
        'Nk': Nk,
        'nawf': nawf,
        'K': K,
        'E': E,
        'V': V,
        'ef_sig': ef_sig,
        'dk': dk,
        'dos': dos,
        'sigmas': sigmas,
        'shell': shell,
        'fs_window': fs_window,
        'ng_vertex': tuple(int(n) for n in ng_vertex),
        'Wg': Wg,
        'Midxg': Midxg,
        'phkg': phkg,
    }


def lambda_q_dense_ws_fast(gR, electrons, q_cryst, zmass, freqs_thz, kblock=4096):
    """``lambda_{q nu}`` for one q, reusing a :func:`precompute_dense_electrons` cache.

    Numerically identical to :func:`lambda_q_dense_ws` (deterministic ``eigh``),
    but the q-independent dense electron spectrum, Fermi level and Fermi-surface
    delta come from ``electrons``; only the coupling vertex is interpolated per q,
    and ``k+q`` is an index shift (requires ``Nk`` commensurate with ``q``).

    Parameters
    ----------
    gR : ndarray ``(nawf, nawf, ncart, n1, n2, n3)``
        PAO-gauge vertex real-space cells for this q (grid == ``ng_vertex``).
    electrons : dict
        Output of :func:`precompute_dense_electrons`.
    q_cryst : ndarray ``(3,)``
        q-point (crystal coordinates); ``q * Nk`` must be integer.
    zmass : ndarray ``(nmode, ncart)``
    freqs_thz : ndarray ``(nmode,)``
    kblock : int, optional

    Returns
    -------
    dict
        ``{'lambda_qnu', 'gamma_ghz', 'dos_ef', 'nk_dense'}``.
    """
    Nk = electrons['Nk']
    nawf = electrons['nawf']
    nkd = Nk**3
    ncart = gR.shape[2]
    nmode = zmass.shape[0]
    if tuple(int(n) for n in gR.shape[3:6]) != electrons['ng_vertex']:
        raise ValueError(
            'vertex grid %s != cached ng_vertex %s' % (gR.shape[3:6], electrons['ng_vertex'])
        )

    shift = np.asarray(q_cryst, dtype=float) * Nk
    if np.any(np.abs(shift - np.round(shift)) > 1.0e-6):
        raise ValueError(
            'Nk=%d must be commensurate with q=%s for the fast path' % (Nk, tuple(q_cryst))
        )
    shift = np.round(shift).astype(int)
    roll = (-int(shift[0]), -int(shift[1]), -int(shift[2]))

    Vk = electrons['V']
    Vkq = np.roll(Vk.reshape(Nk, Nk, Nk, nawf, nawf), roll, axis=(0, 1, 2)).reshape(nkd, nawf, nawf)
    dk_all = electrons['dk']  # (nsig, nkd, nawf)
    nsig = dk_all.shape[0]
    dkq_all = np.roll(dk_all.reshape(nsig, Nk, Nk, Nk, nawf), roll, axis=(1, 2, 3)).reshape(
        nsig, nkd, nawf
    )

    Midxg, Wg, phkg = electrons['Midxg'], electrons['Wg'], electrons['phkg']
    gn = np.transpose(gR[:, :, :, Midxg[:, 0], Midxg[:, 1], Midxg[:, 2]], (3, 0, 1, 2))
    gn = gn * Wg[:, None, None, None]
    gn_flat = gn.reshape(gn.shape[0], -1)

    # Restrict to the Fermi-surface shell: only k with a band near E_F at BOTH k
    # and k+q give a non-negligible double delta (exact to exp(-fs_window^2)).
    shell = electrons.get('shell')
    if shell is not None:
        keep = np.zeros(nkd, dtype=bool)
        for isig in range(nsig):
            shq = np.roll(shell[isig].reshape(Nk, Nk, Nk), roll, axis=(0, 1, 2)).reshape(nkd)
            keep |= shell[isig] & shq
        idx = np.nonzero(keep)[0]
    else:
        idx = np.arange(nkd)

    omega_ry = np.abs(freqs_thz) / RY_TO_THZ
    safe = omega_ry > 1.0e-8
    ry_to_ghz = RY_TO_EV * 2.417989242e5
    num = np.zeros((nsig, nmode))
    for s0 in range(0, idx.size, kblock):
        ii = idx[s0 : s0 + kblock]
        gk = (phkg[ii] @ gn_flat).reshape(-1, nawf, nawf, ncart)
        tmp = np.einsum('bim,bijc->bmjc', Vkq[ii].conj(), gk, optimize=True)
        gband = np.einsum('bmjc,bjn->bmnc', tmp, Vk[ii], optimize=True)
        gnu = np.einsum('bmnc,vc->bvmn', gband, zmass, optimize=True)
        absg2 = np.abs(gnu) ** 2
        for isig in range(nsig):
            num[isig] += np.einsum(
                'bvmn,bm,bn->v', absg2, dkq_all[isig][ii], dk_all[isig][ii], optimize=True
            )

    lam = np.zeros((nsig, nmode))
    gam_ghz = np.zeros((nsig, nmode))
    dos = electrons['dos']
    for isig in range(nsig):
        gam_ghz[isig] = np.pi * num[isig] / nkd * ry_to_ghz
        lam[isig, safe] = num[isig, safe] / (nkd * dos[isig] * omega_ry[safe] ** 2)
    return {'lambda_qnu': lam, 'gamma_ghz': gam_ghz, 'dos_ef': dos, 'nk_dense': Nk}
