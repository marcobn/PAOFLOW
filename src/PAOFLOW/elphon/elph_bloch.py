"""Bloch-space electron-phonon matrix elements from QE DFPT output (Phase 1A, EPW route).

Builds the coarse-grid electron-phonon vertex

.. math::

    g_{mn,\\nu}(k, q) = \\sum_{\\kappa\\alpha}
        \\frac{z^{\\nu}_{\\kappa\\alpha}(q)}{\\sqrt{2 M_\\kappa \\omega_{q\\nu}}}\\,
        \\langle \\psi_{m,k+q} | \\partial_{\\kappa\\alpha} V_q | \\psi_{n,k}\\rangle

directly from the QE ``nscf`` wavefunctions and the ``ph.x`` ``dvscf`` files --
the PAOFLOW replacement for ``pw2wannier90`` + EPW's Bloch vertex.  The Cartesian
deformation-potential matrix elements

.. math::

    d_{mn,\\kappa\\alpha}(k,q) = \\frac1{N_r}\\sum_r
        [e^{-i G_0\\cdot r} u_{m,k'}(r)]^*\\, \\partial_{\\kappa\\alpha}V_q(r)\\, u_{n,k}(r)

are contracted with the phonon eigenvectors ``z`` (from the QE dynamical matrix)
and summed over the Fermi surface with the double-delta approximation to give the
mode-resolved coupling ``lambda_{q\\nu}`` at each (irreducible) coarse ``q``.

Only the *local* self-consistent part of the perturbation (``dvscf``) is included
here; the nonlocal beta-projector derivative is a separate correction.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from ..projection.do_atwfc_proj import fft_allwfc_G2R, read_QE_wfc

HARTREE_TO_RY = 2.0
RY_TO_THZ = 3289.842
RY_TO_EV = 13.605693122994
AMU_RY = 911.4442421


class _DC:
    """Minimal data-controller shim for :func:`read_QE_wfc`."""

    def __init__(self, fpath, nspin=1):
        self.data_attributes = {'fpath': fpath, 'nspin': nspin}
        self.data_arrays = {}

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


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


def _umklapp_phase(G0, fft):
    """``e^{-i G0.r}`` on the FFT grid for an integer reciprocal vector ``G0``."""
    nr1, nr2, nr3 = fft
    if not np.any(G0):
        return None
    a1 = np.exp(-2j * np.pi * G0[0] * np.arange(nr1) / nr1)
    a2 = np.exp(-2j * np.pi * G0[1] * np.arange(nr2) / nr2)
    a3 = np.exp(-2j * np.pi * G0[2] * np.arange(nr3) / nr3)
    return a1[:, None, None] * a2[None, :, None] * a3[None, None, :]


def _load_all_ur(save_dir, nk, fft, nspin=1):
    """FFT every ``nscf`` wavefunction to real space: ``ur[ik]`` shape ``(nbnd, nr1,nr2,nr3)``."""
    dc = _DC(save_dir, nspin=nspin)
    ur = []
    for ik in range(nk):
        gk, wf = read_QE_wfc(dc, ik, 0)
        ur.append(fft_allwfc_G2R(wf['wfc'], gk, fft[0], fft[1], fft[2], 1.0))
    return ur


def deformation_potential_q(ur, ikq, G0, dvscf_cart, fft):
    """Cartesian deformation-potential matrix elements ``d_{mn,c}(k)`` for one q.

    Parameters
    ----------
    ur : list of ndarray
        Real-space cell-periodic wavefunctions ``u_{n,k}(r)`` per k
        (``(nbnd, nr1, nr2, nr3)``).
    ikq, G0 : ndarray
        Output of :func:`kq_index_map`.
    dvscf_cart : ndarray ``(3*nat, nr1, nr2, nr3)``
        Cartesian ``dV/du`` for this q (from :func:`PAOFLOW.elphon.qe_dvscf`).
    fft : tuple(int, int, int)

    Returns
    -------
    ndarray ``(nk, nbnd, nbnd, 3*nat)`` complex
        ``d[k, m, n, c] = <u_{m,k+q}| dvscf_c | u_{n,k}> / N_r``.
    """
    nk = len(ur)
    nbnd = ur[0].shape[0]
    ncart = dvscf_cart.shape[0]
    nr = fft[0] * fft[1] * fft[2]
    d = np.zeros((nk, nbnd, nbnd, ncart), dtype=complex)
    dvflat = dvscf_cart.reshape(ncart, nr)
    for ik in range(nk):
        u_k = ur[ik].reshape(nbnd, nr)  # (n, r)
        ph = _umklapp_phase(G0[ik], fft)
        u_kq = ur[ikq[ik]]
        if ph is not None:
            u_kq = u_kq * ph
        u_kq = u_kq.reshape(nbnd, nr)  # (m, r)
        # d[m,n,c] = sum_r conj(u_kq[m,r]) dvflat[c,r] u_k[n,r] / nr
        # = sum_r conj(u_kq[m,r]) * (dvflat[c,r] * u_k[n,r])
        for c in range(ncart):
            tmp = dvflat[c][None, :] * u_k  # (n, r)
            d[ik, :, :, c] = (u_kq.conj() @ tmp.T) / nr  # (m, n)
    return d


def lambda_qnu(d, eigs_ry, ikq, zmass, freqs_thz, dos_ef, sigmas_ry):
    """Mode-resolved ``lambda_{q nu}`` (per smearing) from the deformation potentials.

    .. math::

        \\lambda_{q\\nu}(\\sigma) = \\frac{1}{N_k N(E_F) \\omega_{q\\nu}^2}
            \\sum_{k,m,n} \\Big| \\sum_{\\kappa\\alpha}
            d_{mn,\\kappa\\alpha}(k)\\, z^{\\nu}_{\\kappa\\alpha}/\\sqrt{M_\\kappa}\\Big|^2
            \\delta_\\sigma(\\varepsilon_{nk})\\,\\delta_\\sigma(\\varepsilon_{m,k+q})

    with ``z`` the mass-weighted dynamical-matrix eigenvectors, ``M`` in Ry mass
    units, energies and ``omega`` in Ry, ``N(E_F)`` in states/spin/Ry.

    Parameters
    ----------
    d : ndarray ``(nk, nbnd, nbnd, 3*nat)``
        Deformation potentials from :func:`deformation_potential_q`.
    eigs_ry : ndarray ``(nk, nbnd)``
        Band energies (Ry, referred to E_F).
    ikq : ndarray ``(nk,)``
    zmass : ndarray ``(nmode, 3*nat)``
        Mass-weighted eigenvectors ``z^{nu}_{kappa alpha}/sqrt(M_kappa)`` already
        divided by ``sqrt(M)`` (Ry mass units).
    freqs_thz : ndarray ``(nmode,)``
    dos_ef : float or ndarray ``(nsigma,)``
        DOS at E_F (states/spin/Ry); scalar or per smearing.
    sigmas_ry : ndarray ``(nsigma,)``

    Returns
    -------
    ndarray ``(nsigma, nmode)``
        ``lambda_{q nu}`` per smearing.
    """
    nk, nbnd = eigs_ry.shape
    nmode = zmass.shape[0]
    sigmas = np.atleast_1d(sigmas_ry)
    dos = np.atleast_1d(dos_ef)
    if dos.size == 1:
        dos = np.repeat(dos, sigmas.size)

    # g_nu[k,m,n] = sum_c d[k,m,n,c] zmass[nu,c]
    g = np.einsum('kmnc,vc->kvmn', d, zmass)  # (nk, nmode, m, n)
    absg2 = np.abs(g) ** 2

    e_k = eigs_ry  # (nk, nbnd) at k, index n
    e_kq = eigs_ry[ikq]  # (nk, nbnd) at k+q, index m

    omega_ry = np.abs(freqs_thz) / RY_TO_THZ
    out = np.zeros((sigmas.size, nmode))
    for isig, sig in enumerate(sigmas):
        dk = np.exp(-((e_k / sig) ** 2)) / (sig * np.sqrt(np.pi))  # (nk, n)
        dkq = np.exp(-((e_kq / sig) ** 2)) / (sig * np.sqrt(np.pi))  # (nk, m)
        # sum over k, m, n: |g|^2 delta(e_kq[m]) delta(e_k[n])
        num = np.einsum('kvmn,km,kn->v', absg2, dkq, dk)  # (nmode,)
        safe = omega_ry > 1.0e-8
        lam = np.zeros(nmode)
        lam[safe] = num[safe] / (nk * dos[isig] * omega_ry[safe] ** 2)
        out[isig] = lam
    return out


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
        Band-basis Cartesian deformation potentials (:func:`deformation_potential_q`).
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
    gk = np.zeros((nawf, nawf, ncart, ng[0], ng[1], ng[2]), dtype=complex)
    for k in range(nk):
        Ak = A[:, :, k]  # (nbnd, nawf)
        Akq = A[:, :, ikq[k]]  # (nbnd, nawf)
        i1, i2, i3 = kgrid_idx[k]
        # (nawf, nawf, ncart): A_{k+q}^dagger d_c A_k for each cart c
        gk[:, :, :, i1, i2, i3] = np.einsum('mi,mnc,nj->ijc', Akq.conj(), d[k], Ak)
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


def lambda_q_dense(gR, HRs, q_int, ng_coarse, zmass, freqs_thz, sigmas_ry, Nk, ispin=0, nelec=None):
    """Converged ``lambda_{q nu}`` by interpolating the vertex to a dense ``Nk^3`` grid.

    The PAO-gauge vertex ``gR`` (from :func:`vertex_pao_R`) and the electronic
    states (from ``HRs``) are Fourier-interpolated to a dense k-grid; at each
    dense k the vertex is rotated back to the interpolated band basis, contracted
    with the phonon eigenvectors and summed over the Fermi surface with the
    double-delta.  ``N(E_F)`` is evaluated on the same dense grid.

    Parameters
    ----------
    gR : ndarray ``(nawf, nawf, ncart, n1, n2, n3)``
        PAO-gauge vertex in real space for this q.
    HRs : ndarray ``(nawf, nawf, n1, n2, n3, nspin)``
        PAO Hamiltonian (E_F at 0), from ``pao_hamiltonian``.
    q_int : tuple(int, int, int)
        ``q`` in coarse-grid integer units (``round(q_cryst * ng_coarse)``).
    ng_coarse : tuple(int, int, int)
        Coarse k-grid dimensions.
    zmass : ndarray ``(nmode, ncart)``
        Mass-weighted phonon eigenvectors ``z / sqrt(M)`` (Ry mass units).
    freqs_thz : ndarray ``(nmode,)``
    sigmas_ry : ndarray ``(nsigma,)``
    Nk : int
        Dense grid size (must be a multiple of every ``ng_coarse`` entry).
    ispin : int, optional

    Returns
    -------
    dict
        ``{'lambda_qnu' (nsigma, nmode), 'dos_ef' (nsigma,), 'nk_dense'}``.
    """
    from .eph_kq import _embed_fftfreq, estates_on_grid

    for s in ng_coarse:
        if Nk % s != 0:
            raise ValueError('Nk=%d must be divisible by coarse grid %s' % (Nk, ng_coarse))

    E, V = estates_on_grid(HRs, Nk)  # (Nk,Nk,Nk,nspin,nawf), (...,nawf,nawf)
    E = E[:, :, :, ispin, :] / RY_TO_EV  # (Nk,Nk,Nk,nawf), E_F at 0, converted eV->Ry
    Vg = V[:, :, :, ispin, :, :]  # (Nk,Nk,Nk,nawf,nawf)
    nawf = E.shape[-1]
    ncart = gR.shape[2]
    nmode = zmass.shape[0]

    # Interpolate the vertex to the dense grid (band-limited, same convention as H).
    gRe = _embed_fftfreq(gR, (3, 4, 5), (Nk, Nk, Nk))
    gkd = np.fft.ifftn(gRe, axes=(3, 4, 5)) * (Nk**3)  # (nawf,nawf,ncart,Nk,Nk,Nk)

    # k+q as an index shift on the dense grid.
    shift = tuple((Nk // ng_coarse[d]) * q_int[d] for d in range(3))
    roll = (-shift[0], -shift[1], -shift[2])
    Vkq = np.roll(Vg, roll, axis=(0, 1, 2))
    Ekq = np.roll(E, roll, axis=(0, 1, 2))

    nkd = Nk**3
    Ef = E.reshape(nkd, nawf)
    Ekqf = Ekq.reshape(nkd, nawf)
    Vk = Vg.reshape(nkd, nawf, nawf)
    Vkqf = Vkq.reshape(nkd, nawf, nawf)
    gkf = np.moveaxis(gkd.reshape(nawf, nawf, ncart, nkd), 3, 0)  # (nkd,nawf,nawf,ncart)

    # Rotate the PAO-gauge vertex into the interpolated band basis.
    tmp = np.einsum('kim,kijc->kmjc', Vkqf.conj(), gkf)
    gband = np.einsum('kmjc,kjn->kmnc', tmp, Vk)  # (nkd, m, n, c)
    gnu = np.einsum('kmnc,vc->kvmn', gband, zmass)  # (nkd, nmode, m, n)
    absg2 = np.abs(gnu) ** 2

    omega_ry = np.abs(freqs_thz) / RY_TO_THZ
    sigmas = np.atleast_1d(sigmas_ry)
    lam = np.zeros((sigmas.size, nmode))
    gam_ghz = np.zeros((sigmas.size, nmode))
    dos_out = np.zeros(sigmas.size)
    safe = omega_ry > 1.0e-8
    ry_to_ghz = RY_TO_EV * 2.417989242e5  # 1 Ry -> GHz
    for isig, sig in enumerate(sigmas):
        ef = _fermi_level(Ef, sig, nelec) if nelec is not None else 0.0
        dk = np.exp(-(((Ef - ef) / sig) ** 2)) / (sig * np.sqrt(np.pi))
        dkq = np.exp(-(((Ekqf - ef) / sig) ** 2)) / (sig * np.sqrt(np.pi))
        nef = dk.sum() / nkd  # DOS per spin at E_F (states/Ry)
        dos_out[isig] = nef
        num = np.einsum('kvmn,km,kn->v', absg2, dkq, dk)
        # phonon linewidth gamma_qnu = pi * (1/Nk) sum_k |g|^2 delta delta  (Ry) -> GHz
        gam_ghz[isig] = np.pi * num / nkd * ry_to_ghz
        lam[isig, safe] = num[safe] / (nkd * nef * omega_ry[safe] ** 2)
    return {'lambda_qnu': lam, 'gamma_ghz': gam_ghz, 'dos_ef': dos_out, 'nk_dense': Nk}


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
