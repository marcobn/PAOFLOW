"""Bare nonlocal (Kleinman-Bylander) projector derivative for the el-ph vertex.

Adds the nonlocal contribution ``<psi_{k+q}| dV_NL/du | psi_k>`` that QE applies
via ``dvqpsi_us_only`` (scalar, norm-conserving branch).  The displacement
derivative of a KB projector is purely a rigid shift of the atom, i.e. in
G-space ``d beta_I^k(G)/du_alpha = -i (k+G)_alpha beta_I^k(G)``.  The matrix
element therefore factorises into per-k projector overlaps

.. math::

    g^{NL}_{\\alpha,a}(m,n) = \\sum_{I\\in a} D_I\\Big[
        \\overline{b^{k+q}_{I,m}}\\, db^{k}_{\\alpha,I,n}
      + \\overline{db^{k+q}_{\\alpha,I,m}}\\, b^{k}_{I,n}\\Big]

with ``b^p_{I,n} = <beta_I^p|psi_{n,p}>`` and
``db^p_{alpha,I,n} = <d_u beta_I^p|psi_{n,p}> = +i sum_G (p+G)_alpha
conj(vkb_I^p(G)) c_{n,p}(G)``.

Because ``D_I`` is m-diagonal and the whole l-shell is summed, the result is
independent of the spherical-harmonic basis convention and of the ``(-i)^l``
prefactor, so a real cubic-harmonic basis (reused from the projection code) is
sufficient.

For a fully-relativistic pseudopotential used in a *scalar* calculation the
j-resolved projectors are first averaged into l-channels exactly as in QE's
``average_pp``.  The nonlinear-core correction is *not* included.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.special import spherical_jn

from ..projection.do_atwfc_proj import calc_ylmg

FPI = 4.0 * np.pi


def _upf_find(root, tag):
    for el in root.iter():
        if el.tag.split('}')[-1] == tag:
            return el
    return None


def read_upf_beta(upf_path):
    """Read the KB projectors, ``D`` matrix and (l, j) labels from a UPF (v2).

    Returns
    -------
    dict
        ``{'r', 'rab', 'beta' (nbeta, mesh), 'lll', 'jjj', 'dij' (nbeta,),
        'kkbeta'}``.  ``jjj`` is zero for a scalar pseudopotential.
    """
    root = ET.parse(upf_path).getroot()
    r = np.array(_upf_find(root, 'PP_R').text.split(), dtype=float)
    rab = np.array(_upf_find(root, 'PP_RAB').text.split(), dtype=float)

    betas, lll, kk = [], [], []
    i = 1
    while True:
        el = _upf_find(root, 'PP_BETA.%d' % i)
        if el is None:
            break
        betas.append(np.array(el.text.split(), dtype=float))
        lll.append(int(el.attrib['angular_momentum']))
        kk.append(int(el.attrib.get('cutoff_radius_index', len(r))))
        i += 1
    nbeta = len(betas)
    mesh = max(b.size for b in betas)
    beta = np.zeros((nbeta, mesh))
    for j, b in enumerate(betas):
        beta[j, : b.size] = b

    dij_el = _upf_find(root, 'PP_DIJ')
    dij_full = np.array(dij_el.text.split(), dtype=float).reshape(nbeta, nbeta)
    dij = np.diag(dij_full).copy()

    jjj = np.zeros(nbeta)
    so = _upf_find(root, 'PP_SPIN_ORB')
    if so is not None:
        for i in range(nbeta):
            rb = _upf_find(root, 'PP_RELBETA.%d' % (i + 1))
            if rb is not None:
                jjj[i] = float(rb.attrib['jjj'])

    return {
        'r': r,
        'rab': rab,
        'beta': beta,
        'lll': np.array(lll),
        'jjj': jjj,
        'dij': dij,
        'kkbeta': int(max(kk)),
    }


def average_pp_beta(bd):
    """Average a fully-relativistic projector set into scalar l-channels.

    Implements QE ``average_pp``: for each ``l != 0`` the ``j = l +- 1/2``
    projectors are combined with weights ``(l+1)`` and ``l``.

    Parameters
    ----------
    bd : dict
        Output of :func:`read_upf_beta`.

    Returns
    -------
    list of dict
        One entry per averaged channel: ``{'l', 'beta' (mesh,), 'D'}``.
    """
    lll, jjj, dij, beta = bd['lll'], bd['jjj'], bd['dij'], bd['beta']
    has_so = np.any(jjj > 0)
    channels = []
    if not has_so:
        for nb in range(len(lll)):
            channels.append({'l': int(lll[nb]), 'beta': beta[nb].copy(), 'D': float(dij[nb])})
        return channels

    used = np.zeros(len(lll), dtype=bool)
    for nb in range(len(lll)):
        if used[nb]:
            continue
        l = int(lll[nb])
        if l == 0:
            channels.append({'l': 0, 'beta': beta[nb].copy(), 'D': float(dij[nb])})
            used[nb] = True
            continue
        # find the j = l+1/2 (ind) and j = l-1/2 (ind1) partners
        partners = [
            m
            for m in range(len(lll))
            if lll[m] == l and not used[m] and abs(jjj[m] - jjj[nb]) < 1.0 + 1e-6
        ]
        ind = next(m for m in partners if abs(jjj[m] - l - 0.5) < 1e-6)
        ind1 = next(m for m in partners if abs(jjj[m] - l + 0.5) < 1e-6)
        V = ((l + 1.0) * dij[ind] + l * dij[ind1]) / (2.0 * l + 1.0)
        b = (1.0 / (2.0 * l + 1.0)) * (
            (l + 1.0) * np.sqrt(dij[ind] / V) * beta[ind] + l * np.sqrt(dij[ind1] / V) * beta[ind1]
        )
        channels.append({'l': l, 'beta': b, 'D': float(V)})
        used[ind] = used[ind1] = True
    return channels


def _simpson(f, rab, n):
    """QE-style Simpson integral over the first ``n`` points (n forced odd)."""
    if n % 2 == 0:
        n -= 1
    w = np.ones(n)
    w[1 : n - 1 : 2] = 4.0
    w[2 : n - 1 : 2] = 2.0
    return np.sum(f[..., :n] * (w * rab[:n]), axis=-1) / 3.0


def beta_of_g(channel, qphys, r, rab, omega, kkbeta):
    """Radial KB projector in reciprocal space, ``f_l(|k+G|)``.

    ``f_l(q) = (4 pi / sqrt(Omega)) \\int beta_l(r) j_l(q r) r dr`` (QE
    ``init_tab_beta`` convention; ``beta_l`` is the raw ``PP_BETA`` array).

    Parameters
    ----------
    channel : dict
        Averaged channel from :func:`average_pp_beta`.
    qphys : ndarray
        Magnitudes ``|k+G|`` in inverse Bohr.
    r, rab : ndarray
        Radial grid and integration weights (Bohr).
    omega : float
        Cell volume (Bohr^3).
    kkbeta : int
        Number of radial points to integrate.

    Returns
    -------
    ndarray
        ``f_l`` for each ``qphys`` (same shape).
    """
    l = channel['l']
    beta = channel['beta']
    q = np.atleast_1d(np.asarray(qphys, dtype=float))
    n = min(kkbeta, r.size)
    rr = r[:n]
    integ_pref = beta[:n] * rr  # beta_l(r) * r
    # j_l(q r): shape (nq, n)
    jl = spherical_jn(l, np.outer(q, rr))
    aux = jl * integ_pref[None, :]
    tab = _simpson(aux, rab[:n], n) * FPI / np.sqrt(omega)
    return tab if tab.size > 1 else float(tab[0])


def build_projectors(channels, nat):
    """Flat projector index tables (one entry per (atom, channel, m)).

    Returns
    -------
    dict
        ``{'l', 'D', 'atom', 'lm', 'nkb'}`` arrays of length ``nkb``.
    """
    proj_l, proj_D, proj_atom, proj_lm = [], [], [], []
    for na in range(nat):
        for ch in channels:
            l = ch['l']
            for mm in range(2 * l + 1):
                proj_l.append(l)
                proj_D.append(ch['D'])
                proj_atom.append(na)
                proj_lm.append(l * l + mm)  # column into calc_ylmg
    return {
        'l': np.array(proj_l),
        'D': np.array(proj_D),
        'atom': np.array(proj_atom),
        'lm': np.array(proj_lm),
        'nkb': len(proj_l),
    }


def becp_k(gkspace, cwfc, channels, proj, tau_cryst, tpiba, omega, r, rab, kkbeta):
    """Projector overlaps ``becp`` and displacement derivatives ``dbecp`` at one k.

    Parameters
    ----------
    gkspace : dict
        ``read_QE_wfc`` descriptor (``xk``, ``mill``, ``bg`` in 2*pi/alat units).
    cwfc : ndarray ``(nbnd, igwx)``
        (Orthonormalised) wavefunction coefficients on the k-sphere.
    channels : list of dict
        Averaged channels (:func:`average_pp_beta`).
    proj : dict
        Output of :func:`build_projectors`.
    tau_cryst : ndarray ``(nat, 3)``
        Atomic positions in crystal coordinates.
    tpiba : float
        ``2*pi/alat`` (inverse Bohr).
    omega : float
        Cell volume (Bohr^3).
    r, rab, kkbeta :
        Radial grid data for :func:`beta_of_g`.

    Returns
    -------
    becp : ndarray ``(nkb, nbnd)`` complex
    dbecp : ndarray ``(3, nkb, nbnd)`` complex
        ``dbecp[alpha] = +i sum_G (k+G)_alpha conj(vkb) c`` (physical (k+G)).
    """
    xk = gkspace['xk'][:3]
    mill = gkspace['mill']
    bg = gkspace['bg']
    hkl = mill.T
    kpg_tpiba = hkl @ bg.T + xk  # (npw, 3), 2*pi/alat units
    qphys = np.linalg.norm(kpg_tpiba, axis=1) * tpiba
    ylmg = calc_ylmg(kpg_tpiba, np.linalg.norm(kpg_tpiba, axis=1))
    kpg_phys = kpg_tpiba * tpiba  # physical (k+G) in 1/Bohr
    # crystal (k+G) for the structure phase: k_cryst + mill
    k_cryst = np.linalg.solve(bg.T, xk)
    kpg_cryst = hkl + k_cryst  # (npw, 3)

    # radial form factor per channel, evaluated at qphys
    fl = {id(ch): beta_of_g(ch, qphys, r, rab, omega, kkbeta) for ch in channels}

    npw = kpg_tpiba.shape[0]
    nkb = proj['nkb']
    nbnd = cwfc.shape[0]
    becp = np.zeros((nkb, nbnd), dtype=complex)
    dbecp = np.zeros((3, nkb, nbnd), dtype=complex)

    # map channel-per-l back for each projector: reconstruct in build order
    ci = 0
    order = []  # channel object per (atom, channel, m) matching build_projectors
    nat = tau_cryst.shape[0]
    for na in range(nat):
        for ch in channels:
            for mm in range(2 * ch['l'] + 1):
                order.append(ch)

    for I in range(nkb):
        na = proj['atom'][I]
        lm = proj['lm'][I]
        ch = order[I]
        strf = np.exp(-2j * np.pi * (kpg_cryst @ tau_cryst[na]))  # e^{-i(k+G).tau}
        vkb = fl[id(ch)] * ylmg[:, lm] * strf  # (npw,)
        vkb_c = vkb.conj()
        becp[I] = (vkb_c[None, :] * cwfc).sum(axis=1)
        # dbecp[alpha] = +i sum_G (k+G)_alpha conj(vkb) c
        for a in range(3):
            dbecp[a, I] = 1j * ((kpg_phys[:, a] * vkb_c)[None, :] * cwfc).sum(axis=1)
    return becp, dbecp


def nonlocal_dq(becp_k_, dbecp_k_, becp_kq_, dbecp_kq_, proj, nat):
    """Nonlocal deformation-potential matrix element for one (k, q).

    Returns
    -------
    ndarray ``(nbnd, nbnd, 3*nat)`` complex
        ``d_NL[m, n, 3*na+alpha]`` to be added to the local ``d``.
    """
    nbnd = becp_k_.shape[1]
    d = np.zeros((nbnd, nbnd, 3 * nat), dtype=complex)
    D = proj['D']
    atom = proj['atom']
    for I in range(proj['nkb']):
        na = atom[I]
        DI = D[I]
        bkq = becp_kq_[I]  # (nbnd,) at k+q
        bk = becp_k_[I]  # (nbnd,) at k
        for a in range(3):
            # term A: conj(becp_kq) (bra, no deriv) * dbecp_k (ket deriv)
            # term B: conj(dbecp_kq) (bra deriv) * becp_k (ket, no deriv)
            tA = np.outer(bkq.conj(), dbecp_k_[a, I])
            tB = np.outer(dbecp_kq_[a, I].conj(), bk)
            d[:, :, 3 * na + a] += DI * (tA + tB)
    return d
