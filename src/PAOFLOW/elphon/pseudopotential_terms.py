"""Bare pseudopotential terms of the electron-phonon vertex (norm-conserving).

The induced potential of ``ph.x`` (``dvscf``) excludes the bare ionic contribution; the
vertex adds, as EPW's ``dvqpsi_us3`` / ``dvqpsi_us_only3`` and ``compute_dvloc``:

* the derivative of the local pseudopotential,
  ``dV^loc_{kappa alpha}(q + G) = -i (q+G)_alpha V^loc_kappa(|q+G|) exp(-i (q+G).tau_kappa)``;
* the derivative of the Kleinman-Bylander projectors,
  ``sum_ij [ |d beta_i> D_ij <beta_j| + |beta_i> D_ij <d beta_j| ]`` with
  ``d beta(k+G) = -i (k+G) beta(k+G)``.

The radial tables ``V^loc(q)`` and ``beta(q)`` mirror QE's (``upflib/vloc_mod.f90``,
``upflib/beta_mod.f90``: Simpson integration on the UPF mesh, ``dq = 0.01`` bohr^-1,
four-point Lagrange interpolation) so that the vertex reproduces EPW's ``epmatq`` to the
rounding level.  Fully relativistic projectors used in a scalar run are averaged as QE's
``average_pp``.  Spherical harmonics follow ``ylmr2`` (real, QE ordering).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf, spherical_jn

E2 = 2.0  # e^2 in Rydberg units
FPI = 4.0 * np.pi
DQ = 0.01  # table step (bohr^-1), QE
RCUT_LOCAL = 10.0  # bohr, QE's msh cutoff of the local-potential integrals


def qe_simpson(func: NDArray[np.float64], rab: NDArray[np.float64]) -> float:
    """Simpson integration on a logarithmic mesh, as QE's ``simpson`` (odd or even mesh).

    Parameters
    ----------
    func : ndarray ``(mesh,)``
    rab : ndarray ``(mesh,)``
        ``dr/di`` of the mesh.
    """
    mesh = func.shape[-1]
    f = func * rab
    if mesh % 2 == 1:
        weights = np.full(mesh, 2.0)
        weights[1::2] = 4.0
        weights[0] = weights[-1] = 1.0
        return float(np.dot(f, weights) / 3.0)
    weights = np.full(mesh, 2.0)
    weights[1::2] = 4.0
    weights[0] = 1.0
    # QE: inner points 2..mesh-1 with alternating 4/2, then the end corrections.
    weights[-1] = 0.0
    inner = np.dot(f[1:-1], weights[1:-1])
    return float((inner + f[0] - f[-3] * 0.25 + f[-2] + f[-1] * 1.25) / 3.0)


def radial_cutoff_index(r: NDArray[np.float64], rcut: float = RCUT_LOCAL) -> int:
    """QE's ``msh``: number of mesh points up to ``rcut``, made odd."""
    beyond = np.nonzero(r > rcut)[0]
    msh = int(beyond[0]) + 1 if beyond.size else int(r.size)
    return 2 * ((msh + 1) // 2) - 1


def lagrange4(
    table: NDArray[np.float64], q: NDArray[np.float64], dq: float = DQ
) -> NDArray[np.float64]:
    """Four-point Lagrange interpolation of a table ``table[i] = f(i dq)`` (QE ``interp_beta``).

    Parameters
    ----------
    table : ndarray ``(..., nq)``
    q : ndarray ``(n,)``
        Abscissae (``>= 0``); values whose stencil leaves the table are zero.

    Returns
    -------
    ndarray ``(..., n)``
    """
    q = np.asarray(q, dtype=float)
    i0 = (q / dq).astype(int)
    px = q / dq - i0
    ux, vx, wx = 1.0 - px, 2.0 - px, 3.0 - px
    nq = table.shape[-1]
    ok = i0 + 3 < nq
    i0c = np.where(ok, i0, 0)
    out = (
        table[..., i0c] * ux * vx * wx / 6.0
        + table[..., i0c + 1] * px * vx * wx / 2.0
        - table[..., i0c + 2] * px * ux * wx / 2.0
        + table[..., i0c + 3] * px * ux * vx / 6.0
    )
    return np.where(ok, out, 0.0)


def real_spherical_harmonics_qe(lmax: int, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Real spherical harmonics in QE's ``ylmr2`` ordering and sign convention.

    Parameters
    ----------
    lmax : int
    vectors : ndarray ``(n, 3)``
        Directions (any length; the zero vector gives the ``cos theta = 0`` limit).

    Returns
    -------
    ndarray ``(n, (lmax+1)^2)``
        Column ``lm = l^2 + 1 + 2m`` (1-based) is ``cos(m phi)`` type, ``lm + 1`` ``sin(m phi)``.
    """
    v = np.asarray(vectors, dtype=float)
    n = v.shape[0]
    lmax2 = (lmax + 1) ** 2
    ylm = np.zeros((n, lmax2))
    if lmax == 0:
        ylm[:, 0] = np.sqrt(1.0 / FPI)
        return ylm
    gmod = np.sqrt(np.einsum('ni,ni->n', v, v))
    cost = np.where(gmod < 1e-9, 0.0, v[:, 2] / np.where(gmod < 1e-9, 1.0, gmod))
    sent = np.sqrt(np.maximum(0.0, 1.0 - cost * cost))
    # associated Legendre part with QE's recursion (0-based columns: lm-1)
    ylm[:, 0] = 1.0
    ylm[:, 1] = cost
    ylm[:, 3] = -sent / np.sqrt(2.0)
    for l in range(2, lmax + 1):
        for m in range(l - 1):
            lm, lm1, lm2 = l * l + 2 * m, (l - 1) ** 2 + 2 * m, (l - 2) ** 2 + 2 * m
            ylm[:, lm] = (
                cost * (2 * l - 1) / np.sqrt(l * l - m * m) * ylm[:, lm1]
                - np.sqrt((l - 1) ** 2 - m * m) / np.sqrt(l * l - m * m) * ylm[:, lm2]
            )
        lm, lm1, lm2 = l * l + 2 * l, l * l + 2 * (l - 1), (l - 1) ** 2 + 2 * (l - 1)
        ylm[:, lm1] = cost * np.sqrt(2 * l - 1) * ylm[:, lm2]
        ylm[:, lm] = -np.sqrt(2 * l - 1) / np.sqrt(2 * l) * sent * ylm[:, lm2]
    eps = 1e-9
    phi = np.where(
        v[:, 0] > eps,
        np.arctan(v[:, 1] / np.where(np.abs(v[:, 0]) > eps, v[:, 0], 1.0)),
        np.where(
            v[:, 0] < -eps,
            np.arctan(v[:, 1] / np.where(np.abs(v[:, 0]) > eps, v[:, 0], 1.0)) + np.pi,
            np.sign(v[:, 1]) * np.pi / 2.0 + (v[:, 1] == 0) * np.pi / 2.0,
        ),
    )
    ylm[:, 0] /= np.sqrt(FPI)
    for l in range(1, lmax + 1):
        c = np.sqrt((2 * l + 1) / FPI)
        base = l * l  # 0-based column of m = 0
        ylm[:, base] *= c
        for m in range(1, l + 1):
            col = base + 2 * m  # 0-based column of the sin term; cos term at col - 1
            leg = ylm[:, col].copy()
            ylm[:, col - 1] = c * np.sqrt(2.0) * leg * np.cos(m * phi)
            ylm[:, col] = c * np.sqrt(2.0) * leg * np.sin(m * phi)
    return ylm


def average_relativistic_projectors(upf) -> dict:
    """Scalar projectors and ``D`` of a fully relativistic UPF used without spin-orbit (QE ``average_pp``).

    Parameters
    ----------
    upf : UPF
        :class:`PAOFLOW.inputs.read_upf.UPF` with ``beta[i]['j']`` set.

    Returns
    -------
    dict
        ``betas`` (list of ``r beta(r)`` arrays), ``lll`` (list of int), ``cutoff`` (list of int),
        ``dion_ry`` ``(nbeta, nbeta)``.
    """
    betas = [np.asarray(b['wfc'], dtype=float) for b in upf.beta]
    lll = [int(b['l']) for b in upf.beta]
    jjj = [float(b['j']) for b in upf.beta]
    cut = [
        int(b['cutoff_index']) if b['cutoff_index'] else betas[i].size
        for i, b in enumerate(upf.beta)
    ]
    dion = np.asarray(upf.dion, dtype=float) * 2.0  # Hartree -> Ry
    out_b, out_l, out_c, out_d = [], [], [], []
    nb = 0
    while nb < len(betas):
        l = lll[nb]
        if l == 0:
            (
                out_b.append(betas[nb]),
                out_l.append(0),
                out_c.append(cut[nb]),
                out_d.append(dion[nb, nb]),
            )
            nb += 1
            continue
        if nb + 1 >= len(betas) or lll[nb + 1] != l:
            raise ValueError(
                'fully relativistic UPF: projector %d (l=%d) has no j partner' % (nb, l)
            )
        if abs(jjj[nb] - l + 0.5) < 1e-7:  # this one is j = l - 1/2
            ind, ind1 = nb + 1, nb
        else:
            ind, ind1 = nb, nb + 1
        vionl = ((l + 1.0) * dion[ind, ind] + l * dion[ind1, ind1]) / (2.0 * l + 1.0)
        beta = (
            (l + 1.0) * np.sqrt(dion[ind, ind] / vionl) * betas[ind]
            + l * np.sqrt(dion[ind1, ind1] / vionl) * betas[ind1]
        ) / (2.0 * l + 1.0)
        (
            out_b.append(beta),
            out_l.append(l),
            out_c.append(max(cut[ind], cut[ind1])),
            out_d.append(vionl),
        )
        nb += 2
    return {'betas': out_b, 'lll': out_l, 'cutoff': out_c, 'dion_ry': np.diag(out_d)}


def scalar_projectors(upf) -> dict:
    """Projectors and ``D`` (Ry) of a UPF as used in a scalar-relativistic run."""
    if getattr(upf, 'has_spinorbit', False) and all(b.get('j') is not None for b in upf.beta):
        return average_relativistic_projectors(upf)
    betas = [np.asarray(b['wfc'], dtype=float) for b in upf.beta]
    cut = [
        int(b['cutoff_index']) if b['cutoff_index'] else betas[i].size
        for i, b in enumerate(upf.beta)
    ]
    return {
        'betas': betas,
        'lll': [int(b['l']) for b in upf.beta],
        'cutoff': cut,
        'dion_ry': np.asarray(upf.dion, dtype=float) * 2.0,
    }


LMAXX = 3  # QE upf_params lmaxx: the rot_ylm matrix is built for this l


def spinor_coefficient(l: int, j: float, m: int, spin: int) -> float:
    """Clebsch-Gordan coefficient of QE's ``spinor`` (``spin`` = 1 up, 2 down; ``m`` = m_j - 1/2)."""
    denom = 1.0 / (2.0 * l + 1.0)
    if abs(j - l - 0.5) < 1e-8:
        return np.sqrt((l + m + 1.0) * denom) if spin == 1 else np.sqrt((l - m) * denom)
    if abs(j - l + 0.5) < 1e-8:
        if m < -l + 1:
            return 0.0
        return np.sqrt((l - m + 1.0) * denom) if spin == 1 else -np.sqrt((l + m) * denom)
    raise ValueError('j = %s and l = %d are not compatible' % (j, l))


def spherical_index(l: int, j: float, m: int, spin: int) -> int:
    """QE ``sph_ind``: the orbital ``m_l`` of the spin component of the ``|l j m_j>`` spinor."""
    if abs(j - l - 0.5) < 1e-8:
        out = m if spin == 1 else m + 1
    elif abs(j - l + 0.5) < 1e-8:
        out = 0 if m < -l + 1 else (m - 1 if spin == 1 else m)
    else:
        raise ValueError('j = %s and l = %d are not compatible' % (j, l))
    return 0 if (out < -l or out > l) else out


def real_to_complex_harmonics(lmax: int = LMAXX) -> NDArray[np.complex128]:
    """QE ``rot_ylm``: ``rot[lmax + m_complex, real index]`` for the ``ylmr2`` real harmonics."""
    n = 2 * lmax + 1
    rot = np.zeros((n, n), dtype=np.complex128)
    rot[lmax, 0] = 1.0
    sqrt2 = np.sqrt(2.0)
    for m in range(1, lmax + 1):
        n1 = 2 * m - 1  # cos column (0-based); sin column n1 + 1
        rot[lmax - m, n1] = (-1.0) ** m / sqrt2
        rot[lmax - m, n1 + 1] = -1j * (-1.0) ** m / sqrt2
        rot[lmax + m, n1] = 1.0 / sqrt2
        rot[lmax + m, n1 + 1] = 1j / sqrt2
    return rot


def spin_orbit_coefficient(
    l: int, j: float, mi: int, mk: int, s1: int, s2: int, rot: NDArray[np.complex128]
) -> complex:
    """QE ``fcoef(ih, kh, is1, is2)`` for two real-harmonic projectors ``mi, mk`` (0-based) of the same ``(l, j)``."""
    lmax = (rot.shape[0] - 1) // 2
    out = 0.0 + 0.0j
    for m in range(-l - 1, l + 1):
        m0 = lmax + spherical_index(l, j, m, s1)
        m1 = lmax + spherical_index(l, j, m, s2)
        out += (
            rot[m0, mi]
            * spinor_coefficient(l, j, m, s1)
            * np.conj(rot[m1, mk])
            * spinor_coefficient(l, j, m, s2)
        )
    return out


def relativistic_projectors(upf) -> dict:
    """Projectors of a fully relativistic UPF used with spin-orbit coupling (no averaging)."""
    betas = [np.asarray(b['wfc'], dtype=float) for b in upf.beta]
    cut = [
        int(b['cutoff_index']) if b['cutoff_index'] else betas[i].size
        for i, b in enumerate(upf.beta)
    ]
    if any(b.get('j') is None for b in upf.beta):
        raise ValueError('spin-orbit coupling needs a fully relativistic UPF (PP_SPIN_ORB)')
    return {
        'betas': betas,
        'lll': [int(b['l']) for b in upf.beta],
        'jjj': [float(b['j']) for b in upf.beta],
        'cutoff': cut,
        'dion_ry': np.asarray(upf.dion, dtype=float) * 2.0,
    }


class LocalPotentialTable:
    """``V^loc(q)`` of one species on QE's interpolation table (Ry, includes the Coulomb tail).

    Parameters
    ----------
    upf : UPF
        Pseudopotential (``vloc`` in Hartree on the UPF mesh, ``val`` the valence charge).
    omega : float
        Cell volume (bohr^3).
    qmax : float
        Largest ``|q + G|`` needed (bohr^-1); the table extends four steps beyond.
    dq : float, optional
        Table step (bohr^-1).
    """

    def __init__(self, upf, omega: float, qmax: float, dq: float = DQ):
        self.dq = float(dq)
        self.omega = float(omega)
        self.zp = float(upf.val)
        r = np.asarray(upf.r, dtype=float)
        msh = radial_cutoff_index(r)
        r, rab = r[:msh], np.asarray(upf.rab, dtype=float)[:msh]
        vloc = np.asarray(upf.vloc, dtype=float)[:msh] * 2.0  # Hartree -> Ry
        nqx = int(qmax / self.dq + 4)
        q = np.arange(1, nqx + 1) * self.dq  # table entries 1..nqx: q = (i-1) dq -> i = 1 is q = 0
        q = q - self.dq
        base = r * vloc + self.zp * E2 * erf(r)
        tab = np.empty(nqx + 1)
        tab[1] = qe_simpson(r * base, rab) * FPI / self.omega  # q = 0 entry (iq = 1)
        qq = q[1:]
        aux = base[None, :] * np.sin(qq[:, None] * r[None, :]) / qq[:, None]
        tab[2:] = np.array([qe_simpson(a, rab) for a in aux]) * FPI / self.omega
        tab[0] = qe_simpson(r * (r * vloc + self.zp * E2), rab) * FPI / self.omega  # G = 0 value
        self.table = tab

    def __call__(self, q_bohr: NDArray[np.float64]) -> NDArray[np.float64]:
        """``V^loc(|q+G|)`` in Ry for lengths in bohr^-1 (``|q+G| < 1e-4`` gives the G = 0 value)."""
        q = np.asarray(q_bohr, dtype=float)
        small = q * q < 1e-8
        out = lagrange4(self.table[1:], np.where(small, 0.0, q), self.dq)
        tail = np.where(
            small,
            0.0,
            -FPI
            / self.omega
            * self.zp
            * E2
            * np.exp(-np.where(small, 0.0, q * q) * 0.25)
            / np.where(small, 1.0, q * q),
        )
        return np.where(small, self.table[0], out + tail)


def bare_local_perturbation(
    tables: dict[str, LocalPotentialTable],
    species_of_atom: Sequence[str],
    tau_alat: NDArray[np.float64],
    q_cart: NDArray[np.float64],
    g_miller: NDArray[np.int_],
    g_cart: NDArray[np.float64],
    tpiba: float,
    fft_grid: tuple[int, int, int],
) -> NDArray[np.complex128]:
    """Periodic part of ``dV^loc / du_{kappa alpha}(q)`` on the grid (EPW ``compute_dvloc``).

    Parameters
    ----------
    tables : dict
        Species -> :class:`LocalPotentialTable`.
    species_of_atom : sequence of str
    tau_alat : ndarray ``(nat, 3)``
        Positions (Cartesian, alat).
    q_cart : ndarray ``(3,)``
        q (Cartesian, ``2 pi / alat``).
    g_miller, g_cart : ndarray ``(3, ng)`` int and ``(ng, 3)``
        The smooth G sphere (:func:`PAOFLOW.elphon.plane_waves.g_sphere`).
    tpiba : float
        ``2 pi / alat`` (bohr^-1).
    fft_grid : tuple of int
        Smooth FFT grid.

    Returns
    -------
    ndarray ``(3 nat, nr1, nr2, nr3)``, complex
        ``dvloc_{3 kappa + alpha}(r)`` in Ry/bohr.
    """
    from .plane_waves import box_indices

    tau = np.asarray(tau_alat, dtype=float)
    nat = tau.shape[0]
    qg = g_cart + np.asarray(q_cart, dtype=float)  # (ng, 3)
    qg_norm = np.sqrt(np.einsum('gi,gi->g', qg, qg)) * tpiba
    box = np.zeros((3 * nat,) + tuple(int(n) for n in fft_grid), dtype=np.complex128)
    i1, i2, i3 = box_indices(g_miller, fft_grid)
    for kappa in range(nat):
        vq = tables[species_of_atom[kappa]](qg_norm)  # (ng,)
        phase = np.exp(-2j * np.pi * (qg @ tau[kappa]))  # e^{-i (q+G).tau}
        common = -1j * tpiba * vq * phase  # (ng,)
        for alpha in range(3):
            box[3 * kappa + alpha, i1, i2, i3] = common * qg[:, alpha]
    return np.fft.ifftn(box, axes=(1, 2, 3)) * int(np.prod(fft_grid))


class KleinmanBylanderProjectors:
    """Kleinman-Bylander projectors ``beta_i(k+G)`` and ``D_ij`` (Ry) of all atoms (QE ``init_us_2``).

    Parameters
    ----------
    upf_by_species : dict
        Species -> :class:`PAOFLOW.inputs.read_upf.UPF` (norm-conserving).
    species_of_atom : sequence of str
        Species of every atom.
    omega : float
        Cell volume (bohr^3).
    qmax : float
        Largest ``|k+G|`` needed (bohr^-1): ``sqrt(ecutwfc)`` for the stored spheres.
    dq : float, optional
        Table step (bohr^-1).
    spin_orbit : bool, optional
        Fully relativistic projectors with the spinor coupling matrix (QE ``dvan_so``);
        otherwise fully relativistic UPFs are averaged as in a scalar run.

    Attributes
    ----------
    atom_of_projector : ndarray ``(nproj,)`` int
    l_of_projector : ndarray ``(nproj,)`` int
    D : ndarray ``(nproj, nproj)`` or None
        Block diagonal per atom, coupling projectors with equal ``(l, m)`` (Ry); ``None`` with
        spin-orbit coupling.
    D_so : ndarray ``(2, 2, nproj, nproj)`` complex or None
        Spinor coupling ``D^{s1 s2}_{ij} = dion_{nb_i nb_j} fcoef(i, j, s1, s2)`` (Ry).
    """

    def __init__(
        self,
        upf_by_species: dict,
        species_of_atom: Sequence[str],
        omega: float,
        qmax: float,
        dq: float = DQ,
        spin_orbit: bool = False,
    ):
        self.dq = float(dq)
        self.omega = float(omega)
        self.spin_orbit = bool(spin_orbit)
        self.species_of_atom = list(species_of_atom)
        nqx = int(qmax / self.dq + 4)
        self.tables: dict[str, NDArray[np.float64]] = {}
        self.lll: dict[str, list[int]] = {}
        self.jjj: dict[str, list[float]] = {}
        self.dion: dict[str, NDArray[np.float64]] = {}
        pref = FPI / np.sqrt(self.omega)
        for name, upf in upf_by_species.items():
            if getattr(upf, 'ptype', 'NC') != 'NC' or getattr(upf, 'has_augmentation', False):
                raise NotImplementedError(
                    '%s: only norm-conserving pseudopotentials are supported' % name
                )
            proj = relativistic_projectors(upf) if self.spin_orbit else scalar_projectors(upf)
            self.jjj[name] = list(proj.get('jjj', [None] * len(proj['lll'])))
            r = np.asarray(upf.r, dtype=float)
            rab = np.asarray(upf.rab, dtype=float)
            kkbeta = max(proj['cutoff']) if proj['cutoff'] else 0
            tab = np.zeros((len(proj['betas']), nqx))
            q = np.arange(nqx) * self.dq
            for nb, (beta, l) in enumerate(zip(proj['betas'], proj['lll'])):
                rr = r[:kkbeta]
                bes = spherical_jn(l, q[:, None] * rr[None, :])  # (nqx, kkbeta)
                aux = beta[None, :kkbeta] * bes * rr[None, :]
                tab[nb] = np.array([qe_simpson(a, rab[:kkbeta]) for a in aux]) * pref
            self.tables[name] = tab
            self.lll[name] = list(proj['lll'])
            self.dion[name] = np.asarray(proj['dion_ry'], dtype=float)
        # projector list: atom by atom, projector nb, m = 1..2l+1 (ylmr2 order)
        atom_of, l_of, nb_of, lm_of, j_of = [], [], [], [], []
        for kappa, name in enumerate(self.species_of_atom):
            for nb, l in enumerate(self.lll[name]):
                for m in range(2 * l + 1):
                    atom_of.append(kappa), l_of.append(l), nb_of.append(nb), lm_of.append(l * l + m)
                    j_of.append(self.jjj[name][nb] if self.jjj[name][nb] is not None else -1.0)
        self.atom_of_projector = np.array(atom_of, dtype=int)
        self.l_of_projector = np.array(l_of, dtype=int)
        self.j_of_projector = np.array(j_of, dtype=float)
        self._nb = np.array(nb_of, dtype=int)
        self._lm = np.array(lm_of, dtype=int)
        self.lmax = int(max([0] + [l for ls in self.lll.values() for l in ls]))
        nproj = self.atom_of_projector.size
        if self.spin_orbit:
            rot = real_to_complex_harmonics(max(LMAXX, self.lmax))
            D = np.zeros((2, 2, nproj, nproj), dtype=np.complex128)
            for i in range(nproj):
                for j in range(nproj):
                    same = (
                        self.atom_of_projector[i] == self.atom_of_projector[j]
                        and self.l_of_projector[i] == self.l_of_projector[j]
                        and abs(self.j_of_projector[i] - self.j_of_projector[j]) < 1e-7
                    )
                    if not same:
                        continue
                    name = self.species_of_atom[self.atom_of_projector[i]]
                    dion = self.dion[name][self._nb[i], self._nb[j]]
                    if dion == 0.0:
                        continue
                    l = int(self.l_of_projector[i])
                    mi, mk = int(self._lm[i] - l * l), int(self._lm[j] - l * l)
                    for s1 in (1, 2):
                        for s2 in (1, 2):
                            D[s1 - 1, s2 - 1, i, j] = dion * spin_orbit_coefficient(
                                l, float(self.j_of_projector[i]), mi, mk, s1, s2, rot
                            )
            self.D_so = D
            self.D = None
        else:
            D = np.zeros((nproj, nproj))
            for i in range(nproj):
                for j in range(nproj):
                    if (
                        self.atom_of_projector[i] == self.atom_of_projector[j]
                        and self._lm[i] == self._lm[j]
                    ):
                        name = self.species_of_atom[self.atom_of_projector[i]]
                        D[i, j] = self.dion[name][self._nb[i], self._nb[j]]
            self.D = D
            self.D_so = None

    @property
    def nproj(self) -> int:
        """Number of projectors (all atoms, all ``m``)."""
        return int(self.atom_of_projector.size)

    def at(
        self, kpg_cart: NDArray[np.float64], tau_alat: NDArray[np.float64], tpiba: float
    ) -> NDArray[np.complex128]:
        """``beta_i(k+G)`` for the vectors of one sphere.

        Parameters
        ----------
        kpg_cart : ndarray ``(npw, 3)``
            ``k + G`` (Cartesian, ``2 pi / alat``).
        tau_alat : ndarray ``(nat, 3)``
            Positions (Cartesian, alat).
        tpiba : float
            ``2 pi / alat`` (bohr^-1).

        Returns
        -------
        ndarray ``(nproj, npw)``, complex
            ``(4 pi / sqrt(Omega)) (-i)^l beta_l(|k+G|) Y_lm(k+G) exp(-i (k+G).tau)``.
        """
        kpg = np.asarray(kpg_cart, dtype=float)
        npw = kpg.shape[0]
        qn = np.sqrt(np.einsum('gi,gi->g', kpg, kpg)) * tpiba
        ylm = real_spherical_harmonics_qe(self.lmax, kpg)  # (npw, (lmax+1)^2)
        out = np.empty((self.nproj, npw), dtype=np.complex128)
        radial: dict[tuple[str, int], NDArray[np.float64]] = {}
        phases: dict[int, NDArray[np.complex128]] = {}
        for i in range(self.nproj):
            kappa = int(self.atom_of_projector[i])
            name = self.species_of_atom[kappa]
            key = (name, int(self._nb[i]))
            if key not in radial:
                radial[key] = lagrange4(self.tables[name][self._nb[i]], qn, self.dq)
            if kappa not in phases:
                phases[kappa] = np.exp(
                    -2j * np.pi * (kpg @ np.asarray(tau_alat[kappa], dtype=float))
                )
            l = int(self.l_of_projector[i])
            out[i] = radial[key] * ylm[:, self._lm[i]] * phases[kappa] * (-1j) ** l
        return out


def nonlocal_perturbation(
    kb: KleinmanBylanderProjectors,
    coefficients_k: NDArray[np.complex128],
    kpg_k: NDArray[np.float64],
    kpg_kq: NDArray[np.float64],
    tau_alat: NDArray[np.float64],
    tpiba: float,
    beta_k: NDArray[np.complex128] | None = None,
    beta_kq: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128]:
    """``dV^NL / du_{kappa alpha} |psi_nk>`` on the ``k+q+G`` sphere (EPW ``dvqpsi_us_only3``).

    Parameters
    ----------
    kb : KleinmanBylanderProjectors
    coefficients_k : ndarray ``(nb, npol * npw_k)``, complex
        Bands at k (spinor components concatenated for ``npol = 2``).
    kpg_k, kpg_kq : ndarray ``(npw_k, 3)``, ``(npw_kq, 3)``
        ``k + G`` and ``k + q + G'`` (Cartesian, ``2 pi / alat``); the latter are the
        vectors of the stored block at ``k' = k + q - G0``, so no umklapp relabelling is needed.
    tau_alat : ndarray ``(nat, 3)``
    tpiba : float
    beta_k, beta_kq : ndarray, optional
        Precomputed ``kb.at(...)`` of the two spheres.

    Returns
    -------
    ndarray ``(3 nat, nb, npol * npw_kq)``, complex
        ``sum_{i in kappa} beta_i(k+q+G') [ ps1_{i n alpha} + (k+q+G')_alpha ps2_{i n alpha} ]`` with
        ``ps1 = D alphap_alpha``, ``alphap_alpha = <beta| i tpiba (k+G)_alpha |psi>``,
        ``ps2 = -i tpiba D becp``, ``becp = <beta|psi>``; with spin-orbit coupling ``D`` is the
        spinor matrix ``D^{s1 s2}`` and the spin components are coupled (EPW ``ps1_nc``).
    """
    if beta_k is None:
        beta_k = kb.at(kpg_k, tau_alat, tpiba)
    if beta_kq is None:
        beta_kq = kb.at(kpg_kq, tau_alat, tpiba)
    c = coefficients_k
    nb = c.shape[0]
    npw_k, npw_kq = kpg_k.shape[0], kpg_kq.shape[0]
    npol = c.shape[1] // npw_k
    nat = len(kb.species_of_atom)
    comps = [c[:, s * npw_k : (s + 1) * npw_k] for s in range(npol)]
    becp = [beta_k.conj() @ cs.T for cs in comps]  # per spin: (nproj, nb)
    if kb.spin_orbit:
        if npol != 2:
            raise ValueError('spin-orbit projectors need spinor wavefunctions (npol = 2)')
        apply_d = lambda vec: [sum(kb.D_so[s1, s2] @ vec[s2] for s2 in range(2)) for s1 in range(2)]
    else:
        apply_d = lambda vec: [kb.D @ v for v in vec]
    ps2 = [-1j * tpiba * v for v in apply_d(becp)]
    out = np.zeros((3 * nat, nb, npol * npw_kq), dtype=np.complex128)
    for alpha in range(3):
        alphap = [beta_k.conj() @ (1j * tpiba * kpg_k[:, alpha][None, :] * cs).T for cs in comps]
        ps1 = apply_d(alphap)
        for kappa in range(nat):
            sel = kb.atom_of_projector == kappa
            if not sel.any():
                continue
            for s in range(npol):
                term = (
                    ps1[s][sel].T @ beta_kq[sel]
                    + (ps2[s][sel].T @ beta_kq[sel]) * kpg_kq[:, alpha][None, :]
                )
                out[3 * kappa + alpha, :, s * npw_kq : (s + 1) * npw_kq] = term
    return out
