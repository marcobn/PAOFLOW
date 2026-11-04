r"""Coupled linear response of a crystal in the PAO basis: local fields, ALDA and long-range kernels.

The independent-particle dielectric function of PAOFLOW lets the electrons respond to the macroscopic
field only. Here the response density of every :math:`\mathbf k` of a dense grid builds a lattice-periodic
induced potential that acts back on the electrons: the crystal local fields (the :math:`\mathbf G \neq 0`
Hartree potential), the adiabatic-LDA kernel and a long-range term :math:`-\alpha/q^2` (the LRC kernel of
Botti et al. 2004, or the head of the bootstrap kernel of Sharma et al. 2011). Atomic units inside (Hartree,
bohr, electron charge :math:`-1`); energies enter and leave in eV.

Notes
-----
For a field :math:`E_b(\omega)` the band-basis density matrix of every :math:`\mathbf k` obeys

.. math::

    \delta P_{nm}(\mathbf k) = \frac{f_m - f_n}{\omega - \epsilon_n(\mathbf k) + \epsilon_m(\mathbf k) + i\eta}
    \Big[r^b + \delta V_{\rm Hxc}[\delta n] + \alpha\,\mathbf P\cdot\mathbf r\Big]_{nm}(\mathbf k),

with :math:`r^b_{nm}` the interband position matrix of the band geometry (zero on the diagonal: the
macroscopic field is the external one), the induced density
:math:`\delta n(\mathbf r) = \sum_{\mathbf k} w_{\mathbf k}\sum_{\mu\nu}u^*_{\mu\mathbf k}u_{\nu\mathbf k}\,
\delta P^{\rm orb}_{\nu\mu}(\mathbf k)` on the cell grid, :math:`\delta V_H(\mathbf G\neq0) = 4\pi\,\delta n(\mathbf G)/G^2`,
:math:`\delta V_H(\mathbf G = 0) = 0`, the ALDA term :math:`f_{xc}[n_0]\,\delta n` and the macroscopic
polarization :math:`\mathbf P = -\Omega^{-1}\sum_{\mathbf k}w_{\mathbf k}\mathrm{Tr}[\delta P(\mathbf k)\,\mathbf r(\mathbf k)]`.
The unknown of the GMRES solve is the pair :math:`(\delta n, \mathbf P)`; the macroscopic dielectric function is
:math:`\epsilon_{M,ab}(\omega) = \delta_{ab} + 4\pi P_a/E_b`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .coupled_response import HARTREE_EV, lda_kernels

CRYSTAL_KERNELS = ('none', 'hartree', 'hartree+alda', 'hartree+alda+lrc', 'hartree+alda+bootstrap')


# --------------------------------------------------------------------------- periodic parts
def miller_sphere(
    k_cart: NDArray[np.float64], bg: NDArray[np.float64], ecut_scaled: float
) -> NDArray[np.int64]:
    r"""Miller indices of the plane waves with :math:`|\mathbf k + \mathbf G|^2 \le` ``ecut_scaled``.

    Parameters
    ----------
    k_cart : ndarray ``(3,)``
        k-point, Cartesian, :math:`2\pi/a` units.
    bg : ndarray ``(3, 3)``
        Reciprocal lattice vectors (rows, :math:`2\pi/a` units).
    ecut_scaled : float
        Cutoff in :math:`(2\pi/a)^2` units (``ecutwfc / (2 pi / alat)^2``).

    Returns
    -------
    ndarray ``(3, npw)`` int
    """
    at = np.linalg.inv(np.asarray(bg, dtype=float)).T  # rows: direct vectors (alat units)
    gmax = float(np.sqrt(ecut_scaled))
    bounds = [
        int(np.ceil(gmax * np.linalg.norm(at[i]) + abs(float(k_cart @ at[i])))) + 1
        for i in range(3)
    ]
    axes = [np.arange(-b, b + 1) for b in bounds]
    m = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    kg = m @ np.asarray(bg, dtype=float) + np.asarray(k_cart, dtype=float)[None, :]
    keep = np.sum(kg * kg, axis=1) <= ecut_scaled
    return np.ascontiguousarray(m[keep].T.astype(np.int64))


def periodic_parts_at_k(
    data_controller, k_cryst
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    """Plane-wave coefficients of the orthonormal PAO Bloch functions at an arbitrary k-point.

    Parameters
    ----------
    data_controller : DataController
        After ``projections`` with the internal basis.
    k_cryst : ndarray ``(3,)``
        k-point in crystal coordinates.

    Returns
    -------
    coeff : ndarray ``(nawf, npw)``, complex
        Coefficients of the periodic parts on the sphere ``|k+G|^2 <= ecutwfc`` (Löwdin basis at k).
    mill : ndarray ``(3, npw)`` int
    """
    from ..projection.do_atwfc_proj import calc_atwfc_k, ortho_atwfc_k

    arry, attr = data_controller.data_dicts()
    bg = np.asarray(arry['b_vectors'], dtype=float)
    alat = float(attr['alat'])
    tpiba = 2.0 * np.pi / alat
    ecut = float(attr['ecutwfc']) / tpiba**2
    k_cart = np.asarray(k_cryst, dtype=float) @ bg
    mill = miller_sphere(k_cart, bg, ecut)
    gkspace = {
        'xk': k_cart * tpiba,
        'igwx': int(mill.shape[1]),
        'mill': mill,
        'bg': bg,
        'gamma_only': False,
    }
    atwfc = calc_atwfc_k(arry['basis'], gkspace, dftSO=bool(attr.get('dftSO', False)))
    return np.asarray(ortho_atwfc_k(atwfc), dtype=complex), mill


def sphere_to_grid(
    coeff: NDArray, mill: NDArray[np.int64], nr: tuple[int, int, int], omega: float
) -> NDArray[np.complex128]:
    r"""Periodic parts on the FFT grid, ``(N_r, nawf)``, normalised as :math:`\sum_r|u|^2\,\Omega/N_r = 1`."""
    n = coeff.shape[0]
    box = np.zeros((n,) + tuple(nr), dtype=complex)
    box[:, mill[0] % nr[0], mill[1] % nr[1], mill[2] % nr[2]] = coeff
    u = np.fft.ifftn(box, axes=(1, 2, 3)) * (float(np.prod(nr)) / np.sqrt(float(omega)))
    return np.ascontiguousarray(u.reshape(n, -1).T)


def grid_for_sphere(mill: NDArray[np.int64], factor: float = 1.0) -> tuple[int, int, int]:
    """FFT grid holding the sphere (``factor`` 1: the wavefunction grid, 2: alias-free products)."""
    return tuple(int(factor * (2 * int(np.abs(mill[i]).max()) + 1)) for i in range(3))


def read_charge_density(
    savedir: str,
) -> tuple[NDArray[np.int64], NDArray[np.complex128], NDArray[np.float64]]:
    """Read QE's ``charge-density.dat`` (Fortran unformatted): ``(mill (3, ngm), rho(G) (nspin, ngm), b (3, 3))``.

    ``rho(G)`` is in electrons per cell volume (QE's convention: :math:`n(\\mathbf r) = \\sum_G \\rho(G)e^{iGr}`).
    """
    import os

    path = os.path.join(savedir, 'charge-density.dat')
    with open(path, 'rb') as fh:
        data = fh.read()

    def records(buf):
        pos = 0
        while pos < len(buf):
            (length,) = np.frombuffer(buf[pos : pos + 4], dtype='<i4')
            yield buf[pos + 4 : pos + 4 + int(length)]
            pos += 8 + int(length)

    rec = records(data)
    gamma_only, ngm, nspin = np.frombuffer(next(rec), dtype='<i4')[:3]
    b = np.frombuffer(next(rec), dtype='<f8').reshape(3, 3)
    mill = np.frombuffer(next(rec), dtype='<i4').reshape(int(ngm), 3).T
    rho = np.stack([np.frombuffer(next(rec), dtype='<c16') for _ in range(int(nspin))])
    if gamma_only:
        raise NotImplementedError('gamma-tricks charge density: unfold the half sphere first')
    return mill.astype(np.int64), rho, b


# --------------------------------------------------------------------------- kernel
class PeriodicKernel:
    r"""Local-field, ALDA and long-range kernels of a crystal on the cell FFT grid.

    Parameters
    ----------
    nr : tuple of int
        FFT grid of the cell.
    at_bohr : ndarray ``(3, 3)``
        Cell vectors (rows, bohr).
    omega : float
        Cell volume (bohr^3).
    xc : {None, 'lda'}, optional
    n0 : ndarray ``(N_r,)``, optional
        Ground-state density on the grid (for ``xc``).
    lrc_alpha : float, optional
        Long-range kernel :math:`-\alpha/q^2`, applied as the macroscopic field :math:`\alpha\mathbf P`.
    hartree : bool, optional
        Include the :math:`\mathbf G \neq 0` Hartree potential (``False`` for tests).
    """

    def __init__(
        self,
        nr: tuple[int, int, int],
        at_bohr: NDArray[np.float64],
        omega: float,
        xc: str | None = None,
        n0: NDArray | None = None,
        lrc_alpha: float = 0.0,
        hartree: bool = True,
        density_threshold: float = 1.0e-8,
    ) -> None:
        self.nr = tuple(int(x) for x in nr)
        self.nr_total = int(np.prod(self.nr))
        self.omega = float(omega)
        self.dv = self.omega / self.nr_total
        at = np.asarray(at_bohr, dtype=float)
        bg = 2.0 * np.pi * np.linalg.inv(at).T
        m = np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n) for n in self.nr], indexing='ij')
        G = sum(m[i][..., None] * bg[i] for i in range(3))
        G2 = np.sum(G * G, axis=-1)
        with np.errstate(divide='ignore'):
            v = np.where(G2 > 0, 4.0 * np.pi / np.where(G2 > 0, G2, 1.0), 0.0)
        self.v_coulomb = v if hartree else np.zeros_like(v)
        self.hartree = bool(hartree)
        self.lrc_alpha = float(lrc_alpha)
        self.xc = xc
        self.f_xc = self.g_xc = None
        if xc is not None:
            if xc != 'lda':
                raise ValueError("xc must be None or 'lda'")
            if n0 is None:
                raise ValueError("xc='lda' needs the ground-state density n0")
            self.f_xc, self.g_xc = lda_kernels(
                np.asarray(n0, dtype=float).reshape(-1), threshold=density_threshold
            )
        parts = []
        if hartree:
            parts.append('hartree')
        if xc:
            parts.append('alda')
        if lrc_alpha:
            parts.append('lrc(%.3f)' % lrc_alpha)
        self.label = '+'.join(parts) if parts else 'none'

    def apply_density(self, dn: NDArray) -> NDArray[np.complex128]:
        """Induced potential on the grid (flat) of the induced density ``dn`` (flat)."""
        dn = np.asarray(dn, dtype=complex).reshape(-1)
        V = np.fft.ifftn(self.v_coulomb * np.fft.fftn(dn.reshape(self.nr))).reshape(-1)
        if self.f_xc is not None:
            V = V + self.f_xc * dn
        return V


# --------------------------------------------------------------------------- channel
@dataclass
class CrystalChannel:
    """Band-basis data of the local k-points of a dense grid (one spin channel).

    Attributes
    ----------
    energies : ndarray ``(nk, n)``
        Band energies (Hartree, scissor included).
    eigenvectors : ndarray ``(nk, n, n)``
        Columns, orbital basis.
    position : ndarray ``(nk, 3, n, n)``
        Interband position matrix (bohr, band basis).
    occupations : ndarray ``(n,)``
    weights : ndarray ``(nk,)``
        k-weights (sum to 1 over the whole grid).
    coeff, mill : lists
        Periodic parts on the plane-wave sphere of every local k (:func:`periodic_parts_at_k`).
    nr : tuple of int
        Cell FFT grid of the density.
    omega : float
    at_bohr : ndarray ``(3, 3)``
    comm : MPI communicator or None
    """

    energies: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    position: NDArray[np.complex128]
    occupations: NDArray[np.float64]
    weights: NDArray[np.float64]
    coeff: list
    mill: list
    nr: tuple
    omega: float
    at_bohr: NDArray[np.float64]
    comm: object = None
    n0: NDArray | None = field(default=None, repr=False)

    @property
    def nk_local(self) -> int:
        return int(self.energies.shape[0])

    @property
    def n(self) -> int:
        return int(self.energies.shape[1])

    def periodic_table(self, ik: int) -> NDArray[np.complex128]:
        """``(N_r, n)`` periodic parts of local k-point ``ik`` on the grid."""
        return sphere_to_grid(self.coeff[ik], self.mill[ik], self.nr, self.omega)

    def coefficients(self, omega_ha: float, eta_ha: float) -> NDArray[np.complex128]:
        r""":math:`c_{nm}(\mathbf k) = (f_m - f_n)/(\omega - \epsilon_n + \epsilon_m + i\eta)`."""
        e, f = self.energies, self.occupations
        d = float(omega_ha) - (e[:, :, None] - e[:, None, :]) + 1j * float(eta_ha)
        return (f[None, None, :] - f[None, :, None]) / d

    def allreduce(self, array: NDArray) -> NDArray:
        if self.comm is None:
            return array
        from mpi4py import MPI

        self.comm.Allreduce(MPI.IN_PLACE, array, op=MPI.SUM)
        return array

    def ground_state_density(self) -> NDArray[np.float64]:
        """Ground-state density on the grid from the occupied states of the local k (allreduced)."""
        if self.n0 is not None:
            return self.n0
        n0 = np.zeros(int(np.prod(self.nr)))
        occ = self.occupations > 0
        for ik in range(self.nk_local):
            u = self.periodic_table(ik)  # (N_r, n)
            psi = u @ self.eigenvectors[ik][:, occ]  # (N_r, nocc)
            n0 += self.weights[ik] * ((np.abs(psi) ** 2) @ self.occupations[occ])
        self.n0 = self.allreduce(n0)
        return self.n0

    def density_and_polarization(
        self, dP_band: NDArray
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Induced density (flat, allreduced) and macroscopic polarization of ``dP_band`` ``(nk, n, n)``."""
        dn = np.zeros(int(np.prod(self.nr)), dtype=complex)
        P = np.zeros(3, dtype=complex)
        for ik in range(self.nk_local):
            U = self.eigenvectors[ik]
            dP_orb = U @ dP_band[ik] @ U.conj().T
            u = self.periodic_table(ik)
            dn += self.weights[ik] * np.einsum('rn,rn->r', u, u.conj() @ dP_orb.T)
            P -= self.weights[ik] * np.einsum('aij,ji->a', self.position[ik], dP_band[ik])
        self.allreduce(dn)
        self.allreduce(P)
        return dn, P / self.omega

    def band_potential(self, V_grid: NDArray, ik: int) -> NDArray[np.complex128]:
        r""":math:`U^\dagger\,[\int u^*_\mu V u_\nu]\,U` of local k-point ``ik``."""
        u = self.periodic_table(ik)
        V_orb = (u.conj().T @ (np.asarray(V_grid).reshape(-1)[:, None] * u)) * (
            self.omega / u.shape[0]
        )
        U = self.eigenvectors[ik]
        return U.conj().T @ V_orb @ U


# --------------------------------------------------------------------------- solver
def first_order_crystal(
    ch: CrystalChannel,
    kernel: PeriodicKernel,
    direction: int,
    omega_ha: float,
    eta_ha: float,
    tol: float = 1.0e-8,
    maxiter: int = 200,
) -> dict:
    """Coupled first-order response of the crystal to a unit field along ``direction``.

    Returns
    -------
    dict
        ``'dP'`` ``(nk_local, n, n)`` band-basis density matrices, ``'density'`` (flat, complex),
        ``'polarization'`` ``(3,)`` (a.u., per volume: the macroscopic susceptibility column),
        ``'iterations'``.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    c = ch.coefficients(omega_ha, eta_ha)
    r_b = ch.position[:, int(direction)]
    nr_tot = int(np.prod(ch.nr))
    count = [0]

    def respond(V_band_fn):
        dP = np.empty_like(c)
        for ik in range(ch.nk_local):
            dP[ik] = c[ik] * V_band_fn(ik)
        return dP

    def induced(X):
        dn, P = X[:nr_tot], X[nr_tot:]
        V_grid = kernel.apply_density(dn)

        def V_band(ik):
            V = ch.band_potential(V_grid, ik)
            if kernel.lrc_alpha:
                V = V + kernel.lrc_alpha * np.einsum('a,aij->ij', P, ch.position[ik])
            return V

        return respond(V_band)

    dP0 = respond(lambda ik: r_b[ik])
    dn0, P0 = ch.density_and_polarization(dP0)
    X0 = np.concatenate([dn0, P0])
    if kernel.label == 'none':
        return {'dP': dP0, 'density': dn0, 'polarization': P0, 'iterations': 0}

    def matvec(X):
        count[0] += 1
        dP = induced(X)
        dn, P = ch.density_and_polarization(dP)
        return X - np.concatenate([dn, P])

    op = LinearOperator((X0.size, X0.size), matvec=matvec, dtype=complex)
    try:
        X, info = gmres(op, X0, x0=X0, rtol=tol, maxiter=maxiter)
    except TypeError:
        X, info = gmres(op, X0, x0=X0, tol=tol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError('crystal response did not converge (GMRES info %d)' % info)
    dP = dP0 + induced(X)
    dn, P = ch.density_and_polarization(dP)
    return {'dP': dP, 'density': dn, 'polarization': P, 'iterations': count[0]}


def dielectric_function_crystal(
    ch: CrystalChannel,
    kernel: PeriodicKernel,
    energies_ev,
    eta_ev: float,
    directions=(0, 1, 2),
    tol: float = 1.0e-8,
    progress: bool = False,
) -> dict:
    r"""Macroscopic dielectric tensor :math:`\epsilon_{M,ab}(\omega) = \delta_{ab} + 4\pi P_a/E_b` with the kernel."""
    energies = np.atleast_1d(np.asarray(energies_ev, dtype=float))
    eps = np.zeros((energies.size, 3, 3), dtype=complex)
    rank = 0 if ch.comm is None else int(ch.comm.Get_rank())
    for ie, w in enumerate(energies):
        for b in directions:
            out = first_order_crystal(
                ch, kernel, b, w / HARTREE_EV, float(eta_ev) / HARTREE_EV, tol=tol
            )
            eps[ie, :, b] = 4.0 * np.pi * out['polarization']
            eps[ie, b, b] += 1.0
            if progress and rank == 0:
                print(
                    '  crystal response (%s): %.3f eV  E_%s  eps %.4f%+.4fj  (%d matvecs)'
                    % (
                        kernel.label,
                        w,
                        'xyz'[b],
                        eps[ie, b, b].real,
                        eps[ie, b, b].imag,
                        out['iterations'],
                    ),
                    flush=True,
                )
    return {'energy_ev': energies, 'epsilon': eps, 'kernel': kernel.label}


def bootstrap_alpha(eps_ipa_static: float, eps_lfe_static: float) -> float:
    r"""Head of the RPA-bootstrap kernel, :math:`\alpha = 4\pi\,\epsilon_M^{-1}(0)/[\epsilon_{\rm IPA}(0) - 1]`.

    ``eps_ipa_static`` is the independent-particle static dielectric constant (no local fields),
    ``eps_lfe_static`` the RPA one with local fields (Rigamonti et al. 2015, one-shot form).
    """
    return 4.0 * np.pi / (float(eps_lfe_static) * (float(eps_ipa_static) - 1.0))


# --------------------------------------------------------------------------- builder
def build_crystal_channel(
    data_controller,
    nk,
    scissor_ev: float = 0.0,
    position_dipoles=False,
    degeneracy_tol: float = 1.0e-4,
    hopping_cutoff: float | None = None,
    kblock: int = 64,
    grid_factor: float = 1.0,
    coeff_dtype=np.complex128,
    comm=None,
    nelec: float | None = None,
    progress: bool = False,
) -> CrystalChannel:
    """Dense-grid channel of a PAOFLOW run: band geometry of the IPA driver plus the periodic parts.

    Parameters
    ----------
    data_controller : DataController
        After ``pao_hamiltonian`` (``HRs``) with the internal basis.
    nk : sequence of int
        Dense k-grid (full grid; the induced density breaks the point group).
    scissor_ev : float, optional
    position_dipoles : bool or {'full', 'intra-atomic'}, optional
        Position dipoles in the interband position matrix.
    degeneracy_tol, hopping_cutoff, kblock
        As :func:`~PAOFLOW.response.nonlinear_optics.nonlinear_optical_spectra`.
    grid_factor : float, optional
        Density grid relative to the wavefunction sphere (1: coarse, validated; 2: alias-free).
    coeff_dtype : optional
        Storage of the periodic-part coefficients (``np.complex64`` halves the memory).
    comm : MPI communicator, optional
        k-points are split over its ranks.
    nelec : float, optional
        Electrons per cell (default ``attr['nelec']``).
    """
    from ..hamiltonian.bloch_derivatives import (
        bloch_terms_from_HRs,
        hamiltonian_derivatives_block,
        orbital_positions_from_data,
        velocity_and_curvature_operators,
    )
    from ..utils.communication import load_balancing
    from .nonlinear_optics import _k_grid, band_geometry, dipole_gradient_by_differences

    arrays, attr = data_controller.data_dicts()
    HRs = arrays['HRs']
    at = np.asarray(arrays['a_vectors'], dtype=float)
    alat = float(attr['alat'])
    omega = float(attr['omega'])
    nspin = HRs.shape[-1]
    if nspin != 1:
        raise NotImplementedError('the crystal coupled response handles one spin channel')
    spin_factor = 1.0 if attr.get('dftSO', False) else 2.0
    nelec = float(attr['nelec'] if nelec is None else nelec)
    nocc = int(round(nelec / spin_factor))
    positions = orbital_positions_from_data(data_controller)
    terms = bloch_terms_from_HRs(HRs, at, alat, positions, ispin=0, hopping_cutoff=hopping_cutoff)
    n = terms.nawf
    K = _k_grid(tuple(int(x) for x in nk))
    nkd = K.shape[0]
    size, rank = (1, 0) if comm is None else (comm.Get_size(), comm.Get_rank())
    kstart, kstop = load_balancing(size, rank, nkd)
    dipoles = None
    if position_dipoles:
        from ..hamiltonian.position_dipoles import position_dipole_operator

        level = position_dipoles if isinstance(position_dipoles, str) else 'full'
        dipoles = position_dipole_operator(
            data_controller, np.asarray(arrays['b_vectors'], dtype=float), alat, level=level
        )
    conduction = (np.arange(n) >= nocc).astype(float)
    energies, vectors, position = [], [], []
    for b0 in range(kstart, kstop, int(kblock)):
        kb = K[b0 : min(b0 + int(kblock), kstop)]
        H, dH, d2H = hamiltonian_derivatives_block(terms, kb)
        v_orb, w_orb = velocity_and_curvature_operators(terms, H, dH, d2H)
        E, U = np.linalg.eigh(H)
        if dipoles is not None:
            d_orb = np.asarray(dipoles(kb), dtype=complex)
            dg_orb = dipole_gradient_by_differences(dipoles, kb, at, alat, 1.0e-4)
            dg_orb += 1j * terms.separation_bohr[None, :, None] * d_orb[:, None]
            geometry = band_geometry(E, U, v_orb, w_orb, degeneracy_tol, d_orb, dg_orb)
        else:
            geometry = band_geometry(E, U, v_orb, w_orb, degeneracy_tol)
        energies.append((E + float(scissor_ev) * conduction[None, :]) / HARTREE_EV)
        vectors.append(U)
        position.append(geometry.position)
        if progress and rank == 0:
            print(
                '  crystal channel: k-points %d-%d of %d' % (b0, min(b0 + kblock, kstop), nkd),
                flush=True,
            )
    coeff, mill = [], []
    for ik in range(kstart, kstop):
        cf, ml = periodic_parts_at_k(data_controller, K[ik])
        coeff.append(np.asarray(cf, dtype=coeff_dtype))
        mill.append(ml)
    nr = grid_for_sphere(mill[0], grid_factor) if mill else (1, 1, 1)
    nr_all = np.array(nr)
    if comm is not None:
        from mpi4py import MPI

        comm.Allreduce(MPI.IN_PLACE, nr_all, op=MPI.MAX)
    occupations = np.zeros(n)
    occupations[:nocc] = spin_factor
    return CrystalChannel(
        energies=np.concatenate(energies) if energies else np.zeros((0, n)),
        eigenvectors=np.concatenate(vectors) if vectors else np.zeros((0, n, n), dtype=complex),
        position=np.concatenate(position) if position else np.zeros((0, 3, n, n), dtype=complex),
        occupations=occupations,
        weights=np.full(kstop - kstart, 1.0 / nkd),
        coeff=coeff,
        mill=mill,
        nr=tuple(int(x) for x in nr_all),
        omega=omega,
        at_bohr=at * alat,
        comm=comm,
    )


__all__ = [
    'CRYSTAL_KERNELS',
    'CrystalChannel',
    'PeriodicKernel',
    'bootstrap_alpha',
    'build_crystal_channel',
    'dielectric_function_crystal',
    'first_order_crystal',
    'grid_for_sphere',
    'miller_sphere',
    'periodic_parts_at_k',
    'read_charge_density',
    'sphere_to_grid',
]
