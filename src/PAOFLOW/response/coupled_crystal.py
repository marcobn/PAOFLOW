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
        'bg': bg.T * tpiba,  # calc_gkspace's convention: transposed, bohr^-1
        'gamma_only': False,
    }
    atwfc = calc_atwfc_k(arry['basis'], gkspace, dftSO=bool(attr.get('dftSO', False)))
    return np.asarray(ortho_atwfc_k(atwfc), dtype=complex), mill


def check_periodic_parts(data_controller, ik: int = 0) -> float:
    """Largest deviation between :func:`periodic_parts_at_k` at nscf k-point ``ik`` and the projection route.

    Both build the orthonormal PAO Bloch functions on the plane-wave sphere; the sets may order the
    sphere differently, so the comparison matches the Miller indices.  Returns ``max |c - c_ref|``
    (0 up to round-off when the descriptors agree).
    """
    from ..projection.do_atwfc_proj import calc_atwfc_k, calc_gkspace, ortho_atwfc_k

    arry, attr = data_controller.data_dicts()
    gk = calc_gkspace(data_controller, int(ik), gamma_only=False)
    ref = ortho_atwfc_k(calc_atwfc_k(arry['basis'], gk, dftSO=bool(attr.get('dftSO', False))))
    bg = np.asarray(arry['b_vectors'], dtype=float)
    k_cryst = np.linalg.solve(bg.T, np.asarray(arry['kgrid'][:, int(ik)], dtype=float))
    mine, mill = periodic_parts_at_k(data_controller, k_cryst)
    key_ref = {tuple(m): i for i, m in enumerate(np.asarray(gk['mill']).T)}
    order = np.array([key_ref[tuple(m)] for m in mill.T])
    return float(np.abs(mine - ref[:, order]).max())


def sphere_to_grid(
    coeff: NDArray, mill: NDArray[np.int64], nr: tuple[int, int, int], omega: float
) -> NDArray[np.complex128]:
    r"""Periodic parts on the FFT grid, ``(N_r, nawf)``, normalised as :math:`\sum_r|u|^2\,\Omega/N_r = 1`."""
    from scipy import fft as sfft

    n = coeff.shape[0]
    box = np.zeros((n,) + tuple(nr), dtype=complex)
    box[:, mill[0] % nr[0], mill[1] % nr[1], mill[2] % nr[2]] = coeff
    u = sfft.ifftn(box, axes=(1, 2, 3), workers=-1) * (float(np.prod(nr)) / np.sqrt(float(omega)))
    return np.ascontiguousarray(u.reshape(n, -1).T)


def smooth_size(n: int) -> int:
    """Smallest 5-smooth integer (factors 2, 3, 5 only) that is ``>= n``."""
    n = max(int(n), 1)
    while True:
        m = n
        for f in (2, 3, 5):
            while m % f == 0:
                m //= f
        if m == 1:
            return n
        n += 1


def grid_for_sphere(mill: NDArray[np.int64], factor: float = 1.0) -> tuple[int, int, int]:
    """FFT grid holding the sphere (``factor`` 1: the wavefunction grid, 2: alias-free products), 5-smooth sizes."""
    return tuple(
        smooth_size(int(np.ceil(factor * (2 * int(np.abs(mill[i]).max()) + 1)))) for i in range(3)
    )


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
        #: the kernel acts on the induced density on the grid (Hartree G != 0 or ALDA); a
        #: long-range term alone only needs the macroscopic polarization
        self.grid_active = self.hartree or self.f_xc is not None
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

    def second_order_density(self, dn1: NDArray, dn2: NDArray) -> NDArray[np.complex128] | None:
        r""":math:`g_{xc}\,\delta n_1\delta n_2` on the grid (``None`` without an xc kernel)."""
        if self.g_xc is None:
            return None
        return self.g_xc * np.asarray(dn1).reshape(-1) * np.asarray(dn2).reshape(-1)


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
    tables: dict = field(default_factory=dict, repr=False)
    position_derivative: NDArray | None = field(default=None, repr=False)
    position_delta: NDArray | None = field(default=None, repr=False)
    velocity: NDArray | None = field(default=None, repr=False)
    velocity_intra: NDArray | None = field(default=None, repr=False)
    energies_bare: NDArray | None = field(default=None, repr=False)
    k_cryst: NDArray | None = field(default=None, repr=False)
    parts_at: object = field(default=None, repr=False)

    @property
    def nk_local(self) -> int:
        return int(self.energies.shape[0])

    @property
    def n(self) -> int:
        return int(self.energies.shape[1])

    def cache_tables(self, max_gb: float = 8.0, dtype=np.complex128) -> bool:
        """Keep the grid tables of the local k-points in memory when they fit ``max_gb`` (per rank).

        Returns ``True`` when cached; otherwise the tables are rebuilt by FFT at every use.
        """
        per_k = int(np.prod(self.nr)) * self.n * np.dtype(dtype).itemsize
        if per_k * self.nk_local > float(max_gb) * 1e9:
            self.tables = {}
            return False
        self.tables = {
            ik: np.asarray(
                sphere_to_grid(self.coeff[ik], self.mill[ik], self.nr, self.omega), dtype=dtype
            )
            for ik in range(self.nk_local)
        }
        return True

    def periodic_table(self, ik: int) -> NDArray[np.complex128]:
        """``(N_r, n)`` periodic parts of local k-point ``ik`` on the grid (cached when available)."""
        if ik in self.tables:
            return self.tables[ik]
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

    def polarization_of(self, dP_band: NDArray) -> NDArray[np.complex128]:
        """Macroscopic polarization of ``dP_band`` ``(nk, n, n)`` (allreduced), no density."""
        P = np.zeros(3, dtype=complex)
        for ik in range(self.nk_local):
            P -= self.weights[ik] * np.einsum('aij,ji->a', self.position[ik], dP_band[ik])
        self.allreduce(P)
        return P / self.omega

    def band_potential(self, V_grid: NDArray, ik: int) -> NDArray[np.complex128]:
        r""":math:`U^\dagger\,[\int u^*_\mu V u_\nu]\,U` of local k-point ``ik``."""
        u = self.periodic_table(ik)
        V_orb = (u.conj().T @ (np.asarray(V_grid).reshape(-1)[:, None] * u)) * (
            self.omega / u.shape[0]
        )
        U = self.eigenvectors[ik]
        return U.conj().T @ V_orb @ U

    def band_potential_derivative(
        self, V_grid: NDArray, ik: int, step: float = 1.0e-3
    ) -> NDArray[np.complex128]:
        r"""Covariant derivative :math:`\delta V_{nm;b}` of the induced potential at local k-point ``ik``.

        :math:`[U^\dagger(\partial_b \delta V_{\rm orb})U]_{nm} + i[r^b, \delta V]_{nm}`, the orbital-basis
        derivative by central differences of the projection on the periodic parts at
        :math:`\mathbf k \pm h\,\hat e_b` (``step`` in bohr^-1); ``(3, n, n)``.
        """
        if self.parts_at is None or self.k_cryst is None:
            raise ValueError(
                'the channel has no periodic-part builder (build it from a data controller)'
            )
        B = 2.0 * np.pi * np.linalg.inv(self.at_bohr).T  # rows: reciprocal vectors (bohr^-1)
        Binv = np.linalg.inv(B)
        V = np.asarray(V_grid).reshape(-1)
        U = self.eigenvectors[ik]
        dV = self.band_potential(V, ik)
        out = np.empty((3, self.n, self.n), dtype=complex)
        for b in range(3):
            shift = step * Binv[b]  # crystal coordinates of h e_b
            orb = []
            for sign in (1.0, -1.0):
                cf, ml = self.parts_at(self.k_cryst[ik] + sign * shift)
                u = sphere_to_grid(cf, ml, self.nr, self.omega)
                orb.append((u.conj().T @ (V[:, None] * u)) * (self.omega / u.shape[0]))
            d_orb = (orb[0] - orb[1]) / (2.0 * step)
            out[b] = U.conj().T @ d_orb @ U + 1j * (
                self.position[ik, b] @ dV - dV @ self.position[ik, b]
            )
        return out


# --------------------------------------------------------------------------- solver
def first_order_crystal(
    ch: CrystalChannel,
    kernel: PeriodicKernel,
    direction: int,
    omega_ha: float,
    eta_ha: float,
    tol: float = 1.0e-8,
    maxiter: int = 200,
    cache_gb: float = 8.0,
    cache_dtype=np.complex128,
) -> dict:
    """Coupled first-order response of the crystal to a unit field along ``direction``.

    Parameters
    ----------
    cache_gb : float, optional
        Memory cap (per rank) for keeping the grid tables of the local k-points
        (:meth:`CrystalChannel.cache_tables`); 0 recomputes them at every use.
    cache_dtype : optional
        Storage of the cached tables (``np.complex64`` halves the memory).

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
    if not ch.tables and cache_gb > 0 and ch.coeff:
        ch.cache_tables(cache_gb, dtype=cache_dtype)

    def respond(V_band_fn):
        dP = np.empty_like(c)
        for ik in range(ch.nk_local):
            dP[ik] = c[ik] * V_band_fn(ik)
        return dP

    grid = kernel.grid_active

    def density_polarization(dP):
        # the first-order density matrix has no diagonal (populations): Tr[r dP] is the full
        # macroscopic polarization; the density is only needed by a grid kernel
        if grid:
            return ch.density_and_polarization(dP)
        return np.zeros(nr_tot, dtype=complex), ch.polarization_of(dP)

    def induced(X):
        dn, P = X[:nr_tot], X[nr_tot:]
        V_grid = kernel.apply_density(dn) if grid else None

        def V_band(ik):
            V = ch.band_potential(V_grid, ik) if grid else 0.0
            if kernel.lrc_alpha:
                V = V + kernel.lrc_alpha * np.einsum('a,aij->ij', P, ch.position[ik])
            return V

        return respond(V_band)

    dP0 = respond(lambda ik: r_b[ik])
    if kernel.label == 'none':
        return {
            'dP': dP0,
            'density': None,
            'polarization': ch.polarization_of(dP0),
            'iterations': 0,
        }
    dn0, P0 = density_polarization(dP0)
    X0 = np.concatenate([dn0, P0])

    def matvec(X):
        count[0] += 1
        dP = induced(X)
        dn, P = density_polarization(dP)
        return X - np.concatenate([dn, P])

    op = LinearOperator((X0.size, X0.size), matvec=matvec, dtype=complex)
    try:
        X, info = gmres(op, X0, x0=X0, rtol=tol, maxiter=maxiter)
    except TypeError:
        X, info = gmres(op, X0, x0=X0, tol=tol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError('crystal response did not converge (GMRES info %d)' % info)
    dP = dP0 + induced(X)
    dn, P = density_polarization(dP)
    return {'dP': dP, 'density': dn if grid else None, 'polarization': P, 'iterations': count[0]}


def dielectric_function_crystal(
    ch: CrystalChannel,
    kernel: PeriodicKernel,
    energies_ev,
    eta_ev: float,
    directions=(0, 1, 2),
    tol: float = 1.0e-8,
    progress: bool = False,
    cache_gb: float = 8.0,
    cache_dtype=np.complex128,
) -> dict:
    r"""Macroscopic dielectric tensor :math:`\epsilon_{M,ab}(\omega) = \delta_{ab} + 4\pi P_a/E_b` with the kernel."""
    energies = np.atleast_1d(np.asarray(energies_ev, dtype=float))
    eps = np.zeros((energies.size, 3, 3), dtype=complex)
    rank = 0 if ch.comm is None else int(ch.comm.Get_rank())
    for ie, w in enumerate(energies):
        for b in directions:
            out = first_order_crystal(
                ch, kernel, b, w / HARTREE_EV, float(eta_ev) / HARTREE_EV, tol=tol,
                cache_gb=cache_gb, cache_dtype=cache_dtype,
            )  # fmt: skip
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


def second_order_crystal(
    ch: CrystalChannel,
    kernel: PeriodicKernel,
    b: int,
    c: int,
    omega_ha: float,
    eta_ha: float,
    first_b: dict,
    first_c: dict,
    tol: float = 1.0e-8,
    maxiter: int = 200,
    cache_gb: float = 8.0,
    cache_dtype=np.complex128,
    derivative_step: float = 1.0e-3,
) -> dict:
    r"""Second-order response of the crystal to :math:`E_b(\omega)E_c(\omega)` (second-harmonic generation).

    Parameters
    ----------
    ch, kernel, tol, maxiter, cache_gb, cache_dtype
        As :func:`first_order_crystal`.
    b, c : int
        Field directions.
    omega_ha, eta_ha : float
        Photon energy and damping (Hartree); the response is at :math:`2\omega`.
    first_b, first_c : dict
        Results of :func:`first_order_crystal` for the two directions at ``omega_ha``.
    derivative_step : float, optional
        Step (bohr^-1) of the covariant derivative of the induced potential.

    Returns
    -------
    dict
        ``'dP'`` ``(nk_local, n, n)``, ``'polarization'`` ``(3,)`` (second-order macroscopic polarization
        per unit fields, a.u.: :math:`\chi^{(2)}_{abc}` in atomic units), ``'iterations'``.

    Notes
    -----
    Length gauge (Aversa and Sipe 1995): with the dressed first-order potentials
    :math:`V_b = r^b + \delta V_b` and densities :math:`\delta P_b`, the particle-hole blocks obey

    .. math::

        (2\omega - \omega'_{nm} + i\eta)\,\delta P^{(2)}_{nm} = S_{nm} + (f_m - f_n)\big[K[\delta P^{(2)}] + \tfrac12 V_g\big]_{nm},\quad
        S = \tfrac12\big([V_b,\delta P_c] + [V_c,\delta P_b]\big) + \tfrac{i}{2}\big[(\delta P_c)_{;b} + (\delta P_b)_{;c}\big],

    with the covariant derivative of the first-order density
    :math:`(\delta P_c)_{nm;b} = (f_m - f_n)\,[(V_c)_{nm;b}/d_{nm} + (v^b_{\rm intra}V_c - V_cv^b_{\rm intra})_{nm}/d_{nm}^2]`,
    :math:`d_{nm} = \omega - \omega'_{nm} + i\eta`, carrying the intraband (k-derivative) term of the
    length gauge; the occupied-occupied and empty-empty blocks follow from :math:`P^2 = g_sP`. The
    polarization is the current, :math:`P^{(2)}_a = J_a/(-2i\omega)`,
    :math:`J_a = -\Omega^{-1}\sum_{\mathbf k}w_{\mathbf k}[\sum_{n\neq m}\delta P^{(2)}_{nm}v'^a_{mn} + \sum_n\delta P^{(2)}_{nn}v^a_{nn}]`,
    :math:`v'^a_{mn} = i\omega'_{mn}r^a_{mn}` (scissor-consistent), reported in the sign convention of
    :func:`~PAOFLOW.response.nonlinear_optics.nonlinear_optical_spectra` (charge :math:`e > 0`).  With the
    kernel off this reproduces that driver: real parts to 0.1-0.4 % off resonance and 1-4 % near the
    :math:`2\omega` resonance; the imaginary parts differ off resonance by the damping placement
    (:math:`\eta` in every denominator here, in the resonant ones there), a few per cent of the
    dominant component.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    if ch.position_derivative is None:
        raise ValueError(
            'the channel lacks the second-order geometry (build it with channel_from_HRs)'
        )
    if not ch.tables and cache_gb > 0 and ch.coeff:
        ch.cache_tables(cache_gb, dtype=cache_dtype)
    w, eta = float(omega_ha), float(eta_ha)
    e, f = ch.energies, ch.occupations
    fnm = f[None, :] - f[:, None]  # f_m - f_n
    occ = f > 0
    oo = occ[:, None] & occ[None, :]
    vv = ~occ[:, None] & ~occ[None, :]
    d1 = w - (e[:, :, None] - e[:, None, :]) + 1j * eta
    d2 = 2.0 * w - (e[:, :, None] - e[:, None, :]) + 1j * eta
    c2 = fnm[None] / d2
    nr_tot = int(np.prod(ch.nr))
    active = kernel.label != 'none'
    grid = active and kernel.grid_active
    alpha = kernel.lrc_alpha if active else 0.0
    V_grid = (
        {
            'b': kernel.apply_density(first_b['density']),
            'c': kernel.apply_density(first_c['density']),
        }
        if grid
        else {}
    )
    V_g = kernel.second_order_density(first_b['density'], first_c['density']) if grid else None

    def density_polarization(dP):
        # the macroscopic polarization of a second-order density matrix is the one of its
        # current, J/(-2 i omega): the diagonal (population) part carries the intraband
        # polarization that Tr[r_inter dP] misses; the density on the grid only for a grid
        # kernel.  _current_polarization carries the e > 0 output convention (-1): the field
        # alpha P of the kernel needs the physical sign of the first-order solve
        dn = ch.density_and_polarization(dP)[0] if grid else np.zeros(nr_tot, dtype=complex)
        return dn, -_current_polarization(ch, dP, w)

    P1 = {'b': first_b['dP'], 'c': first_c['dP']}
    pol1 = {'b': first_b['polarization'], 'c': first_c['polarization']}
    dirs = {'b': int(b), 'c': int(c)}
    S_e = np.empty_like(c2)
    P_block = np.empty_like(c2)
    for ik in range(ch.nk_local):
        r = ch.position[ik]
        vin = ch.velocity_intra[ik]
        Vt, Vd = {}, {}
        for lab in ('b', 'c'):
            a_ = dirs[lab]
            Vt[lab] = r[a_].copy()
            Vd[lab] = ch.position_derivative[ik][:, a_].copy()  # [direction] -> r^a_{;direction}
            if grid:
                Vt[lab] = Vt[lab] + ch.band_potential(V_grid[lab], ik)
                Vd[lab] = Vd[lab] + ch.band_potential_derivative(V_grid[lab], ik, derivative_step)
            if active:
                if alpha:
                    # macroscopic field alpha P of the long-range kernel: alpha P.r and its
                    # covariant derivative alpha P_b r^b_{;d}, the same dressing as the grid potential
                    Vt[lab] = Vt[lab] + alpha * np.einsum('a,aij->ij', pol1[lab], r)
                    Vd[lab] = Vd[lab] + alpha * np.einsum(
                        'b,dbij->dij', pol1[lab], ch.position_derivative[ik]
                    )
        dd = d1[ik]
        D = {}
        for lab, other in (('c', 'b'), ('b', 'c')):
            # derivative of leg `lab` along the macroscopic field of the other leg: the bare
            # axis plus, with the long-range kernel, alpha times its induced polarization
            e_eff = np.zeros(3, dtype=complex)
            e_eff[dirs[other]] = 1.0
            if alpha:
                e_eff = e_eff + alpha * pol1[other]
            v_eff = np.einsum('d,dij->ij', e_eff, vin)
            comm_v = v_eff @ Vt[lab] - Vt[lab] @ v_eff
            D[lab] = fnm * (np.einsum('d,dij->ij', e_eff, Vd[lab]) / dd + comm_v / dd**2)
        Pb, Pc = P1['b'][ik], P1['c'][ik]
        S = 0.5 * (Vt['b'] @ Pc - Pc @ Vt['b'] + Vt['c'] @ Pb - Pb @ Vt['c']) + 0.5j * (
            D['c'] + D['b']
        )
        Q = 0.5 * (Pb @ Pc + Pc @ Pb) / float(ch.occupations.max())
        P_block[ik] = np.where(vv, Q, 0.0) - np.where(oo, Q, 0.0)
        S_e[ik] = np.where(oo | vv, 0.0, S / d2[ik])
    if active:
        dn_blk, P_blk = density_polarization(P_block)
        Vb_grid = None
        if grid:
            Vb_grid = kernel.apply_density(dn_blk)
            if V_g is not None:
                Vb_grid = Vb_grid + 0.5 * V_g
        for ik in range(ch.nk_local):
            Vk = ch.band_potential(Vb_grid, ik) if grid else 0.0
            if alpha:
                Vk = Vk + alpha * np.einsum('a,aij->ij', P_blk, ch.position[ik])
            S_e[ik] = S_e[ik] + c2[ik] * Vk
    count = [0]

    def induced(X):
        dn, P = X[:nr_tot], X[nr_tot:]
        Vg = kernel.apply_density(dn) if grid else None
        out = np.empty_like(c2)
        for ik in range(ch.nk_local):
            Vk = ch.band_potential(Vg, ik) if grid else 0.0
            if alpha:
                Vk = Vk + alpha * np.einsum('a,aij->ij', P, ch.position[ik])
            out[ik] = c2[ik] * Vk
        return out

    if active:
        dn0, P0 = density_polarization(S_e)
        X0 = np.concatenate([dn0, P0])

        def matvec(X):
            count[0] += 1
            dn, P = density_polarization(induced(X))
            return X - np.concatenate([dn, P])

        op = LinearOperator((X0.size, X0.size), matvec=matvec, dtype=complex)
        try:
            X, info = gmres(op, X0, x0=X0, rtol=tol, maxiter=maxiter)
        except TypeError:
            X, info = gmres(op, X0, x0=X0, tol=tol, maxiter=maxiter)
        if info != 0:
            raise RuntimeError(
                'second-order crystal response did not converge (GMRES info %d)' % info
            )
        dP = S_e + induced(X) + P_block
    else:
        dP = S_e + P_block
    return {'dP': dP, 'polarization': _current_polarization(ch, dP, w), 'iterations': count[0]}


def _current_polarization(ch: CrystalChannel, dP: NDArray, w: float) -> NDArray[np.complex128]:
    r"""Macroscopic polarization :math:`P_a = J_a/(-2i\omega)` of a second-order density matrix.

    The current :math:`J_a = -\sum_k w_k\,{\rm Tr}[\delta P\, v^a]/\Omega` uses the interband
    velocity :math:`v^a_{nm} = i\omega_{nm} r^a_{nm}` and the intraband (diagonal) velocity; the
    latter carries the polarization of the population changes, absent from
    :math:`{\rm Tr}[r_{\rm inter}\,\delta P]`.  Sign convention of
    :func:`~PAOFLOW.response.nonlinear_optics.nonlinear_optical_spectra` (the charge enters as
    :math:`e^3` with :math:`e > 0`): the electronic polarization (charge -1) is multiplied by -1.
    """
    e = ch.energies
    J = np.zeros(3, dtype=complex)
    for ik in range(ch.nk_local):
        om = e[ik][:, None] - e[ik][None, :]  # omega'_nm
        for a_ in range(3):
            v = 1j * om * ch.position[ik, a_]  # v'^a_nm = i omega'_nm r^a_nm
            np.fill_diagonal(v, np.diag(ch.velocity[ik, a_]))
            J[a_] -= ch.weights[ik] * np.einsum('nm,mn->', dP[ik], v)
    ch.allreduce(J)
    J /= ch.omega
    return -J / (-2.0j * w)


#: chi^(2) in pm/V per atomic unit (P = eps0 chi E E; e^3 / (eps0 E_h^2) = 24.43 pm/V)
CHI2_AU_TO_PM_V = 24.43


def shg_crystal(
    ch: CrystalChannel,
    kernel: PeriodicKernel,
    components,
    energies_ev,
    eta_ev: float,
    tol: float = 1.0e-8,
    progress: bool = False,
    cache_gb: float = 8.0,
    cache_dtype=np.complex128,
) -> dict:
    r""":math:`\chi^{(2)}_{abc}(-2\omega;\omega,\omega)` of the crystal with the kernel, in pm/V (:math:`P = \epsilon_0\chi EE`).

    Parameters
    ----------
    components : sequence of ``(a, b, c)``
    energies_ev : array_like
    eta_ev : float

    Returns
    -------
    dict
        ``'energy_ev'``, ``'chi'`` ``(ncomp, ne)`` complex (pm/V), ``'chi_au'``, ``'kernel'``.
    """
    energies = np.atleast_1d(np.asarray(energies_ev, dtype=float))
    comps = [tuple(int(x) for x in cmp) for cmp in components]
    chi = np.zeros((len(comps), energies.size), dtype=complex)
    rank = 0 if ch.comm is None else int(ch.comm.Get_rank())
    for ie, w in enumerate(energies):
        wh, eh = w / HARTREE_EV, float(eta_ev) / HARTREE_EV
        needed = sorted({bb for _, bb, _ in comps} | {cc for _, _, cc in comps})
        first = {
            d: first_order_crystal(
                ch, kernel, d, wh, eh, tol=tol, cache_gb=cache_gb, cache_dtype=cache_dtype
            )
            for d in needed
        }
        pairs = {}
        for ic, (a, bb, cc) in enumerate(comps):
            key = (min(bb, cc), max(bb, cc))
            if key not in pairs:
                pairs[key] = second_order_crystal(
                    ch, kernel, key[0], key[1], wh, eh, first[key[0]], first[key[1]], tol=tol,
                    cache_gb=cache_gb, cache_dtype=cache_dtype,
                )  # fmt: skip
            chi[ic, ie] = pairs[key]['polarization'][a] * CHI2_AU_TO_PM_V
            if progress and rank == 0:
                print(
                    '  crystal SHG (%s): %.3f eV  chi_%s%s%s = %.3f%+.3fj pm/V'
                    % (
                        kernel.label,
                        w,
                        'xyz'[a],
                        'xyz'[bb],
                        'xyz'[cc],
                        chi[ic, ie].real,
                        chi[ic, ie].imag,
                    ),
                    flush=True,
                )
    return {
        'energy_ev': energies,
        'chi': chi,
        'chi_au': chi / CHI2_AU_TO_PM_V,
        'kernel': kernel.label,
    }


def bootstrap_alpha(eps_ipa_static: float, eps_lfe_static: float) -> float:
    r"""Head of the RPA-bootstrap kernel, :math:`\alpha = 4\pi\,\epsilon_M^{-1}(0)/[\epsilon_{\rm IPA}(0) - 1]`.

    ``eps_ipa_static`` is the independent-particle static dielectric constant (no local fields),
    ``eps_lfe_static`` the RPA one with local fields (Rigamonti et al. 2015, one-shot form).
    """
    return 4.0 * np.pi / (float(eps_lfe_static) * (float(eps_ipa_static) - 1.0))


# --------------------------------------------------------------------------- builder
def channel_from_HRs(
    HRs: NDArray,
    a_vectors: NDArray[np.float64],
    alat: float,
    omega: float,
    positions_cryst,
    nelec: float,
    nk,
    scissor_ev: float = 0.0,
    spin_factor: float = 2.0,
    dipoles=None,
    degeneracy_tol: float = 1.0e-4,
    hopping_cutoff: float | None = None,
    kblock: int = 64,
    comm=None,
    progress: bool = False,
) -> CrystalChannel:
    """Dense-grid channel from a real-space PAO Hamiltonian (the band geometry of the IPA driver).

    Parameters
    ----------
    HRs : ndarray ``(nawf, nawf, n1, n2, n3, nspin)``
        Real-space PAO Hamiltonian (eV).
    a_vectors, alat, omega
        Direct lattice (rows, alat units), lattice parameter (bohr), cell volume (bohr^3).
    positions_cryst : ndarray ``(nawf, 3)`` or None
        Orbital centres (crystal coordinates).
    nelec : float
        Electrons per cell.
    nk : sequence of int
        Dense k-grid (full grid).
    scissor_ev, spin_factor, degeneracy_tol, hopping_cutoff, kblock
        As :func:`~PAOFLOW.response.nonlinear_optics.nonlinear_optical_spectra`.
    dipoles : callable, optional
        Position-dipole operator (``K_cryst -> (nk, 3, nawf, nawf)``).
    comm : MPI communicator, optional
        k-points are split over its ranks.

    Returns
    -------
    CrystalChannel
        Without periodic parts (``coeff``/``mill`` empty, ``nr = (1, 1, 1)``): enough for the
        kernel-off responses and the tests; :func:`build_crystal_channel` adds them.
    """
    from ..hamiltonian.bloch_derivatives import (
        bloch_terms_from_HRs,
        hamiltonian_derivatives_block,
        velocity_and_curvature_operators,
    )
    from ..utils.communication import load_balancing
    from .nonlinear_optics import _k_grid, band_geometry, dipole_gradient_by_differences

    HRs = np.asarray(HRs)
    if HRs.shape[-1] != 1:
        raise NotImplementedError('the crystal coupled response handles one spin channel')
    at = np.asarray(a_vectors, dtype=float)
    nocc = int(round(float(nelec) / float(spin_factor)))
    terms = bloch_terms_from_HRs(
        HRs, at, float(alat), positions_cryst, ispin=0, hopping_cutoff=hopping_cutoff
    )
    n = terms.nawf
    K = _k_grid(tuple(int(x) for x in nk))
    nkd = K.shape[0]
    size, rank = (1, 0) if comm is None else (comm.Get_size(), comm.Get_rank())
    kstart, kstop = load_balancing(size, rank, nkd)
    conduction = (np.arange(n) >= nocc).astype(float)
    energies, bare, vectors, position, pderiv, pdelta, velocity, vintra = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for b0 in range(kstart, kstop, int(kblock)):
        kb = K[b0 : min(b0 + int(kblock), kstop)]
        H, dH, d2H = hamiltonian_derivatives_block(terms, kb)
        v_orb, w_orb = velocity_and_curvature_operators(terms, H, dH, d2H)
        E, U = np.linalg.eigh(H)
        if dipoles is not None:
            d_orb = np.asarray(dipoles(kb), dtype=complex)
            dg_orb = dipole_gradient_by_differences(dipoles, kb, at, float(alat), 1.0e-4)
            dg_orb += 1j * terms.separation_bohr[None, :, None] * d_orb[:, None]
            geometry = band_geometry(E, U, v_orb, w_orb, degeneracy_tol, d_orb, dg_orb)
        else:
            geometry = band_geometry(E, U, v_orb, w_orb, degeneracy_tol)
        energies.append((E + float(scissor_ev) * conduction[None, :]) / HARTREE_EV)
        bare.append(E / HARTREE_EV)
        vectors.append(U)
        position.append(geometry.position)
        pderiv.append(geometry.position_derivative)
        pdelta.append(geometry.position_delta / HARTREE_EV)
        velocity.append(geometry.velocity / HARTREE_EV)
        vintra.append(geometry.velocity_intra / HARTREE_EV)
        if progress and rank == 0:
            print(
                '  crystal channel: k-points %d-%d of %d' % (b0, min(b0 + kblock, kstop), nkd),
                flush=True,
            )
    occupations = np.zeros(n)
    occupations[:nocc] = float(spin_factor)
    cat = (
        lambda arrs, shape, dtype=complex: np.concatenate(arrs)
        if arrs
        else np.zeros(shape, dtype=dtype)
    )  # noqa: E731
    return CrystalChannel(
        energies=cat(energies, (0, n), float),
        eigenvectors=cat(vectors, (0, n, n)),
        position=cat(position, (0, 3, n, n)),
        occupations=occupations,
        weights=np.full(kstop - kstart, 1.0 / nkd),
        coeff=[],
        mill=[],
        nr=(1, 1, 1),
        omega=float(omega),
        at_bohr=at * float(alat),
        comm=comm,
        position_derivative=cat(pderiv, (0, 3, 3, n, n)),
        position_delta=cat(pdelta, (0, 3, 3, n, n)),
        velocity=cat(velocity, (0, 3, n, n)),
        velocity_intra=cat(vintra, (0, 3, n, n)),
        energies_bare=cat(bare, (0, n), float),
        k_cryst=K[kstart:kstop],
    )


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
    """Dense-grid channel of a PAOFLOW run: :func:`channel_from_HRs` plus the periodic parts.

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
    from ..hamiltonian.bloch_derivatives import orbital_positions_from_data
    from ..utils.communication import load_balancing
    from .nonlinear_optics import _k_grid

    arrays, attr = data_controller.data_dicts()
    at = np.asarray(arrays['a_vectors'], dtype=float)
    alat = float(attr['alat'])
    spin_factor = 1.0 if attr.get('dftSO', False) else 2.0
    dipoles = None
    if position_dipoles:
        from ..hamiltonian.position_dipoles import position_dipole_operator

        level = position_dipoles if isinstance(position_dipoles, str) else 'full'
        dipoles = position_dipole_operator(
            data_controller, np.asarray(arrays['b_vectors'], dtype=float), alat, level=level
        )
    ch = channel_from_HRs(
        arrays['HRs'], at, alat, float(attr['omega']), orbital_positions_from_data(data_controller),
        float(attr['nelec'] if nelec is None else nelec), nk, scissor_ev=scissor_ev,
        spin_factor=spin_factor, dipoles=dipoles, degeneracy_tol=degeneracy_tol,
        hopping_cutoff=hopping_cutoff, kblock=kblock, comm=comm, progress=progress,
    )  # fmt: skip
    K = _k_grid(tuple(int(x) for x in nk))
    size, rank = (1, 0) if comm is None else (comm.Get_size(), comm.Get_rank())
    kstart, kstop = load_balancing(size, rank, K.shape[0])
    coeff, mill = [], []
    for ik in range(kstart, kstop):
        cf, ml = periodic_parts_at_k(data_controller, K[ik])
        coeff.append(np.asarray(cf, dtype=coeff_dtype))
        mill.append(ml)
    nr_all = np.array(grid_for_sphere(mill[0], grid_factor) if mill else (1, 1, 1))
    if comm is not None:
        from mpi4py import MPI

        comm.Allreduce(MPI.IN_PLACE, nr_all, op=MPI.MAX)
    ch.coeff, ch.mill, ch.nr = coeff, mill, tuple(int(x) for x in nr_all)
    ch.parts_at = lambda k_cryst: periodic_parts_at_k(data_controller, k_cryst)
    return ch


__all__ = [
    'CRYSTAL_KERNELS',
    'CrystalChannel',
    'PeriodicKernel',
    'bootstrap_alpha',
    'build_crystal_channel',
    'channel_from_HRs',
    'check_periodic_parts',
    'dielectric_function_crystal',
    'first_order_crystal',
    'grid_for_sphere',
    'miller_sphere',
    'periodic_parts_at_k',
    'read_charge_density',
    'second_order_crystal',
    'shg_crystal',
    'sphere_to_grid',
]
