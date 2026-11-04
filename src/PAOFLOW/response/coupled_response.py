r"""Coupled (RPA / adiabatic-LDA level) second-order response of a molecule in the PAO basis.

The independent-particle hyperpolarizability of :mod:`PAOFLOW.response.nonlinear_optics` lets the
electrons respond to the external field only.  Here the induced Hartree (and optionally
exchange-correlation) potential of the first- and second-order density acts back on the
electrons: the "local-field" screening that reduces :math:`\beta` of push-pull chromophores by a
factor 3-5.  Everything is written in the orthonormal PAO basis of one spin channel with the
Gamma-point Hamiltonian :math:`H` (Hartree), the dipole operator :math:`D_a` (bohr) and the
density matrix :math:`P = g_s\sum_{\rm occ}|n\rangle\langle n|` (:math:`g_s` the spin factor);
atomic units throughout, electrons carry the charge :math:`-1`, so the coupling to a field
:math:`\mathbf E` is :math:`+\mathbf E\cdot\mathbf D` and the dipole is
:math:`\mu_a = -\mathrm{Tr}[P D_a] + \sum_I Z_I R_{I,a}`.

Notes
-----
First order (frequency :math:`\omega`, damping :math:`\eta`), in the eigenbasis of :math:`H`:

.. math::

    \delta P_{nm}(\omega) = \frac{f_m - f_n}{\omega - \epsilon_n + \epsilon_m + i\eta}
    \left[V^{\rm ext} + K[\delta P(\omega)]\right]_{nm},

a linear system in the particle-hole space solved with GMRES; :math:`K` is the kernel
(:class:`MonopoleKernel`: Hartree of the atomic monopoles; :class:`GridKernel`: Hartree of the
density on the real-space grid, optionally with the ALDA :math:`f_{xc}`).

Second order, for the fields :math:`E_b(\omega_1)` and :math:`E_c(\omega_2)`,
:math:`\omega_\sigma = \omega_1 + \omega_2`:

.. math::

    (\omega_\sigma - \epsilon_n + \epsilon_m + i\eta)\,\delta P^{(2)}_{nm} =
    S_{nm} + (f_m - f_n)\left[K[\delta P^{(2)}] + V_g\right]_{nm},\qquad
    S = \tfrac12\left([V^{(1)}_b, \delta P_c] + [V^{(1)}_c, \delta P_b]\right),

with :math:`V^{(1)} = V^{\rm ext} + K[\delta P]` the effective first-order potentials and
:math:`V_g = \tfrac12 g_{xc}\,\delta n_b\delta n_c` the second functional derivative (zero at the
Hartree level); the occupied-occupied and empty-empty blocks follow from :math:`P^2 = g_sP`.  The Taylor
hyperpolarizability is :math:`\beta^T_{abc}(-\omega_\sigma;\omega_1,\omega_2) =
-2\,\mathrm{Tr}[\delta P^{(2)}_{bc} D_a]` (the "B" one is half of it), which the static
finite-field route :func:`static_hyperpolarizability` reproduces at :math:`\omega \to 0`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903

#: Hubbard parameters (Hartree) of the atomic monopoles, the DFTB ``mio-1-1`` values; other
#: elements fall back on :data:`HUBBARD_U_DEFAULT_HA`.
HUBBARD_U_HA = {'H': 0.4195, 'C': 0.3647, 'N': 0.4304, 'O': 0.4954, 'S': 0.3288}
HUBBARD_U_DEFAULT_HA = 0.40
KERNELS = ('none', 'monopole', 'hartree', 'hartree+alda')


# --------------------------------------------------------------------------- the system
@dataclass
class MolecularChannel:
    """One spin channel of a molecule: Gamma Hamiltonian, dipole operator, occupations.

    Parameters
    ----------
    H : ndarray ``(n, n)``
        Hermitian Hamiltonian in the orthonormal PAO basis (Hartree).
    dipole : ndarray ``(3, n, n)``
        Position (dipole) operator, Hermitian, in bohr: the orbital centres on the diagonal for a
        tight-binding model, or the position dipoles of the PAO basis.
    nocc : int
        Occupied states of the channel.
    spin_factor : float, optional
        Electrons per occupied state (2 for a closed shell, 1 per channel otherwise).
    active : int, optional
        Keep only the lowest ``active`` eigenstates in the response (particle-hole space); the
        others (e.g. the shifted complement of a PAO Hamiltonian) neither respond nor screen.
    """

    H: NDArray
    dipole: NDArray
    nocc: int
    spin_factor: float = 2.0
    active: int | None = None

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=complex)
        self.dipole = np.asarray(self.dipole, dtype=complex)
        n = self.H.shape[0]
        if self.H.shape != (n, n) or self.dipole.shape != (3, n, n):
            raise ValueError('H must be (n, n) and dipole (3, n, n)')
        if not 0 < int(self.nocc) <= n:
            raise ValueError('nocc must be between 1 and n')
        self.energies, self.states = np.linalg.eigh(self.H)
        self.occupations = np.zeros(n)
        self.occupations[: int(self.nocc)] = float(self.spin_factor)
        self.active_mask = np.ones(n, dtype=bool)
        if self.active is not None:
            if not int(self.nocc) < int(self.active) <= n:
                raise ValueError('active must be between nocc + 1 and n')
            self.active_mask[int(self.active) :] = False

    @property
    def n(self) -> int:
        """Number of orbitals."""
        return int(self.H.shape[0])

    @property
    def density(self) -> NDArray[np.complex128]:
        """Ground-state density matrix :math:`P_0 = C f C^\\dagger` (orbital basis)."""
        C = self.states
        return (C * self.occupations[None, :]) @ C.conj().T

    @property
    def gap(self) -> float:
        """HOMO-LUMO gap (Hartree)."""
        k = int(self.nocc)
        return float(self.energies[k] - self.energies[k - 1]) if k < self.n else np.inf

    def electronic_dipole(self, P: NDArray | None = None) -> NDArray[np.complex128]:
        """:math:`-\\mathrm{Tr}[P D_a]`, the electronic dipole of ``P`` (default: the ground state)."""
        P = self.density if P is None else P
        return -np.einsum('aij,ji->a', self.dipole, P)


# --------------------------------------------------------------------------- kernels
class CoulombKernel:
    """Base kernel: no induced potential (the independent-particle response)."""

    label = 'none'

    def apply(self, dP: NDArray) -> NDArray[np.complex128]:
        """Induced potential (orbital basis) of the density-matrix change ``dP``."""
        return np.zeros_like(np.asarray(dP, dtype=complex))

    def second_order(self, dP1: NDArray, dP2: NDArray) -> NDArray[np.complex128] | None:
        """Second functional derivative term :math:`g_{xc}\\delta n_1\\delta n_2` (``None``: zero)."""
        return None


class MonopoleKernel(CoulombKernel):
    r"""Hartree kernel of the atomic monopoles (self-consistent-charge DFTB form).

    Parameters
    ----------
    atom_index : ndarray ``(n,)`` int
        Atom of every orbital.
    positions_bohr : ndarray ``(nat, 3)``
        Atomic positions.
    hubbard_ha : ndarray ``(nat,)``
        On-site parameters :math:`U_A` (Hartree): the energy of a charge fluctuation on one atom.
    model : {'ohno', 'point'}, optional
        Off-site interaction: Ohno-Klopman :math:`\gamma_{AB} = [R_{AB}^2 + (2/(U_A+U_B))^2]^{-1/2}`
        (interpolates to :math:`U` at short range) or bare point charges :math:`1/R_{AB}`.

    Notes
    -----
    With :math:`\delta q_A = \sum_{\mu\in A}\delta P_{\mu\mu}` the electron-number change of atom
    :math:`A` (orthonormal basis, Mulliken populations), the induced potential is diagonal,

    .. math::

        \delta V_{\mu\mu} = \sum_B \gamma_{A(\mu)B}\,\delta q_B .
    """

    label = 'monopole'

    def __init__(
        self,
        atom_index: NDArray[np.int64],
        positions_bohr: NDArray[np.float64],
        hubbard_ha: NDArray[np.float64],
        model: str = 'ohno',
    ) -> None:
        self.atom_index = np.asarray(atom_index, dtype=int)
        R = np.asarray(positions_bohr, dtype=float).reshape(-1, 3)
        U = np.asarray(hubbard_ha, dtype=float).reshape(-1)
        if R.shape[0] != U.size or self.atom_index.max() >= U.size:
            raise ValueError('positions, hubbard parameters and atom indices disagree')
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        if model == 'ohno':
            a = 2.0 / (U[:, None] + U[None, :])
            gamma = 1.0 / np.sqrt(dist**2 + a**2)
        elif model == 'point':
            with np.errstate(divide='ignore'):
                gamma = np.where(dist > 0, 1.0 / np.where(dist > 0, dist, 1.0), 0.0)
        else:
            raise ValueError("model must be 'ohno' or 'point'")
        np.fill_diagonal(gamma, U)
        self.gamma = gamma
        self.nat = int(U.size)

    def charges(self, dP: NDArray) -> NDArray[np.complex128]:
        """Electron-number change per atom of ``dP``."""
        d = np.diagonal(np.asarray(dP, dtype=complex))
        return np.bincount(self.atom_index, weights=d.real, minlength=self.nat) + 1j * np.bincount(
            self.atom_index, weights=d.imag, minlength=self.nat
        )

    def apply(self, dP: NDArray) -> NDArray[np.complex128]:
        """Diagonal induced potential :math:`\\sum_B\\gamma_{A(\\mu)B}\\delta q_B`."""
        site = self.gamma @ self.charges(dP)
        return np.diag(site[self.atom_index]).astype(complex)


def kernel_from_species(
    atom_index: NDArray[np.int64],
    positions_bohr: NDArray[np.float64],
    species: list[str],
    hubbard_u: dict | None = None,
    model: str = 'ohno',
) -> MonopoleKernel:
    """:class:`MonopoleKernel` with the Hubbard parameters of the species labels.

    Parameters
    ----------
    atom_index, positions_bohr, model
        As :class:`MonopoleKernel`.
    species : list of str
        Element label of every atom.
    hubbard_u : dict, optional
        ``{label: U (Hartree)}`` overriding :data:`HUBBARD_U_HA`.
    """
    table = dict(HUBBARD_U_HA)
    table.update({str(k): float(v) for k, v in (hubbard_u or {}).items()})
    U = np.array(
        [table.get(str(s), table.get(str(s).capitalize(), HUBBARD_U_DEFAULT_HA)) for s in species]
    )
    return MonopoleKernel(atom_index, positions_bohr, U, model=model)


# --------------------------------------------------------------------------- LDA kernel
def pw92_correlation(rs: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perdew-Wang 1992 correlation energy per electron (paramagnetic) and its :math:`r_s` derivative (Hartree)."""
    rs = np.asarray(rs, dtype=float)
    A, a1, b1, b2, b3, b4 = 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
    x = np.sqrt(rs)
    den = 2.0 * A * (b1 * x + b2 * rs + b3 * rs * x + b4 * rs * rs)
    log = np.log(1.0 + 1.0 / den)
    e_c = -2.0 * A * (1.0 + a1 * rs) * log
    dden = 2.0 * A * (0.5 * b1 / x + b2 + 1.5 * b3 * x + 2.0 * b4 * rs)
    de_c = -2.0 * A * a1 * log + 2.0 * A * (1.0 + a1 * rs) * dden / (den * den + den)
    return e_c, de_c


def lda_potential(n: NDArray[np.float64]) -> NDArray[np.float64]:
    """Spin-unpolarised LDA exchange-correlation potential (Hartree; PW92 correlation)."""
    n = np.maximum(np.asarray(n, dtype=float), 1.0e-30)
    rs = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    v_x = -((3.0 * n / np.pi) ** (1.0 / 3.0))
    e_c, de_c = pw92_correlation(rs)
    return v_x + e_c - rs * de_c / 3.0


def lda_kernels(
    n: NDArray[np.float64], threshold: float = 1.0e-8, delta: float = 1.0e-3
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ALDA :math:`f_{xc} = dv_{xc}/dn` and :math:`g_{xc} = d^2v_{xc}/dn^2` (zero below ``threshold``)."""
    n = np.asarray(n, dtype=float)
    keep = n > threshold
    f = np.zeros_like(n)
    g = np.zeros_like(n)
    nk = n[keep]
    vp = lda_potential(nk * (1.0 + delta))
    vm = lda_potential(nk * (1.0 - delta))
    v0 = lda_potential(nk)
    f[keep] = (vp - vm) / (2.0 * delta * nk)
    g[keep] = (vp - 2.0 * v0 + vm) / (delta * nk) ** 2
    return f, g


# --------------------------------------------------------------------------- grid kernel
class GridKernel(CoulombKernel):
    r"""Hartree (+ ALDA) kernel of the density on the real-space grid of a box.

    Parameters
    ----------
    phi : ndarray ``(n, nr1, nr2, nr3)``
        Orthonormal orbitals on the FFT grid (:func:`~PAOFLOW.hamiltonian.do_real_space.pao_orbitals_on_grid`).
    at_bohr : ndarray ``(3, 3)``
        Cell vectors (rows, bohr).
    omega : float
        Cell volume (bohr^3).
    xc : {None, 'lda'}, optional
        Add the adiabatic LDA kernel evaluated at the density of ``P0``.
    P0 : ndarray ``(n, n)``, optional
        Ground-state density matrix (needed for ``xc``).
    cutoff_radius : float, optional
        Radius of the spherical Coulomb cutoff (bohr); default half the shortest cell height,
        which removes the periodic images exactly for a molecule inside that sphere.
    density_threshold : float, optional
        Density below which the xc kernels are set to zero.

    Notes
    -----
    .. math::

        \delta n(\mathbf r) = \sum_{\mu\nu}\phi_\mu(\mathbf r)\phi_\nu(\mathbf r)\,\delta P_{\nu\mu},\qquad
        \delta V_H(\mathbf G) = \frac{4\pi}{G^2}\left[1-\cos(GR_c)\right]\delta n(\mathbf G),\qquad
        \delta V_{\mu\nu} = \int\phi_\mu\,[\delta V_H + f_{xc}\delta n]\,\phi_\nu .
    """

    def __init__(
        self,
        phi: NDArray,
        at_bohr: NDArray[np.float64],
        omega: float,
        xc: str | None = None,
        P0: NDArray | None = None,
        cutoff_radius: float | None = None,
        density_threshold: float = 1.0e-8,
    ) -> None:
        phi = np.asarray(phi)
        self.n = int(phi.shape[0])
        self.nr = tuple(int(x) for x in phi.shape[1:])
        table = phi.reshape(self.n, -1).T  # (N_r, n)
        scale = float(np.abs(table).max())
        if np.abs(table.imag).max() < 1.0e-8 * scale:
            table = np.ascontiguousarray(table.real)
        self.table = table
        self.omega = float(omega)
        self.nr_total = int(np.prod(self.nr))
        self.dv = self.omega / self.nr_total
        at = np.asarray(at_bohr, dtype=float)
        bg = 2.0 * np.pi * np.linalg.inv(at).T  # rows, bohr^-1
        if cutoff_radius is None:
            heights = [
                self.omega / np.linalg.norm(np.cross(at[j], at[k]))
                for j, k in ((1, 2), (2, 0), (0, 1))
            ]
            cutoff_radius = 0.5 * min(heights)
        self.cutoff_radius = float(cutoff_radius)
        m = np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n) for n in self.nr], indexing='ij')
        G = sum(m[i][..., None] * bg[i] for i in range(3))
        G2 = np.sum(G * G, axis=-1)
        Gn = np.sqrt(G2)
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.where(
                G2 > 0,
                4.0 * np.pi * (1.0 - np.cos(Gn * self.cutoff_radius)) / np.where(G2 > 0, G2, 1.0),
                0.0,
            )
        v[0, 0, 0] = 2.0 * np.pi * self.cutoff_radius**2
        self.v_coulomb = v
        self.xc = xc
        self.f_xc = self.g_xc = None
        if xc is not None:
            if xc != 'lda':
                raise ValueError("xc must be None or 'lda'")
            if P0 is None:
                raise ValueError("xc='lda' needs the ground-state density matrix P0")
            n0 = self.density(P0).real
            self.f_xc, self.g_xc = lda_kernels(n0, threshold=density_threshold)
            self.label = 'hartree+alda'
        else:
            self.label = 'hartree'

    #: grid points per block of the orbital-product sums (bounds the temporaries)
    block_rows = 200_000

    def _blocks(self):
        for start in range(0, self.nr_total, int(self.block_rows)):
            yield start, min(start + int(self.block_rows), self.nr_total)

    def _density_real(self, M: NDArray[np.float64], out: NDArray) -> None:
        """``out[r] = sum_mn t[r, m] M[m, n] t[r, n]`` for a real table and real ``M``."""
        t = self.table
        for a, b in self._blocks():
            tb = t[a:b]
            out[a:b] = np.einsum('rm,rm->r', tb @ M.T, tb)

    def density(self, dP: NDArray) -> NDArray[np.complex128]:
        r""":math:`\delta n(\mathbf r) = \sum_{\mu\nu}\phi_\mu\phi_\nu\delta P_{\nu\mu}` on the grid (flat, ``N_r``)."""
        dP = np.asarray(dP)
        if np.isrealobj(self.table):
            out = np.empty(self.nr_total, dtype=complex)
            tmp = np.empty(self.nr_total)
            self._density_real(np.ascontiguousarray(dP.real), tmp)
            out[:] = tmp
            if np.iscomplexobj(dP):
                self._density_real(np.ascontiguousarray(dP.imag), tmp)
                out += 1j * tmp
            return out
        t = self.table
        out = np.empty(self.nr_total, dtype=complex)
        M = np.asarray(dP, dtype=complex).T
        for a, b in self._blocks():
            tb = t[a:b]
            out[a:b] = np.einsum('rm,rm->r', tb @ M, tb.conj())
        return out

    def _potential_real(self, V: NDArray[np.float64]) -> NDArray[np.float64]:
        t = self.table
        out = np.zeros((self.n, self.n))
        for a, b in self._blocks():
            tb = t[a:b]
            out += tb.T @ (V[a:b, None] * tb)
        return out * self.dv

    def potential_matrix(self, V: NDArray) -> NDArray[np.complex128]:
        r""":math:`\int\phi_\mu V\phi_\nu` for a potential on the grid (flat)."""
        V = np.asarray(V).reshape(-1)
        if np.isrealobj(self.table):
            out = self._potential_real(np.ascontiguousarray(V.real)).astype(complex)
            if np.iscomplexobj(V):
                out += 1j * self._potential_real(np.ascontiguousarray(V.imag))
            return out
        t = self.table
        out = np.zeros((self.n, self.n), dtype=complex)
        for a, b in self._blocks():
            tb = t[a:b]
            out += tb.conj().T @ (V[a:b, None] * tb)
        return out * self.dv

    def hartree(self, dn: NDArray) -> NDArray[np.complex128]:
        """Hartree potential (flat) of a density on the grid with the cutoff Coulomb kernel."""
        dn_G = np.fft.fftn(np.asarray(dn, dtype=complex).reshape(self.nr))
        return np.fft.ifftn(self.v_coulomb * dn_G).reshape(-1)

    def apply(self, dP: NDArray) -> NDArray[np.complex128]:
        dn = self.density(dP)
        V = self.hartree(dn)
        if self.f_xc is not None:
            V = V + self.f_xc * dn
        return self.potential_matrix(V)

    def second_order(self, dP1: NDArray, dP2: NDArray) -> NDArray[np.complex128] | None:
        if self.g_xc is None:
            return None
        return self.potential_matrix(self.g_xc * self.density(dP1) * self.density(dP2))

    def coulomb_integrals(self, pairs) -> NDArray[np.complex128]:
        """``(mu nu|kappa lambda)`` for index quadruples ``pairs`` (test helper, one Poisson solve each)."""
        out = []
        for mu, nu, kappa, lam in pairs:
            dP = np.zeros((self.n, self.n))
            dP[lam, kappa] = 1.0  # delta n = phi_kappa phi_lambda
            out.append(self.potential_matrix(self.hartree(self.density(dP)))[mu, nu])
        return np.array(out)


# --------------------------------------------------------------------------- PAOFLOW-facing builders
def atoms_from_basis(data_controller) -> tuple[NDArray[np.int64], NDArray[np.float64], list[str]]:
    """``(atom_index (n,), positions (nat, 3) bohr, species labels)`` from ``arrays['basis']``."""
    arrays, _ = data_controller.data_dicts()
    basis = arrays.get('basis')
    if not basis or 'tau' not in basis[0] or 'atom' not in basis[0]:
        raise ValueError(
            'the monopole kernel needs the PAO basis records (atom labels and centres)'
        )
    seen: dict = {}
    index, positions, species = [], [], []
    for orbital in basis:
        key = tuple(np.round(np.asarray(orbital['tau'], dtype=float), 6))
        if key not in seen:
            seen[key] = len(positions)
            positions.append(np.asarray(orbital['tau'], dtype=float))
            species.append(str(orbital['atom']))
        index.append(seen[key])
    return np.array(index, dtype=int), np.array(positions), species


def grid_kernel_from_data(
    data_controller, P0: NDArray, xc: str | None = None, grid='coarse', verbose: bool = False
) -> GridKernel:
    """:class:`GridKernel` of the molecule held by ``data_controller`` (internal PAO basis).

    Parameters
    ----------
    data_controller : DataController
    P0 : ndarray ``(n, n)``
        Ground-state density matrix (for the ALDA kernel).
    xc : {None, 'lda'}, optional
    grid : 'coarse', 'full' or tuple of int, optional
        FFT grid: ``'coarse'`` (default) the wavefunction sphere, which reproduces the alias-free
        ``'full'`` grid (twice the sphere, eight times the memory) to three digits on PNA; or an
        explicit ``(n1, n2, n3)``.
    verbose : bool, optional
    """
    from ..hamiltonian.do_real_space import pao_orbitals_on_grid

    nr = None
    if grid == 'coarse':
        phi, info = pao_orbitals_on_grid(data_controller, nr=None)
        nr = tuple(max(1, n // 2) for n in info['nr'])
        del phi
    elif grid not in (None, 'full'):
        nr = tuple(int(n) for n in grid)
    phi, info = pao_orbitals_on_grid(data_controller, nr=nr)
    if verbose:
        print(
            '  grid kernel: %d orbitals on a %s grid (%.1f GB), cutoff radius %.1f bohr'
            % (phi.shape[0], 'x'.join(str(n) for n in info['nr']), phi.nbytes / 2e9, 0.0),
            flush=True,
        )
    return GridKernel(phi, info['at_bohr'], info['omega'], xc=xc, P0=P0)


def coupled_molecule(
    data_controller,
    HRs: NDArray,
    positions_bohr: NDArray[np.float64],
    dipoles,
    nocc_channels,
    spin_factor: float,
    coupling: str,
    eta_ev: float,
    energies_ev,
    scissor_ev: float = 0.0,
    hubbard_u: dict | None = None,
    grid='coarse',
    active: int | str | None = None,
    static_step='auto',
    tol: float = 1.0e-10,
    progress: bool = False,
    comm=None,
    n_threads: int | None = None,
) -> dict:
    """Coupled SHG hyperpolarizability of the molecule of a PAOFLOW run (one closed-shell channel).

    Parameters
    ----------
    data_controller : DataController
    HRs : ndarray ``(n, n, 1, 1, 1, nspin)``
        Gamma-point PAO Hamiltonian (eV).
    positions_bohr : ndarray ``(n, 3)``
        Orbital centres (the tight-binding dipole operator).
    dipoles : callable or None
        Position-dipole operator of :func:`~PAOFLOW.hamiltonian.position_dipoles.position_dipole_operator`;
        ``None`` uses the orbital centres.
    nocc_channels : sequence of int
    spin_factor : float
    coupling : {'none', 'monopole', 'hartree', 'hartree+alda'}
    eta_ev : float
    energies_ev : array_like
        Photon energies (eV).
    scissor_ev : float, optional
        Rigid shift of the empty states.
    hubbard_u : dict, optional
        ``{species: U (Hartree)}`` of the monopole kernel.
    grid : optional
        See :func:`grid_kernel_from_data`.
    active : int or 'projected', optional
        Response space: the lowest ``active`` eigenstates, or ``'projected'`` for the states with good
        projectability (``attr['bnd']``), excluding the complement that the projectability filter puts
        at the shift energy.  Default: all states.  The finite-field static reference is skipped when
        an active space is set.
    static_step, tol, progress, comm, n_threads
        See :func:`coupled_hyperpolarizability` (collective over ``comm``: every rank builds the
        channel and the kernel and returns the full result).

    Returns
    -------
    dict
        Of :func:`coupled_hyperpolarizability` plus ``'energy_ev'``, ``'gap_ev'`` and ``'active'``.
    """
    if coupling not in KERNELS:
        raise ValueError('coupling must be one of %s' % (KERNELS,))
    HRs = np.asarray(HRs)
    if HRs.shape[-1] != 1:
        raise NotImplementedError('the coupled response handles one (closed-shell) spin channel')
    if tuple(HRs.shape[2:5]) != (1, 1, 1):
        raise ValueError('the coupled response needs the Gamma-point Hamiltonian of a box')
    H = np.asarray(HRs[:, :, 0, 0, 0, 0], dtype=complex) / HARTREE_EV
    nocc = int(nocc_channels[0])
    if scissor_ev:
        e, C = np.linalg.eigh(H)
        e[nocc:] += float(scissor_ev) / HARTREE_EV
        H = (C * e[None, :]) @ C.conj().T
    if dipoles is not None:
        D = np.asarray(dipoles(np.zeros((1, 3)))[0], dtype=complex)
    else:
        pos = np.asarray(positions_bohr, dtype=float)
        D = np.stack([np.diag(pos[:, a]) for a in range(3)]).astype(complex)
    n_active = None
    if active is not None:
        if active == 'projected':
            _, attr = data_controller.data_dicts()
            n_active = int(attr['bnd'])
        else:
            n_active = int(active)
        static_step = None
    ch = MolecularChannel(H, D, nocc, float(spin_factor), active=n_active)
    if coupling == 'none':
        kernel = CoulombKernel()
    elif coupling == 'monopole':
        index, atoms, species = atoms_from_basis(data_controller)
        kernel = kernel_from_species(index, atoms, species, hubbard_u=hubbard_u)
    else:
        kernel = grid_kernel_from_data(
            data_controller, ch.density, xc=('lda' if coupling == 'hartree+alda' else None),
            grid=grid, verbose=progress,
        )  # fmt: skip
    energies = np.atleast_1d(np.asarray(energies_ev, dtype=float))
    out = coupled_hyperpolarizability(
        ch, kernel, energies / HARTREE_EV, float(eta_ev) / HARTREE_EV, static_step=static_step,
        tol=tol, progress=progress, comm=comm, n_threads=n_threads,
    )  # fmt: skip
    out['energy_ev'] = energies
    out['gap_ev'] = ch.gap * HARTREE_EV
    out['active'] = n_active
    return out


# --------------------------------------------------------------------------- solvers
def _coefficients(ch: MolecularChannel, omega: float, eta: float):
    """``(c, d, mask)``: :math:`c_{nm} = (f_m - f_n)/d_{nm}`, :math:`d_{nm} = \\omega - \\epsilon_n + \\epsilon_m + i\\eta`."""
    e, f = ch.energies, ch.occupations
    d = float(omega) - (e[:, None] - e[None, :]) + 1j * float(eta)
    df = (f[None, :] - f[:, None]) * (ch.active_mask[:, None] & ch.active_mask[None, :])
    return df / d, d, df != 0


def solve_coupled(
    ch: MolecularChannel,
    kernel: CoulombKernel,
    x0: NDArray[np.complex128],
    omega: float,
    eta: float,
    tol: float = 1.0e-10,
    maxiter: int = 300,
) -> NDArray[np.complex128]:
    """Solve :math:`x - c\\circ(C^\\dagger K[CxC^\\dagger]C) = x_0` for the eigenbasis matrix ``x``.

    Parameters
    ----------
    ch : MolecularChannel
    kernel : CoulombKernel
    x0 : ndarray ``(n, n)``
        Source, non-zero on the particle-hole blocks only.
    omega, eta : float
        Frequency and damping (Hartree) of the coefficients :math:`c_{nm}`.
    tol, maxiter
        GMRES relative tolerance and iteration cap.

    Returns
    -------
    ndarray ``(n, n)``
        The particle-hole blocks of the coupled response (eigenbasis).
    """
    if type(kernel) is CoulombKernel:
        return np.array(x0, dtype=complex)
    from scipy.sparse.linalg import LinearOperator, gmres

    C = ch.states
    Ch = C.conj().T
    c, _, _ = _coefficients(ch, omega, eta)
    n = ch.n

    def matvec(v):
        x = v.reshape(n, n)
        dV = kernel.apply(C @ x @ Ch)
        return (x - c * (Ch @ dV @ C)).ravel()

    op = LinearOperator((n * n, n * n), matvec=matvec, dtype=complex)
    b = np.asarray(x0, dtype=complex).ravel()
    try:
        x, info = gmres(op, b, x0=b, rtol=tol, maxiter=maxiter)
    except TypeError:  # scipy < 1.12
        x, info = gmres(op, b, x0=b, tol=tol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError('the coupled response did not converge (GMRES info %d)' % info)
    return x.reshape(n, n)


def first_order_density(
    ch: MolecularChannel,
    kernel: CoulombKernel,
    V_ext: NDArray,
    omega: float,
    eta: float,
    tol: float = 1.0e-10,
) -> NDArray[np.complex128]:
    """First-order density matrix (orbital basis) driven by ``V_ext`` at ``omega``."""
    C = ch.states
    Ch = C.conj().T
    c, _, _ = _coefficients(ch, omega, eta)
    x0 = c * (Ch @ np.asarray(V_ext, dtype=complex) @ C)
    x = solve_coupled(ch, kernel, x0, omega, eta, tol=tol)
    return C @ x @ Ch


def second_order_density(
    ch: MolecularChannel,
    kernel: CoulombKernel,
    V_b: NDArray,
    V_c: NDArray,
    omega1: float,
    omega2: float,
    eta: float,
    tol: float = 1.0e-10,
    first_order: tuple[NDArray, NDArray] | None = None,
) -> NDArray[np.complex128]:
    """Second-order density matrix (orbital basis) for the fields ``V_b(omega1)`` and ``V_c(omega2)``.

    Parameters
    ----------
    ch, kernel, eta, tol
        As :func:`first_order_density`.
    V_b, V_c : ndarray ``(n, n)``
        External potentials of the two fields (unit amplitude).
    omega1, omega2 : float
        Their frequencies (Hartree); the response is at :math:`\\omega_1 + \\omega_2`.
    first_order : tuple of ndarray, optional
        ``(dP_b(omega1), dP_c(omega2))`` already computed.

    Returns
    -------
    ndarray ``(n, n)``
        :math:`\\delta P^{(2)}_{bc}`, symmetric in :math:`(b,\\omega_1)\\leftrightarrow(c,\\omega_2)`.
    """
    C = ch.states
    Ch = C.conj().T
    V_b = np.asarray(V_b, dtype=complex)
    V_c = np.asarray(V_c, dtype=complex)
    if first_order is None:
        P_b = first_order_density(ch, kernel, V_b, omega1, eta, tol=tol)
        P_c = first_order_density(ch, kernel, V_c, omega2, eta, tol=tol)
    else:
        P_b, P_c = first_order
    V1_b = V_b + kernel.apply(P_b)
    V1_c = V_c + kernel.apply(P_c)
    S = 0.5 * (V1_b @ P_c - P_c @ V1_b + V1_c @ P_b - P_b @ V1_c)
    omega = float(omega1) + float(omega2)
    c, d, mask = _coefficients(ch, omega, eta)
    # occupied-occupied and empty-empty blocks from P^2 = g_s P (known a priori)
    Pb_e, Pc_e = Ch @ P_b @ C, Ch @ P_c @ C
    Q = 0.5 * (Pb_e @ Pc_e + Pc_e @ Pb_e) / float(ch.spin_factor)
    occ = ch.occupations > 0
    oo = occ[:, None] & occ[None, :]
    vv = ~occ[:, None] & ~occ[None, :]
    x_block = np.where(vv, Q, 0.0) - np.where(oo, Q, 0.0)
    P_block = C @ x_block @ Ch
    # particle-hole blocks: the source, the induced potential of the known blocks and the
    # second functional derivative; the induced potential of the particle-hole blocks is
    # solved self-consistently
    S_e = Ch @ S @ C
    x0 = np.where(mask, S_e / d, 0.0) + c * (Ch @ kernel.apply(P_block) @ C)
    V_g = kernel.second_order(P_b, P_c)
    if V_g is not None:
        x0 = x0 + 0.5 * c * (Ch @ V_g @ C)
    x = solve_coupled(ch, kernel, x0, omega, eta, tol=tol)
    return C @ x @ Ch + P_block


# --------------------------------------------------------------------------- finite field
def finite_field_density(
    ch: MolecularChannel,
    kernel: CoulombKernel,
    field: NDArray[np.float64],
    mixing: float = 0.3,
    tol: float = 1.0e-11,
    maxiter: int = 500,
    history: int = 12,
) -> NDArray[np.complex128]:
    """Self-consistent ground-state density matrix in a static field (orbital basis).

    Parameters
    ----------
    ch, kernel
    field : ndarray ``(3,)``
        Static field (a.u.); the Hamiltonian is :math:`H + \\mathbf E\\cdot\\mathbf D + K[P - P_0]`.
    mixing : float, optional
        Mixing of the Pulay (DIIS) update.
    tol : float, optional
        Convergence of :math:`\\max|P_{\\rm out} - P_{\\rm in}|`.
    maxiter : int, optional
    history : int, optional
        Length of the DIIS history.

    Raises
    ------
    RuntimeError
        If the self-consistency does not converge.
    """
    field = np.asarray(field, dtype=float).reshape(3)
    V_ext = np.einsum('a,aij->ij', field, ch.dipole)
    P0 = ch.density
    nocc = int(ch.nocc)
    g = float(ch.spin_factor)

    def output(P):
        Hs = ch.H + V_ext + kernel.apply(P - P0)
        _, C = np.linalg.eigh(0.5 * (Hs + Hs.conj().T))
        occ = C[:, :nocc]
        return g * (occ @ occ.conj().T)

    P = P0.copy()
    inputs: list = []
    residuals: list = []
    err = np.inf
    for _ in range(int(maxiter)):
        R = output(P) - P
        err = float(np.abs(R).max())
        if err < tol:
            return P
        inputs.append(P)
        residuals.append(R)
        if len(inputs) > int(history):
            inputs.pop(0)
            residuals.pop(0)
        m = len(inputs)
        if m == 1:
            P = P + float(mixing) * R
            continue
        # Pulay: minimise |sum_i c_i R_i| with sum_i c_i = 1
        B = np.zeros((m + 1, m + 1), dtype=complex)
        for i in range(m):
            for j in range(m):
                B[i, j] = np.vdot(residuals[i], residuals[j])
        B[m, :m] = B[:m, m] = 1.0
        rhs = np.zeros(m + 1, dtype=complex)
        rhs[m] = 1.0
        try:
            coef = np.linalg.solve(B, rhs)[:m]
        except np.linalg.LinAlgError:
            coef = np.zeros(m)
            coef[-1] = 1.0
        P = sum(c * (Pi + float(mixing) * Ri) for c, Pi, Ri in zip(coef, inputs, residuals))
        P = 0.5 * (P + P.conj().T)
    raise RuntimeError(
        'finite-field self-consistency did not converge (residual %.1e): the field step is too '
        'large for the gap (occupations swap between iterations) or the mixing too aggressive' % err
    )


def auto_field_step(ch: MolecularChannel, fraction: float = 0.02) -> float:
    """Field step with :math:`h\\,\\Delta D \\le` ``fraction`` of the gap (:math:`\\Delta D` the dipole span)."""
    span = max(float(np.ptp(np.linalg.eigvalsh(ch.dipole[a]))) for a in range(3))
    gap = ch.gap if np.isfinite(ch.gap) else 1.0
    return min(1.0e-3, float(fraction) * gap / max(span, 1.0e-12))


def _work_split(n_items: int, comm):
    """``(rank, size)`` of ``comm`` (``(0, 1)`` without MPI)."""
    if comm is None:
        return 0, 1
    return int(comm.Get_rank()), int(comm.Get_size())


def _thread_map(fn, tasks, n_threads):
    """Map ``fn`` over ``tasks`` with a thread pool of ``n_threads`` workers (serial for 1)."""
    if n_threads <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=int(n_threads)) as pool:
        return list(pool.map(fn, tasks))


def _allreduce(comm, array: NDArray) -> NDArray:
    """Sum ``array`` over ``comm`` in place (no-op without MPI)."""
    if comm is None:
        return array
    from mpi4py import MPI

    comm.Allreduce(MPI.IN_PLACE, array, op=MPI.SUM)
    return array


def static_hyperpolarizability(
    ch: MolecularChannel,
    kernel: CoulombKernel,
    step: float | None = None,
    richardson: bool = True,
    comm=None,
    n_threads: int | None = None,
    **scf,
) -> dict:
    """Static :math:`\\alpha` and Taylor :math:`\\beta` from finite fields.

    Parameters
    ----------
    ch, kernel
    step : float, optional
        Field step :math:`h` (a.u.); default :func:`auto_field_step`.
    richardson : bool, optional
        Combine the steps :math:`h` and :math:`2h` to cancel the :math:`O(h^2)` error.
    comm : MPI communicator, optional
        The field settings are distributed over its ranks (every rank returns the full result).
    n_threads : int, optional
        Thread pool over the field settings of this rank (joblib-style request, see
        :func:`~PAOFLOW.utils.parallel_resources.resolve_n_jobs`; ``None``: serial).
    **scf
        Options of :func:`finite_field_density`.

    Returns
    -------
    dict
        ``'alpha'`` ``(3, 3)`` (a.u.), ``'beta_T'`` ``(3, 3, 3)`` (a.u., Taylor convention,
        :math:`\\partial^2\\mu_a/\\partial E_b\\partial E_c`), ``'dipole'`` ``(3,)`` electronic.
    """
    from ..utils.parallel_resources import resolve_n_jobs

    step = auto_field_step(ch) if step is None else float(step)
    steps = [step, 2.0 * step] if richardson else [step]
    eye = np.eye(3)
    fields = [np.zeros(3)]
    index: dict = {}
    for h in steps:
        for b in range(3):
            for sign in (1, -1):
                index[(h, b, sign)] = len(fields)
                fields.append(sign * h * eye[b])
        for b in range(3):
            for c in range(b + 1, 3):
                for sb, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    index[(h, b, c, sb, sc)] = len(fields)
                    fields.append(h * (sb * eye[b] + sc * eye[c]))
    rank, size = _work_split(len(fields), comm)
    mine = list(range(rank, len(fields), size))
    workers = resolve_n_jobs(n_threads, n_tasks=len(mine))
    mu = np.zeros((len(fields), 3))
    values = _thread_map(
        lambda k: ch.electronic_dipole(finite_field_density(ch, kernel, fields[k], **scf)).real,
        mine,
        workers,
    )
    for k, v in zip(mine, values):
        mu[k] = v
    _allreduce(comm, mu)

    def tensors(h):
        alpha = np.zeros((3, 3))
        beta = np.zeros((3, 3, 3))
        for b in range(3):
            plus, minus = mu[index[(h, b, 1)]], mu[index[(h, b, -1)]]
            alpha[:, b] = (plus - minus) / (2 * h)
            beta[:, b, b] = (plus + minus - 2 * mu[0]) / (h * h)
        for b in range(3):
            for c in range(b + 1, 3):
                acc = sum(
                    sb * sc * mu[index[(h, b, c, sb, sc)]]
                    for sb, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                )
                beta[:, b, c] = beta[:, c, b] = acc / (4 * h * h)
        return alpha, beta

    alpha, beta = tensors(steps[0])
    if richardson:
        alpha2, beta2 = tensors(steps[1])
        alpha = (4.0 * alpha - alpha2) / 3.0
        beta = (4.0 * beta - beta2) / 3.0
    return {'alpha': alpha, 'beta_T': beta, 'dipole': mu[0]}


# --------------------------------------------------------------------------- driver
def coupled_hyperpolarizability(
    ch: MolecularChannel,
    kernel: CoulombKernel,
    energies_ha: NDArray[np.float64],
    eta_ha: float,
    static_step: float | str | None = 'auto',
    tol: float = 1.0e-10,
    progress: bool = False,
    comm=None,
    n_threads: int | None = None,
) -> dict:
    r"""SHG hyperpolarizability :math:`\beta^T(-2\omega;\omega,\omega)` and :math:`\alpha(\omega)` with the kernel.

    Parameters
    ----------
    ch : MolecularChannel
    kernel : CoulombKernel
        :class:`CoulombKernel` (independent particles), :class:`MonopoleKernel` or
        :class:`GridKernel`.
    energies_ha : array_like
        Photon energies :math:`\omega` (Hartree); 0 gives the static limit.
    eta_ha : float
        Damping (Hartree).
    static_step : float, 'auto' or None, optional
        Finite-field step of the static reference (:func:`auto_field_step` for ``'auto'``,
        ``None`` skips it).
    tol : float, optional
        GMRES tolerance.
    progress : bool, optional
        Print one line per energy (rank 0).
    comm : MPI communicator, optional
        The independent solves (three field directions and six second-order pairs per energy)
        are distributed over its ranks; every rank returns the full result.
    n_threads : int, optional
        Thread pool over the solves of this rank (joblib-style request within the core budget,
        :func:`~PAOFLOW.utils.parallel_resources.resolve_n_jobs`; ``None``: serial).  The BLAS
        threads of the kernel products add to it: set ``OMP_NUM_THREADS`` accordingly.

    Returns
    -------
    dict
        ``'energy_ha'``, ``'alpha'`` ``(ne, 3, 3)`` complex, ``'beta_T'`` ``(ne, 3, 3, 3)`` complex
        (Taylor, a.u., electrons), ``'beta_T_static'`` ``(3, 3, 3)`` and ``'alpha_static'`` from
        the finite-field route, ``'kernel'`` label.
    """
    from ..utils.parallel_resources import resolve_n_jobs

    energies = np.atleast_1d(np.asarray(energies_ha, dtype=float))
    ne = energies.size
    alpha = np.zeros((ne, 3, 3), dtype=complex)
    beta = np.zeros((ne, 3, 3, 3), dtype=complex)
    D = ch.dipole
    pairs = [(b, c) for b in range(3) for c in range(b, 3)]
    items = [(ie, b, c) for ie in range(ne) for b, c in pairs]
    rank, size = _work_split(len(items), comm)
    mine = items[rank::size]
    workers = resolve_n_jobs(n_threads, n_tasks=max(len(mine), 1))
    needed = sorted({ie for ie, _, _ in mine})
    first = _thread_map(
        lambda t: first_order_density(ch, kernel, D[t[1]], energies[t[0]], eta_ha, tol=tol),
        [(ie, b) for ie in needed for b in range(3)],
        workers,
    )
    P1 = {}
    for (ie, b), P in zip([(ie, b) for ie in needed for b in range(3)], first):
        P1.setdefault(ie, [None] * 3)[b] = P
    for ie in needed:
        if (ie, 0, 0) in mine:  # one owner per energy for alpha
            for b in range(3):
                alpha[ie, :, b] = -np.einsum('aij,ji->a', D, P1[ie][b])

    def second(t):
        ie, b, c = t
        P2 = second_order_density(
            ch, kernel, D[b], D[c], energies[ie], energies[ie], eta_ha, tol=tol,
            first_order=(P1[ie][b], P1[ie][c]),
        )  # fmt: skip
        return -2.0 * np.einsum('aij,ji->a', D, P2)

    for (ie, b, c), val in zip(mine, _thread_map(second, mine, workers)):
        beta[ie, :, b, c] = val
        if c != b:
            beta[ie, :, c, b] = val
    _allreduce(comm, alpha)
    _allreduce(comm, beta)
    if progress and rank == 0:
        for ie, w in enumerate(energies):
            print(
                '  coupled response (%s): %.3f eV  beta_xxx %.1f  alpha_xx %.2f'
                % (kernel.label, w * HARTREE_EV, beta[ie, 0, 0, 0].real, alpha[ie, 0, 0].real),
                flush=True,
            )
    out = {'energy_ha': energies, 'alpha': alpha, 'beta_T': beta, 'kernel': kernel.label}
    if static_step is not None:
        ff = static_hyperpolarizability(
            ch, kernel, step=None if static_step == 'auto' else float(static_step), comm=comm,
            n_threads=n_threads,
        )  # fmt: skip
        out['beta_T_static'] = ff['beta_T']
        out['alpha_static'] = ff['alpha']
    return out


__all__ = [
    'HUBBARD_U_HA',
    'KERNELS',
    'CoulombKernel',
    'MolecularChannel',
    'GridKernel',
    'MonopoleKernel',
    'atoms_from_basis',
    'auto_field_step',
    'coupled_molecule',
    'coupled_hyperpolarizability',
    'finite_field_density',
    'first_order_density',
    'grid_kernel_from_data',
    'kernel_from_species',
    'lda_kernels',
    'lda_potential',
    'pw92_correlation',
    'second_order_density',
    'solve_coupled',
    'static_hyperpolarizability',
]
