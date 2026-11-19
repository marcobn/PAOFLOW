"""Model crystal for electron-phonon coupling: electrons, phonons and the vertex of the same crystal.

A :class:`ElectronPhononModel` is a tight-binding crystal whose Slater-Koster hoppings depend on
the bond length, with harmonic phonons from springs between the same atoms.  Moving an atom
changes the hoppings of its bonds, so the electron-phonon vertex :math:`\\partial H/\\partial u`
follows from the model itself, with no further parameter.  The property functions of
:mod:`PAOFLOW.models.electron_phonon.properties` hand the model's states, phonons and coupling to
PAOFLOW's electron-phonon engines (:mod:`PAOFLOW.elphon`).

Units: energies in eV, lengths in bohr, masses in amu, wave vectors in 1/bohr (crystal
coordinates where stated), force constants in eV/bohr^2; inside the model :math:`\\hbar = 1`, so a
band velocity is :math:`\\hbar v` in eV bohr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from ..models import HoppingModel

BOHR_M = 0.529177210903e-10
E_CHARGE = 1.602176634e-19
HBAR_EV_S = 6.582119569e-16
AMU_KG = 1.66053906660e-27
KB_EV = 8.617333262e-5
RY_EV = 13.605693122994
EPS0 = 8.8541878128e-12
#: ``(hbar omega)^2 = HBAR2_K_OVER_M * K / M`` in eV^2 for ``K`` in eV/bohr^2 and ``M`` in amu.
HBAR2_K_OVER_M = (HBAR_EV_S**2) * E_CHARGE / (BOHR_M**2 * AMU_KG)

#: Orbital kinds of the Slater-Koster table and their Cartesian axis.
P_AXIS = {'px': 0, 'py': 1, 'pz': 2}
ORBITAL_KINDS = ('s', 'px', 'py', 'pz')

#: Central finite-difference step (bohr) of the hopping gradient.
GRADIENT_STEP = 1.0e-6


def slater_koster_hopping(
    orbital_i: str, orbital_j: str, bond: NDArray[np.float64], parameters: dict[str, float]
) -> float:
    """Two-centre hopping :math:`\\langle i|H|j\\rangle` of s and p orbitals along a bond.

    Parameters
    ----------
    orbital_i, orbital_j : {'s', 'px', 'py', 'pz'}
        Orbital on the first and on the second atom.
    bond : ndarray, shape ``(3,)``
        Bond vector from the atom of ``orbital_i`` to the atom of ``orbital_j`` (bohr).
    parameters : dict
        Slater-Koster integrals (eV): ``'sss'`` (:math:`V_{ss\\sigma}`), ``'sps'``
        (:math:`V_{sp\\sigma}`), ``'pss'`` (:math:`V_{ps\\sigma}`, default ``'sps'``), ``'pps'``
        (:math:`V_{pp\\sigma}`), ``'ppp'`` (:math:`V_{pp\\pi}`); missing ones are zero.

    Returns
    -------
    float
        The hopping (eV).

    Notes
    -----
    With the direction cosines :math:`\\ell = \\mathbf d/|\\mathbf d|`,

    .. math::

        t_{s,s} = V_{ss\\sigma},\\quad t_{s,p_a} = \\ell_a V_{sp\\sigma},\\quad
        t_{p_a,s} = -\\ell_a V_{ps\\sigma},\\quad
        t_{p_a,p_b} = \\ell_a\\ell_b V_{pp\\sigma} + (\\delta_{ab} - \\ell_a\\ell_b) V_{pp\\pi}.
    """
    direction = bond / np.linalg.norm(bond)
    if orbital_i == 's' and orbital_j == 's':
        return parameters.get('sss', 0.0)
    if orbital_i == 's':
        return direction[P_AXIS[orbital_j]] * parameters.get('sps', 0.0)
    if orbital_j == 's':
        return -direction[P_AXIS[orbital_i]] * parameters.get('pss', parameters.get('sps', 0.0))
    axis_i, axis_j = P_AXIS[orbital_i], P_AXIS[orbital_j]
    cosines = direction[axis_i] * direction[axis_j]
    return cosines * parameters.get('pps', 0.0) + ((axis_i == axis_j) - cosines) * parameters.get(
        'ppp', 0.0
    )


@dataclass(frozen=True)
class HoppingTable:
    """Every hopping of a model with its gradient (the vertex).

    Attributes
    ----------
    orbital_i, orbital_j : ndarray of int, shape ``(nhop,)``
        Orbitals of the hopping :math:`\\langle i|H|j\\rangle`.
    bonds : ndarray, shape ``(nhop, 3)``
        Bond vector from the atom of ``i`` to the atom of ``j`` (bohr).
    hoppings : ndarray, shape ``(nhop,)``
        Hopping (eV).
    gradients : ndarray, shape ``(nhop, 3)``
        :math:`\\nabla_{\\mathbf d} t_{ij}(\\mathbf d)` (eV/bohr).
    """

    orbital_i: NDArray[np.int_]
    orbital_j: NDArray[np.int_]
    bonds: NDArray[np.float64]
    hoppings: NDArray[np.float64]
    gradients: NDArray[np.float64]


@dataclass(frozen=True)
class ForceConstantTable:
    """Force-constant blocks :math:`\\Phi_{ab}(\\mathbf R)` of the springs (acoustic sum rule built in).

    Attributes
    ----------
    atom_a, atom_b : ndarray of int, shape ``(nblock,)``
        Atoms of the block.
    bonds : ndarray, shape ``(nblock, 3)``
        Vector from atom ``a`` to the image of atom ``b`` (bohr); zero for the self terms.
    blocks : ndarray, shape ``(nblock, 3, 3)``
        :math:`\\Phi_{ab}` (eV/bohr^2).
    """

    atom_a: NDArray[np.int_]
    atom_b: NDArray[np.int_]
    bonds: NDArray[np.float64]
    blocks: NDArray[np.float64]


@dataclass
class ElectronPhononModel:
    """Tight-binding electrons, harmonic phonons and the electron-phonon vertex of one crystal.

    Parameters
    ----------
    name : str
        Name of the model (file names of the outputs).
    lattice : array_like, shape ``(3, 3)``
        Lattice vectors (rows) in units of ``alat``; a non-periodic direction gets a long vector.
    alat : float
        Lattice parameter (bohr).
    positions : array_like, shape ``(natom, 3)``
        Atomic positions in crystal coordinates.
    masses : array_like, shape ``(natom,)``
        Atomic masses (amu).
    orbitals : list of (int, str)
        ``(atom, kind)`` of every orbital, kind one of ``'s'``, ``'px'``, ``'py'``, ``'pz'``.
    onsite : array_like, shape ``(norb,)``
        On-site energies (eV).
    bonds : list of dict
        Hopping shells: ``{'atoms': (a, b), 'distance': d0, 'V': {...}, 'eta': eta}``, the
        Slater-Koster integrals ``V`` (eV, :func:`slater_koster_hopping`) at the bond length ``d0``
        (bohr), scaled as :math:`(d_0/d)^\\eta` at other lengths (``eta`` defaults to 2,
        Harrison's rule).
    springs : list of dict
        Spring shells: ``{'atoms': (a, b), 'distance': d0, 'KL': K_L, 'KT': K_T}``, longitudinal
        and transverse spring constants (eV/bohr^2) of every pair at distance ``d0``.
    nelec : float
        Electrons per cell (both spins).
    dimension : int, optional
        Number of periodic directions (the first ``dimension`` lattice vectors).
    neighbour_search : int, optional
        Cells searched along each periodic direction for the bonds and the springs.

    Notes
    -----
    The Hamiltonian is in the atomic gauge, with the phase of the actual bond vector:

    .. math::

        H_{ij}(\\mathbf k) = \\varepsilon_i\\delta_{ij}
            + \\sum_{\\mathbf d} t_{ij}(\\mathbf d)\\, e^{i\\mathbf k\\cdot\\mathbf d}.

    The vertex is the derivative of the two-centre hoppings with respect to the displacement
    :math:`u_{\\kappa\\beta}(\\mathbf R) = e^{i\\mathbf q\\cdot(\\mathbf R+\\boldsymbol\\tau_\\kappa)}`
    of atom :math:`\\kappa` along :math:`\\beta`:

    .. math::

        \\langle i,\\mathbf k+\\mathbf q|\\partial_{\\mathbf q\\kappa\\beta}V|j,\\mathbf k\\rangle
        = \\sum_{\\mathbf d} \\partial_\\beta t_{ij}(\\mathbf d)
        \\left[\\delta_{\\kappa\\kappa(j)} e^{i(\\mathbf k+\\mathbf q)\\cdot\\mathbf d}
        - \\delta_{\\kappa\\kappa(i)} e^{i\\mathbf k\\cdot\\mathbf d}\\right].

    The springs give the force constants
    :math:`\\Phi_{ab}(\\mathbf d) = -[K_L \\hat d\\hat d + K_T(1 - \\hat d\\hat d)]` and the self
    terms that impose the acoustic sum rule.
    """

    name: str
    lattice: NDArray[np.float64]
    alat: float
    positions: NDArray[np.float64]
    masses: NDArray[np.float64]
    orbitals: list[tuple[int, str]]
    onsite: NDArray[np.float64]
    bonds: list[dict[str, Any]]
    springs: list[dict[str, Any]]
    nelec: float
    dimension: int = 3
    neighbour_search: int = 2
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.lattice = np.asarray(self.lattice, dtype=float)
        self.positions = np.atleast_2d(np.asarray(self.positions, dtype=float))
        self.masses = np.atleast_1d(np.asarray(self.masses, dtype=float))
        self.onsite = np.atleast_1d(np.asarray(self.onsite, dtype=float))
        self.orbitals = [(int(atom), str(kind)) for atom, kind in self.orbitals]
        if self.lattice.shape != (3, 3):
            raise ValueError('lattice must be 3x3 (rows are the lattice vectors)')
        if self.positions.shape != (len(self.masses), 3):
            raise ValueError('one position (crystal coordinates) per mass')
        if len(self.onsite) != len(self.orbitals):
            raise ValueError('one on-site energy per orbital')
        if not 1 <= int(self.dimension) <= 3:
            raise ValueError('dimension must be 1, 2 or 3')
        for atom, kind in self.orbitals:
            if kind not in ORBITAL_KINDS or not 0 <= atom < len(self.masses):
                raise ValueError(
                    'orbital (%d, %r) is not an s or p orbital of an atom' % (atom, kind)
                )
        if np.any(self.masses <= 0.0):
            raise ValueError('masses must be positive')

    # ------------------------------------------------------------------------------- geometry
    @property
    def number_of_atoms(self) -> int:
        """Atoms per cell."""
        return len(self.masses)

    @property
    def number_of_orbitals(self) -> int:
        """Orbitals per cell (the number of bands)."""
        return len(self.orbitals)

    @property
    def number_of_modes(self) -> int:
        """Phonon branches, three per atom."""
        return 3 * self.number_of_atoms

    @property
    def periodic_axes(self) -> tuple[bool, bool, bool]:
        """Which lattice directions are periodic."""
        return tuple(axis < self.dimension for axis in range(3))

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        """Lattice vectors (rows, alat units), as :class:`PAOFLOW.models.models.HoppingModel`."""
        return self.lattice

    @property
    def lattice_vectors_bohr(self) -> NDArray[np.float64]:
        """Lattice vectors (rows, bohr)."""
        return self.alat * self.lattice

    @property
    def reciprocal_vectors(self) -> NDArray[np.float64]:
        """Reciprocal vectors (rows, 1/bohr, with the factor 2 pi)."""
        return 2 * np.pi * np.linalg.inv(self.lattice_vectors_bohr).T

    @property
    def cell_volume(self) -> float:
        """Volume of the cell (bohr^3), the long vectors of the non-periodic directions included."""
        return float(abs(np.linalg.det(self.lattice_vectors_bohr)))

    @property
    def orbital_atoms(self) -> NDArray[np.int_]:
        """Atom of every orbital."""
        return np.array([atom for atom, _kind in self.orbitals], dtype=int)

    def to_cartesian(self, k_crystal: ArrayLike) -> NDArray[np.float64]:
        """Cartesian wave vectors (1/bohr) of crystal coordinates, shape ``(nk, 3)``."""
        return np.atleast_2d(np.asarray(k_crystal, dtype=float)) @ self.reciprocal_vectors

    def _pairs(self, shells: list[dict[str, Any]]) -> list[tuple[int, int, NDArray, dict]]:
        """``(a, b, bond, shell)`` of every pair of atoms at the distance of a shell."""
        positions = self.positions
        ranges = [
            range(-self.neighbour_search, self.neighbour_search + 1) if periodic else range(1)
            for periodic in self.periodic_axes
        ]
        cells = np.stack(np.meshgrid(*ranges, indexing='ij'), -1).reshape(-1, 3)
        pairs = []
        for shell in shells:
            atom_a, atom_b = shell['atoms']
            for cell in cells:
                bond = (cell + positions[atom_b] - positions[atom_a]) @ self.lattice_vectors_bohr
                length = np.linalg.norm(bond)
                if length > 1e-8 and abs(length - shell['distance']) < 1e-6:
                    pairs.append((atom_a, atom_b, bond, shell))
                    if atom_a != atom_b:
                        pairs.append((atom_b, atom_a, -bond, shell))
        return pairs

    # ------------------------------------------------------------------------------ the tables
    def hopping_table(self) -> HoppingTable:
        """Every hopping of the model with its gradient (cached).

        Returns
        -------
        HoppingTable

        Notes
        -----
        A hopping of the shell with bond length :math:`d_0` and exponent :math:`\\eta` is
        :math:`t_{ij}(\\mathbf d) = t^{SK}_{ij}(\\hat d)\\,(d_0/|\\mathbf d|)^\\eta`; its gradient is
        taken by central finite differences with a step of 1e-6 bohr.
        """
        if 'hoppings' in self._cache:
            return self._cache['hoppings']
        rows = []
        for atom_a, atom_b, bond, shell in self._pairs(self.bonds):
            exponent = shell.get('eta', 2.0)
            for index_i, (atom_i, kind_i) in enumerate(self.orbitals):
                if atom_i != atom_a:
                    continue
                for index_j, (atom_j, kind_j) in enumerate(self.orbitals):
                    if atom_j != atom_b:
                        continue

                    def hopping(
                        vector, kind_i=kind_i, kind_j=kind_j, shell=shell, exponent=exponent
                    ):
                        scale = (shell['distance'] / np.linalg.norm(vector)) ** exponent
                        return slater_koster_hopping(kind_i, kind_j, vector, shell['V']) * scale

                    value = hopping(bond)
                    gradient = np.array(
                        [
                            (hopping(bond + step) - hopping(bond - step)) / (2 * GRADIENT_STEP)
                            for step in GRADIENT_STEP * np.eye(3)
                        ]
                    )
                    if abs(value) > 1e-14 or np.abs(gradient).max() > 1e-14:
                        rows.append((index_i, index_j, bond, value, gradient))
        if not rows:
            raise ValueError('the model has no hopping: check the bond distances')
        table = HoppingTable(
            np.array([row[0] for row in rows]),
            np.array([row[1] for row in rows]),
            np.array([row[2] for row in rows]),
            np.array([row[3] for row in rows]),
            np.array([row[4] for row in rows]),
        )
        self._cache['hoppings'] = table
        return table

    def force_constant_table(self) -> ForceConstantTable:
        """Force-constant blocks of the springs and the self terms (cached).

        Returns
        -------
        ForceConstantTable
        """
        if 'force_constants' in self._cache:
            return self._cache['force_constants']
        atom_a, atom_b, bonds, blocks = [], [], [], []
        self_terms = np.zeros((self.number_of_atoms, 3, 3))
        for first, second, bond, shell in self._pairs(self.springs):
            direction = bond / np.linalg.norm(bond)
            projector = np.outer(direction, direction)
            spring = shell['KL'] * projector + shell.get('KT', 0.0) * (np.eye(3) - projector)
            atom_a.append(first)
            atom_b.append(second)
            bonds.append(bond)
            blocks.append(-spring)
            self_terms[first] += spring
        for atom in range(self.number_of_atoms):
            atom_a.append(atom)
            atom_b.append(atom)
            bonds.append(np.zeros(3))
            blocks.append(self_terms[atom])
        table = ForceConstantTable(
            np.array(atom_a), np.array(atom_b), np.array(bonds), np.array(blocks)
        )
        self._cache['force_constants'] = table
        return table

    # ------------------------------------------------------------------------------- k-space
    def hamiltonian(
        self, k_crystal: ArrayLike, derivative: bool = False
    ) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Bloch Hamiltonian in the atomic gauge (and its k-derivative).

        Parameters
        ----------
        k_crystal : array_like, shape ``(nk, 3)`` or ``(3,)``
            Wave vectors in crystal coordinates.
        derivative : bool, optional
            Also return :math:`\\partial H/\\partial k_a` (eV bohr).

        Returns
        -------
        ndarray, shape ``(nk, norb, norb)``
            :math:`H(\\mathbf k)` (eV); with ``derivative`` a tuple with the velocity matrices
            of shape ``(nk, 3, norb, norb)``.
        """
        table = self.hopping_table()
        k_cartesian = self.to_cartesian(k_crystal)
        phase = np.exp(1j * k_cartesian @ table.bonds.T)  # (nk, nhop)
        size = self.number_of_orbitals
        flat = table.orbital_i * size + table.orbital_j
        hamiltonian = np.zeros((len(k_cartesian), size * size), complex)
        np.add.at(hamiltonian.T, flat, (phase * table.hoppings).T)
        hamiltonian = hamiltonian.reshape(-1, size, size) + np.diag(self.onsite)
        if not derivative:
            return hamiltonian
        velocity = np.zeros((len(k_cartesian), 3, size * size), complex)
        for axis in range(3):
            np.add.at(
                velocity[:, axis].T, flat, (1j * phase * table.hoppings * table.bonds[:, axis]).T
            )
        return hamiltonian, velocity.reshape(-1, 3, size, size)

    def vertex(self, k_crystal: ArrayLike, q_crystal: ArrayLike) -> NDArray[np.complex128]:
        """Deformation potential in the orbital basis, one displacement pattern per atom and axis.

        Parameters
        ----------
        k_crystal : array_like, shape ``(nk, 3)``
            Initial wave vectors (crystal coordinates).
        q_crystal : array_like, shape ``(3,)``
            Phonon wave vector (crystal coordinates).

        Returns
        -------
        ndarray, shape ``(nk, 3 natom, norb, norb)``
            :math:`\\langle i,\\mathbf k+\\mathbf q|\\partial_{\\mathbf q\\kappa\\beta}V|j,
            \\mathbf k\\rangle` (eV/bohr), the pattern index ``3 kappa + beta``.
        """
        table = self.hopping_table()
        k_cartesian = self.to_cartesian(k_crystal)
        kq_cartesian = k_cartesian + self.to_cartesian(q_crystal)
        phase_kq = np.exp(1j * kq_cartesian @ table.bonds.T)
        phase_k = np.exp(1j * k_cartesian @ table.bonds.T)
        atom_of = self.orbital_atoms
        atom_i, atom_j = atom_of[table.orbital_i], atom_of[table.orbital_j]
        size = self.number_of_orbitals
        flat = table.orbital_i * size + table.orbital_j
        result = np.zeros((len(k_cartesian), self.number_of_modes, size * size), complex)
        for atom in range(self.number_of_atoms):
            on_j = (atom_j == atom).astype(float)
            on_i = (atom_i == atom).astype(float)
            for axis in range(3):
                gradient = table.gradients[:, axis]
                values = phase_kq * (gradient * on_j) - phase_k * (gradient * on_i)
                np.add.at(result[:, 3 * atom + axis].T, flat, values.T)
        return result.reshape(len(k_cartesian), self.number_of_modes, size, size)

    def phonons(self, q_crystal: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Phonon energies and eigenvectors of the dynamical matrix.

        Parameters
        ----------
        q_crystal : array_like, shape ``(nq, 3)`` or ``(3,)``
            Wave vectors (crystal coordinates).

        Returns
        -------
        energies : ndarray, shape ``(nq, nmode)``
            :math:`\\hbar\\omega_{\\mathbf q\\nu}` (eV), negative for an unstable mode.
        eigenvectors : ndarray, shape ``(nq, 3 natom, nmode)``
            :math:`e_{\\kappa\\beta,\\nu}(\\mathbf q)`, the row index ``3 kappa + beta``.

        Notes
        -----
        .. math::

            D_{\\kappa\\alpha,\\kappa'\\beta}(\\mathbf q) = \\frac{1}{\\sqrt{M_\\kappa M_{\\kappa'}}}
            \\sum_{\\mathbf d}\\Phi_{\\kappa\\alpha,\\kappa'\\beta}(\\mathbf d)\\,
            e^{i\\mathbf q\\cdot\\mathbf d}.
        """
        table = self.force_constant_table()
        q_cartesian = self.to_cartesian(q_crystal)
        phase = np.exp(1j * q_cartesian @ table.bonds.T)  # (nq, nblock)
        natom = self.number_of_atoms
        dynamical = np.zeros((len(q_cartesian), natom, 3, natom, 3), complex)
        for block in range(len(table.atom_a)):
            first, second = table.atom_a[block], table.atom_b[block]
            dynamical[:, first, :, second, :] += (
                phase[:, block, None, None]
                * table.blocks[block]
                / np.sqrt(self.masses[first] * self.masses[second])
            )
        dynamical = dynamical.reshape(len(q_cartesian), self.number_of_modes, self.number_of_modes)
        dynamical = 0.5 * (dynamical + np.conj(np.swapaxes(dynamical, 1, 2)))
        squared, eigenvectors = np.linalg.eigh(dynamical)
        energies = np.sign(squared) * np.sqrt(np.abs(squared) * HBAR2_K_OVER_M)
        return energies, eigenvectors

    def to_hopping_model(self) -> 'HoppingModel':
        """The electrons as a :class:`PAOFLOW.models.models.HoppingModel`.

        Returns
        -------
        HoppingModel
            For the PAOFLOW tools that take models (``PAOFLOW.PAOFLOW(model=...)``, the band
            paths of ``paoflow-gen``).
        """
        from ..models import HoppingModel

        table = self.hopping_table()
        atom_of = self.orbital_atoms
        hoppings = [(index, index, (0, 0, 0), energy) for index, energy in enumerate(self.onsite)]
        for index_i, index_j, bond, value in zip(
            table.orbital_i, table.orbital_j, table.bonds, table.hoppings
        ):
            cell = np.rint(
                np.linalg.solve(self.lattice_vectors_bohr.T, bond)
                - self.positions[atom_of[index_j]]
                + self.positions[atom_of[index_i]]
            ).astype(int)
            hoppings.append((int(index_i), int(index_j), tuple(cell), value))
        grid = tuple(
            2 * self.neighbour_search + 2 if periodic else 1 for periodic in self.periodic_axes
        )
        return HoppingModel(
            self.name,
            tuple(hoppings),
            self.number_of_orbitals,
            self.lattice,
            self.alat,
            self.positions[atom_of],
            self.nelec,
            grid,
        )
