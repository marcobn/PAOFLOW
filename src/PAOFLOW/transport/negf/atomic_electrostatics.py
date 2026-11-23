"""Three-dimensional electrostatics of the junction: one Gaussian charge per atom.

The bias-induced change of the electron number on every atom of the conductor, :math:`\\delta q_A`, is a normalised
Gaussian of width :math:`\\sigma_A` (:mod:`~PAOFLOW.transport.negf.charge_model`). The potential energy of an electron
solves the Poisson equation in the slab between the electrode planes, periodic across the transverse cell, with
Dirichlet values :math:`\\mu_{L,R} - E_F` at the planes (:mod:`~PAOFLOW.transport.negf.poisson_3d`). Averaged over
the charge of every atom,

.. math::

    U_A = U_L + (U_R - U_L)\\,\\langle g_A|\\tfrac{z - a}{b - a}\\rangle + \\sum_B P_{AB}\\,\\delta q_B,

and :math:`U_A` acts on all the orbitals of atom :math:`A`. The electrode planes are those of the planar
electrostatics, half a layer spacing outside the conductor. The planar average of the three-dimensional potential
is the planar potential smoothed by the Gaussians, and the in-plane structure adds the on-site Hartree energy of
every atom.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from PAOFLOW.transport.negf.electrostatics import planar_layers
from PAOFLOW.transport.negf.poisson_3d import DirichletSlab, GaussianChargesInSlab
from PAOFLOW.transport.partition.types import AtomOrbitals, block_atom_index


class AtomicElectrostatics:
    """Charge groups = atoms, three-dimensional Poisson equation between the electrode planes.

    Parameters
    ----------
    group_of_orbital : ndarray, shape ``(dimC,)``, int
        Atom of every orbital of the conductor block.
    positions : ndarray, shape ``(n_atoms, 3)``
        Cartesian positions (bohr).
    labels : sequence of str
        Element of every atom.
    widths : ndarray, shape ``(n_atoms,)``
        Gaussian widths (bohr).
    transport_axis : int
        Cartesian axis of transport.
    in_plane_vectors : ndarray, shape ``(2, 3)``
        Transverse lattice vectors (bohr), perpendicular to the transport axis.
    left_boundary, right_boundary : float
        Electrode planes along the transport axis (bohr).
    comm : MPI.Comm, optional
        The kernel is computed on rank 0 and broadcast.
    """

    name = '3d'

    def __init__(
        self,
        group_of_orbital: NDArray[np.int_],
        positions: NDArray[np.float64],
        labels: Sequence[str],
        widths: NDArray[np.float64],
        transport_axis: int,
        in_plane_vectors: NDArray[np.float64],
        left_boundary: float,
        right_boundary: float,
        comm: MPI.Comm | None = None,
    ) -> None:
        self.group_of_orbital = np.asarray(group_of_orbital, dtype=int)
        self._positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        self.labels = [str(label) for label in labels]
        self.widths = np.asarray(widths, dtype=float).ravel()
        self.transport_axis = int(transport_axis)
        self.in_plane_axes = [axis for axis in range(3) if axis != self.transport_axis]
        vectors = np.asarray(in_plane_vectors, dtype=float).reshape(2, 3)
        self.in_plane_vectors_cartesian = vectors
        self.slab = DirichletSlab(
            in_plane_vectors=vectors[:, self.in_plane_axes],
            lower=float(left_boundary),
            upper=float(right_boundary),
        )
        self.charges = GaussianChargesInSlab(
            self.slab,
            self._positions[:, self.in_plane_axes],
            self._positions[:, self.transport_axis],
            self.widths,
        )
        self._ramp = self.charges.ramp_fractions()
        kernel = None
        if comm is None or comm.Get_rank() == 0:
            kernel = self.charges.kernel()
        if comm is not None and comm.Get_size() > 1:
            kernel = comm.bcast(kernel, root=0)
        self._kernel = kernel

    @property
    def n_groups(self) -> int:
        return len(self._positions)

    @property
    def coordinates(self) -> NDArray[np.float64]:
        return self._positions

    def external_potential(
        self, potential_left: float, potential_right: float
    ) -> NDArray[np.float64]:
        """The linear ramp between the planes averaged over every atomic charge (eV).

        Parameters
        ----------
        potential_left, potential_right : float
            Potential energy of an electron in the left and right electrodes (eV).

        Returns
        -------
        ndarray, shape ``(n_atoms,)``
        """
        return potential_left + (potential_right - potential_left) * self._ramp

    def kernel(self) -> NDArray[np.float64]:
        """:math:`P_{AB}` (eV per electron), grounded planes."""
        return self._kernel

    def planar_average(
        self,
        group_charge: NDArray[np.float64],
        potential_left: float,
        potential_right: float,
        heights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Planar average of :math:`U(\\mathbf r)` along the transport axis (eV).

        Parameters
        ----------
        group_charge : ndarray, shape ``(n_atoms,)``
        potential_left, potential_right : float
            Electrode potentials (eV).
        heights : ndarray
            Positions along the transport axis (bohr), between the planes.

        Returns
        -------
        ndarray, shape like ``heights``
        """
        z = np.asarray(heights, dtype=float)
        ramp = potential_left + (potential_right - potential_left) * (z - self.slab.lower) / (
            self.slab.length
        )
        return ramp + self.charges.planar_average(group_charge, z)

    def potential_on_grid(
        self,
        group_charge: NDArray[np.float64],
        potential_left: float,
        potential_right: float,
        spacing: float = 0.3,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """:math:`U(\\mathbf r)` on a grid of the slab, planes included (eV).

        Parameters
        ----------
        group_charge : ndarray, shape ``(n_atoms,)``
        potential_left, potential_right : float
            Electrode potentials (eV).
        spacing : float, optional
            Approximate grid spacing (bohr).

        Returns
        -------
        origin : ndarray, shape ``(3,)``
            Cartesian origin of the grid (bohr): the left plane.
        spanning_vectors : ndarray, shape ``(3, 3)``
            :math:`\\mathbf a_1, \\mathbf a_2, L\\hat{\\mathbf z}` as rows (bohr).
        values : ndarray, shape ``(N_1, N_2, N_z + 1)``
            Periodic in-plane (the last point of each in-plane axis is not repeated); along the transport axis from
            the left to the right plane.
        """
        lengths = np.linalg.norm(self.in_plane_vectors_cartesian, axis=1)
        in_plane_points = tuple(max(2, int(np.ceil(length / spacing))) for length in lengths)
        n_z = max(2, int(np.ceil(self.slab.length / spacing)))
        z = self.slab.lower + self.slab.length * np.arange(n_z + 1) / n_z
        values = self.charges.potential_on_grid(group_charge, in_plane_points, z)
        values += (
            potential_left
            + (potential_right - potential_left) * (z - self.slab.lower) / self.slab.length
        )[None, None, :]
        axis_vector = np.zeros(3)
        axis_vector[self.transport_axis] = self.slab.length
        origin = np.zeros(3)
        origin[self.transport_axis] = self.slab.lower
        spanning = np.vstack([self.in_plane_vectors_cartesian, axis_vector])
        return origin, spanning, values


def transverse_lattice_vectors(
    cell_vectors_bohr: NDArray[np.float64], transport_axis: int, tolerance: float = 1.0e-6
) -> NDArray[np.float64]:
    """The two lattice vectors of the conductor cell across the transport axis.

    Parameters
    ----------
    cell_vectors_bohr : ndarray, shape ``(3, 3)``
        Lattice vectors as rows (bohr).
    transport_axis : int
    tolerance : float, optional
        Largest component along the transport axis, relative to the length of the vector.

    Returns
    -------
    ndarray, shape ``(2, 3)``

    Raises
    ------
    NotImplementedError
        If they are not perpendicular to the transport axis.
    """
    cell = np.asarray(cell_vectors_bohr, dtype=float)
    along_axis = int(np.argmax(np.abs(cell[:, transport_axis])))
    vectors = np.delete(cell, along_axis, axis=0)
    if np.any(np.abs(vectors[:, transport_axis]) > tolerance * np.linalg.norm(vectors, axis=1)):
        raise NotImplementedError(
            'the three-dimensional electrostatics needs transverse lattice vectors perpendicular to the '
            'transport axis'
        )
    return vectors


def atomic_electrostatics(
    central_atoms: Sequence[AtomOrbitals],
    transport_axis: int,
    cell_vectors_bohr: NDArray[np.float64],
    widths: NDArray[np.float64],
    comm: MPI.Comm | None = None,
) -> AtomicElectrostatics:
    """The three-dimensional electrostatics of a conductor partition.

    Parameters
    ----------
    central_atoms : sequence of AtomOrbitals
        Atoms of the conductor in the orbital order of its block (``HamiltonianBlockPartition.central_atoms``).
    transport_axis : int
        Cartesian axis of transport (0, 1, 2).
    cell_vectors_bohr : ndarray, shape ``(3, 3)``
        Lattice vectors of the conductor cell as rows (bohr).
    widths : ndarray, shape ``(n_atoms,)``
        Gaussian width of every atom (bohr), :func:`~PAOFLOW.transport.negf.charge_model.atomic_charge_widths`.
    comm : MPI.Comm, optional

    Returns
    -------
    AtomicElectrostatics
    """
    layers = planar_layers(central_atoms, transport_axis, cell_vectors_bohr)
    group_of_orbital = block_atom_index(central_atoms)  # rows of an atom may be split by spin
    return AtomicElectrostatics(
        group_of_orbital=group_of_orbital,
        positions=np.array([atom.position for atom in central_atoms], dtype=float),
        labels=[atom.element for atom in central_atoms],
        widths=widths,
        transport_axis=transport_axis,
        in_plane_vectors=transverse_lattice_vectors(cell_vectors_bohr, transport_axis),
        left_boundary=layers.left_boundary,
        right_boundary=layers.right_boundary,
        comm=comm,
    )
