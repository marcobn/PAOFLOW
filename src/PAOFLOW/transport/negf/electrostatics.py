"""Electrostatics of the junction: the potential of the bias-induced charge between two electrodes.

The self-consistent loop (:mod:`~PAOFLOW.transport.negf.self_consistency`) sees the electrostatics only through an
:class:`ElectrostaticModel`: the orbitals are grouped into charge groups (atomic layers, atoms), and the potential
energy of an electron on the groups is linear in their charges,

.. math::

    U_g = U^{\rm ext}_g + \sum_h P_{gh}\,\delta q_h,

with :math:`U^{\rm ext}` the solution of the Laplace equation with the electrode potentials as Dirichlet values and
:math:`P` the Poisson kernel with grounded electrodes (eV per electron).

Two models:

- :class:`PlanarLayers` (here): the change of the electron number on each atomic layer is a sheet of charge. The
  planar average of the Hartree potential solves a one-dimensional Poisson equation along the transport axis; the
  potential is piecewise linear, with kinks at the charged layers. Exact in planar average for a junction uniform
  across the transverse cell.
- :class:`~PAOFLOW.transport.negf.atomic_electrostatics.AtomicElectrostatics`: one Gaussian charge per atom and
  the three-dimensional Poisson equation, periodic across the transverse cell (molecules, wires).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.transport.partition.types import AtomOrbitals

HARTREE_EV = 27.211386245988


@runtime_checkable
class ElectrostaticModel(Protocol):
    """What the self-consistent loop needs from the electrostatics.

    Attributes
    ----------
    group_of_orbital : ndarray, shape ``(dimC,)``, int
        Charge group of every orbital of the conductor block.
    n_groups : int
    coordinates : ndarray
        Positions of the groups (bohr): ``(n_groups,)`` along the transport axis for layers, ``(n_groups, 3)`` for
        atoms.
    """

    group_of_orbital: NDArray[np.int_]

    @property
    def n_groups(self) -> int: ...

    @property
    def coordinates(self) -> NDArray[np.float64]: ...

    def external_potential(
        self, potential_left: float, potential_right: float
    ) -> NDArray[np.float64]:
        """Potential energy on the groups with no charge (eV): the Laplace solution, shape ``(n_groups,)``."""
        ...

    def kernel(self) -> NDArray[np.float64]:
        """Potential energy on the groups per extra electron on each group, grounded electrodes (eV)."""
        ...


@dataclass(frozen=True)
class PlanarLayers:
    """Atomic layers of the conductor along the transport axis.

    Attributes
    ----------
    layer_of_orbital : ndarray, shape ``(dimC,)``, int
        Layer of every orbital of the conductor block.
    coordinates : ndarray, shape ``(n_layers,)``
        Position of each layer along the transport axis (bohr), increasing.
    left_boundary, right_boundary : float
        Electrode planes (bohr): half a layer spacing outside the first and the last layer.
    area : float
        Area of the transverse cell (bohr²).
    """

    layer_of_orbital: NDArray[np.int_]
    coordinates: NDArray[np.float64]
    left_boundary: float
    right_boundary: float
    area: float

    @property
    def n_layers(self) -> int:
        return len(self.coordinates)

    @property
    def group_of_orbital(self) -> NDArray[np.int_]:
        return self.layer_of_orbital

    @property
    def n_groups(self) -> int:
        return self.n_layers

    def external_potential(
        self, potential_left: float, potential_right: float
    ) -> NDArray[np.float64]:
        """The linear ramp between the electrode planes at the layers (eV).

        Parameters
        ----------
        potential_left, potential_right : float
            Potential energy of an electron in the left and right electrodes (eV).

        Returns
        -------
        ndarray, shape ``(n_layers,)``
        """
        z = self.coordinates
        a, b = self.left_boundary, self.right_boundary
        return potential_left + (potential_right - potential_left) * (z - a) / (b - a)

    def kernel(self) -> NDArray[np.float64]:
        """The planar Poisson kernel (:func:`poisson_kernel`)."""
        return poisson_kernel(self)


def planar_layers(
    central_atoms: Sequence[AtomOrbitals],
    transport_axis: int,
    cell_vectors_bohr: NDArray[np.float64],
    tolerance: float = 1.0e-3,
) -> PlanarLayers:
    """Group the conductor orbitals into layers and find the electrode planes.

    Parameters
    ----------
    central_atoms : sequence of AtomOrbitals
        Atoms of the conductor in the orbital order of its block (``HamiltonianBlockPartition.central_atoms``).
    transport_axis : int
        Cartesian axis of transport (0, 1, 2).
    cell_vectors_bohr : ndarray, shape ``(3, 3)``
        Lattice vectors as rows (bohr).
    tolerance : float, optional
        Atoms closer than this along the axis (bohr) share a layer.

    Returns
    -------
    PlanarLayers

    Raises
    ------
    ValueError
        With fewer than two layers.
    """
    coordinates_of_atoms = np.array(
        [atom.position[transport_axis] for atom in central_atoms], dtype=float
    )
    order = np.argsort(coordinates_of_atoms, kind='stable')
    layer_coordinates: list[float] = []
    layer_of_atom = np.empty(len(central_atoms), dtype=int)
    for index in order:
        coordinate = coordinates_of_atoms[index]
        if not layer_coordinates or coordinate - layer_coordinates[-1] > tolerance:
            layer_coordinates.append(coordinate)
        layer_of_atom[index] = len(layer_coordinates) - 1
    if len(layer_coordinates) < 2:
        raise ValueError('the planar electrostatics needs at least two layers in the conductor')
    layer_of_orbital = np.concatenate(
        [
            np.full(atom.dimension, layer_of_atom[i], dtype=int)
            for i, atom in enumerate(central_atoms)
        ]
    )
    coordinates = np.asarray(layer_coordinates)
    cell = np.asarray(cell_vectors_bohr, dtype=float)
    along_axis = int(np.argmax(np.abs(cell[:, transport_axis])))
    area = abs(float(np.linalg.det(cell))) / abs(float(cell[along_axis, transport_axis]))
    return PlanarLayers(
        layer_of_orbital=layer_of_orbital,
        coordinates=coordinates,
        left_boundary=float(coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])),
        right_boundary=float(coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])),
        area=area,
    )


def group_sums(
    orbital_values: NDArray[np.float64], model: ElectrostaticModel
) -> NDArray[np.float64]:
    """Sum an orbital-resolved quantity over each charge group.

    Parameters
    ----------
    orbital_values : ndarray, shape ``(dimC,)``
    model : ElectrostaticModel

    Returns
    -------
    ndarray, shape ``(n_groups,)``
    """
    return np.bincount(model.group_of_orbital, weights=orbital_values, minlength=model.n_groups)


def layer_sums(orbital_values: NDArray[np.float64], layers: PlanarLayers) -> NDArray[np.float64]:
    """Sum an orbital-resolved quantity over each layer (:func:`group_sums`).

    Parameters
    ----------
    orbital_values : ndarray, shape ``(dimC,)``
    layers : PlanarLayers

    Returns
    -------
    ndarray, shape ``(n_layers,)``
    """
    return group_sums(orbital_values, layers)


def electrostatic_potential(
    group_charge: NDArray[np.float64],
    model: ElectrostaticModel,
    potential_left: float,
    potential_right: float,
) -> NDArray[np.float64]:
    """Electron potential energy on the groups from their charges and the electrode potentials.

    Parameters
    ----------
    group_charge : ndarray, shape ``(n_groups,)``
        Change of the electron number on each group, per transverse cell (positive: more electrons).
    model : ElectrostaticModel
    potential_left, potential_right : float
        Potential energy of an electron in the left and right electrodes (eV): :math:`\mu_{L,R} - E_F`.

    Returns
    -------
    ndarray, shape ``(n_groups,)``
        :math:`U = U^{\rm ext} + P\,\delta q` (eV).
    """
    return model.external_potential(potential_left, potential_right) + model.kernel() @ np.asarray(
        group_charge, dtype=float
    )


def planar_potential(
    layer_charge: NDArray[np.float64],
    layers: PlanarLayers,
    potential_left: float,
    potential_right: float,
) -> NDArray[np.float64]:
    """Electron potential energy at every layer from the layer charges and the electrode potentials.

    Parameters
    ----------
    layer_charge : ndarray, shape ``(n_layers,)``
        Change of the electron number on each layer, per transverse cell (positive: more electrons).
    layers : PlanarLayers
    potential_left, potential_right : float
        Potential energy of an electron in the left and right electrodes (eV): their shifts
        :math:`\\mu_L - E_F`, :math:`\\mu_R - E_F`.

    Returns
    -------
    ndarray, shape ``(n_layers,)``
        Potential energy :math:`U` at the layers (eV).

    Notes
    -----
    In Hartree atomic units the electron potential energy obeys :math:`U'' = -4\\pi\\,\\delta n` with
    :math:`\\delta n(z) = \\sum_m (\\delta q_m/A)\\,\\delta(z - z_m)`, :math:`U(a) = U_L`, :math:`U(b) = U_R`.
    With the Dirichlet Green's function of the interval :math:`[a, b]`,

    .. math::

        U(z) = U_L + (U_R - U_L)\\frac{z - a}{b - a}
            + \\frac{4\\pi}{A}\\sum_m \\delta q_m\\,\\frac{(z_< - a)(b - z_>)}{b - a},

    :math:`z_< = \\min(z, z_m)`, :math:`z_> = \\max(z, z_m)`: extra electrons raise the potential energy around
    them. The electrodes are ideal metals: they screen at their planes ``a``, ``b``.
    """
    z = layers.coordinates
    a, b = layers.left_boundary, layers.right_boundary
    ramp = potential_left + (potential_right - potential_left) * (z - a) / (b - a)
    return ramp + poisson_kernel(layers) @ np.asarray(layer_charge, dtype=float)


def poisson_kernel(layers: PlanarLayers) -> NDArray[np.float64]:
    """Potential energy on every layer per extra electron on every layer, with grounded electrodes.

    Parameters
    ----------
    layers : PlanarLayers

    Returns
    -------
    ndarray, shape ``(n_layers, n_layers)``
        :math:`P_{lm} = (4\\pi/A)(z_< - a)(b - z_>)/(b - a)` in eV per electron (symmetric, positive definite):
        the response part of :func:`planar_potential`.
    """
    z = layers.coordinates
    a, b = layers.left_boundary, layers.right_boundary
    lower = np.minimum(z[:, None], z[None, :]) - a
    upper = b - np.maximum(z[:, None], z[None, :])
    return 4.0 * np.pi / layers.area * lower * upper / (b - a) * HARTREE_EV
