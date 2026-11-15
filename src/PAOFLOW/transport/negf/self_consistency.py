"""Self-consistent potential of a conductor at finite bias, and the current.

The PAO Hamiltonian of the DFT run at zero bias is the self-consistent reference. At a bias V the leads shift to
their electrochemical potentials, the conductor gets a potential energy :math:`U` on every orbital, and :math:`U`
follows from the Hartree response to the bias-induced change of the electron numbers,
:math:`\\delta n = n(V) - n(0)`, both computed with the same open-system Green's functions (so that the errors of
the projection cancel and :math:`V = 0` gives back the DFT Hamiltonian):

.. math::

    H(V) = H_0 + \\tfrac12\\{U, S\\},\\qquad U = U_{\\rm ramp} + v_H[\\delta n(U)].

The exchange-correlation response is neglected. The fixed point is found with Anderson mixing of :math:`U`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from PAOFLOW.transport.negf.contour import KB_EV, EnergyQuadrature, bias_window, equilibrium_contour
from PAOFLOW.transport.negf.density_matrix import (
    NonEquilibriumDensity,
    OpenConductor,
    landauer_current,
    local_density_of_states,
    open_system_density,
)
from PAOFLOW.transport.negf.electrostatics import (
    PlanarLayers,
    layer_sums,
    planar_potential,
    poisson_kernel,
)

PROFILES = ('self_consistent', 'linear')


@dataclass(frozen=True)
class SelfConsistencySettings:
    """Numerical settings of the finite-bias calculation.

    Attributes
    ----------
    temperature : float
        Electron temperature of the leads (K).
    fermi_energy : float
        Fermi level of the leads at zero bias (eV).
    arc_points, line_points, fermi_poles : int
        Equilibrium contour (:func:`~PAOFLOW.transport.negf.contour.equilibrium_contour`).
    window_spacing : float
        Energy step in the bias window (eV).
    mixing : float
        Anderson mixing parameter (applied to the preconditioned residual).
    history : int
        Previous iterations kept by the Anderson mixing.
    tolerance : float
        Convergence threshold on :math:`\\max|U_{\\rm out} - U_{\\rm in}|` (eV).
    max_iterations : int
        Iteration limit per bias.
    energy_margin : float
        The contour starts this far below the lowest state of the conductor (eV).
    thomas_fermi_preconditioner : bool
        Precondition the residual with :math:`(1 + P\\chi_0)^{-1}` (see :func:`finite_bias_scan`); without it a
        metal needs mixing below about :math:`2/(1 + P\\chi_0)`, a few hundredths.
    verbose : bool
        Print one line per iteration on rank 0.
    """

    temperature: float = 300.0
    fermi_energy: float = 0.0
    arc_points: int = 30
    line_points: int = 40
    fermi_poles: int = 10
    window_spacing: float = 0.005
    mixing: float = 0.5
    history: int = 6
    tolerance: float = 1.0e-3
    max_iterations: int = 80
    energy_margin: float = 2.0
    thomas_fermi_preconditioner: bool = True
    verbose: bool = False

    @property
    def temperature_ev(self) -> float:
        return KB_EV * self.temperature


@dataclass
class BiasPointResult:
    """The converged state of the junction at one bias.

    Attributes
    ----------
    bias : float
        V (volt).
    chemical_potential_left, chemical_potential_right : float
        :math:`\\mu_L`, :math:`\\mu_R` (eV).
    layer_potential : ndarray, shape ``(n_layers,)``
        Electron potential energy on every layer (eV).
    layer_charge : ndarray, shape ``(n_layers,)``
        Change of the electron number on every layer with respect to zero bias, per transverse cell.
    window_energies, transmission : ndarray, shape ``(n_window,)``
        :math:`T(E; V)` per spin and transverse cell in the bias window.
    current : float
        :math:`\\int dE\\, T(E;V)\\,[f_L - f_R]` (eV): the current in units of :math:`2e/h`, per transverse cell.
    iterations : int
        Iterations used (0 for a fixed profile).
    converged : bool
    residual : float
        Last :math:`\\max|U_{\\rm out} - U_{\\rm in}|` (eV).
    """

    bias: float
    chemical_potential_left: float
    chemical_potential_right: float
    layer_potential: NDArray[np.float64]
    layer_charge: NDArray[np.float64]
    window_energies: NDArray[np.float64]
    transmission: NDArray[np.float64]
    current: float
    iterations: int
    converged: bool
    residual: float


@dataclass
class FiniteBiasResults:
    """A bias scan.

    Attributes
    ----------
    profile : str
        ``'self_consistent'`` or ``'linear'``.
    layer_coordinates : ndarray, shape ``(n_layers,)``
        Layer positions along the transport axis (bohr).
    points : list of BiasPointResult
    """

    profile: str
    layer_coordinates: NDArray[np.float64]
    points: list[BiasPointResult] = field(default_factory=list)

    @property
    def bias(self) -> NDArray[np.float64]:
        return np.array([point.bias for point in self.points])

    @property
    def current(self) -> NDArray[np.float64]:
        return np.array([point.current for point in self.points])


def anderson_update(
    inputs: Sequence[NDArray[np.float64]],
    residuals: Sequence[NDArray[np.float64]],
    mixing: float,
    preconditioner: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Next input of a fixed-point iteration :math:`x = F(x)` by Anderson (Pulay) mixing.

    Parameters
    ----------
    inputs : sequence of ndarray
        Previous inputs :math:`x_j`, the last one current.
    residuals : sequence of ndarray
        Their residuals :math:`r_j = F(x_j) - x_j`.
    mixing : float
        :math:`\\beta`.
    preconditioner : ndarray, shape ``(n, n)``, optional
        :math:`M`, an approximation of :math:`(1 - \\partial F/\\partial x)^{-1}` applied to the residual.

    Returns
    -------
    ndarray

    Notes
    -----
    With :math:`\\Delta x_j = x_n - x_j`, :math:`\\Delta r_j = r_n - r_j`, the coefficients :math:`\\theta`
    minimise :math:`|r_n - \\Delta r\\,\\theta|`, and
    :math:`x_{n+1} = x_n - \\Delta x\\,\\theta + \\beta M(r_n - \\Delta r\\,\\theta)`. With one entry it is
    (preconditioned) linear mixing.
    """
    x, r = inputs[-1], residuals[-1]
    step = (lambda v: v) if preconditioner is None else (lambda v: preconditioner @ v)
    if len(inputs) == 1:
        return x + mixing * step(r)
    delta_inputs = np.stack([x - previous for previous in inputs[:-1]], axis=1)
    delta_residuals = np.stack([r - previous for previous in residuals[:-1]], axis=1)
    theta = np.linalg.lstsq(delta_residuals, r, rcond=None)[0]
    return x - delta_inputs @ theta + mixing * step(r - delta_residuals @ theta)


class ConductorPotential:
    """A potential energy on the layers of the conductor, added to its Hamiltonian block.

    Parameters
    ----------
    blc_blocks : Mapping[str, Any]
        Block operators; ``blc_00C.H`` is changed in place by :meth:`apply` and given back by :meth:`restore`.
    layers : PlanarLayers

    Notes
    -----
    :math:`H = H_0 + \\tfrac12(U_i + U_j)S_{ij}`, which is :math:`H_0 + \\mathrm{diag}(U)` for orthonormal
    orbitals.
    """

    def __init__(self, blc_blocks: Mapping[str, Any], layers: PlanarLayers) -> None:
        self._block = blc_blocks['blc_00C']
        self._reference = np.array(self._block.H, copy=True)
        self._overlap = np.array(self._block.S, copy=True)
        self._layers = layers

    def apply(self, layer_potential: NDArray[np.float64]) -> None:
        """Set :math:`H = H_0 + \\tfrac12\\{U, S\\}`.

        Parameters
        ----------
        layer_potential : ndarray, shape ``(n_layers,)``
            Potential energy of every layer (eV).
        """
        orbital = np.asarray(layer_potential, dtype=float)[self._layers.layer_of_orbital]
        pair = 0.5 * (orbital[:, None] + orbital[None, :])
        self._block.H[...] = self._reference + pair[:, :, None] * self._overlap

    def restore(self) -> None:
        """Give back the zero-bias Hamiltonian."""
        self._block.H[...] = self._reference


def lowest_conductor_energy(blc_blocks: Mapping[str, Any]) -> float:
    """Lowest eigenvalue of the conductor Hamiltonian over the transverse k-points (eV).

    Parameters
    ----------
    blc_blocks : Mapping[str, Any]

    Returns
    -------
    float
    """
    hamiltonian = blc_blocks['blc_00C'].H
    lowest = np.inf
    for ik in range(hamiltonian.shape[-1]):
        block = hamiltonian[..., ik]
        lowest = min(lowest, float(np.linalg.eigvalsh(0.5 * (block + block.conj().T))[0]))
    return lowest


def finite_bias_scan(
    system: OpenConductor,
    layers: PlanarLayers,
    settings: SelfConsistencySettings,
    bias_values: Sequence[float],
    mu_left_coefficient: float = 0.5,
    mu_right_coefficient: float = -0.5,
    profile: str = 'self_consistent',
    spin_factor: float = 2.0,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> FiniteBiasResults:
    """Current, potential and charge of the junction over a list of biases.

    Parameters
    ----------
    system : OpenConductor
    layers : PlanarLayers
        Layers of the conductor and the electrode planes.
    settings : SelfConsistencySettings
    bias_values : sequence of float
        V (volt), in the order they are solved (each starts from the previous one).
    mu_left_coefficient, mu_right_coefficient : float, optional
        :math:`\\mu_{L,R} = E_F + c_{L,R}\\,eV` (the convention of ``Transport.compute_current``).
    profile : {'self_consistent', 'linear'}, optional
        ``'linear'`` keeps the ramp between the electrodes, without the response of the charge (one evaluation
        per bias); ``'self_consistent'`` adds the Hartree potential of :math:`\\delta n` and iterates.
    spin_factor : float, optional
        2 for spin-degenerate orbitals, 1 for spin-orbitals.
    comm : MPI.Comm, optional

    Returns
    -------
    FiniteBiasResults

    Raises
    ------
    ValueError
        For an unknown profile.

    Notes
    -----
    The conductor Hamiltonian is changed during the scan and restored at the end.

    In a metal the fixed-point map :math:`U \\mapsto U_{\\rm ramp} + P\\,\\delta q(U)` has the Jacobian
    :math:`J \\approx -P\\chi_0`, with :math:`P` the Poisson kernel (:func:`poisson_kernel`, tens of eV per electron
    and layer) and :math:`\\chi_0` the local density of states of each layer at :math:`E_F`: plain mixing then
    oscillates and diverges (charge sloshing). The residual is preconditioned with :math:`(1 + P\\chi_0)^{-1}`, the
    exact inverse for a linear Thomas-Fermi response; :math:`\\chi_0` is
    :math:`-\\mathrm{Im}\\,G_{ii}(E_F + i\\eta)/\\pi` summed over each layer, :math:`\\eta = \\max(\\pi k_BT, \\delta)`.
    """
    if profile not in PROFILES:
        raise ValueError(f'profile must be one of {PROFILES}, got {profile!r}')
    temperature_ev = settings.temperature_ev
    fermi = settings.fermi_energy
    potential = ConductorPotential(system.blc_blocks, layers)
    no_window = EnergyQuadrature(
        points=np.zeros(0, dtype=complex), weights=np.zeros(0, dtype=complex)
    )
    results = FiniteBiasResults(profile=profile, layer_coordinates=layers.coordinates.copy())
    lowest = lowest_conductor_energy(system.blc_blocks)
    say = settings.verbose and comm.Get_rank() == 0

    def contour_at(chemical_potential: float, shift_floor: float) -> EnergyQuadrature:
        return equilibrium_contour(
            lowest + min(0.0, shift_floor) - settings.energy_margin,
            chemical_potential,
            temperature_ev,
            arc_points=settings.arc_points,
            line_points=settings.line_points,
            fermi_poles=settings.fermi_poles,
        )

    try:
        zero_bias_contour = contour_at(fermi, 0.0)
        reference = open_system_density(
            system,
            zero_bias_contour,
            zero_bias_contour,
            no_window,
            fermi,
            fermi,
            temperature_ev,
            0.0,
            0.0,
            spin_factor,
            comm,
        ).occupations
        preconditioner = None
        if settings.thomas_fermi_preconditioner:
            broadening = max(np.pi * temperature_ev, system.broadening)
            response = layer_sums(
                local_density_of_states(
                    system, fermi + 1j * broadening, 0.0, 0.0, spin_factor, comm
                ),
                layers,
            )  # states per eV on each layer
            preconditioner = np.linalg.inv(
                np.eye(layers.n_layers) + poisson_kernel(layers) * response[None, :]
            )
        previous: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None
        for bias in bias_values:
            shift_left, shift_right = mu_left_coefficient * bias, mu_right_coefficient * bias
            mu_left, mu_right = fermi + shift_left, fermi + shift_right
            window = bias_window(mu_left, mu_right, temperature_ev, settings.window_spacing)
            ramp = planar_potential(np.zeros(layers.n_layers), layers, shift_left, shift_right)

            def evaluate(
                layer_potential: NDArray[np.float64],
            ) -> tuple[NonEquilibriumDensity, NDArray[np.float64]]:
                potential.apply(layer_potential)
                floor = min(shift_left, shift_right, float(np.min(layer_potential)))
                contour_right = contour_at(mu_right, floor)
                contour_left = contour_right if mu_left == mu_right else contour_at(mu_left, floor)
                density = open_system_density(
                    system, contour_left, contour_right, window, mu_left, mu_right, temperature_ev,
                    shift_left, shift_right, spin_factor, comm,
                )  # fmt: skip
                return density, layer_sums(density.occupations - reference, layers)

            iterations, converged, residual = 0, True, 0.0
            if profile == 'linear':
                layer_potential = ramp
                density, charge = evaluate(layer_potential)
            else:
                if previous is not None and previous[0] != 0.0:
                    # the response to the charge scales with the bias: start from the previous one, rescaled
                    layer_potential = ramp + (previous[1] - previous[2]) * bias / previous[0]
                else:
                    layer_potential = ramp
                inputs: list[NDArray[np.float64]] = []
                residuals: list[NDArray[np.float64]] = []
                converged = False
                for iterations in range(1, settings.max_iterations + 1):
                    density, charge = evaluate(layer_potential)
                    output = planar_potential(charge, layers, shift_left, shift_right)
                    step = output - layer_potential
                    residual = float(np.max(np.abs(step)))
                    if say:
                        print(f'  V = {bias:+.4f} V  iteration {iterations:3d}  max|dU| = {residual:.2e} eV  '
                              f'sum dq = {charge.sum():+.2e}', flush=True)  # fmt: skip
                    if residual < settings.tolerance:
                        converged = True
                        break
                    inputs = (inputs + [layer_potential])[-settings.history :]
                    residuals = (residuals + [step])[-settings.history :]
                    layer_potential = anderson_update(
                        inputs, residuals, settings.mixing, preconditioner
                    )
                previous = (bias, layer_potential.copy(), ramp)
            results.points.append(
                BiasPointResult(
                    bias=float(bias),
                    chemical_potential_left=mu_left,
                    chemical_potential_right=mu_right,
                    layer_potential=np.asarray(layer_potential, dtype=float).copy(),
                    layer_charge=np.asarray(charge, dtype=float),
                    window_energies=window.points.real.copy(),
                    transmission=density.transmission.copy(),
                    current=landauer_current(density, mu_left, mu_right, temperature_ev),
                    iterations=iterations,
                    converged=converged,
                    residual=residual,
                )
            )
    finally:
        potential.restore()
    return results


def check_orthonormal_conductor(blc_blocks: Mapping[str, Any], tolerance: float = 1.0e-8) -> None:
    """Raise unless the conductor orbitals are orthonormal (the occupations are then the diagonal of rho).

    Parameters
    ----------
    blc_blocks : Mapping[str, Any]
    tolerance : float, optional

    Raises
    ------
    NotImplementedError
        For non-orthonormal orbitals (Mulliken occupations are not implemented).
    """
    overlap = blc_blocks['blc_00C'].S
    identity = np.eye(overlap.shape[0])[:, :, None]
    if np.max(np.abs(overlap - identity)) > tolerance:
        raise NotImplementedError(
            'finite-bias self-consistency needs orthonormal orbitals (do_overlap_transformation=False on an '
            'orthonormal PAO Hamiltonian)'
        )
