"""Orbital occupations and transmission of the open conductor at finite bias (non-equilibrium Green's functions).

The density matrix of the conductor between two leads at chemical potentials :math:`\\mu_L`, :math:`\\mu_R` is split
into an equilibrium part, which fills every state up to :math:`\\mu_R` and is integrated on a complex contour, and a
non-equilibrium part, the states injected by the left lead in the bias window (Brandbyge et al., PRB 65, 165401
(2002)):

.. math::

    \\rho = -\\frac{1}{\\pi}\\,\\mathrm{Im}\\int dE\\, G(E)\\, f(E-\\mu_R)
        + \\frac{1}{2\\pi}\\int dE\\, G\\Gamma_L G^\\dagger\\,\\bigl[f(E-\\mu_L) - f(E-\\mu_R)\\bigr].

Only the diagonal is accumulated (orthonormal orbitals), summed over the transverse k-points with their weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from PAOFLOW.transport.conductor_kpoint import compute_kpoint_green, compute_kpoint_self_energies
from PAOFLOW.transport.negf.contour import EnergyQuadrature, fermi_function


@dataclass(frozen=True)
class OpenConductor:
    """The parts of a built Transport calculation that the finite-bias density needs.

    Attributes
    ----------
    blc_blocks : Mapping[str, Any]
        Block operators of the conductor and the leads (``Transport.blc_blocks``).
    k_weights : ndarray, shape ``(nk,)``
        Weights of the transverse k-points (they sum to 1).
    broadening : float
        :math:`\\delta` of the retarded functions on the real axis (eV).
    leads_are_identical : bool
        The two leads have the same Hamiltonian (their transfer matrices may be shared at equal shifts).
    max_iterations, max_failures : int
        Limits of the lead transfer-matrix iteration.
    transfer_threshold : float
        Convergence threshold of the transfer-matrix iteration.
    """

    blc_blocks: Mapping[str, Any]
    k_weights: NDArray[np.float64]
    broadening: float
    leads_are_identical: bool
    max_iterations: int
    transfer_threshold: float
    max_failures: int

    @property
    def n_kpoints(self) -> int:
        return len(self.k_weights)

    @property
    def dimension(self) -> int:
        return int(self.blc_blocks['blc_00C'].H.shape[0])


def open_conductor_from(conductor_data: Any, blc_blocks: Mapping[str, Any]) -> OpenConductor:
    """Collect the finite-bias inputs from a built Transport calculation.

    Parameters
    ----------
    conductor_data : ConductorData
        ``Transport.conductor_data`` after ``build_hamiltonian_blocks``.
    blc_blocks : Mapping[str, Any]
        ``Transport.blc_blocks``.

    Returns
    -------
    OpenConductor
    """
    return OpenConductor(
        blc_blocks=blc_blocks,
        k_weights=np.asarray(conductor_data._runtime.wk_par, dtype=float),
        broadening=float(conductor_data.energy.delta),
        leads_are_identical=bool(conductor_data.advanced.leads_are_identical),
        max_iterations=int(conductor_data.iteration.niterx),
        transfer_threshold=float(conductor_data.iteration.transfer_thr),
        max_failures=int(conductor_data.iteration.nfailx),
    )


def conductor_green_function(
    system: OpenConductor,
    ik: int,
    energy: complex,
    shift_left: float,
    shift_right: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Retarded Green's function and lead self-energies of the conductor at a real or complex energy.

    Parameters
    ----------
    system : OpenConductor
    ik : int
        Transverse k-point.
    energy : complex
        :math:`z` (eV). A real energy gets the broadening :math:`\\delta` of ``system``; a complex one in the
        upper half plane is used as it is (``delta = Im z``).
    shift_left, shift_right : float
        Rigid shifts of the left and right lead bands (eV): their electrochemical potentials under bias.

    Returns
    -------
    G : ndarray, shape ``(dimC, dimC)``, complex
    sigma_left, sigma_right : ndarray, shape ``(dimC, dimC)``, complex

    Notes
    -----
    The kernels build :math:`(E + i\\delta)S - H` from :math:`E S - H` and :math:`\\delta`, both in the lead
    decimation and in :math:`G_C`, so :math:`z = E + i\\eta` is :math:`E` with :math:`\\delta = \\eta`.
    """
    energy = complex(energy)
    delta = energy.imag if energy.imag > 0.0 else system.broadening
    sigma_left, sigma_right, _ = compute_kpoint_self_energies(
        blc_blocks=system.blc_blocks,
        ik=ik,
        ie_g=0,
        egrid=np.array([energy.real]),
        delta=delta,
        shift_L=shift_left,
        shift_C=0.0,
        shift_R=shift_right,
        shift_corr=0.0,
        leads_are_identical=system.leads_are_identical,
        niterx=system.max_iterations,
        transfer_thr=system.transfer_threshold,
        nfailx=system.max_failures,
        surface=False,
    )
    green = compute_kpoint_green(
        blc_00C=system.blc_blocks['blc_00C'].at_k(ik),
        sigma_L=sigma_left,
        sigma_R=sigma_right,
        delta=delta,
        surface=False,
    )
    return green, sigma_left, sigma_right


@dataclass
class NonEquilibriumDensity:
    """Occupations of the conductor orbitals and the transmission in the bias window.

    Attributes
    ----------
    occupations : ndarray, shape ``(dimC,)``
        Electrons on every orbital (spin included), per transverse cell.
    window : EnergyQuadrature
        Real energies of the bias window.
    transmission : ndarray, shape ``(n_window,)``
        :math:`T(E)` per spin and transverse cell on the window energies.
    """

    occupations: NDArray[np.float64]
    window: EnergyQuadrature
    transmission: NDArray[np.float64]


def open_system_density(
    system: OpenConductor,
    contour_left: EnergyQuadrature,
    contour_right: EnergyQuadrature,
    window: EnergyQuadrature,
    chemical_potential_left: float,
    chemical_potential_right: float,
    temperature_ev: float,
    shift_left: float,
    shift_right: float,
    spin_factor: float,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> NonEquilibriumDensity:
    """Occupations of the conductor orbitals at finite bias, and the transmission in the window.

    Parameters
    ----------
    system : OpenConductor
        The conductor, with its current Hamiltonian (any bias potential already applied to its block).
    contour_left, contour_right : EnergyQuadrature
        :func:`~PAOFLOW.transport.negf.contour.equilibrium_contour` at :math:`\\mu_L` and at :math:`\\mu_R`; pass
        the same object at zero bias (it is then evaluated once).
    window : EnergyQuadrature
        :func:`~PAOFLOW.transport.negf.contour.bias_window`.
    chemical_potential_left, chemical_potential_right : float
        :math:`\\mu_L`, :math:`\\mu_R` (eV).
    temperature_ev : float
        :math:`k_BT` (eV).
    shift_left, shift_right : float
        Shifts of the lead bands (eV).
    spin_factor : float
        2 for spin-degenerate orbitals, 1 for spin-orbitals.
    comm : MPI.Comm, optional
        The energy points are shared out over its ranks.

    Returns
    -------
    NonEquilibriumDensity

    Notes
    -----
    Two exact decompositions of the same density, each with an equilibrium part on a contour and the states
    injected by the other lead in the window,

    .. math::

        \\rho^{L} = \\rho_{\\rm eq}(\\mu_R) + \\Delta_L,\\quad
        \\Delta_L = \\frac{1}{2\\pi}\\int dE\\,G\\Gamma_LG^\\dagger\\,(f_L - f_R),\\qquad
        \\rho^{R} = \\rho_{\\rm eq}(\\mu_L) + \\Delta_R,\\quad
        \\Delta_R = \\frac{1}{2\\pi}\\int dE\\,G\\Gamma_RG^\\dagger\\,(f_R - f_L),

    are combined orbital by orbital with :math:`w_L = \\Delta_R^2/(\\Delta_L^2 + \\Delta_R^2)`,
    :math:`n_i = s\\sum_{\\mathbf k}w_{\\mathbf k}[w_L\\rho^L + (1-w_L)\\rho^R]_{ii}` (Brandbyge et al.). The
    real-axis error of each decomposition grows with its :math:`\\Delta`, so the more accurate one gets the larger
    weight. States bound inside the bias window, coupled to neither lead, have :math:`\\Delta_L = \\Delta_R = 0`
    and get the mean occupation :math:`(f_L + f_R)/2`: steady-state NEGF does not fix it, and this choice keeps
    the mirror symmetry of a symmetric junction. The transmission is
    :math:`T(E) = \\sum_{\\mathbf k} w_{\\mathbf k}\\,\\mathrm{Tr}[\\Gamma_R G\\Gamma_L G^\\dagger]`, with
    :math:`\\Gamma = i(\\Sigma - \\Sigma^\\dagger)`.
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    dimension = system.dimension
    zero_bias = contour_left is contour_right
    contours = (
        {'right': contour_right} if zero_bias else {'right': contour_right, 'left': contour_left}
    )
    equilibrium = {name: np.zeros(dimension, dtype=complex) for name in contours}
    injected_left = np.zeros(dimension)
    injected_right = np.zeros(dimension)
    transmission = np.zeros(len(window))
    window_energies = window.points.real
    occupation_difference = (
        fermi_function(window_energies, chemical_potential_left, temperature_ev).real
        - fermi_function(window_energies, chemical_potential_right, temperature_ev).real
    )
    tasks = [(name, j) for name, contour in contours.items() for j in range(len(contour))]
    tasks += [('window', j) for j in range(len(window))]
    for kind, j in tasks[rank::size]:
        for ik in range(system.n_kpoints):
            if kind != 'window':
                contour = contours[kind]
                green, _, _ = conductor_green_function(
                    system, ik, contour.points[j], shift_left, shift_right
                )
                equilibrium[kind] += system.k_weights[ik] * contour.weights[j] * np.diagonal(green)
                continue
            green, sigma_left, sigma_right = conductor_green_function(
                system, ik, window_energies[j], shift_left, shift_right
            )
            gamma_left = 1j * (sigma_left - sigma_left.conj().T)
            gamma_right = 1j * (sigma_right - sigma_right.conj().T)
            spectral_left = green @ gamma_left @ green.conj().T
            spectral_right = green @ gamma_right @ green.conj().T
            weight = (
                system.k_weights[ik]
                * window.weights[j].real
                * occupation_difference[j]
                / (2.0 * np.pi)
            )
            injected_left += weight * np.real(np.diagonal(spectral_left))
            injected_right -= weight * np.real(np.diagonal(spectral_right))
            transmission[j] += system.k_weights[ik] * np.real(np.trace(gamma_right @ spectral_left))
    equilibrium = {name: comm.allreduce(value, op=MPI.SUM) for name, value in equilibrium.items()}
    injected_left = comm.allreduce(injected_left, op=MPI.SUM)
    injected_right = comm.allreduce(injected_right, op=MPI.SUM)
    transmission = comm.allreduce(transmission, op=MPI.SUM)
    from_right = -np.imag(equilibrium['right']) / np.pi + injected_left
    if zero_bias:
        return NonEquilibriumDensity(
            occupations=spin_factor * from_right, window=window, transmission=transmission
        )
    from_left = -np.imag(equilibrium['left']) / np.pi + injected_right
    squares = injected_left**2 + injected_right**2
    weight_right_reference = np.where(
        squares > 1.0e-30, injected_right**2 / np.maximum(squares, 1.0e-300), 0.5
    )
    occupations = weight_right_reference * from_right + (1.0 - weight_right_reference) * from_left
    return NonEquilibriumDensity(
        occupations=spin_factor * occupations, window=window, transmission=transmission
    )


def local_density_of_states(
    system: OpenConductor,
    energy: complex,
    shift_left: float,
    shift_right: float,
    spin_factor: float,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> NDArray[np.float64]:
    """Broadened local density of states of every conductor orbital, :math:`-\\mathrm{Im}\\,G_{ii}(z)/\\pi`.

    Parameters
    ----------
    system : OpenConductor
    energy : complex
        :math:`z = E + i\\eta` (eV); :math:`\\eta` sets the broadening.
    shift_left, shift_right : float
        Shifts of the lead bands (eV).
    spin_factor : float
    comm : MPI.Comm, optional
        The transverse k-points are shared out over its ranks.

    Returns
    -------
    ndarray, shape ``(dimC,)``
        States per eV and transverse cell (spin included).
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    density = np.zeros(system.dimension)
    for ik in range(rank, system.n_kpoints, size):
        green, _, _ = conductor_green_function(system, ik, energy, shift_left, shift_right)
        density -= system.k_weights[ik] * np.imag(np.diagonal(green)) / np.pi
    return spin_factor * comm.allreduce(density, op=MPI.SUM)


def landauer_current(density: NonEquilibriumDensity, chemical_potential_left: float,
                     chemical_potential_right: float, temperature_ev: float) -> float:  # fmt: skip
    """Landauer current from the transmission in the bias window.

    Parameters
    ----------
    density : NonEquilibriumDensity
    chemical_potential_left, chemical_potential_right : float
        :math:`\\mu_L`, :math:`\\mu_R` (eV).
    temperature_ev : float
        :math:`k_BT` (eV).

    Returns
    -------
    float
        :math:`\\int dE\\, T(E)[f(E-\\mu_L) - f(E-\\mu_R)]` in eV, per transverse cell: the current in units of
        :math:`2e/h` (as ``Transport.compute_current``); multiply by :math:`G_0 = 2e^2/h` per volt.
    """
    energies = density.window.points.real
    if energies.size == 0:
        return 0.0
    difference = (
        fermi_function(energies, chemical_potential_left, temperature_ev).real
        - fermi_function(energies, chemical_potential_right, temperature_ev).real
    )
    return float(np.sum(density.window.weights.real * density.transmission * difference))
