"""Electron-phonon properties of a model crystal with PAOFLOW's engines.

The model supplies the states, the phonons and the vertex
(:class:`~PAOFLOW.models.electron_phonon.model.ElectronPhononModel`); these functions build the
coupling :math:`g_{mn\\nu}(\\mathbf k,\\mathbf q)` on uniform grids and pass it to the kernels of
the ab initio route:

* :func:`mode_resolved_coupling`: :math:`\\lambda_{\\mathbf q\\nu}` (and the transport
  :math:`\\lambda^{tr}_{\\mathbf q\\nu}`) by the Fermi-surface double delta, with PAOFLOW's
  adaptive widths (:func:`PAOFLOW.elphon.adaptive_smearing.state_widths`);
* :func:`eliashberg_function`: :func:`PAOFLOW.elphon.eph_kq.eliashberg_from_modes`
  (:math:`\\alpha^2F`, :math:`\\lambda`, :math:`\\omega_{\\log}`, McMillan and Allen-Dynes
  :math:`T_c`);
* :func:`carrier_scattering_rates` and :func:`carrier_mobility`:
  :func:`PAOFLOW.elphon.carrier_scattering.scattering_rate_kernel` (SERTA and MRTA);
* :func:`indirect_absorption`:
  :func:`PAOFLOW.elphon.phonon_assisted_absorption.indirect_absorption_kernel`.

The loops run serially on one core; the grids of the lecture notes take seconds to a minute.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import BOHR_M, E_CHARGE, EPS0, HBAR_EV_S, KB_EV, RY_EV, ElectronPhononModel


# ------------------------------------------------------------------------------ grids and states
def grid_counts(model: ElectronPhononModel, n: int) -> NDArray[np.int_]:
    """Points of an ``n``-grid along each lattice direction (1 along the non-periodic ones)."""
    return np.array([n if periodic else 1 for periodic in model.periodic_axes])


def crystal_grid(model: ElectronPhononModel, n: int) -> NDArray[np.float64]:
    """Gamma-centred uniform grid with ``n`` points along each periodic direction.

    Parameters
    ----------
    model : ElectronPhononModel
    n : int
        Points per periodic direction.

    Returns
    -------
    ndarray, shape ``(n**dimension, 3)``
        Crystal coordinates in [0, 1), the last axis fastest.
    """
    axes = [np.arange(count) / count for count in grid_counts(model, n)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)


def grid_index(model: ElectronPhononModel, n: int, k_crystal: ArrayLike) -> NDArray[np.int_]:
    """Index in :func:`crystal_grid` of points of the grid given in any periodic image."""
    counts = grid_counts(model, n)
    integer = np.rint(np.asarray(k_crystal) * counts).astype(int) % counts
    return (integer[..., 0] * counts[1] + integer[..., 1]) * counts[2] + integer[..., 2]


@dataclass
class ElectronStates:
    """Electrons of a model on a uniform grid.

    Attributes
    ----------
    kpoints : ndarray, shape ``(nk, 3)``
        :func:`crystal_grid` (crystal coordinates).
    grid_size : int
        Points per periodic direction.
    energies : ndarray, shape ``(nk, nbnd)``
        Band energies (eV).
    vectors : ndarray, shape ``(nk, norb, nbnd)``
        Eigenvectors in the atomic gauge.
    velocities : ndarray, shape ``(nk, nbnd, 3)``
        Band velocities :math:`\\hbar v` (eV bohr).
    velocity_matrices : ndarray, shape ``(nk, 3, nbnd, nbnd)`` or None
        Velocity matrices in the band basis (for the optical matrix elements).
    """

    kpoints: NDArray[np.float64]
    grid_size: int
    energies: NDArray[np.float64]
    vectors: NDArray[np.complex128]
    velocities: NDArray[np.float64]
    velocity_matrices: NDArray[np.complex128] | None = None


def electron_states(
    model: ElectronPhononModel, n: int, velocity_matrices: bool = False, block: int = 4096
) -> ElectronStates:
    """Diagonalise the model on an ``n``-grid.

    Parameters
    ----------
    model : ElectronPhononModel
    n : int
        Points per periodic direction.
    velocity_matrices : bool, optional
        Keep the full velocity matrices (needed by :func:`indirect_absorption`).
    block : int, optional
        k-points diagonalised at once (memory).

    Returns
    -------
    ElectronStates
    """
    kpoints = crystal_grid(model, n)
    energies, vectors, velocities, matrices = [], [], [], []
    for start in range(0, len(kpoints), block):
        hamiltonian, velocity = model.hamiltonian(kpoints[start : start + block], derivative=True)
        values, eigenvectors = np.linalg.eigh(hamiltonian)
        band_velocity = np.einsum('kin,kaij,kjm->kanm', eigenvectors.conj(), velocity, eigenvectors)
        energies.append(values)
        vectors.append(eigenvectors)
        velocities.append(np.real(np.einsum('kann->kna', band_velocity)))
        if velocity_matrices:
            matrices.append(band_velocity)
    return ElectronStates(
        kpoints,
        n,
        np.concatenate(energies),
        np.concatenate(vectors),
        np.concatenate(velocities),
        np.concatenate(matrices) if velocity_matrices else None,
    )


def shifted_eigenvectors(
    model: ElectronPhononModel, states: ElectronStates, kq_crystal: NDArray[np.float64]
) -> tuple[NDArray[np.int_], NDArray[np.complex128]]:
    """Eigenvectors at ``k + q`` taken from the grid, with the gauge of the unfolded point.

    Parameters
    ----------
    model : ElectronPhononModel
    states : ElectronStates
    kq_crystal : ndarray, shape ``(nk, 3)``
        Points of the grid, possibly outside [0, 1).

    Returns
    -------
    indices : ndarray of int, shape ``(nk,)``
        Their index on the grid.
    vectors : ndarray, shape ``(nk, norb, nbnd)``
        Atomic-gauge eigenvectors at the unfolded points: the stored ones times
        :math:`e^{-i\\mathbf G\\cdot\\boldsymbol\\tau}` when ``k + q`` wraps by :math:`\\mathbf G`.
    """
    indices = grid_index(model, states.grid_size, kq_crystal)
    wrapped = kq_crystal - np.floor(kq_crystal + 1e-9)
    reciprocal_shift = np.rint(kq_crystal - wrapped)
    centres = model.positions[model.orbital_atoms]  # (norb, 3)
    phase = np.exp(-2j * np.pi * reciprocal_shift @ centres.T)  # (nk, norb)
    return indices, phase[:, :, None] * states.vectors[indices]


def fermi_level(
    energies: NDArray[np.float64], nelec: float, sigma: float, spin_degeneracy: float = 2.0
) -> float:
    """Fermi level of ``nelec`` electrons on a uniform grid, Gaussian occupations (bisection).

    Parameters
    ----------
    energies : ndarray, shape ``(nk, nbnd)``
        Band energies (eV).
    nelec : float
        Electrons per cell.
    sigma : float
        Gaussian width (eV).
    spin_degeneracy : float, optional
        2 for spin-degenerate bands.

    Returns
    -------
    float
        The Fermi level (eV).
    """
    from scipy.special import erfc

    low, high = energies.min() - 1, energies.max() + 1
    for _ in range(100):
        middle = 0.5 * (low + high)
        count = spin_degeneracy * 0.5 * erfc((energies - middle) / sigma).sum() / len(energies)
        low, high = (middle, high) if count < nelec else (low, middle)
    return middle


def band_edges(states: ElectronStates, nelec: float) -> tuple[float, float]:
    """Valence-band maximum and conduction-band minimum of a filling (spin-degenerate bands).

    Parameters
    ----------
    states : ElectronStates
    nelec : float
        Electrons per cell (an even number for an insulator).

    Returns
    -------
    tuple of float
        ``(vbm, cbm)`` in eV; ``cbm <= vbm`` means the filling is metallic.

    Raises
    ------
    ValueError
        If the filling leaves no empty band or no full one.
    """
    filled = int(round(nelec / 2.0))
    if not 0 < filled < states.energies.shape[1]:
        raise ValueError('the filling leaves no empty or no full band')
    return float(states.energies[:, filled - 1].max()), float(states.energies[:, filled].min())


def fermi_occupations(
    energies: ArrayLike, potential: float, temperature: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fermi-Dirac occupation :math:`f` and vacancy :math:`1 - f`, each without cancellation.

    Parameters
    ----------
    energies : array_like
        Energies (eV).
    potential : float
        Chemical potential (eV).
    temperature : float
        Temperature (K).

    Returns
    -------
    tuple of ndarray
        ``(f, 1 - f)``; the vacancy is :math:`1/(e^{-x} + 1)`, so the holes of a band far below
        the chemical potential keep their (tiny) weight instead of rounding to zero.
    """
    reduced = np.clip((np.asarray(energies) - potential) / (KB_EV * temperature), -200, 200)
    return 1.0 / (np.exp(reduced) + 1.0), 1.0 / (np.exp(-reduced) + 1.0)


def gaussian_delta(energy: ArrayLike, width: ArrayLike) -> NDArray[np.float64]:
    """Normalised Gaussian :math:`e^{-x^2/\\sigma^2}/(\\sigma\\sqrt\\pi)` (PAOFLOW's convention)."""
    return np.exp(-((np.asarray(energy) / width) ** 2)) / (width * np.sqrt(np.pi))


# ------------------------------------------------------------------------------------ the coupling
def coupling_matrix_elements(
    model: ElectronPhononModel,
    states: ElectronStates,
    k_indices: NDArray[np.int_],
    q_crystal: ArrayLike,
    phonon: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.int_], NDArray[np.float64]]:
    """Electron-phonon matrix elements :math:`g_{mn\\nu}(\\mathbf k,\\mathbf q)` of grid states.

    Parameters
    ----------
    model : ElectronPhononModel
    states : ElectronStates
    k_indices : ndarray of int, shape ``(nk,)``
        Initial states (indices on the grid).
    q_crystal : array_like, shape ``(3,)``
        Phonon wave vector, a point of the grid (crystal coordinates).
    phonon : tuple, optional
        ``(energies (nmode,), eigenvectors (3 natom, nmode))`` at ``q``; computed if omitted.

    Returns
    -------
    coupling : ndarray, shape ``(nk, nmode, nbnd, nbnd)``
        :math:`g_{mn\\nu}(\\mathbf k,\\mathbf q)` (eV), final band ``m`` at ``k + q``, initial
        band ``n`` at ``k``.
    kq_indices : ndarray of int, shape ``(nk,)``
        Index of ``k + q`` on the grid.
    phonon_energies : ndarray, shape ``(nmode,)``
        :math:`\\hbar\\omega_{\\mathbf q\\nu}` (eV).

    Notes
    -----
    .. math::

        g_{mn\\nu}(\\mathbf k,\\mathbf q) = \\sum_{\\kappa\\beta}
        \\sqrt{\\frac{\\hbar}{2M_\\kappa\\omega_{\\mathbf q\\nu}}}\\, e_{\\kappa\\beta,\\nu}(\\mathbf q)
        \\langle\\psi_{m\\mathbf k+\\mathbf q}|\\partial_{\\mathbf q\\kappa\\beta}V
        |\\psi_{n\\mathbf k}\\rangle,

    with the zero-point amplitude of :func:`PAOFLOW.elphon.eph_kq.zero_point_amplitude`.
    """
    from ...elphon.eph_kq import THZ_TO_EV, zero_point_amplitude

    q_crystal = np.asarray(q_crystal, dtype=float)
    k_crystal = states.kpoints[k_indices]
    kq_indices, vectors_kq = shifted_eigenvectors(model, states, k_crystal + q_crystal)
    vertex = model.vertex(k_crystal, q_crystal)  # (nk, 3 natom, norb, norb)
    vectors_k = states.vectors[k_indices]
    band_vertex = np.einsum('kim,kcij,kjn->kcmn', vectors_kq.conj(), vertex, vectors_k)
    if phonon is None:
        energies, eigenvectors = model.phonons(q_crystal[None])
        energies, eigenvectors = energies[0], eigenvectors[0]
    else:
        energies, eigenvectors = phonon
    frequencies_thz = np.maximum(energies, 0.0) / THZ_TO_EV
    amplitude = zero_point_amplitude(
        np.repeat(model.masses, 3)[:, None], frequencies_thz[None, :]
    )  # (3 natom, nmode), bohr
    pattern = eigenvectors * amplitude
    coupling = np.einsum('cv,kcmn->kvmn', pattern, band_vertex)
    return coupling, kq_indices, energies


# ----------------------------------------------------------------------- metals: the double delta
def grid_spacing(model: ElectronPhononModel, n: int) -> float:
    """Spacing (1/bohr) of an ``n``-grid: the root of the zone measure over the periodic axes."""
    periodic = model.dimension
    reciprocal = model.reciprocal_vectors[:periodic, :periodic]
    return float(abs(np.linalg.det(reciprocal)) ** (1.0 / periodic) / n)


def delta_widths(
    model: ElectronPhononModel, states: ElectronStates, sigma: float, adaptive: float | None
) -> NDArray[np.float64]:
    """Width of the energy delta of every state.

    Parameters
    ----------
    model : ElectronPhononModel
    states : ElectronStates
    sigma : float
        Fixed width, or the floor of the adaptive widths (eV).
    adaptive : float or None
        Prefactor ``a`` of the adaptive widths :math:`a|\\mathbf v_{n\\mathbf k}|\\Delta k`
        (0.7 for Gaussians); ``None``: the fixed width.

    Returns
    -------
    ndarray, shape ``(nk, nbnd)``
        Widths (eV), :func:`PAOFLOW.elphon.adaptive_smearing.state_widths`.
    """
    if adaptive is None:
        return np.full(states.energies.shape, sigma)
    from ...elphon.adaptive_smearing import state_widths

    return state_widths(states.velocities, grid_spacing(model, states.grid_size), adaptive, sigma)


def mode_resolved_coupling(
    model: ElectronPhononModel,
    nk: int,
    q_points: ArrayLike,
    sigma: float,
    fermi_energy: float | None = None,
    window: float = 4.0,
    transport: bool = False,
    adaptive: float | None = None,
) -> dict:
    """Mode-resolved coupling :math:`\\lambda_{\\mathbf q\\nu}` by the Fermi-surface double delta.

    Parameters
    ----------
    model : ElectronPhononModel
        A metal.
    nk : int
        k-points per periodic direction.
    q_points : array_like, shape ``(nq, 3)``
        Phonon wave vectors, points of the k-grid (crystal coordinates).
    sigma : float
        Gaussian width (eV), the floor of the adaptive widths.
    fermi_energy : float, optional
        Fermi level (eV); by default that of ``model.nelec`` with the width ``sigma``.
    window : float, optional
        States farther than ``window`` widths from the Fermi level are left out.
    transport : bool, optional
        Also :math:`\\lambda^{tr}_{\\mathbf q\\nu}`, with the velocity factor of the
        resistivity.
    adaptive : float, optional
        Prefactor of the adaptive widths (0.7); ``None``: the fixed width ``sigma``.

    Returns
    -------
    dict
        ``lambda_qv`` and ``lambda_tr_qv`` (nq, nmode), ``omega_ev`` (nq, nmode) phonon
        energies, ``N_F`` (per spin and cell, 1/eV), ``E_F`` (eV), ``states``
        (:class:`ElectronStates`) and ``v2`` (the Fermi-surface average of :math:`|v|^2`).

    Notes
    -----
    .. math::

        \\lambda_{\\mathbf q\\nu} = \\frac{2}{N_F\\,\\omega_{\\mathbf q\\nu}}\\frac{1}{N_k}
        \\sum_{\\mathbf k mn} |g_{mn\\nu}(\\mathbf k,\\mathbf q)|^2
        \\delta(\\varepsilon_{n\\mathbf k})\\,\\delta(\\varepsilon_{m\\mathbf k+\\mathbf q}),

    energies from the Fermi level; the transport version weighs each pair with
    :math:`\\mathbf v_{n\\mathbf k}\\cdot(\\mathbf v_{n\\mathbf k} - \\mathbf v_{m\\mathbf k+\\mathbf q})
    /\\langle v^2\\rangle`.  Equal to PAOFLOW's production kernel
    :func:`PAOFLOW.elphon.elph_bloch.lambda_q_dense_ws_fast` on the same states.
    """
    states = electron_states(model, nk)
    widths = delta_widths(model, states, sigma, adaptive)
    fermi = (
        fermi_level(states.energies, model.nelec, sigma) if fermi_energy is None else fermi_energy
    )
    relative = states.energies - fermi
    deltas = gaussian_delta(relative, widths) * (np.abs(relative) < window * widths)
    dos_fermi = deltas.sum() / len(states.kpoints)
    velocities = states.velocities
    mean_square_velocity = np.sum(deltas[..., None] * velocities**2) / deltas.sum()
    shell = np.nonzero(deltas.max(1) > 0)[0]
    q_points = np.atleast_2d(np.asarray(q_points, dtype=float))
    coupling_q = np.zeros((len(q_points), model.number_of_modes))
    coupling_tr_q = np.zeros_like(coupling_q)
    phonon_energies = np.zeros_like(coupling_q)
    for iq, q in enumerate(q_points):
        coupling, kq_indices, energies = coupling_matrix_elements(model, states, shell, q)
        phonon_energies[iq] = energies
        squared = np.abs(coupling) ** 2  # (nk, nmode, m, n)
        weight = deltas[kq_indices][:, None, :, None] * deltas[shell][:, None, None, :]
        stable = energies > 1e-4
        coupling_q[iq, stable] = (
            2.0
            / (dos_fermi * energies[stable])
            * (squared * weight).sum((0, 2, 3))[stable]
            / len(states.kpoints)
        )
        if transport:
            velocity_k, velocity_kq = velocities[shell], velocities[kq_indices]
            dot = np.einsum('knc,kmc->kmn', velocity_k, velocity_kq)
            factor = ((np.sum(velocity_k**2, -1)[:, None, :] - dot) / mean_square_velocity)[:, None]
            coupling_tr_q[iq, stable] = (
                2.0
                / (dos_fermi * energies[stable])
                * (squared * weight * factor).sum((0, 2, 3))[stable]
                / len(states.kpoints)
            )
    return {
        'lambda_qv': coupling_q,
        'lambda_tr_qv': coupling_tr_q,
        'omega_ev': phonon_energies,
        'N_F': dos_fermi,
        'E_F': fermi,
        'states': states,
        'v2': mean_square_velocity,
    }


def eliashberg_function(
    lambda_qv: NDArray[np.float64],
    omega_ev: NDArray[np.float64],
    q_weights: NDArray[np.float64] | None = None,
    mu_star: float = 0.1,
    sigma_w_frac: float = 0.03,
    nomega: int = 600,
) -> dict:
    """Eliashberg function, :math:`\\lambda`, :math:`\\omega_{\\log}` and :math:`T_c` of the modes.

    Parameters
    ----------
    lambda_qv : ndarray, shape ``(nq, nmode)``
        :func:`mode_resolved_coupling` (``lambda_qv`` or ``lambda_tr_qv``).
    omega_ev : ndarray, shape ``(nq, nmode)``
        Phonon energies (eV).
    q_weights : ndarray, shape ``(nq,)``, optional
        Star weights of irreducible q-points.
    mu_star : float, optional
        Coulomb pseudopotential of the :math:`T_c` formulae.
    sigma_w_frac : float, optional
        Gaussian width of the modes as a fraction of the highest phonon energy.
    nomega : int, optional
        Frequency points.

    Returns
    -------
    dict
        :func:`PAOFLOW.elphon.eph_kq.eliashberg_from_modes`: ``omega`` (eV), ``a2F``,
        ``lambda``, ``omega_log`` (eV), ``Tc_mcmillan``, ``Tc_allen_dynes`` (K), ...
    """
    from ...elphon.eph_kq import THZ_TO_EV, eliashberg_from_modes

    return eliashberg_from_modes(
        lambda_qv,
        np.maximum(omega_ev, 0) / THZ_TO_EV,
        q_weights=q_weights,
        mu_star=mu_star,
        nomega=nomega,
        sigma_w_frac=sigma_w_frac,
    )


def plasma_energy(
    states: ElectronStates,
    fermi_energy: float,
    temperature: float,
    volume: float,
    spin_degeneracy: float = 2.0,
) -> float:
    """Drude plasma energy :math:`\\hbar\\omega_p` (eV) from the Fermi-surface velocities.

    Parameters
    ----------
    states : ElectronStates
    fermi_energy : float
        Fermi level (eV).
    temperature : float
        Temperature of the Fermi window (K).
    volume : float
        Cell volume (bohr^3).
    spin_degeneracy : float, optional

    Returns
    -------
    float

    Notes
    -----
    .. math::

        \\omega_p^2 = \\frac{e^2 g_s}{\\varepsilon_0 \\Omega N_k}\\sum_{n\\mathbf k}
        \\left(-\\frac{\\partial f}{\\partial\\varepsilon}\\right)\\frac{|\\mathbf v_{n\\mathbf k}|^2}{3}.
    """
    thermal = KB_EV * temperature
    reduced = np.clip((states.energies - fermi_energy) / thermal, -200, 200)
    window = np.exp(reduced) / (np.exp(reduced) + 1) ** 2 / thermal
    weight = (window[..., None] * states.velocities**2).sum() / 3.0 / len(states.kpoints)
    constant = E_CHARGE / (EPS0 * BOHR_M)  # e^2 / (eps0 bohr eV)
    return float(np.sqrt(constant * spin_degeneracy * weight / volume))


# ---------------------------------------------------------------------------- symmetry of q-grids
def signed_permutations() -> list[NDArray[np.int_]]:
    """The 48 signed permutation matrices (the operations of O_h on a cubic lattice)."""
    operations = []
    for order in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            matrix = np.zeros((3, 3), int)
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            operations.append(matrix)
    return operations


def symmetry_operations(model: ElectronPhononModel) -> list[NDArray[np.int_]]:
    """Signed permutations of the crystal axes that are symmetries of the model.

    Parameters
    ----------
    model : ElectronPhononModel

    Returns
    -------
    list of ndarray, shape ``(3, 3)``
        Integer matrices acting on crystal coordinates that keep the lattice metric, keep the
        non-periodic axes, and map every atom onto an atom of the same mass and on-site energies.
        The bonds and the springs depend only on distances and on the atoms, so they follow.
    """
    metric = model.lattice @ model.lattice.T
    signatures = [
        (round(float(model.masses[atom]), 8),)
        + tuple(
            sorted(
                round(float(e), 8) for e, a in zip(model.onsite, model.orbital_atoms) if a == atom
            )
        )
        for atom in range(model.number_of_atoms)
    ]
    kept = []
    for operation in signed_permutations():
        if not np.allclose(operation.T @ metric @ operation, metric):
            continue
        if any(
            operation[axis, axis] != 1
            for axis, periodic in enumerate(model.periodic_axes)
            if not periodic
        ):
            continue
        mapped = True
        for atom in range(model.number_of_atoms):
            image = operation @ model.positions[atom]
            matches = [
                other
                for other in range(model.number_of_atoms)
                if signatures[other] == signatures[atom]
                and np.allclose((image - model.positions[other] + 0.5) % 1.0 - 0.5, 0.0, atol=1e-8)
            ]
            if not matches:
                mapped = False
                break
        if mapped:
            kept.append(operation)
    return kept


def irreducible_q_points(
    model: ElectronPhononModel, n: int, rotations: Sequence[NDArray[np.int_]] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Irreducible points of an ``n``-grid with their star weights.

    Parameters
    ----------
    model : ElectronPhononModel
    n : int
        Points per periodic direction.
    rotations : sequence of ndarray, optional
        Point group (integer crystal-axis matrices); by default :func:`symmetry_operations`.

    Returns
    -------
    q_points : ndarray, shape ``(nir, 3)``
        Crystal coordinates.
    weights : ndarray, shape ``(nir,)``
        Star multiplicities, summing to the number of grid points.

    Notes
    -----
    3D models are folded with :func:`PAOFLOW.elphon.do_pao_eph_dense_q.irreducible_qmesh` (time
    reversal included); 1D and 2D models keep the full grid.
    """
    if model.dimension < 3:
        q_points = crystal_grid(model, n)
        return q_points, np.ones(len(q_points))
    from ...elphon.do_pao_eph_dense_q import irreducible_qmesh

    rotations = symmetry_operations(model) if rotations is None else rotations
    return irreducible_qmesh(n, rotations, model.lattice, np.linalg.inv(model.lattice).T)


# ---------------------------------------------------------------- semiconductors: the pair loops
def group_pairs_by_q(
    model: ElectronPhononModel,
    states: ElectronStates,
    initial: NDArray[np.int_],
    final: NDArray[np.int_],
) -> list[tuple[NDArray[np.float64], NDArray[np.int_]]]:
    """Group the pairs (``k`` in ``initial``, ``k + q`` in ``final``) by their ``q``.

    Parameters
    ----------
    model : ElectronPhononModel
    states : ElectronStates
    initial, final : ndarray of int
        Grid indices of the initial and final k-points.

    Returns
    -------
    list of (ndarray, ndarray)
        ``(q in crystal coordinates, the initial k indices of that q)``.
    """
    counts = grid_counts(model, states.grid_size)
    integer = np.rint(states.kpoints * counts).astype(int)
    difference = (integer[final][None, :, :] - integer[initial][:, None, :]) % counts
    q_ids = (difference[..., 0] * counts[1] + difference[..., 1]) * counts[2] + difference[..., 2]
    rows = np.repeat(np.arange(len(initial)), len(final))
    order = np.argsort(q_ids.ravel(), kind='stable')
    flat, rows = q_ids.ravel()[order], rows[order]
    cuts = np.nonzero(np.diff(flat))[0] + 1
    groups = []
    for ids, members in zip(np.split(flat, cuts), np.split(rows, cuts)):
        q = np.array(np.unravel_index(ids[0], counts)) / counts
        groups.append((q, initial[members]))
    return groups


def carrier_scattering_rates(
    model: ElectronPhononModel,
    states: ElectronStates,
    initial: NDArray[np.int_],
    final: NDArray[np.int_],
    temperatures: ArrayLike,
    chemical_potential: ArrayLike,
    smearing: float = 0.005,
    adaptive: float | None = 0.7,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """SERTA and MRTA scattering rates of the states at the ``initial`` k-points.

    Parameters
    ----------
    model : ElectronPhononModel
    states : ElectronStates
    initial, final : ndarray of int
        Grid indices of the initial k-points (all bands) and of the final ones.
    temperatures : array_like
        Temperatures (K).
    chemical_potential : array_like
        Chemical potential (eV) of the final-state occupations, one or one per temperature.
    smearing : float, optional
        Width of the energy delta, or the floor of the adaptive widths (eV).
    adaptive : float, optional
        Prefactor of the adaptive widths (one per final state); ``None``: fixed ``smearing``.

    Returns
    -------
    serta, mrta : ndarray, shape ``(ntemp, nk, nbnd)``
        :math:`1/\\tau_{n\\mathbf k}` (1/s), zero outside the initial k-points.

    Notes
    -----
    :func:`PAOFLOW.elphon.carrier_scattering.scattering_rate_kernel`:

    .. math::

        \\frac{1}{\\tau_{n\\mathbf k}} = \\frac{2\\pi}{\\hbar}\\frac{1}{N_q}\\sum_{m\\nu\\mathbf q}
        |g_{mn\\nu}|^2\\left[(n_{\\mathbf q\\nu} + 1 - f_{m\\mathbf k+\\mathbf q})
        \\delta(\\varepsilon_{n\\mathbf k} - \\varepsilon_{m\\mathbf k+\\mathbf q} - \\hbar\\omega)
        + (n_{\\mathbf q\\nu} + f_{m\\mathbf k+\\mathbf q})
        \\delta(\\varepsilon_{n\\mathbf k} - \\varepsilon_{m\\mathbf k+\\mathbf q} + \\hbar\\omega)\\right],

    and the momentum-relaxation (MRTA) rate weighs each term with
    :math:`1 - \\mathbf v_{n\\mathbf k}\\cdot\\mathbf v_{m\\mathbf k+\\mathbf q}/|\\mathbf v_{n\\mathbf k}|^2`.
    """
    from ...elphon.carrier_scattering import scattering_rate_kernel

    thermal = KB_EV * np.atleast_1d(np.asarray(temperatures, dtype=float))
    potential = np.broadcast_to(np.asarray(chemical_potential, dtype=float), thermal.shape)
    number_of_q = len(states.kpoints)
    serta = np.zeros((len(thermal), len(states.kpoints), states.energies.shape[1]))
    mrta = np.zeros_like(serta)
    widths = delta_widths(model, states, smearing, adaptive)
    for q, k_indices in group_pairs_by_q(model, states, initial, final):
        coupling, kq_indices, energies = coupling_matrix_elements(model, states, k_indices, q)
        occupations = states.energies[kq_indices][None] - potential[:, None, None]
        width = smearing if adaptive is None else widths[kq_indices]  # one per final state
        rate_serta, rate_mrta = scattering_rate_kernel(
            states.energies[k_indices],
            states.energies[kq_indices],
            states.velocities[k_indices],
            states.velocities[kq_indices],
            np.abs(coupling) ** 2,
            energies,
            thermal,
            occupations,
            width,
            1e-4,
            weight=1.0 / number_of_q,
        )
        serta[:, k_indices] += rate_serta
        mrta[:, k_indices] += rate_mrta
    return serta, mrta


def carrier_mobility(
    states: ElectronStates,
    rates: NDArray[np.float64],
    bands: int | Sequence[int],
    temperatures: ArrayLike,
    chemical_potential: ArrayLike,
    carriers: str = 'electrons',
    direction: int = 0,
) -> NDArray[np.float64]:
    """Drift mobility of electrons or holes in the relaxation-time approximation.

    Parameters
    ----------
    states : ElectronStates
    rates : ndarray, shape ``(ntemp, nk, nbnd)``
        :func:`carrier_scattering_rates` (SERTA or MRTA).
    bands : int or sequence of int
        Bands of the carriers.
    temperatures : array_like
        Temperatures (K), those of ``rates``.
    chemical_potential : array_like
        Chemical potential (eV), one or one per temperature.
    carriers : {'electrons', 'holes'}, optional
    direction : int, optional
        Cartesian axis of the field and the current.

    Returns
    -------
    ndarray, shape ``(ntemp,)``
        Mobility (cm^2/Vs).

    Notes
    -----
    .. math::

        \\mu = \\frac{e\\sum_{n\\mathbf k} v_{n\\mathbf k,a}^2\\,\\tau_{n\\mathbf k}
        \\left(-\\partial f/\\partial\\varepsilon\\right)}{\\sum_{n\\mathbf k} f_{n\\mathbf k}}

    for electrons, with :math:`1 - f` in the denominator for holes; the cell volume and the spin
    degeneracy cancel.  States without a rate (outside the window of the rates) do not conduct.
    """
    if carriers not in ('electrons', 'holes'):
        raise ValueError("carriers must be 'electrons' or 'holes'")
    bands = np.atleast_1d(np.asarray(bands, dtype=int))
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    potentials = np.broadcast_to(np.asarray(chemical_potential, dtype=float), temperatures.shape)
    velocity = states.velocities[:, bands, direction] / HBAR_EV_S * BOHR_M  # m/s
    mobilities = []
    for it, temperature in enumerate(temperatures):
        thermal = KB_EV * temperature
        occupation, vacancy = fermi_occupations(
            states.energies[:, bands], potentials[it], temperature
        )
        rate = rates[it][:, bands]
        lifetime = np.where(rate > 0, 1.0 / np.maximum(rate, 1e-300), 0.0)
        carriers_count = np.sum(occupation if carriers == 'electrons' else vacancy)
        conduction = np.sum(velocity**2 * lifetime * occupation * vacancy / thermal)
        mobilities.append(1e4 * conduction / carriers_count)
    return np.array(mobilities)


def indirect_absorption(
    model: ElectronPhononModel,
    states: ElectronStates,
    initial: NDArray[np.int_],
    final: NDArray[np.int_],
    temperatures: ArrayLike,
    photon_energies: ArrayLike,
    eta_ev: float = 0.02,
    degauss_ev: float = 0.02,
    reference_ev: float = 0.0,
) -> NDArray[np.float64]:
    """Phonon-assisted :math:`\\mathrm{Im}\\,\\varepsilon(\\omega)` across an indirect gap.

    Parameters
    ----------
    model : ElectronPhononModel
        A semiconductor.
    states : ElectronStates
        With ``velocity_matrices`` (:func:`electron_states`).
    initial, final : ndarray of int
        Grid indices of the valence k-points near the top and the conduction ones near the bottom.
    temperatures : array_like
        Temperatures (K).
    photon_energies : array_like
        Photon energies (eV).
    eta_ev : float, optional
        Broadening of the intermediate states (eV).
    degauss_ev : float, optional
        Gaussian width of the energy delta (eV).
    reference_ev : float, optional
        Energy zero of the kernel, mid-gap (eV).

    Returns
    -------
    ndarray, shape ``(ntemp, 3, nomega)``
        :math:`\\mathrm{Im}\\,\\varepsilon` along x, y, z.

    Notes
    -----
    :func:`PAOFLOW.elphon.phonon_assisted_absorption.indirect_absorption_kernel`, the second-order
    (photon + phonon) processes of Noffsinger et al. with both orderings: phonon absorption
    (:math:`\\propto n_B`) opens at :math:`E_g - \\hbar\\Omega`, emission
    (:math:`\\propto n_B + 1`) at :math:`E_g + \\hbar\\Omega`.  The kernel takes Rydberg units and
    the coupling without the factor :math:`1/\\sqrt{2\\omega}`.
    """
    from ...elphon.phonon_assisted_absorption import indirect_absorption_kernel

    if states.velocity_matrices is None:
        raise ValueError('indirect_absorption needs electron_states(..., velocity_matrices=True)')
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    photon_energies = np.asarray(photon_energies, dtype=float)
    number_of_q = len(states.kpoints)
    epsilon = np.zeros((len(temperatures), 3, len(photon_energies)))
    window = (
        max(
            np.abs(states.energies[initial] - reference_ev).max(),
            np.abs(states.energies[final] - reference_ev).max(),
        )
        + 1e-3
    )
    for q, k_indices in group_pairs_by_q(model, states, initial, final):
        coupling, kq_indices, energies = coupling_matrix_elements(model, states, k_indices, q)
        omega_ry = energies / RY_EV
        bare = (
            coupling / RY_EV * np.sqrt(2.0 * np.maximum(omega_ry, 1e-12))[None, :, None, None]
        )  # without 1/sqrt(2 omega)
        contribution, _ = indirect_absorption_kernel(
            (states.energies[k_indices] - reference_ev) / RY_EV,
            (states.energies[kq_indices] - reference_ev) / RY_EV,
            states.velocity_matrices[k_indices] / RY_EV,
            states.velocity_matrices[kq_indices] / RY_EV,
            bare,
            omega_ry,
            KB_EV * temperatures / RY_EV,
            photon_energies / RY_EV,
            [eta_ev / RY_EV],
            degauss_ev / RY_EV,
            window / RY_EV,
            1e-4 / RY_EV,
            weight=1.0 / (number_of_q * len(states.kpoints) * model.cell_volume),
        )
        epsilon += contribution[:, 0]
    return epsilon
