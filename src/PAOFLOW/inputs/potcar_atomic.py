r"""Recover bound pseudo-atomic orbitals from legacy PAW radial datasets.

Old POTCAR headers can omit atomic levels and use zero placeholders for their
reference energies. A second partial wave can be a mixed or energy-derivative
reference, so its energy belongs to a channel matrix rather than an independent
eigenvalue. The AE outer-region equation determines that matrix; a generalized
PAW radial solve then supplies confined, normalized bound pseudo-orbitals.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh

if TYPE_CHECKING:
    from .read_potcar import PotcarDataset


def _valence_shells(dataset: PotcarDataset) -> list[tuple[int, int, float]]:
    """Identify occupied shells from VRHFIN and the frozen-core electron count.

    Parameters
    ----------
    dataset : PotcarDataset
        Dataset with element, valence charge and valence configuration.

    Returns
    -------
    list of tuple
        Principal quantum number, angular momentum and occupation.

    Raises
    ------
    ValueError
        If occupations are missing, inconsistent, or contain several
        occupied radial shells of the same angular-momentum channel.
    """
    from ..basis_gen.radial import lowest_valence_n

    configuration = getattr(dataset, 'valence_configuration', '')
    matches = re.findall(r'([spdf])\s*(\d+(?:\.\d+)?)', configuration.lower())
    if not matches:
        raise ValueError(
            f'{dataset.title}: an Atomic configuration table or a VRHFIN '
            'valence configuration is required to identify occupied orbitals'
        )
    shells: list[tuple[int, int, float]] = []
    seen: set[int] = set()
    for letter, count in matches:
        occupation = float(count)
        if occupation <= 0.0:
            continue
        angular_momentum = 'spdf'.index(letter)
        if occupation > 2 * (2 * angular_momentum + 1) + 1.0e-8:
            raise ValueError(f'{dataset.title}: invalid VRHFIN occupation {letter}{count}')
        if angular_momentum in seen:
            raise ValueError(
                f'{dataset.title}: legacy VRHFIN lists several occupied {letter} shells; '
                'an explicit Atomic configuration table is required for this dataset'
            )
        seen.add(angular_momentum)
        shells.append((lowest_valence_n(dataset, angular_momentum), angular_momentum, occupation))
    missing_electrons = dataset.val - sum(occupation for _, _, occupation in shells)
    if missing_electrons > 1.0e-6 and shells:
        # Some headers describe only the outer atom in VRHFIN even when a
        # filled semicore channel is explicitly in ZVAL (Ga_d: s2p1, ZVAL=13).
        # Accept only a unique set of complete, lower-principal-number shells
        # supported by the project's actual PAW channels.
        outer_principal_number = max(principal for principal, _, _ in shells)
        candidates = [
            (lowest_valence_n(dataset, angular), angular, float(2 * (2 * angular + 1)))
            for angular in sorted(set(dataset.lll) - seen)
            if lowest_valence_n(dataset, angular) < outer_principal_number
        ]
        completions = [
            [shell for index, shell in enumerate(candidates) if mask & (1 << index)]
            for mask in range(1, 1 << len(candidates))
            if abs(
                sum(shell[2] for index, shell in enumerate(candidates) if mask & (1 << index))
                - missing_electrons
            )
            < 1.0e-6
        ]
        if len(completions) == 1:
            shells = sorted(completions[0]) + shells
    if abs(sum(occupation for _, _, occupation in shells) - dataset.val) > 1.0e-6:
        raise ValueError(f'{dataset.title}: VRHFIN occupations do not sum to ZVAL={dataset.val:g}')
    return shells


def _reference_potentials(
    dataset: PotcarDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Reconstruct full screened AE and pseudo reference potentials.

    Parameters
    ----------
    dataset : PotcarDataset
        Parsed radial potentials, densities and ionic local potential.

    Returns
    -------
    all_electron : ndarray, shape (npoints_stored,)
        Full AE screened potential in Hartree.
    pseudo : ndarray, shape (npoints,)
        Full pseudo screened potential in Hartree, continued outside the
        stored sphere by the frozen-density atomic potential.

    Notes
    -----
    The stored potentials have the sign of the negative valence screening:

    .. math::

        V_{AE} = V_H[n_c] - Z/r - P_{AE},\qquad
        V_{PS} = V_{loc} - P_{PS}.

    A correction decaying over one bohr matches the pseudo continuation
    to the stored potential value at the junction.
    """
    from ..basis_gen.atomic_potential import frozen_effective_potential
    from ..basis_gen.radial import _ELEMENTS
    from .read_potcar import HARTREE_EV

    if dataset.element not in _ELEMENTS:
        raise ValueError(f'{dataset.title}: unknown atomic number for reference reconstruction')
    atomic_number = _ELEMENTS.index(dataset.element) + 1
    stored_count = dataset.npoints_stored
    radial_mesh = dataset.r[:stored_count]
    core_radial_density = 4.0 * np.pi * radial_mesh**2 * dataset.rho_core_ae[:stored_count]
    enclosed_core = cumulative_trapezoid(core_radial_density, radial_mesh, initial=0.0)
    outer_core = cumulative_trapezoid(core_radial_density / radial_mesh, radial_mesh, initial=0.0)
    core_potential = enclosed_core / radial_mesh + outer_core[-1] - outer_core
    all_electron = core_potential - atomic_number / radial_mesh - dataset._potae_ev / HARTREE_EV
    pseudo = np.asarray(frozen_effective_potential(dataset), dtype=float).copy()
    stored_pseudo = dataset.vloc[:stored_count] - dataset._potps_ev / HARTREE_EV
    if dataset.r.size > stored_count:
        boundary_offset = stored_pseudo[-1] - pseudo[stored_count - 1]
        distance = dataset.r[stored_count:] - radial_mesh[-1]
        pseudo[stored_count:] += boundary_offset * np.exp(-distance)
    pseudo[:stored_count] = stored_pseudo
    if not np.all(np.isfinite(all_electron)) or not np.all(np.isfinite(pseudo)):
        raise ValueError(f'{dataset.title}: non-finite screened reference potential')
    return all_electron, pseudo


def _channel_reference_matrix(
    dataset: PotcarDataset,
    projector_indices: list[int],
    all_electron_potential: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Fit the AE radial reference operator in its partial-wave span.

    Parameters
    ----------
    dataset : PotcarDataset
        Dataset containing scalar-relativistic AE partial waves.
    projector_indices : list of int
        Projectors of one angular-momentum channel, in file order.
    all_electron_potential : ndarray, shape (npoints_stored,)
        Full screened AE potential in Hartree.

    Returns
    -------
    reference_matrix : ndarray, shape (nchannel, nchannel)
        Matrix satisfying ``(T + V_AE) U = U E`` in the outer sphere.
    residuals : ndarray, shape (nchannel,)
        Weighted RMS equation residual divided by wave RMS, in Hartree.

    Notes
    -----
    The first reference is an eigenfunction. Remaining columns may mix
    partial waves, including derivative references. The outer-region fit
    avoids the strong scalar-relativistic kinetic correction near the
    nucleus; residuals measure this reconstruction's numerical accuracy.
    """
    stored_count = dataset.npoints_stored
    radial_mesh = dataset.r[:stored_count]
    angular_momentum = dataset.lll[projector_indices[0]]
    partial_waves = np.column_stack(
        [dataset.aewfc[index][:stored_count] for index in projector_indices]
    )
    centrifugal = 0.5 * angular_momentum * (angular_momentum + 1) / radial_mesh**2
    actions = np.column_stack(
        [
            -0.5 * CubicSpline(radial_mesh, partial_waves[:, column])(radial_mesh, 2)
            + (centrifugal + all_electron_potential) * partial_waves[:, column]
            for column in range(len(projector_indices))
        ]
    )
    selected = (radial_mesh > 0.55 * radial_mesh[-1]) & (radial_mesh < radial_mesh[-4])
    if np.count_nonzero(selected) < max(8, 2 * len(projector_indices)):
        selected = (radial_mesh > 0.25 * radial_mesh[-1]) & (radial_mesh < radial_mesh[-3])
    if np.count_nonzero(selected) < max(6, 2 * len(projector_indices)):
        raise ValueError(
            f'{dataset.title}: insufficient radial points to recover reference energies'
        )
    weights = np.sqrt(dataset.rab[:stored_count][selected])
    weighted_waves = partial_waves[selected] * weights[:, None]
    weighted_actions = actions[selected] * weights[:, None]
    singular_values = np.linalg.svd(weighted_waves, compute_uv=False)
    if singular_values[-1] <= 1.0e-10 * singular_values[0]:
        raise ValueError(
            f'{dataset.title}: linearly dependent AE reference waves in l={angular_momentum}'
        )
    reference_matrix = np.zeros((len(projector_indices), len(projector_indices)))
    reference_matrix[0, 0] = np.dot(weighted_waves[:, 0], weighted_actions[:, 0]) / np.dot(
        weighted_waves[:, 0], weighted_waves[:, 0]
    )
    if len(projector_indices) > 1:
        reference_matrix[:, 1:] = np.linalg.lstsq(
            weighted_waves, weighted_actions[:, 1:], rcond=None
        )[0]
    residuals = np.linalg.norm(weighted_actions - weighted_waves @ reference_matrix, axis=0)
    residuals /= np.linalg.norm(weighted_waves, axis=0)
    tolerance = max(0.005, 0.02 * float(np.max(np.abs(reference_matrix))))
    if np.any(residuals > tolerance):
        raise ValueError(
            f'{dataset.title}: AE reference reconstruction in l={angular_momentum} has '
            f'a residual of {residuals.max():.4g} Ha (limit {tolerance:.4g} Ha)'
        )
    return reference_matrix, residuals


def _channel_screened_strengths(
    dataset: PotcarDataset,
    projector_indices: list[int],
    reference_matrix: NDArray[np.float64],
    pseudo_potential: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    r"""Recover screened nonlocal strengths with the mixed-reference PAW identity.

    Parameters
    ----------
    dataset : PotcarDataset
        Dataset containing pseudo partial waves and augmentation charges.
    projector_indices : list of int
        Projectors of one angular-momentum channel.
    reference_matrix : ndarray, shape (nchannel, nchannel)
        Recovered AE reference operator in Hartree.
    pseudo_potential : ndarray, shape (npoints,)
        Screened pseudo potential in Hartree.

    Returns
    -------
    strengths : ndarray, shape (nchannel, nchannel)
        Hermitian screened nonlocal strengths in Hartree.
    asymmetry : float
        Maximum asymmetry before Hermitian completion, in Hartree.

    Notes
    -----
    The biorthogonal projector identity is ``D = (B + Q) E - H_PS``:
    ``B`` is the pseudo-wave overlap and ``H_PS`` the local kinetic and
    potential matrix. The first, unmixed reference fixes its Hermitian row.
    """
    stored_count = dataset.npoints_stored
    radial_mesh = dataset.r[:stored_count]
    weights = dataset.rab[:stored_count]
    angular_momentum = dataset.lll[projector_indices[0]]
    partial_waves = np.column_stack(
        [dataset.pswfc_partial[index][:stored_count] for index in projector_indices]
    )
    local_potential = (
        pseudo_potential[:stored_count]
        + 0.5 * angular_momentum * (angular_momentum + 1) / radial_mesh**2
    )
    actions = np.column_stack(
        [
            -0.5 * CubicSpline(radial_mesh, partial_waves[:, column])(radial_mesh, 2)
            + local_potential * partial_waves[:, column]
            for column in range(len(projector_indices))
        ]
    )
    overlap = partial_waves.T @ (weights[:, None] * partial_waves)
    local_hamiltonian = partial_waves.T @ (weights[:, None] * actions)
    augmentation = dataset.qqq[np.ix_(projector_indices, projector_indices)]
    raw_strengths = (overlap + augmentation) @ reference_matrix - local_hamiltonian
    asymmetry = float(np.max(np.abs(raw_strengths - raw_strengths.T)))
    if asymmetry > max(0.02, 0.05 * float(np.max(np.abs(raw_strengths)))):
        raise ValueError(
            f'{dataset.title}: inconsistent legacy PAW reference matrix in l={angular_momentum}; '
            f'nonlocal strengths differ from their transpose by {asymmetry:.4g} Ha'
        )
    strengths = 0.5 * (raw_strengths + raw_strengths.T)
    strengths[:, 0] = raw_strengths[:, 0]
    strengths[0, :] = raw_strengths[:, 0]
    return strengths, asymmetry


def _bound_reference_orbital(
    dataset: PotcarDataset,
    angular_momentum: int,
    projector_index: int,
    reference_energy: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Solve and validate a confined bound pseudo-orbital in the PAW metric.

    Parameters
    ----------
    dataset : PotcarDataset
        Dataset with recovered atomic potential and nonlocal strengths.
    angular_momentum : int
        Orbital angular momentum.
    projector_index : int
        First reference projector of the occupied channel.
    reference_energy : float
        Recovered AE reference energy in Hartree.

    Returns
    -------
    orbital : ndarray, shape (npoints,)
        Radial ``u = r R``, normalized in the PAW metric.
    energy : float
        Generalized radial eigenvalue in Hartree.
    inner_overlap : float
        Absolute normalized overlap with the stored pseudo reference
        inside the stored sphere.

    Raises
    ------
    ValueError
        If no bound solution agrees with the reference energy and shape.

    Notes
    -----
    A finite-difference Dirichlet box uses the extended dataset radius.
    Only the four lowest states are computed. The reconstructed functions
    have the same finite radial support as the dataset.
    """
    radial_box = float(dataset.r[-1])
    interval_count = min(1800, max(800, int(np.ceil(radial_box / 0.0125))))
    spacing = radial_box / interval_count
    radial_mesh = np.arange(1, interval_count) * spacing
    projector_indices = [
        index for index, momentum in enumerate(dataset.lll) if momentum == angular_momentum
    ]
    projectors = np.column_stack(
        [
            CubicSpline(dataset.r, dataset.beta[index]['wfc'])(radial_mesh)
            for index in projector_indices
        ]
    )
    local_potential = CubicSpline(dataset.r, dataset._atomic_effective_potential)(radial_mesh)
    diagonal = (
        spacing**-2
        + local_potential
        + 0.5 * angular_momentum * (angular_momentum + 1) / radial_mesh**2
    )
    off_diagonal = np.full(radial_mesh.size - 1, -0.5 / spacing**2)
    hamiltonian = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    strengths = dataset._atomic_screened_dion[np.ix_(projector_indices, projector_indices)]
    hamiltonian += spacing * projectors @ strengths @ projectors.T
    augmentation = dataset.qqq[np.ix_(projector_indices, projector_indices)]
    overlap_matrix = np.eye(radial_mesh.size) + spacing * projectors @ augmentation @ projectors.T
    try:
        energies, vectors = eigh(
            hamiltonian,
            overlap_matrix,
            subset_by_index=(0, min(3, radial_mesh.size - 1)),
            check_finite=False,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f'{dataset.title}: invalid PAW overlap in the atomic radial problem'
        ) from error
    reference = CubicSpline(dataset.r, dataset.pswfc_partial[projector_index])(radial_mesh)
    inside = radial_mesh <= dataset.r[dataset.npoints_stored - 1]
    inner_products = vectors[inside].T @ reference[inside]
    denominators = np.linalg.norm(vectors[inside], axis=0) * np.linalg.norm(reference[inside])
    inner_overlaps = np.abs(inner_products / denominators)
    energy_tolerance = max(0.02, 0.05 * abs(reference_energy))
    candidates = np.flatnonzero(
        (energies < -1.0e-5)
        & (np.abs(energies - reference_energy) < energy_tolerance)
        & (inner_overlaps > 0.95)
    )
    if reference_energy >= -1.0e-5 or not candidates.size:
        raise ValueError(
            f'{dataset.title}: no bound l={angular_momentum} pseudo-orbital matches '
            f'the recovered reference at {reference_energy:.5f} Ha; '
            'a larger radial grid or explicit atomic reference data is required'
        )
    selected = int(candidates[np.argmin(np.abs(energies[candidates] - reference_energy))])
    values = np.concatenate(([0.0], vectors[:, selected], [0.0]))
    mesh_with_boundaries = np.concatenate(([0.0], radial_mesh, [radial_box]))
    orbital = CubicSpline(mesh_with_boundaries, values)(dataset.r)
    norm = dataset.augmented_norm(orbital, angular_momentum)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f'{dataset.title}: non-positive reconstructed PAW orbital norm')
    orbital /= np.sqrt(norm)
    if inner_products[selected] < 0.0:
        orbital *= -1.0
    return orbital, float(energies[selected]), float(inner_overlaps[selected])


def recover_legacy_atomic_data(dataset: PotcarDataset) -> None:
    """Recover missing legacy atomic metadata and bound PAW orbitals.

    Parameters
    ----------
    dataset : PotcarDataset
        Parsed dataset after radial projectors have been constructed.

    Returns
    -------
    None
        For datasets missing atomic configuration, fills reference
        energies, atomic configuration, beta labels and cached atomic
        potentials, strengths and pseudo-orbitals. Adds diagnostics in
        ``legacy_reconstruction``. Explicit atomic configurations are unchanged.

    Raises
    ------
    ValueError
        If the atomic information cannot be recovered consistently.

    Notes
    -----
    Orbital labels follow VRHFIN and the frozen core. Radial functions use
    the POTCAR reference Hamiltonian and augmentation metric; no generic
    all-electron basis tables are substituted.
    """
    from .read_potcar import HARTREE_EV

    if dataset.atomic_configuration:
        return
    occupied_shells = _valence_shells(dataset)
    missing_channels = {angular for _, angular, _ in occupied_shells} - set(dataset.lll)
    if missing_channels:
        raise ValueError(
            f'{dataset.title}: occupied VRHFIN channels lack PAW projectors: {missing_channels}'
        )
    all_electron_potential, pseudo_potential = _reference_potentials(dataset)
    dataset._atomic_effective_potential = pseudo_potential
    dataset._atomic_screened_dion = np.zeros((dataset.nproj, dataset.nproj))
    reference_energies = np.zeros(dataset.nproj)
    reference_matrices: dict[int, NDArray[np.float64]] = {}
    reference_residuals = np.zeros(dataset.nproj)
    asymmetry = 0.0
    for angular_momentum in sorted(set(dataset.lll)):
        indices = [
            index for index, momentum in enumerate(dataset.lll) if momentum == angular_momentum
        ]
        matrix, residuals = _channel_reference_matrix(dataset, indices, all_electron_potential)
        strengths, channel_asymmetry = _channel_screened_strengths(
            dataset, indices, matrix, pseudo_potential
        )
        dataset._atomic_screened_dion[np.ix_(indices, indices)] = strengths
        reference_energies[indices] = np.diag(matrix)
        reference_residuals[indices] = residuals
        reference_matrices[angular_momentum] = matrix
        asymmetry = max(asymmetry, channel_asymmetry)
    dataset.reference_energies = reference_energies
    dataset._energies_ev = reference_energies * HARTREE_EV
    dataset.dion_screened_asymmetry = asymmetry
    dataset.atomic_configuration = []
    dataset._legacy_pseudo_orbitals = []
    energy_errors: dict[str, float] = {}
    inner_overlaps: dict[str, float] = {}
    for principal_number, angular_momentum, occupation in occupied_shells:
        projector_index = dataset.lll.index(angular_momentum)
        reference_energy = float(reference_energies[projector_index])
        orbital, energy, inner_overlap = _bound_reference_orbital(
            dataset, angular_momentum, projector_index, reference_energy
        )
        label = f'{principal_number}{"SPDF"[angular_momentum]}'
        dataset.atomic_configuration.append(
            (
                principal_number,
                angular_momentum,
                angular_momentum + 0.5,
                reference_energy * HARTREE_EV,
                occupation,
            )
        )
        dataset._legacy_pseudo_orbitals.append(
            {
                'label': label,
                'occ': occupation,
                'wfc': orbital,
                'projector': projector_index,
                'energy': energy,
            }
        )
        energy_errors[label] = abs(energy - reference_energy)
        inner_overlaps[label] = inner_overlap
    for index, projector in enumerate(dataset.beta):
        projector['energy'] = float(reference_energies[index])
        projector['label'] = (
            f'{dataset.name} l={dataset.lll[index]} E={dataset._energies_ev[index]:.3f} eV'
        )
    dataset.legacy_reconstruction = {
        'method': 'outer AE reference operator and generalized PAW box solver',
        'reference_matrices': reference_matrices,
        'reference_residual_hartree': reference_residuals,
        'nonlocal_asymmetry_hartree': asymmetry,
        'orbital_energy_error_hartree': energy_errors,
        'orbital_inner_overlap': inner_overlaps,
        'confinement_radius_bohr': float(dataset.r[-1]),
    }
