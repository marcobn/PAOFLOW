r"""Spin splitting of quantum-oscillation orbits from the zero-field Bloch states.

A Kramers-degenerate band (spin-orbit coupling, inversion and time reversal) splits in a field
:math:`\mathbf B = B\hat{\mathbf n}` by the Zeeman matrix of the degenerate pair,

.. math::

    Z_{ab}(\mathbf k) = \mu_B\langle a|(\mathbf L + g_s\mathbf S)\cdot\hat{\mathbf n}|b\rangle
    - \mu_B\,\mathbf M_{ab}(\mathbf k)\cdot\hat{\mathbf n}\quad(\text{per tesla}),

the atomic term of :func:`PAOFLOW.hamiltonian.magnetic_field.zeeman_matrix` and the itinerant
(Bloch) orbital moment of the pair, the non-Abelian form of
:func:`PAOFLOW.response.orbital_moment.moment_and_curvature`,

.. math::

    M^c_{ab} = -\frac{i}{2}\sum_{l\notin D}
    \frac{V^{a'}_{al}V^{b'}_{lb} - V^{b'}_{al}V^{a'}_{lb}}{\varepsilon_l - \varepsilon}
    \quad(\mu_B,\ (a',b',c)\ \text{cyclic}),

with :math:`V = \hbar\mathbf v` the velocity of the atomic-gauge Hamiltonian (orbital centres
:math:`\boldsymbol\tau_i`, :mod:`PAOFLOW.hamiltonian.bloch_derivatives`; no intra-atomic dipoles, so
the itinerant term is the inter-site circulation and the intra-atomic one is :math:`\mathbf L`).
Around a cyclotron orbit, traversed in the time :math:`dt = \hbar\,dk/(eB|v_\perp|)`, the two
spin-split Landau-level sequences differ by the eigenphases :math:`\lambda_{1,2}` of the
path-ordered propagator of the pair

.. math::

    \mathcal A = \prod_j e^{-i\theta_j Z_j}\,S_{j,j+1},\qquad
    \theta_j = \frac{\hbar}{e}\frac{\Delta k_j}{|V_{\perp,j}|},\qquad
    S_{j,j+1} = \langle u^A_{\mathbf k_j}|u^A_{\mathbf k_{j+1}}\rangle,

which includes the non-Abelian Berry connection through the overlaps :math:`S` (Mikitik and
Sharlai, Phys. Rev. Lett. 93, 106403 (2004); Alexandradinata and Glazman, Phys. Rev. B 97, 144422
(2018)). Both factors are independent of :math:`B`. For a constant :math:`Z = (g\mu_B/2)\sigma_z`
the phase difference is :math:`\pi g m^*/m_e`, so the orbit's :math:`g^*m^*/m_e = (\lambda_1 -
\lambda_2)/\pi` and the spin reduction factor of the first harmonic is
:math:`R_s = \cos[(\lambda_1-\lambda_2)/2]`. The cyclotron mass of the same orbit is
:math:`m^*/m_e = (\hbar^2/m_e)(2\pi)^{-1}\oint dk/|V_\perp|`.

The adiabatic phase :math:`\sum_j\theta_j(z_{+,j} - z_{-,j})`, the local splitting accumulated
along the orbit, is what the Fermi-surface areas of the field-split bands give (a finite-field
calculation with :meth:`PAOFLOW.PAOFLOW.magnetic_field`); the propagator differs from it when the
spin axis of the pair turns along the orbit. Units: eV, bohr, tesla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..hamiltonian.bloch_derivatives import BlochTerms, _hermitian
from ..hamiltonian.magnetic_field import BOHR_MAGNETON_EV_PER_TESLA
from ..pyskeaf.results import OrbitOutline, read_orbit_outlines
from ..response.orbital_moment import CYCLIC, RYDBERG_EV

#: hbar / e in tesla bohr^2: the phase of a Zeeman energy (eV per tesla) over dk / |hbar v|.
HBAR_OVER_E_TESLA_BOHR2 = 1.054571817e-34 / 1.602176634e-19 / (5.29177210903e-11) ** 2
#: hbar^2 / m_e in eV bohr^2 (twice the Rydberg).
HBAR2_OVER_ME_EV_BOHR2 = 2.0 * RYDBERG_EV
BOHR_IN_ANGSTROM = 0.529177210903
#: The levels of the Zeeman matrix, cumulative.
LEVELS = ('spin', 'spin_atomic', 'total')


@dataclass
class OrbitSplitting:
    """Spin splitting of one orbit (:func:`orbit_spin_splitting`).

    Attributes
    ----------
    effective_mass : float
        :math:`m^*/m_e` from the orbit.
    adiabatic_phase : dict
        Per level of :data:`LEVELS`, :math:`\\sum_j\\theta_j(z_+ - z_-)_j` (rad).
    propagator_phase : dict
        Per level, the eigenphase difference of the propagator in ``[0, pi]`` (rad, modulo
        :math:`2\\pi` and sign), with the Berry transport.
    berry_phase_difference : float
        Eigenphase difference of the transport alone (no Zeeman term), ``[0, pi]``.
    berry_phase_mean : float
        Mean eigenphase of the transport alone (the Berry phase of the pair, modulo :math:`2\\pi`).
    max_pair_splitting_ev : float
        Largest energy difference of the pair on the orbit (the degeneracy check).
    max_energy_error_ev : float
        Largest distance of the points from the Fermi energy after the projection.
    npoints : int
    """

    effective_mass: float
    adiabatic_phase: dict = field(default_factory=dict)
    propagator_phase: dict = field(default_factory=dict)
    berry_phase_difference: float = 0.0
    berry_phase_mean: float = 0.0
    max_pair_splitting_ev: float = 0.0
    max_energy_error_ev: float = 0.0
    npoints: int = 0


def field_direction(azimuthal_deg: float, polar_deg: float) -> NDArray[np.float64]:
    """Unit field vector of the pyskeaf angles: ``(cos az sin pol, sin az sin pol, cos pol)``."""
    az, pol = np.radians(azimuthal_deg), np.radians(polar_deg)
    return np.array([np.cos(az) * np.sin(pol), np.sin(az) * np.sin(pol), np.cos(pol)])


def contour_points(outline: OrbitOutline) -> NDArray[np.float64]:
    """Points of an outline (1/Angstrom) without the repeated closing point, if any."""
    points = np.asarray(outline.points_inv_angstrom, dtype=float)
    if points.shape[0] > 2 and np.allclose(points[0], points[-1], atol=1e-10):
        return points[:-1]
    return points


def resample_closed_contour(points: NDArray[np.float64], npoints: int) -> NDArray[np.float64]:
    """Resample a closed polygon to ``npoints`` points equally spaced in arc length."""
    closed = np.vstack([points, points[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    targets = np.linspace(0.0, arc[-1], npoints, endpoint=False)
    return np.column_stack([np.interp(targets, arc, closed[:, a]) for a in range(3)])


def pair_orbital_moment(
    energies: NDArray[np.float64],
    velocities: NDArray[np.complex128],
    pair: NDArray[np.int_],
    degeneracy_tol: float = 1.0e-4,
) -> NDArray[np.complex128]:
    """Non-Abelian itinerant orbital moment of a degenerate group of bands at one k-point.

    Parameters
    ----------
    energies : ndarray ``(nb,)``
        Band energies (eV).
    velocities : ndarray ``(3, nb, nb)``, complex
        :math:`\\hbar v^a_{nm}` in the band basis (eV bohr).
    pair : ndarray of int
        Indices of the group (its mean energy is :math:`\\varepsilon`).
    degeneracy_tol : float, optional
        States closer than this to :math:`\\varepsilon` (eV) are left out of the sum.

    Returns
    -------
    ndarray ``(3, d, d)``, complex
        :math:`M^c_{ab}` (:math:`\\mu_B`), Hermitian; its diagonal for a single band is the
        moment of :func:`PAOFLOW.response.orbital_moment.moment_and_curvature`.
    """
    pair = np.asarray(pair, dtype=int)
    energy = float(np.mean(energies[pair]))
    gap = energies - energy
    outside = np.abs(gap) > degeneracy_tol
    outside[pair] = False
    inverse = np.where(outside, 1.0 / np.where(outside, gap, 1.0), 0.0)
    moment = np.zeros((3, pair.size, pair.size), dtype=complex)
    for c, (a, b) in enumerate(CYCLIC):
        left_a = velocities[a][pair][:, outside] * inverse[outside][None, :]
        left_b = velocities[b][pair][:, outside] * inverse[outside][None, :]
        moment[c] = (
            -0.5j
            * (left_a @ velocities[b][outside][:, pair] - left_b @ velocities[a][outside][:, pair])
            / RYDBERG_EV
        )
    return _hermitian(moment)


def _hamiltonian_and_velocity(
    terms: BlochTerms, k_cryst: NDArray[np.float64]
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """``H(k)`` (lattice gauge) and the atomic-gauge velocity ``dH + i dtau o H`` at k-points."""
    k = np.atleast_2d(np.asarray(k_cryst, dtype=float))
    n = terms.nawf
    phases = np.exp(-2j * np.pi * (k @ terms.cells.T))
    hamiltonian = _hermitian((phases @ terms.hamiltonian_cells).reshape(-1, n, n))
    velocity = np.empty((k.shape[0], 3, n, n), dtype=complex)
    for a in range(3):
        derivative = _hermitian(
            (
                (phases * (-1j * terms.cell_vectors_bohr[:, a])[None, :]) @ terms.hamiltonian_cells
            ).reshape(-1, n, n)
        )
        velocity[:, a] = derivative + 1j * terms.separation_bohr[a][None] * hamiltonian
    return hamiltonian, velocity


def _pair_states(
    terms: BlochTerms,
    points_bohr: NDArray[np.float64],
    lattice_bohr: NDArray[np.float64],
    pair: NDArray[np.int_],
    degeneracy_tol: float,
    with_moment: bool,
    chunk: int = 32,
) -> tuple[
    NDArray[np.float64], NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]
]:
    """Energies, states, band gradient and itinerant moment of the pair at the contour points.

    The Fourier sums run over blocks of ``chunk`` points and only the pair is kept, so the memory
    is that of one block of ``H`` and ``v`` (``4 chunk nawf^2`` complex numbers).

    Returns
    -------
    energies : ndarray ``(npts, d)``
    states : ndarray ``(npts, nawf, d)``
        Eigenvectors of the pair (lattice gauge, orbital basis).
    gradients : ndarray ``(npts, 3)``
        :math:`\\nabla_{\\mathbf k}\\varepsilon` of the pair (eV bohr).
    moments : ndarray ``(npts, 3, d, d)``
        Itinerant moment (:math:`\\mu_B`); zeros without ``with_moment``.
    """
    npts, d = points_bohr.shape[0], pair.size
    nawf = terms.nawf
    energies = np.zeros((npts, d))
    states = np.zeros((npts, nawf, d), dtype=complex)
    gradients = np.zeros((npts, 3))
    moments = np.zeros((npts, 3, d, d), dtype=complex)
    k_cryst = (points_bohr @ np.asarray(lattice_bohr, dtype=float).T) / (2.0 * np.pi)
    for start in range(0, npts, chunk):
        hamiltonian, velocity = _hamiltonian_and_velocity(terms, k_cryst[start : start + chunk])
        values, vectors = np.linalg.eigh(hamiltonian)
        for offset in range(values.shape[0]):
            j = start + offset
            basis = vectors[offset][:, pair]
            energies[j], states[j] = values[offset, pair], basis
            if with_moment:
                band_velocity = (
                    np.conjugate(vectors[offset].T)[None] @ velocity[offset] @ vectors[offset][None]
                )
                pair_block = band_velocity[:, pair][:, :, pair]
                moments[j] = pair_orbital_moment(
                    values[offset], band_velocity, pair, degeneracy_tol
                )
            else:
                pair_block = np.conjugate(basis.T)[None] @ velocity[offset] @ basis[None]
            gradients[j] = np.real(np.trace(pair_block, axis1=1, axis2=2)) / d
    return energies, states, gradients, moments


def _closest_unitary(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    left, _, right = np.linalg.svd(matrix)
    return left @ right


def _evolution(zeeman: NDArray[np.complex128], theta: float) -> NDArray[np.complex128]:
    values, vectors = np.linalg.eigh(zeeman)
    return (vectors * np.exp(-1j * theta * values)[None, :]) @ np.conjugate(vectors.T)


def loop_eigenphases(
    zeeman: list[NDArray[np.complex128]] | None,
    links: list[NDArray[np.complex128]],
    thetas: NDArray[np.float64],
) -> tuple[float, float]:
    """Eigenphase difference and mean of the closed-loop propagator.

    Parameters
    ----------
    zeeman : list of ndarray ``(d, d)`` or None
        Zeeman matrix at every point, in the basis of that point (eV per tesla); ``None`` for the
        transport alone.
    links : list of ndarray ``(d, d)``
        Unitary overlaps from point ``j`` to ``j + 1`` (the last one closes the loop).
    thetas : ndarray ``(npts,)``
        :math:`\\theta_j` (1 / (eV per tesla)).

    Returns
    -------
    tuple of float
        :math:`|\\lambda_1 - \\lambda_2|` folded into ``[0, pi]`` and the mean eigenphase (rad).
    """
    dim = links[0].shape[0]
    product = np.eye(dim, dtype=complex)
    for j, link in enumerate(links):
        if zeeman is not None:
            product = product @ _evolution(zeeman[j], thetas[j])
        product = product @ link
    phases = np.angle(np.linalg.eigvals(product))
    if dim == 1:
        return 0.0, float(phases[0])
    difference = abs(float(np.angle(np.exp(1j * (phases[0] - phases[1])))))
    mean = float(np.angle(np.exp(0.5j * (phases[0] + phases[1]))))
    return difference, mean


def orbit_spin_splitting(
    terms: BlochTerms,
    zeeman_spin: NDArray[np.complex128] | None,
    zeeman_atomic: NDArray[np.complex128] | None,
    orbital_positions_bohr: NDArray[np.float64],
    lattice_bohr: NDArray[np.float64],
    kpoints_bohr: NDArray[np.float64],
    field_unit: NDArray[np.float64],
    band: int,
    spinor: bool = True,
    fermi_energy: float = 0.0,
    g_s: float = 2.00231930436256,
    newton_steps: int = 2,
    degeneracy_tol: float = 1.0e-4,
) -> OrbitSplitting:
    """Spin-splitting phases and cyclotron mass of one closed orbit.

    Parameters
    ----------
    terms : BlochTerms
        Of :func:`PAOFLOW.hamiltonian.bloch_derivatives.bloch_terms_from_HRs` (with the orbital
        positions, so that the velocity is that of the atomic gauge).
    zeeman_spin, zeeman_atomic : ndarray ``(nawf, nawf)`` or None
        :math:`\\mu_B g_s\\mathbf S\\cdot\\hat{\\mathbf n}` and :math:`\\mu_B\\mathbf L\\cdot\\hat{\\mathbf n}`
        in the orbital basis (eV per tesla); ``zeeman_atomic`` may be ``None`` (no atomic term).
        Ignored for a scalar basis (``spinor=False``).
    orbital_positions_bohr : ndarray ``(nawf, 3)``
        Cartesian orbital centres (bohr), for the overlaps of the atomic gauge.
    lattice_bohr : ndarray ``(3, 3)``
        Direct lattice vectors (rows, bohr).
    kpoints_bohr : ndarray ``(npts, 3)``
        Closed contour, Cartesian (1/bohr), in the plane normal to ``field_unit``.
    field_unit : ndarray ``(3,)``
        Field direction.
    band : int
        Zero-based index of the band (either member of the Kramers pair).
    spinor : bool, optional
        Spinor basis (Kramers pair ``band - band % 2, + 1``); ``False`` for a spin-degenerate scalar
        band, whose spin factor is analytic.
    fermi_energy : float, optional
        Energy of the contour (eV).
    g_s : float, optional
        Spin g-factor (scalar bases only; spinor bases carry it in ``zeeman_spin``).
    newton_steps : int, optional
        In-plane Newton steps that move the points onto the energy contour.
    degeneracy_tol : float, optional
        Degeneracy threshold of the itinerant moment (eV).

    Returns
    -------
    OrbitSplitting

    Notes
    -----
    The weights :math:`\\theta_j` use the arc length halfway to the two neighbours and
    :math:`|V_\\perp|` of the pair at the point; the links are the closest unitary matrices to the
    overlaps of the pair in the atomic gauge,
    :math:`\\sum_i c^*_{ia}(\\mathbf k_j)\\,e^{i(\\mathbf k_j-\\mathbf k_{j+1})\\cdot\\boldsymbol\\tau_i}
    c_{ib}(\\mathbf k_{j+1})`.
    """
    n_hat = np.asarray(field_unit, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    lattice = np.asarray(lattice_bohr, dtype=float)
    pair = np.array([band - band % 2, band - band % 2 + 1]) if spinor else np.array([band])
    points = np.array(kpoints_bohr, dtype=float)

    for _ in range(max(0, int(newton_steps))):
        pair_energies, _, gradients, _ = _pair_states(
            terms, points, lattice, pair, degeneracy_tol, with_moment=False
        )
        in_plane = gradients - (gradients @ n_hat)[:, None] * n_hat[None, :]
        error = np.mean(pair_energies, axis=1) - fermi_energy
        points -= (error / np.maximum(np.sum(in_plane**2, axis=1), 1e-30))[:, None] * in_plane

    pair_energies, bases, gradients, moments = _pair_states(
        terms, points, lattice, pair, degeneracy_tol, with_moment=True
    )
    npts = points.shape[0]
    segment = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    arc = 0.5 * (segment + np.roll(segment, 1))
    speeds = np.linalg.norm(gradients - (gradients @ n_hat)[:, None] * n_hat[None, :], axis=1)
    max_split = float(np.max(np.ptp(pair_energies, axis=1)))
    max_error = float(np.max(np.abs(np.mean(pair_energies, axis=1) - fermi_energy)))

    states, zeeman_levels = [], {level: [] for level in LEVELS}
    spin_pair = (0.5 * g_s * BOHR_MAGNETON_EV_PER_TESLA) * np.diag([1.0, -1.0]).astype(complex)
    for j in range(npts):
        basis = bases[j]
        itinerant = -BOHR_MAGNETON_EV_PER_TESLA * np.tensordot(n_hat, moments[j], axes=(0, 0))
        if spinor:
            spin = np.conjugate(basis.T) @ zeeman_spin @ basis
            atomic = (
                np.conjugate(basis.T) @ zeeman_atomic @ basis
                if zeeman_atomic is not None
                else np.zeros_like(spin)
            )
        else:  # band x spin: the orbital terms shift both spins alike
            spin = spin_pair
            atomic = np.zeros((2, 2), dtype=complex)
            itinerant = np.real(itinerant[0, 0]) * np.eye(2)
        zeeman_levels['spin'].append(spin)
        zeeman_levels['spin_atomic'].append(spin + atomic)
        zeeman_levels['total'].append(spin + atomic + itinerant)
        states.append(basis)

    thetas = HBAR_OVER_E_TESLA_BOHR2 * arc / np.maximum(speeds, 1e-30)
    positions = np.asarray(orbital_positions_bohr, dtype=float)
    links = []
    for j in range(npts):
        nxt = (j + 1) % npts
        phase = np.exp(1j * (positions @ (points[j] - points[nxt])))
        overlap = np.conjugate(states[j].T) @ (phase[:, None] * states[nxt])
        link = _closest_unitary(overlap)
        links.append(link if spinor else np.kron(link, np.eye(2)))

    result = OrbitSplitting(
        effective_mass=float(HBAR2_OVER_ME_EV_BOHR2 / (2.0 * np.pi) * np.sum(arc / speeds)),
        max_pair_splitting_ev=max_split,
        max_energy_error_ev=max_error,
        npoints=npts,
    )
    result.berry_phase_difference, result.berry_phase_mean = loop_eigenphases(None, links, thetas)
    for level in LEVELS:
        local = np.array([np.ptp(np.linalg.eigvalsh(z)) for z in zeeman_levels[level]])
        result.adiabatic_phase[level] = float(np.sum(thetas * local))
        result.propagator_phase[level] = loop_eigenphases(zeeman_levels[level], links, thetas)[0]
    return result


def splitting_summary(result: OrbitSplitting) -> dict:
    """Derived quantities of an :class:`OrbitSplitting`: g*m*, g*, R_s per level.

    The propagator phase is known modulo :math:`2\\pi` and sign; it is unfolded to the branch
    closest to the adiabatic phase of the same level (``*_prop``), and ``|R_s|`` of the propagator
    needs no unfolding.
    """
    mass = result.effective_mass
    summary = {
        'm_star': mass,
        'npoints': result.npoints,
        'berry_phase_mean': result.berry_phase_mean,
        'berry_rotation': result.berry_phase_difference,
        'max_pair_splitting_meV': 1e3 * result.max_pair_splitting_ev,
        'max_energy_error_meV': 1e3 * result.max_energy_error_ev,
    }
    for level in LEVELS:
        adiabatic = result.adiabatic_phase[level]
        folded = result.propagator_phase[level]
        turns = int(np.floor(adiabatic / (2.0 * np.pi)))
        candidates = np.array(
            [
                sign * folded + 2.0 * np.pi * m
                for sign in (1.0, -1.0)
                for m in range(turns - 1, turns + 3)
            ]
        )
        unfolded = float(candidates[np.argmin(np.abs(candidates - adiabatic))])
        summary[f'gm_{level}'] = adiabatic / np.pi
        summary[f'g_{level}'] = adiabatic / (np.pi * mass)
        summary[f'Rs_{level}'] = float(np.cos(0.5 * adiabatic))
        summary[f'gm_{level}_prop'] = abs(unfolded) / np.pi
        summary[f'g_{level}_prop'] = abs(unfolded) / (np.pi * mass)
        summary[f'absRs_{level}_prop'] = float(abs(np.cos(0.5 * folded)))
    return summary


__all__ = [
    'HBAR_OVER_E_TESLA_BOHR2',
    'LEVELS',
    'OrbitOutline',
    'OrbitSplitting',
    'contour_points',
    'field_direction',
    'loop_eigenphases',
    'orbit_spin_splitting',
    'pair_orbital_moment',
    'read_orbit_outlines',
    'resample_closed_contour',
    'splitting_summary',
]
