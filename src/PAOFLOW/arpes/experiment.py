"""The ARPES experiment: geometry and settings, the simulation driver and its result."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .basis import PhotoemissionBasis
from .bloch import BlochHamiltonian
from .kinematics import (
    final_state_directions,
    kinetic_energy,
    kz_free_electron,
    momentum_mesh,
    photoelectron_momentum,
)
from .matrix_elements import dipole_matrix_elements, polarization_weights
from .radial import BOHR_ANGSTROM, RadialIntegrals
from .spectral import SIGMA_PER_FWHM, assemble_intensity, gaussian_broadening


@dataclass
class ArpesExperiment:
    """Settings of an ARPES simulation (energies in eV, momenta in 1/Å, angles in radians).

    Parameters
    ----------
    photon_ev : float
        Photon energy.
    work_function_ev : float, optional
    inner_potential_ev : float, optional
        With a value, ``k_z`` of every k-point follows from the photon energy
        (:func:`~PAOFLOW.arpes.kinematics.kz_free_electron`); otherwise ``kz`` is used.
    temperature_k : float, optional
        Fermi-Dirac cut-off; ``None`` keeps all states.
    polarization : sequence of 3 complex
        Light polarisation in the sample frame (Cartesian), e.g. ``(1, 0, 0)`` or circular
        ``(1, 1j, 0) / sqrt 2``.
    energy : tuple ``(min, max, n)``
        Energy window relative to the Fermi level.
    kx, ky : tuple ``(min, max, n)``
        In-plane momentum mesh (laboratory frame).
    kz : float, optional
        Out-of-plane momentum without an inner potential.
    kpoints : ndarray ``(N, 3)``, optional
        Explicit momenta (a path or any list); overrides ``kx``/``ky``; ``kz`` of the rows is
        replaced when an inner potential is given.
    azimuth : float, optional
        Rotation of the sample about its normal.
    resolution : tuple ``(dE, dk)``
        Gaussian FWHM in energy (eV) and momentum (1/Å); 0 disables.
    escape_depth : float, optional
        Photoelectron escape depth ``lambda`` (Å) for slab depths; ``None`` disables.
    radial : str or RadialIntegrals
        ``'pao'`` (the basis tables), ``'slater'``, ``'hydrogenic'`` or a prepared
        :class:`~PAOFLOW.arpes.radial.RadialIntegrals`.
    phase_shifts, fixed_radial : dict, optional
        Passed to :class:`~PAOFLOW.arpes.radial.RadialIntegrals` (keys ``(radial_key, l')``).
    position_phase : bool, optional
        Apply the gauge phase ``e^{-i (k_f - k).tau_i}`` (``False``: chinook's convention).
    kz_at_binding_energy : bool, optional
        Evaluate the free-electron ``k_z`` at each state's binding energy instead of at E_F.
    window_margin_ev : float, optional
        Bands are kept when they enter ``[E_min - margin, E_max + margin]``.
    kblock : int, optional
        k-points per block.
    """

    photon_ev: float
    work_function_ev: float = 4.0
    inner_potential_ev: float | None = None
    temperature_k: float | None = None
    polarization: tuple = (1.0, 0.0, 0.0)
    energy: tuple[float, float, int] = (-1.0, 0.1, 400)
    kx: tuple[float, float, int] | None = (-1.0, 1.0, 101)
    ky: tuple[float, float, int] | None = (-1.0, 1.0, 101)
    kz: float = 0.0
    kpoints: NDArray[np.float64] | None = None
    azimuth: float = 0.0
    resolution: tuple[float, float] = (0.01, 0.01)
    escape_depth: float | None = None
    radial: object = 'pao'
    phase_shifts: dict | None = None
    fixed_radial: dict | None = None
    position_phase: bool = False
    kz_at_binding_energy: bool = False
    window_margin_ev: float = 0.5
    kblock: int = 256

    def momenta(self) -> tuple[NDArray[np.float64], tuple[int, int] | None, dict]:
        """``(kpoints (N, 3), grid shape or None, axes)`` in the sample frame."""
        if self.kpoints is not None:
            kp = np.array(self.kpoints, dtype=float)
            if kp.ndim != 2 or kp.shape[1] != 3:
                raise ValueError('kpoints must be (N, 3)')
            return kp, None, {}
        kp, kx_axis, ky_axis = momentum_mesh(self.kx, self.ky, self.kz, self.azimuth)
        return kp, (int(self.kx[2]), int(self.ky[2])), {'kx': kx_axis, 'ky': ky_axis}

    def energy_grid(self) -> NDArray[np.float64]:
        """Uniform energy grid of the window."""
        return np.linspace(self.energy[0], self.energy[1], int(self.energy[2]))


@dataclass
class IntensityMap:
    """Result of :func:`simulate`.

    Attributes
    ----------
    kpoints : ndarray ``(N, 3)``
        Sample-frame momenta (1/Å) of the simulation (``k_z`` as used by the Hamiltonian).
    w : ndarray ``(nw,)``
        Energy grid (eV, relative to E_F).
    intensity : ndarray ``(N, nw)``
        Resolution-broadened intensity.
    raw : ndarray ``(N, nw)``
        Intensity before broadening.
    energies, weights : ndarray ``(N, nb)``
        Band energies of the window bands and their matrix-element weights.
    bands : ndarray ``(nb,)``
        Band indices kept.
    grid : tuple ``(nx, ny)`` or None
        Mesh shape (``kx`` fastest) when the momenta form a mesh.
    axes : dict
        ``'kx'``, ``'ky'`` axes of the mesh (laboratory frame).
    metadata : dict
        The experiment settings.
    """

    kpoints: NDArray[np.float64]
    w: NDArray[np.float64]
    intensity: NDArray[np.float64]
    raw: NDArray[np.float64]
    energies: NDArray[np.float64]
    weights: NDArray[np.float64]
    bands: NDArray[np.int_]
    grid: tuple[int, int] | None = None
    axes: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def as_grid(self, broadened: bool = True) -> NDArray[np.float64]:
        """``(nx, ny, nw)`` view of the intensity of a mesh simulation."""
        if self.grid is None:
            raise ValueError('not a mesh simulation')
        data = self.intensity if broadened else self.raw
        return data.reshape(self.grid[0], self.grid[1], self.w.size)

    def energy_distribution_curve(self, ik: int, broadened: bool = True) -> NDArray[np.float64]:
        """``I(w)`` at k-point ``ik``."""
        return (self.intensity if broadened else self.raw)[ik]

    def momentum_distribution_curve(
        self, energy_ev: float, broadened: bool = True
    ) -> NDArray[np.float64]:
        """``I(k)`` at the energy closest to ``energy_ev``."""
        iw = int(np.argmin(np.abs(self.w - energy_ev)))
        return (self.intensity if broadened else self.raw)[:, iw]

    def save(self, path: str) -> str:
        """Write everything to an ``.npz`` file; returns the path."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path, kpoints=self.kpoints, w=self.w, intensity=self.intensity, raw=self.raw,
            energies=self.energies, weights=self.weights, bands=self.bands,
            grid=np.array(self.grid if self.grid is not None else (0, 0)),
            kx=self.axes.get('kx', np.zeros(0)), ky=self.axes.get('ky', np.zeros(0)),
            metadata=np.array(repr(self.metadata)),
        )  # fmt: skip
        return path


def _window_bands(energies: NDArray[np.float64], emin: float, emax: float) -> NDArray[np.int_]:
    """Bands with at least one state in ``[emin, emax]``."""
    inside = (energies >= emin) & (energies <= emax)
    return np.nonzero(inside.any(axis=0))[0]


def simulate(
    experiment: ArpesExperiment,
    basis: PhotoemissionBasis,
    bloch: BlochHamiltonian,
    self_energy=None,
) -> IntensityMap:
    """Simulate the ARPES intensity of a PAO model.

    Parameters
    ----------
    experiment : ArpesExperiment
    basis : PhotoemissionBasis
        Orbitals of the model (with radial tables, or ``experiment.radial`` set to an atomic
        fallback).
    bloch : BlochHamiltonian
        The PAO Hamiltonian (energies relative to its ``fermi_level``).
    self_energy : complex, ndarray or callable, optional
        ``Sigma`` of the window states: a constant, an array ``(N, nb, nw)``, or a callable
        ``f(w, energies, kpoints, bands)`` evaluated per k-block (see
        :func:`~PAOFLOW.arpes.spectral.assemble_intensity`); ``None`` for the bare bands.

    Returns
    -------
    IntensityMap

    Notes
    -----
    Steps (chinook ``datacube`` + ``spectral``): momenta of the mesh, free-electron ``k_z``
    when an inner potential is given, diagonalisation of ``H(k)`` in blocks, selection of the
    bands entering the energy window, photoelectron momentum per state from
    ``h nu + E_b - W``, dipole matrix elements, ``|eps.M|^2``, spectral function with the
    self-energy and the Fermi factor, Gaussian resolution on the mesh (energy, ``k_x``, ``k_y``)
    or along the energy and the path.
    """
    kpoints, grid, axes = experiment.momenta()
    nkpt = kpoints.shape[0]
    kpar = np.linalg.norm(kpoints[:, :2], axis=1)
    if experiment.inner_potential_ev is not None:
        kpoints = kpoints.copy()
        kpoints[:, 2] = kz_free_electron(
            experiment.photon_ev, kpar, experiment.work_function_ev, experiment.inner_potential_ev
        )
    w = experiment.energy_grid()
    margin = experiment.window_margin_ev + 6.0 * SIGMA_PER_FWHM * experiment.resolution[0]

    k_cryst = bloch.crystal_coordinates(kpoints)
    energies_all = np.empty((nkpt, bloch.nawf))
    vectors_all = np.empty((nkpt, bloch.nawf, bloch.nawf), dtype=complex)
    for s0 in range(0, nkpt, experiment.kblock):
        s1 = min(s0 + experiment.kblock, nkpt)
        energies_all[s0:s1], vectors_all[s0:s1] = bloch.eigenstates(k_cryst[s0:s1])
    bands = _window_bands(energies_all, w[0] - margin, w[-1] + margin)
    if bands.size == 0:
        raise ValueError('no band enters the energy window %s' % (experiment.energy,))
    energies = energies_all[:, bands]

    # photoelectron momentum of every state (vacuum): |k_f| from the kinetic energy, the
    # in-plane part conserved, the out-of-plane part from the remainder
    kinetic = kinetic_energy(experiment.photon_ev, energies, experiment.work_function_ev)
    kf_norm = photoelectron_momentum(kinetic)
    directions, valid = final_state_directions(
        np.broadcast_to(kpoints[:, None, :2], energies.shape + (2,)), kf_norm
    )
    k_final = directions * kf_norm[..., None]

    if isinstance(experiment.radial, RadialIntegrals):
        radial = experiment.radial
        work_basis = basis
    else:
        work_basis = basis
        if experiment.radial in ('slater', 'hydrogenic'):
            work_basis = basis.with_atomic_radial(experiment.radial)
        elif experiment.radial != 'pao':
            raise ValueError("radial must be 'pao', 'slater', 'hydrogenic' or a RadialIntegrals")
        radial = RadialIntegrals(
            work_basis.radial_functions(), float(kf_norm.max()) * BOHR_ANGSTROM * 1.05,
            phase_shifts=experiment.phase_shifts, fixed=experiment.fixed_radial,
        )  # fmt: skip

    weights = np.zeros(energies.shape)
    eps = np.asarray(experiment.polarization, dtype=complex)
    for s0 in range(0, nkpt, experiment.kblock):
        s1 = min(s0 + experiment.kblock, nkpt)
        M = dipole_matrix_elements(
            work_basis, vectors_all[s0:s1][:, :, bands], k_final[s0:s1], radial,
            escape_depth=experiment.escape_depth,
            k_hamiltonian=kpoints[s0:s1] if experiment.position_phase else None,
        )  # fmt: skip
        weights[s0:s1] = polarization_weights(M, eps)
    weights[~valid] = 0.0

    sigma = 0.0 if self_energy is None else self_energy
    raw = assemble_intensity(
        weights, energies, w, sigma, experiment.temperature_k, kpoints, experiment.kblock, bands
    )

    dE, dk = experiment.resolution
    dw = (w[-1] - w[0]) / max(w.size - 1, 1)
    if grid is not None:
        cube = raw.reshape(grid[0], grid[1], w.size)
        spacing = [
            (axes['kx'][-1] - axes['kx'][0]) / max(grid[0] - 1, 1),
            (axes['ky'][-1] - axes['ky'][0]) / max(grid[1] - 1, 1),
            dw,
        ]
        broadened = gaussian_broadening(cube, (dk, dk, dE), spacing).reshape(nkpt, w.size)
    else:
        step = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
        path_spacing = float(np.mean(step)) if step.size and np.all(step > 0) else None
        broadened = gaussian_broadening(raw, (dk, dE), (path_spacing, dw))

    metadata = {k: v for k, v in experiment.__dict__.items() if k != 'kpoints'}
    metadata['valid_fraction'] = float(np.mean(valid))
    return IntensityMap(kpoints, w, broadened, raw, energies, weights, bands, grid, axes, metadata)
