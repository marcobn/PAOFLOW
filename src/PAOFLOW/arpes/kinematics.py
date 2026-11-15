"""Photoelectron kinematics: kinetic energy, momentum, free-electron ``k_z`` and emission meshes.

Adapted from chinook (``klib.kz_kpt``, ``klib.kmesh``, ``tilt.ang_mesh``, ``tilt.k_mesh``;
R. P. Day et al., npj Quantum Materials 4, 54 (2019), MIT licence, Copyright (c) 2018 Ryan
Patrick Day).  Energies in eV, momenta in 1/Å, angles in radians.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: ``sqrt(2 m_e e) / hbar`` in 1/(Å sqrt(eV)): ``|k| = KINETIC_PREFACTOR sqrt(E_kin)``.
KINETIC_PREFACTOR = 0.5123167219


def photoelectron_momentum(kinetic_ev: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """``|k_f|`` (1/Å) of a free electron with kinetic energy ``kinetic_ev`` (eV), 0 below zero."""
    kinetic = np.maximum(np.asarray(kinetic_ev, dtype=float), 0.0)
    return KINETIC_PREFACTOR * np.sqrt(kinetic)


def kinetic_energy(
    photon_ev: float, binding_ev: NDArray[np.float64] | float, work_function_ev: float
) -> NDArray[np.float64]:
    """``E_kin = h nu + E_b - W`` with ``E_b`` the band energy relative to the Fermi level (eV).

    ``binding_ev`` is negative for occupied states, so the kinetic energy decreases with the
    binding energy.
    """
    return photon_ev + np.asarray(binding_ev, dtype=float) - work_function_ev


def kz_free_electron(
    photon_ev: float,
    k_parallel: NDArray[np.float64] | float,
    work_function_ev: float,
    inner_potential_ev: float,
    binding_ev: NDArray[np.float64] | float = 0.0,
) -> NDArray[np.float64]:
    """Out-of-plane momentum (1/Å) of the free-electron final state inside the crystal.

    Parameters
    ----------
    photon_ev : float
        Photon energy.
    k_parallel : array_like
        In-plane momentum (1/Å).
    work_function_ev : float
        Work function.
    inner_potential_ev : float
        Inner potential ``V_0`` (positive for a potential well).
    binding_ev : array_like, optional
        Band energy relative to the Fermi level (eV); chinook evaluates ``k_z`` at the Fermi
        level (default 0).

    Returns
    -------
    ndarray
        ``k_z`` (1/Å), 0 where the state is evanescent.

    Notes
    -----
    .. math::

        k_z = \\frac{\\sqrt{2 m_e}}{\\hbar}\\sqrt{E_{\\rm kin}\\Big(1 - \\frac{k_\\parallel^2}{k_n^2}\\Big) + V_0},
        \\qquad k_n = \\frac{\\sqrt{2 m_e E_{\\rm kin}}}{\\hbar},

    chinook ``kz_kpt`` with ``E_kin = h nu + E_b - W``.
    """
    kinetic = kinetic_energy(photon_ev, binding_ev, work_function_ev)
    kn2 = (KINETIC_PREFACTOR**2) * kinetic
    kpar2 = np.asarray(k_parallel, dtype=float) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        perpendicular = np.where(kn2 > 0.0, kinetic * (1.0 - kpar2 / np.maximum(kn2, 1e-300)), 0.0)
    return KINETIC_PREFACTOR * np.sqrt(np.maximum(perpendicular + inner_potential_ev, 0.0))


def momentum_mesh(
    kx: tuple[float, float, int],
    ky: tuple[float, float, int],
    kz: float = 0.0,
    azimuth: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Uniform in-plane momentum mesh at fixed ``k_z``, rotated by ``azimuth`` about the normal.

    Parameters
    ----------
    kx, ky : tuple ``(min, max, n)``
        Axes of the mesh in the laboratory frame (1/Å).
    kz : float, optional
        Out-of-plane momentum (1/Å) assigned to every point.
    azimuth : float, optional
        Rotation of the sample about its normal (radians): a point at polar angle ``phi`` in the
        laboratory frame lies at ``phi - azimuth`` in the sample frame (chinook ``kmesh``).

    Returns
    -------
    kpoints : ndarray ``(nx * ny, 3)``
        Sample-frame momenta, ``x`` fastest.
    kx_axis, ky_axis : ndarray
        The laboratory-frame axes.
    """
    kx_axis = np.linspace(kx[0], kx[1], int(kx[2]))
    ky_axis = np.linspace(ky[0], ky[1], int(ky[2]))
    X, Y = np.meshgrid(kx_axis, ky_axis, indexing='ij')
    if abs(azimuth) > 0.0:
        kp = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X) - azimuth
        X, Y = kp * np.cos(phi), kp * np.sin(phi)
    kpoints = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(kz))], axis=1)
    return kpoints, kx_axis, ky_axis


def angle_mesh(
    kinetic_ev: float,
    theta: tuple[float, float, int],
    phi: tuple[float, float, int],
    kz: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """In-plane momenta of a mesh of emission angles (chinook ``tilt.gen_kpoints``).

    Parameters
    ----------
    kinetic_ev : float
        Photoelectron kinetic energy (eV) at which the angles are converted.
    theta, phi : tuple ``(min, max, n)``
        Ranges (radians) of the two rotation angles of the detector: ``theta`` about the
        vertical axis, ``phi`` about the horizontal one.
    kz : float, optional
        Out-of-plane momentum assigned to every point.

    Returns
    -------
    kpoints : ndarray ``(n_theta * n_phi, 3)``
        ``k_parallel = |k| (-sin theta, -cos theta sin phi)``, ``theta`` fastest.
    theta_axis, phi_axis : ndarray
    """
    theta_axis = np.linspace(theta[0], theta[1], int(theta[2]))
    phi_axis = np.linspace(phi[0], phi[1], int(phi[2]))
    T, P = np.meshgrid(theta_axis, phi_axis, indexing='ij')
    klen = float(photoelectron_momentum(kinetic_ev))
    kx = -klen * np.sin(T)
    ky = -klen * np.cos(T) * np.sin(P)
    kpoints = np.stack([kx.ravel(), ky.ravel(), np.full(T.size, float(kz))], axis=1)
    return kpoints, theta_axis, phi_axis


def final_state_directions(
    k_parallel: NDArray[np.float64], kf_norm: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Unit vectors of the photoelectron momentum from its in-plane part and its modulus.

    Parameters
    ----------
    k_parallel : ndarray ``(..., 2)`` or ``(..., 3)``
        In-plane momentum (1/Å); a third component is ignored.
    kf_norm : ndarray ``(...)``
        ``|k_f|`` (1/Å).

    Returns
    -------
    directions : ndarray ``(..., 3)``
        ``(sin theta cos phi, sin theta sin phi, cos theta)`` with ``cos theta =
        sqrt(k_f^2 - k_par^2) / k_f`` (chinook ``datacube``); zero where the state cannot emit.
    valid : ndarray of bool ``(...)``
        ``k_f^2 >= k_par^2``.
    """
    kpar = np.asarray(k_parallel, dtype=float)[..., :2]
    kf = np.asarray(kf_norm, dtype=float)
    kpar2 = np.sum(kpar**2, axis=-1)
    valid = (kf > 0.0) & (kf**2 >= kpar2)
    safe = np.where(valid, kf, 1.0)
    kz = np.sqrt(np.maximum(kf**2 - kpar2, 0.0))
    directions = np.stack([kpar[..., 0], kpar[..., 1], kz], axis=-1) / safe[..., None]
    return np.where(valid[..., None], directions, 0.0), valid
