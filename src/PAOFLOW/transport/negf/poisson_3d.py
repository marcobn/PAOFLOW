"""Poisson equation between two grounded planes, periodic across the transverse cell, for Gaussian charges.

The slab is the transverse cell (in-plane lattice vectors :math:`\\mathbf a_1, \\mathbf a_2`, area :math:`A`) times the
interval :math:`a < z < b` between the electrode planes (:math:`L = b - a`). With homogeneous Dirichlet conditions at
the planes the electron potential energy of a density :math:`\\delta n`, :math:`\\nabla^2 U = -4\\pi\\delta n`
(hartree units), is diagonal in in-plane plane waves times sines:

.. math::

    U(\\mathbf r) = \\frac{4\\pi}{A}\\sum_{\\mathbf G}\\sum_{n\\ge1}
    \\frac{e^{i\\mathbf G\\cdot\\mathbf r_\\parallel}\\,\\phi_n(z)\\,\\hat n(\\mathbf G, n)}{G^2 + k_n^2},\\qquad
    \\phi_n(z) = \\sqrt{2/L}\\,\\sin k_n(z - a),\\quad k_n = n\\pi/L,

with :math:`\\hat n(\\mathbf G, n) = \\int_{\\rm cell} d^2r_\\parallel\\int dz\\, e^{-i\\mathbf G\\cdot\\mathbf r_\\parallel}\\phi_n(z)\\,
\\delta n`. A Gaussian of width :math:`\\sigma` at :math:`\\mathbf R` (with its periodic images) has
:math:`\\hat n = e^{-i\\mathbf G\\cdot\\mathbf R_\\parallel}e^{-G^2\\sigma^2/2}\\,\\gamma(n)`: the in-plane factor is analytic,
and the sine coefficients :math:`\\gamma(n)` of its profile along :math:`z` are computed on a fine grid. Near a plane the
profile is the Gaussian minus its mirror image across the plane, renormalised to one electron: it vanishes at the
planes, so the sine series converges spectrally. The interaction of two Gaussians is then

.. math::

    P_{AB} = \\frac{4\\pi}{A}\\sum_{\\mathbf G} e^{i\\mathbf G\\cdot(\\mathbf R_A - \\mathbf R_B)}
    e^{-G^2(\\sigma_A^2 + \\sigma_B^2)/2}\\sum_n \\frac{\\gamma_A(n)\\gamma_B(n)}{G^2 + k_n^2},

exact in the plane (periodic images included) and converged along :math:`z` with the grid. The :math:`\\mathbf G = 0`
term is the planar kernel smoothed by the profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dst, next_fast_len

HARTREE_EV = 27.211386245988
GAUSSIAN_TAIL = 5.26  # sqrt(-ln 1e-12): exp(-x**2) < 1e-12 beyond this


@dataclass(frozen=True)
class DirichletSlab:
    """Transverse cell and electrode planes.

    Attributes
    ----------
    in_plane_vectors : ndarray, shape ``(2, 2)``
        :math:`\\mathbf a_1, \\mathbf a_2` as rows, in the in-plane Cartesian coordinates (bohr).
    lower, upper : float
        Positions of the electrode planes along the transport axis (bohr), :math:`a < b`.
    """

    in_plane_vectors: NDArray[np.float64]
    lower: float
    upper: float

    @property
    def length(self) -> float:
        return self.upper - self.lower

    @property
    def area(self) -> float:
        return abs(float(np.linalg.det(self.in_plane_vectors)))

    @property
    def reciprocal_vectors(self) -> NDArray[np.float64]:
        """:math:`\\mathbf b_i` as rows, :math:`\\mathbf b_i\\cdot\\mathbf a_j = 2\\pi\\delta_{ij}` (1/bohr)."""
        return 2.0 * np.pi * np.linalg.inv(self.in_plane_vectors).T


def in_plane_wavevectors(
    slab: DirichletSlab, cutoff: float, half_plane: bool = True
) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
    """Reciprocal vectors of the transverse cell inside a sphere.

    Parameters
    ----------
    slab : DirichletSlab
    cutoff : float
        :math:`|\\mathbf G| \\le` cutoff (1/bohr).
    half_plane : bool, optional
        Keep one of every pair :math:`\\pm\\mathbf G` (multiplicity 2) and :math:`\\mathbf G = 0` (multiplicity 1);
        otherwise every vector with multiplicity 1.

    Returns
    -------
    indices : ndarray, shape ``(n, 2)``, int
        :math:`(m_1, m_2)`, :math:`\\mathbf G = m_1\\mathbf b_1 + m_2\\mathbf b_2`.
    vectors : ndarray, shape ``(n, 2)``
    multiplicity : ndarray, shape ``(n,)``
    """
    reciprocal = slab.reciprocal_vectors
    bounds = [
        int(np.floor(cutoff * np.linalg.norm(vector) / (2.0 * np.pi))) + 1
        for vector in slab.in_plane_vectors
    ]
    m1, m2 = np.meshgrid(
        np.arange(-bounds[0], bounds[0] + 1), np.arange(-bounds[1], bounds[1] + 1), indexing='ij'
    )
    indices = np.column_stack([m1.ravel(), m2.ravel()])
    vectors = indices @ reciprocal
    keep = np.einsum('ij,ij->i', vectors, vectors) <= cutoff**2 * (1.0 + 1e-12)
    if half_plane:
        keep &= (indices[:, 0] > 0) | ((indices[:, 0] == 0) & (indices[:, 1] >= 0))
    indices, vectors = indices[keep], vectors[keep]
    multiplicity = np.ones(len(indices))
    if half_plane:
        multiplicity[np.any(indices != 0, axis=1)] = 2.0
    return indices, vectors, multiplicity


class GaussianChargesInSlab:
    """Gaussian charges between two grounded planes, periodic across the transverse cell.

    Parameters
    ----------
    slab : DirichletSlab
    in_plane_positions : ndarray, shape ``(n_charges, 2)``
        In-plane coordinates of the centres (bohr).
    heights : ndarray, shape ``(n_charges,)``
        Positions along the transport axis (bohr), strictly between the planes.
    widths : ndarray, shape ``(n_charges,)``
        :math:`\\sigma` (bohr).
    sampling : float, optional
        Grid spacing along the transport axis in units of the smallest width (default 1/8).

    Raises
    ------
    ValueError
        For a centre outside the slab or a non-positive width.
    """

    def __init__(
        self,
        slab: DirichletSlab,
        in_plane_positions: NDArray[np.float64],
        heights: NDArray[np.float64],
        widths: NDArray[np.float64],
        sampling: float = 0.125,
    ) -> None:
        self.slab = slab
        self.in_plane_positions = np.asarray(in_plane_positions, dtype=float).reshape(-1, 2)
        self.heights = np.asarray(heights, dtype=float).ravel()
        self.widths = np.asarray(widths, dtype=float).ravel()
        if np.any(self.widths <= 0.0):
            raise ValueError('Gaussian widths must be positive')
        if np.any(self.heights <= slab.lower) or np.any(self.heights >= slab.upper):
            raise ValueError('every charge must lie strictly between the electrode planes')
        length = slab.length
        intervals = next_fast_len(int(np.ceil(length / (sampling * self.widths.min()))) + 1)
        self.grid_spacing = length / intervals
        self.grid = slab.lower + self.grid_spacing * np.arange(1, intervals)  # interior points
        self.wavenumbers = np.pi * np.arange(1, intervals) / length
        profiles = np.zeros((len(self.heights), len(self.grid)))
        for centres, sign in (
            (self.heights, 1.0),
            (2.0 * slab.lower - self.heights, -1.0),
            (2.0 * slab.upper - self.heights, -1.0),
        ):  # the Gaussian and its mirror images across the two planes: zero at the planes
            profiles += sign * np.exp(
                -0.5 * ((self.grid[None, :] - centres[:, None]) / self.widths[:, None]) ** 2
            )
        profiles /= self.grid_spacing * profiles.sum(axis=1, keepdims=True)  # one electron inside
        self.profiles = profiles
        # orthonormal sine coefficients: gamma(n) = h sum_k profile(z_k) phi_n(z_k), DST-I = 2 sum sin
        self.sine_coefficients = (
            np.sqrt(2.0 / length) * self.grid_spacing * 0.5 * dst(profiles, type=1, axis=1)
        )

    @property
    def n_charges(self) -> int:
        return len(self.heights)

    def in_plane_cutoff(self) -> float:
        """:math:`|\\mathbf G|` beyond which :math:`e^{-G^2\\sigma_{\\min}^2} < 10^{-12}` (1/bohr)."""
        return GAUSSIAN_TAIL / self.widths.min()

    def kernel(self, in_plane_cutoff: float | None = None) -> NDArray[np.float64]:
        """Interaction of the charges with grounded planes (eV per electron).

        Parameters
        ----------
        in_plane_cutoff : float, optional
            :math:`|\\mathbf G|` cutoff (1/bohr); default :meth:`in_plane_cutoff`. Zero keeps only the planar
            average.

        Returns
        -------
        ndarray, shape ``(n_charges, n_charges)``
            :math:`P_{AB}` (symmetric, positive definite).
        """
        cutoff = self.in_plane_cutoff() if in_plane_cutoff is None else float(in_plane_cutoff)
        _, vectors, multiplicity = in_plane_wavevectors(self.slab, cutoff)
        squared = np.einsum('ij,ij->i', vectors, vectors)
        shells = np.round(squared, 10)
        coefficients = self.sine_coefficients
        kernel = np.zeros((self.n_charges, self.n_charges))
        for shell in np.unique(shells):
            members = np.flatnonzero(shells == shell)
            g2 = float(squared[members[0]])
            along = (coefficients / (g2 + self.wavenumbers**2)) @ coefficients.T
            phases = np.exp(1j * self.in_plane_positions @ vectors[members].T) * np.sqrt(
                multiplicity[members]
            )
            in_plane = (phases @ phases.conj().T).real
            form = np.exp(-0.5 * g2 * self.widths**2)
            kernel += in_plane * np.outer(form, form) * along
        kernel *= 4.0 * np.pi / self.slab.area * HARTREE_EV
        return 0.5 * (kernel + kernel.T)

    def ramp_fractions(self) -> NDArray[np.float64]:
        """:math:`\\langle g_A|(z - a)/L\\rangle`: the linear Laplace solution averaged over every charge.

        Returns
        -------
        ndarray, shape ``(n_charges,)``
        """
        fraction = (self.grid - self.slab.lower) / self.slab.length
        return self.grid_spacing * self.profiles @ fraction

    def planar_average(
        self, charges: NDArray[np.float64], heights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Planar average of the potential energy of the charges (grounded planes), eV.

        Parameters
        ----------
        charges : ndarray, shape ``(n_charges,)``
            Electrons per transverse cell.
        heights : ndarray
            Positions along the transport axis (bohr), within the planes.

        Returns
        -------
        ndarray, shape like ``heights``
        """
        z = np.asarray(heights, dtype=float)
        modes = np.sqrt(2.0 / self.slab.length) * np.sin(
            np.multiply.outer(z - self.slab.lower, self.wavenumbers)
        )
        amplitude = (
            np.asarray(charges, dtype=float) @ self.sine_coefficients
        ) / self.wavenumbers**2
        return 4.0 * np.pi / self.slab.area * HARTREE_EV * (modes @ amplitude)

    def potential_on_grid(
        self,
        charges: NDArray[np.float64],
        in_plane_points: tuple[int, int],
        heights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Potential energy of the charges on a grid (grounded planes), eV.

        Parameters
        ----------
        charges : ndarray, shape ``(n_charges,)``
            Electrons per transverse cell.
        in_plane_points : tuple of int
            :math:`(N_1, N_2)`: the points :math:`(j_1/N_1)\\mathbf a_1 + (j_2/N_2)\\mathbf a_2`; raised to resolve
            every :math:`\\mathbf G` of the charges.
        heights : ndarray, shape ``(n_z,)``
            Positions along the transport axis (bohr).

        Returns
        -------
        ndarray, shape ``(N_1, N_2, n_z)``
        """
        indices, vectors, _ = in_plane_wavevectors(
            self.slab, self.in_plane_cutoff(), half_plane=False
        )
        points = [
            max(int(n), 2 * int(np.abs(indices[:, i]).max()) + 1)
            for i, n in enumerate(in_plane_points)
        ]
        z = np.asarray(heights, dtype=float)
        modes = np.sqrt(2.0 / self.slab.length) * np.sin(
            np.multiply.outer(z - self.slab.lower, self.wavenumbers)
        )
        squared = np.einsum('ij,ij->i', vectors, vectors)
        weights = (
            np.asarray(charges, dtype=float)[None, :]
            * np.exp(-1j * vectors @ self.in_plane_positions.T)
            * np.exp(-0.5 * np.outer(squared, self.widths**2))
        )  # (n_G, n_charges)
        amplitudes = (weights @ self.sine_coefficients) / (
            squared[:, None] + self.wavenumbers[None, :] ** 2
        )  # (n_G, n_modes)
        spectrum = np.zeros((points[0], points[1], len(z)), dtype=complex)
        spectrum[indices[:, 0] % points[0], indices[:, 1] % points[1], :] = amplitudes @ modes.T
        grid = np.fft.ifft2(spectrum, axes=(0, 1)).real * (points[0] * points[1])
        return 4.0 * np.pi / self.slab.area * HARTREE_EV * grid
