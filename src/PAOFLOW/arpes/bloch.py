"""Bloch Hamiltonian of the PAO model at arbitrary k, in PAOFLOW's conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..elphon.elph_bloch import _ws_lattice
from .radial import BOHR_ANGSTROM


@dataclass
class BlochHamiltonian:
    """``H(k) = sum_n W_n H(R_n) e^{-2 pi i k.n}`` from the real-space PAO Hamiltonian.

    Parameters
    ----------
    hn : ndarray ``(nws, nawf, nawf)``, complex
        Wigner-Seitz-weighted ``H(R_n)`` (eV).
    cells : ndarray ``(nws, 3)``, int
        Integer lattice cells ``n`` of the Wigner-Seitz images.
    at : ndarray ``(3, 3)``
        Lattice vectors (rows) in units of ``alat``.
    alat_bohr : float
        Lattice parameter (bohr).
    fermi_level : float
        Energy subtracted from the eigenvalues (eV).

    Notes
    -----
    PAOFLOW stores ``HRs = ifftn(Hks)``, hence :math:`H(\\mathbf k) = \\sum_R H(R)\\,
    e^{-2\\pi i\\,\\mathbf k\\cdot R}` with ``k`` in crystal coordinates; the FFT cells are folded
    into the Wigner-Seitz cell of the supercell with degeneracy weights
    (:func:`~PAOFLOW.elphon.elph_bloch._ws_lattice`), which gives the smooth interpolation of the
    electron-phonon drivers rather than the periodic FFT grid of ``bands``.
    """

    hn: NDArray[np.complex128]
    cells: NDArray[np.int_]
    at: NDArray[np.float64]
    alat_bohr: float
    fermi_level: float = 0.0

    @classmethod
    def from_hrs(
        cls,
        HRs: NDArray[np.complex128],
        at: NDArray[np.float64],
        alat_bohr: float,
        fermi_level: float = 0.0,
        ispin: int = 0,
    ) -> BlochHamiltonian:
        """From ``HRs`` ``(nawf, nawf, n1, n2, n3[, nspin])`` on the FFT grid."""
        H = np.asarray(HRs)
        if H.ndim == 6:
            H = H[..., ispin]
        ng = H.shape[2:5]
        cells, weights, grid = _ws_lattice(ng, np.asarray(at, dtype=float))
        hn = (
            np.transpose(H[:, :, grid[:, 0], grid[:, 1], grid[:, 2]], (2, 0, 1))
            * weights[:, None, None]
        )
        return cls(
            np.ascontiguousarray(hn),
            cells,
            np.asarray(at, dtype=float),
            float(alat_bohr),
            fermi_level,
        )

    @classmethod
    def from_data_controller(
        cls, data_controller, ispin: int = 0, fermi_level: float = 0.0
    ) -> BlochHamiltonian:
        """From a PAOFLOW run after ``pao_hamiltonian``.

        PAOFLOW's band energies are already relative to the DFT Fermi level (``read_QE_xml``
        subtracts it), so ``fermi_level`` is 0 unless the Hamiltonian was shifted otherwise.
        """
        arry, attr = data_controller.data_dicts()
        return cls.from_hrs(arry['HRs'], arry['a_vectors'], attr['alat'], fermi_level, ispin)

    @property
    def nawf(self) -> int:
        """Number of basis functions."""
        return self.hn.shape[1]

    @property
    def lattice_angstrom(self) -> NDArray[np.float64]:
        """Lattice vectors (rows) in Å."""
        return self.at * self.alat_bohr * BOHR_ANGSTROM

    def crystal_coordinates(self, k_cart_inv_angstrom: NDArray[np.float64]) -> NDArray[np.float64]:
        """Crystal coordinates ``k.a_i / 2 pi`` of Cartesian momenta (1/Å)."""
        return (
            np.asarray(k_cart_inv_angstrom, dtype=float) @ self.lattice_angstrom.T / (2.0 * np.pi)
        )

    def hamiltonian(self, k_cryst: NDArray[np.float64]) -> NDArray[np.complex128]:
        """``H(k)`` ``(nk, nawf, nawf)`` at crystal-coordinate k-points ``(nk, 3)``, hermitised."""
        k = np.atleast_2d(np.asarray(k_cryst, dtype=float))
        phases = np.exp(-2j * np.pi * (k @ self.cells.T))  # (nk, nws)
        Hk = (phases @ self.hn.reshape(self.hn.shape[0], -1)).reshape(
            k.shape[0], self.nawf, self.nawf
        )
        return 0.5 * (Hk + np.conjugate(np.transpose(Hk, (0, 2, 1))))

    def eigenstates(
        self, k_cryst: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Band energies relative to the Fermi level ``(nk, nawf)`` and eigenvectors ``(nk, nawf, nawf)``."""
        energies, vectors = np.linalg.eigh(self.hamiltonian(k_cryst))
        return energies - self.fermi_level, vectors
