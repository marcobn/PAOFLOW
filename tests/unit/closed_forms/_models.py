"""Small model Hamiltonians with closed-form properties, for the tests of this folder."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from PAOFLOW.models.models import HoppingModel

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]])
SIGMA_Z = np.diag([1.0, -1.0]).astype(complex)


def _spinful(name: str, terms: list, nelec: float, grid: tuple[int, int, int]) -> HoppingModel:
    """One orbital, two spin-orbitals (PAOFLOW's block order), on a square lattice with vacuum."""
    hoppings = []
    for cell, matrix in terms:
        for i in range(2):
            for j in range(2):
                if abs(matrix[i, j]) > 0:
                    hoppings.append((i, j, np.array(cell), matrix[i, j]))
    return HoppingModel(
        name, tuple(hoppings), 2, np.diag([1.0, 1.0, 10.0]), 5.0, np.zeros((2, 3)), nelec, grid,
        spinful=True,
    )  # fmt: skip


def rashba_model(t: float = 1.0, alpha: float = 0.3, nelec: float = 0.6) -> HoppingModel:
    r"""Square-lattice Rashba gas: :math:`-2t(\cos k_x + \cos k_y) + \alpha(\sin k_y\sigma_x - \sin k_x\sigma_y)`.

    The spin of the two bands is :math:`\mp\hat g/2`, :math:`\mathbf g = (\sin k_y, -\sin k_x, 0)`.
    """
    terms = []
    for axis in (0, 1):
        unit = tuple(np.eye(3, dtype=int)[axis])
        terms += [(unit, -t * np.eye(2)), (tuple(-np.array(unit)), -t * np.eye(2))]
    terms += [((0, 1, 0), alpha * SIGMA_X / 2j), ((0, -1, 0), -alpha * SIGMA_X / 2j)]
    terms += [((1, 0, 0), -alpha * SIGMA_Y / 2j), ((-1, 0, 0), alpha * SIGMA_Y / 2j)]
    return _spinful('rashba', terms, nelec, (3, 3, 1))


def qwz_model(m: float = 1.0, t: float = 1.0) -> HoppingModel:
    r"""Qi-Wu-Zhang Chern insulator: :math:`t(\sin k_x\sigma_x + \sin k_y\sigma_y) + (m + t\cos k_x + t\cos k_y)\sigma_z`.

    Chern number :math:`-\mathrm{sign}(m)` of the lower band for :math:`0 < |m| < 2t`, 0 for
    :math:`|m| > 2t`; one electron per cell.
    """
    terms = [((0, 0, 0), m * SIGMA_Z)]
    for axis, sigma in ((0, SIGMA_X), (1, SIGMA_Y)):
        unit = np.eye(3, dtype=int)[axis]
        terms += [(tuple(unit), t * sigma / 2j), (tuple(-unit), -t * sigma / 2j)]
        terms += [(tuple(unit), t * SIGMA_Z / 2), (tuple(-unit), t * SIGMA_Z / 2)]
    return _spinful('qwz', terms, 1.0, (3, 3, 1))


def model_run(model: HoppingModel, nfft: tuple[int, int, int] | None = None, **kw):
    """``PAOFLOW.PAOFLOW(model=...)`` in a fresh output folder (interpolated and diagonalized)."""
    from PAOFLOW import PAOFLOW

    p = PAOFLOW.PAOFLOW(model=model, outputdir=tempfile.mkdtemp(), verbose=False, **kw)
    p.data_controller.data_attributes['abort_on_exception'] = True
    if nfft is not None:
        p.interpolated_hamiltonian(nfft1=nfft[0], nfft2=nfft[1], nfft3=nfft[2])
        p.pao_eigh()
    return p


def output(p, name: str) -> np.ndarray:
    """A text output of the run."""
    return np.loadtxt(os.path.join(p.data_controller.data_attributes['opath'], name))


def bloch(model: HoppingModel, k_cart: np.ndarray, order: int = 0) -> np.ndarray:
    r"""The Bloch Hamiltonian of a model, or its ``order``-th derivative along x (atomic gauge).

    ``k_cart`` in 1/bohr; derivatives in eV bohr^order.
    """
    a, A = model.alat, model.lattice_vectors
    H = np.zeros((model.number_of_orbitals,) * 2, dtype=complex)
    for i, j, cell, h in model.hoppings:
        d = a * (np.asarray(cell) + model.orbital_positions[j] - model.orbital_positions[i]) @ A
        H[i, j] += (1j * d[0]) ** order * h * np.exp(1j * k_cart @ d)
    return H
