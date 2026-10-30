"""Bloch Hamiltonian and its first and second k-derivatives at arbitrary k from ``HRs``.

The real-space PAO Hamiltonian ``HRs`` of :meth:`PAOFLOW.pao_hamiltonian` defines the lattice-gauge
Bloch Hamiltonian

.. math::

    H(\\mathbf k) = \\sum_{\\mathbf n} e^{-i\\mathbf k\\cdot\\mathbf R_{\\mathbf n}} H(\\mathbf n),
    \\qquad \\mathbf R_{\\mathbf n} = a\\,(\\mathbf n\\cdot\\mathbf a_i),

with the pair-resolved Wigner-Seitz images of the real-space cells (as the electron-phonon
interpolation of :mod:`PAOFLOW.elphon.elph_bloch`).  Its k-derivatives are explicit sums,

.. math::

    \\partial_a H = \\sum_{\\mathbf n} (-iR^a_{\\mathbf n})\\,e^{-i\\mathbf k\\cdot\\mathbf R_{\\mathbf n}} H(\\mathbf n),
    \\qquad
    \\partial_a\\partial_b H = \\sum_{\\mathbf n} (-R^a_{\\mathbf n}R^b_{\\mathbf n})\\,
    e^{-i\\mathbf k\\cdot\\mathbf R_{\\mathbf n}} H(\\mathbf n).

The velocity and the "curvature" operators of the atomic-gauge Hamiltonian
:math:`H^A_{ij} = e^{i\\mathbf k\\cdot(\\boldsymbol\\tau_j-\\boldsymbol\\tau_i)}H_{ij}` (the position
operator diagonal on the orbital centres :math:`\\boldsymbol\\tau_i`) are, expressed in the lattice
gauge,

.. math::

    v_a = \\partial_a H + i\\,\\Delta\\tau_a\\circ H,\\qquad
    w_{ab} = \\partial_a\\partial_b H + i\\,\\Delta\\tau_b\\circ\\partial_a H
    + i\\,\\Delta\\tau_a\\circ\\partial_b H - \\Delta\\tau_a\\Delta\\tau_b\\circ H,

with :math:`(\\Delta\\tau_a)_{ij} = (\\tau_j-\\tau_i)_a` and :math:`\\circ` the element-wise product;
their band-basis matrices :math:`U^\\dagger v_a U` equal those of :math:`\\partial_a H^A` and
:math:`\\partial_a\\partial_b H^A` in the eigenbasis of :math:`H^A`.  Units: eV, bohr, 1/bohr;
``hbar = 1`` (``v`` is :math:`\\hbar\\mathbf v` in eV bohr, ``w`` in eV bohr^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..elphon.elph_bloch import _ws_lattice, _ws_lattice_pairs

#: Index pairs of the six independent second derivatives.
PAIRS_AB = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@dataclass
class BlochTerms:
    """Real-space ingredients of the Bloch Hamiltonian and its derivatives.

    Attributes
    ----------
    cells : ndarray ``(nws, 3)``, int
        Integer cells ``n`` of the Wigner-Seitz images (phase ``exp(-2 pi i k.n)``).
    cell_vectors_bohr : ndarray ``(nws, 3)``
        Cartesian ``R_n`` (bohr).
    hamiltonian_cells : ndarray ``(nws, nawf*nawf)``, complex
        Wigner-Seitz weighted ``H(n)`` (eV), flattened over the orbital pair.
    separation_bohr : ndarray ``(3, nawf, nawf)``
        ``tau_j - tau_i`` (bohr); zero without orbital positions.
    nawf : int
    """

    cells: NDArray[np.int_]
    cell_vectors_bohr: NDArray[np.float64]
    hamiltonian_cells: NDArray[np.complex128]
    separation_bohr: NDArray[np.float64]
    nawf: int


def bloch_terms_from_HRs(
    HRs: NDArray[np.complex128],
    a_vectors: NDArray[np.float64],
    alat: float,
    orbital_positions_cryst: NDArray[np.float64] | None = None,
    ispin: int = 0,
    hopping_cutoff: float | None = None,
) -> BlochTerms:
    """Wigner-Seitz real-space terms of ``HRs`` for :func:`hamiltonian_derivatives_block`.

    Parameters
    ----------
    HRs : ndarray ``(nawf, nawf, n1, n2, n3, nspin)``
        PAOFLOW's real-space Hamiltonian (eV), ``H(k) = sum_n exp(-2 pi i k.n) HRs[..., n]``.
    a_vectors : ndarray ``(3, 3)``
        Direct lattice vectors (rows, alat units).
    alat : float
        Lattice parameter (bohr).
    orbital_positions_cryst : ndarray ``(nawf, 3)``, optional
        Orbital centres in crystal coordinates (pair-resolved Wigner-Seitz cells and the
        intersite position term); ``None`` puts every orbital at the origin.
    ispin : int, optional
        Spin channel.
    hopping_cutoff : float, optional
        Drop the cells whose largest weighted ``|H(n)|`` is below this value (eV).

    Returns
    -------
    BlochTerms
    """
    H = np.asarray(HRs)[:, :, :, :, :, ispin]
    nawf = H.shape[0]
    ng = H.shape[2:5]
    at = np.asarray(a_vectors, dtype=float)
    if orbital_positions_cryst is None:
        cells, weights, grid_index = _ws_lattice(ng, at)
        Hn = np.transpose(H[:, :, grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]], (2, 0, 1))
        Hn = Hn * weights[:, None, None]
        separation = np.zeros((3, nawf, nawf))
    else:
        positions = np.asarray(orbital_positions_cryst, dtype=float)
        cells, weights, grid_index = _ws_lattice_pairs(ng, at, positions, sign=-1)
        Hn = np.transpose(H[:, :, grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]], (2, 0, 1))
        Hn = Hn * weights
        tau_cart = float(alat) * (positions @ at)  # (nawf, 3) bohr
        separation = np.transpose(tau_cart[None, :, :] - tau_cart[:, None, :], (2, 0, 1))
    if hopping_cutoff is not None:
        keep = np.max(np.abs(Hn), axis=(1, 2)) >= float(hopping_cutoff)
        cells, Hn = cells[keep], Hn[keep]
    vectors = float(alat) * (np.asarray(cells, dtype=float) @ at)
    return BlochTerms(
        cells=np.asarray(cells, dtype=int),
        cell_vectors_bohr=vectors,
        hamiltonian_cells=np.ascontiguousarray(Hn.reshape(Hn.shape[0], -1)),
        separation_bohr=separation,
        nawf=nawf,
    )


def _hermitian(block: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (block + np.conjugate(np.swapaxes(block, -1, -2)))


def hamiltonian_derivatives_block(
    terms: BlochTerms, k_cryst: NDArray[np.float64]
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """``H``, ``dH/dk_a`` and ``d^2H/dk_a dk_b`` (orbital basis) at a block of k-points.

    Parameters
    ----------
    terms : BlochTerms
    k_cryst : ndarray ``(nb, 3)``
        Crystal coordinates.

    Returns
    -------
    H : ndarray ``(nb, nawf, nawf)``
        eV, Hermitised.
    dH : ndarray ``(nb, 3, nawf, nawf)``
        eV bohr.
    d2H : ndarray ``(nb, 6, nawf, nawf)``
        eV bohr^2, the pairs of :data:`PAIRS_AB`.
    """
    k = np.atleast_2d(np.asarray(k_cryst, dtype=float))
    n = terms.nawf
    phases = np.exp(-2j * np.pi * (k @ terms.cells.T))  # (nb, nws)
    R = terms.cell_vectors_bohr
    H = _hermitian((phases @ terms.hamiltonian_cells).reshape(-1, n, n))
    dH = np.empty((k.shape[0], 3, n, n), dtype=complex)
    for a in range(3):
        dH[:, a] = _hermitian(
            ((phases * (-1j * R[:, a])[None, :]) @ terms.hamiltonian_cells).reshape(-1, n, n)
        )
    d2H = np.empty((k.shape[0], 6, n, n), dtype=complex)
    for p, (a, b) in enumerate(PAIRS_AB):
        d2H[:, p] = _hermitian(
            ((phases * (-R[:, a] * R[:, b])[None, :]) @ terms.hamiltonian_cells).reshape(-1, n, n)
        )
    return H, dH, d2H


def velocity_and_curvature_operators(
    terms: BlochTerms,
    H: NDArray[np.complex128],
    dH: NDArray[np.complex128],
    d2H: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Atomic-gauge velocity ``v_a`` and curvature ``w_ab`` operators in the lattice gauge.

    Parameters
    ----------
    terms : BlochTerms
    H, dH, d2H
        Of :func:`hamiltonian_derivatives_block`.

    Returns
    -------
    v : ndarray ``(nb, 3, nawf, nawf)``
        :math:`\\partial_a H + i\\Delta\\tau_a\\circ H` (eV bohr).
    w : ndarray ``(nb, 3, 3, nawf, nawf)``
        :math:`\\partial_a\\partial_b H + i\\Delta\\tau_b\\circ\\partial_a H + i\\Delta\\tau_a\\circ\\partial_b H
        - \\Delta\\tau_a\\Delta\\tau_b\\circ H` (eV bohr^2), symmetric in ``(a, b)``.
    """
    sep = terms.separation_bohr
    nb, n = H.shape[0], terms.nawf
    v = np.empty((nb, 3, n, n), dtype=complex)
    for a in range(3):
        v[:, a] = dH[:, a] + 1j * sep[a][None] * H
    w = np.empty((nb, 3, 3, n, n), dtype=complex)
    for p, (a, b) in enumerate(PAIRS_AB):
        block = (
            d2H[:, p]
            + 1j * sep[b][None] * dH[:, a]
            + 1j * sep[a][None] * dH[:, b]
            - (sep[a] * sep[b])[None] * H
        )
        w[:, a, b] = block
        w[:, b, a] = block
    return v, w


def orbital_positions_from_data(data_controller) -> NDArray[np.float64] | None:
    """Orbital centres (crystal coordinates) of a PAOFLOW run or a tight-binding model.

    QE runs hold the Cartesian centre of every PAO orbital in ``arrays['basis']``
    (:func:`PAOFLOW.elphon.elph_bloch.pao_orbital_positions`); tight-binding models hold the
    atom positions ``arrays['tau']`` (Cartesian, alat units) with ``arrays['naw']`` orbitals
    per atom.  Returns ``None`` when neither is available.
    """
    arrays, attributes = data_controller.data_dicts()
    at = np.asarray(arrays['a_vectors'], dtype=float)
    if 'basis' in arrays and len(arrays['basis']) and 'tau' in arrays['basis'][0]:
        from ..elphon.elph_bloch import pao_orbital_positions

        return pao_orbital_positions(data_controller, at)
    if 'tau' in arrays and 'naw' in arrays:
        tau = np.atleast_2d(np.asarray(arrays['tau'], dtype=float))
        counts = np.asarray(arrays['naw'], dtype=int)
        if counts.sum() == int(attributes.get('nawf', counts.sum())):
            per_orbital = np.repeat(tau, counts, axis=0)
            return np.linalg.solve(at.T, per_orbital.T).T
    return None


__all__ = [
    'PAIRS_AB',
    'BlochTerms',
    'bloch_terms_from_HRs',
    'hamiltonian_derivatives_block',
    'orbital_positions_from_data',
    'velocity_and_curvature_operators',
]
