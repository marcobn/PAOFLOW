"""Dipole photoemission matrix elements of PAO Bloch states with a plane-wave final state.

Notes
-----
For a state :math:`|n\\mathbf k\\rangle = \\sum_i c_{i,n\\mathbf k}|\\phi_i,\\mathbf k\\rangle`
(cell-periodic gauge, :class:`~PAOFLOW.arpes.bloch.BlochHamiltonian`) and a photoelectron of
momentum :math:`\\mathbf k_f`,

.. math::

    M^\\alpha_{n\\mathbf k} = \\sum_i c_{i,n\\mathbf k}\\,\\sigma_i\\,e^{-d_i/2\\lambda}\\,
    e^{-i(\\mathbf k_f-\\mathbf k)\\cdot\\boldsymbol\\tau_i}\\,s_i
    \\sum_{l'=l_i\\pm1} B^{l'}_i(|\\mathbf k_f|)\\sum_{m'} G^\\alpha_{l_i m_i, l' m'}\\,
    Y_{l'm'}(\\hat{\\mathbf k}_f),

with :math:`s_i` the sign of the QE harmonic, :math:`B` the radial integrals
(:mod:`~PAOFLOW.arpes.radial`), :math:`G` the dipole Gaunt coefficients
(:mod:`~PAOFLOW.arpes.harmonics`), :math:`\\sigma_i` the cross-section factor and
:math:`d_i / \\lambda` the depth over the escape depth (chinook ``M_compute``, vectorised over
k-points and bands).  The position phase is the gauge term that chinook absorbs into its
atomic-gauge Bloch sums; it is unity for :math:`\\mathbf k_f = \\mathbf k`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .basis import PhotoemissionBasis
from .harmonics import dipole_gaunt_table, qe_harmonic, real_harmonics
from .radial import BOHR_ANGSTROM, RadialIntegrals


def dipole_matrix_elements(
    basis: PhotoemissionBasis,
    eigenvectors: NDArray[np.complex128],
    k_final: NDArray[np.float64],
    radial: RadialIntegrals,
    escape_depth: float | None = None,
    k_hamiltonian: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """``M[k, n, s, alpha]`` for a block of k-points and bands.

    Parameters
    ----------
    basis : PhotoemissionBasis
    eigenvectors : ndarray ``(nk, nawf, nb)``, complex
        Columns are the states (cell-periodic gauge).
    k_final : ndarray ``(nk, nb, 3)``
        Photoelectron momenta (1/Å); rows of zero norm are states that do not emit.
    radial : RadialIntegrals
        Radial integrals keyed by :attr:`Orbital.radial_key`.
    escape_depth : float, optional
        ``lambda`` (Å) of the attenuation ``e^{-d_i / 2 lambda}``; ``None`` disables it.
    k_hamiltonian : ndarray ``(nk, 3)``, optional
        Momenta at which the states were computed (1/Å); when given, the gauge phase
        ``e^{-i (k_f - k).tau_i}`` is applied (``None``: chinook's atomic-gauge convention, no
        phase).

    Returns
    -------
    ndarray ``(nk, nb, nspin, 3)``, complex
        ``nspin = 1`` for a spinless basis, 2 (up, down) for a spinor basis; last axis ``x, y, z``.
    """
    c = np.asarray(eigenvectors)
    nk, nawf, nb = c.shape
    if nawf != basis.size:
        raise ValueError('eigenvectors have %d rows, basis %d orbitals' % (nawf, basis.size))
    kf = np.asarray(k_final, dtype=float)
    kf_norm = np.linalg.norm(kf, axis=-1)  # (nk, nb)
    valid = kf_norm > 0.0
    directions = kf / np.where(valid, kf_norm, 1.0)[..., None]
    kf_bohr = kf_norm * BOHR_ANGSTROM  # 1/bohr for the radial tables

    lprimes = sorted({lp for o in basis.orbitals for lp in (o.l - 1, o.l + 1) if lp >= 0})
    harmonics = {lp: real_harmonics(lp, directions) for lp in lprimes}  # (nk, nb, 2lp+1)

    nspin = 2 if basis.is_spinor else 1
    M = np.zeros((nk, nb, nspin, 3), dtype=complex)
    for o in basis.orbitals:
        radial_terms = {}
        for lp in (o.l - 1, o.l + 1):
            if lp >= 0:
                radial_terms[lp] = radial(o.radial_key, lp, kf_bohr)  # (nk, nb)
        prefactor = o.cross_section
        if escape_depth is not None and o.depth != 0.0:
            prefactor *= np.exp(-abs(o.depth) / (2.0 * escape_depth))
        phase = c[:, o.index, :] * prefactor  # (nk, nb)
        if k_hamiltonian is not None:
            dk = kf - np.asarray(k_hamiltonian, dtype=float)[:, None, :]  # (nk, nb, 3)
            phase = phase * np.exp(-1j * (dk @ o.position))
        for m_qe, spin, coefficient in o.terms:
            m_std, sign = qe_harmonic(o.l, m_qe)
            im = m_std + o.l
            angular = np.zeros((nk, nb, 3), dtype=complex)
            for lp, B in radial_terms.items():
                gaunt = dipole_gaunt_table(o.l, lp)[:, im, :]  # (3, 2lp+1)
                angular += B[..., None] * np.einsum('ap,knp->kna', gaunt, harmonics[lp])
            spin_index = 0 if spin >= 0 else 1
            M[:, :, spin_index, :] += (phase * (sign * coefficient))[..., None] * angular
    M[~valid] = 0.0
    return M


def polarization_weights(
    M: NDArray[np.complex128], polarization: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """``sum_s |eps . M[k, n, s]|^2`` for a polarisation vector ``eps`` (Cartesian, complex)."""
    eps = np.asarray(polarization, dtype=complex)
    amplitude = np.einsum('knsa,a->kns', M, eps)
    return np.sum(np.abs(amplitude) ** 2, axis=-1)
