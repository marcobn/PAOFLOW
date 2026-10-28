"""Augmented overlap of ultrasoft and PAW pseudopotentials for the internal projections.

The pseudo-wavefunctions of an ultrasoft or PAW calculation are orthonormal in the metric

.. math::

    S = 1 + \\sum_{a}\\sum_{ij} q^{a}_{ij}\\,|\\beta^{a}_i\\rangle\\langle\\beta^{a}_j|,

not in the plain plane-wave one, so the projection of a Bloch state on an atomic orbital is
:math:`\\langle\\phi|S|\\psi\\rangle`, the Gram matrix of the atomic orbitals is
:math:`\\langle\\phi|S|\\phi\\rangle` (what ``projwfc.x`` uses and writes with
``lwrite_overlaps``) and the bands must not be re-orthonormalised. Re-orthonormalising the
truncated set of ``nbnd`` pseudo-wavefunctions with the plain overlap, the former workaround of
:func:`PAOFLOW.projection.do_atwfc_proj.read_QE_wfc`, mixes the bands where the ``nbnd`` cut
crosses a multiplet and breaks the point symmetry of the PAO Hamiltonian at those k-points
(GaAs, October 2026: 0.65 eV at 16 of 4096 grid points, a 5-25 meV spread of the interpolated
eigenvalues between equivalent k and a 2-5 % violation of the tensor symmetry of the
second-order optical responses).

The projectors :math:`\\beta^{a}_i(\\mathbf k+\\mathbf G)` and the integrated augmentation charges
:math:`q^{a}_{ij}` (``PP_Q`` of the UPF) come from
:class:`PAOFLOW.elphon.pseudopotential_terms.KleinmanBylanderProjectors`.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

from ..elphon.pseudopotential_terms import KleinmanBylanderProjectors
from ..inputs.read_upf import UPF


def k_plus_g_vectors(gkspace: dict) -> NDArray[np.float64]:
    """``k + G`` of a wavefunction block (bohr^-1), as built by ``calc_atwfc_k``.

    Parameters
    ----------
    gkspace : dict
        G-space descriptor of :func:`PAOFLOW.projection.do_atwfc_proj.read_QE_wfc`
        (``'xk'``, ``'mill'``, ``'bg'`` in bohr^-1).

    Returns
    -------
    ndarray ``(npw, 3)``
    """
    k_plus_g = np.dot(gkspace['mill'].T, gkspace['bg'].T)
    return k_plus_g + np.asarray(gkspace['xk'], dtype=float)[None, :]


def is_augmented(upf: UPF) -> bool:
    """Whether a UPF carries augmentation charges (ultrasoft or PAW)."""
    return getattr(upf, 'ptype', 'NC') != 'NC' or bool(getattr(upf, 'has_augmentation', False))


class AugmentedOverlap:
    """The operator ``S - 1`` of a cell: projectors at ``k + G`` and the charges ``q_ij``.

    Parameters
    ----------
    kb : KleinmanBylanderProjectors
        Built with ``require_norm_conserving=False``; its :attr:`Q` is the block matrix of the
        integrated augmentation charges.
    tau_alat : ndarray ``(nat, 3)``
        Atomic positions (Cartesian, alat units).
    tpiba : float
        ``2 pi / alat`` (bohr^-1).
    """

    def __init__(self, kb: KleinmanBylanderProjectors, tau_alat: NDArray[np.float64], tpiba: float):
        if kb.Q is None:
            raise ValueError('the projectors carry no augmentation charges')
        self.kb = kb
        self.Q = np.asarray(kb.Q, dtype=float)
        self.tau_alat = np.asarray(tau_alat, dtype=float)
        self.tpiba = float(tpiba)

    @property
    def nproj(self) -> int:
        """Number of projectors (all atoms, all ``m``)."""
        return self.kb.nproj

    def beta(self, k_plus_g: NDArray[np.float64]) -> NDArray[np.complex128]:
        """``beta_p(k+G)`` on one sphere.

        Parameters
        ----------
        k_plus_g : ndarray ``(npw, 3)``
            ``k + G`` (Cartesian, bohr^-1).

        Returns
        -------
        ndarray ``(nproj, npw)``, complex
        """
        return self.kb.at(np.asarray(k_plus_g, dtype=float) / self.tpiba, self.tau_alat, self.tpiba)

    def overlap(
        self,
        a: NDArray[np.complex128],
        b: NDArray[np.complex128],
        beta: NDArray[np.complex128],
        becp_a: NDArray[np.complex128] | None = None,
        becp_b: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """``<a_i|S|b_j>`` for two sets of functions on the same sphere.

        Parameters
        ----------
        a, b : ndarray ``(na, npw)``, ``(nb, npw)``
            Plane-wave coefficients (rows = functions).
        beta : ndarray ``(nproj, npw)``
            :meth:`beta` on the same sphere.
        becp_a, becp_b : ndarray, optional
            ``<beta_p|a_i>`` ``(nproj, na)`` and ``<beta_p|b_j>`` ``(nproj, nb)`` when already
            available.

        Returns
        -------
        ndarray ``(na, nb)``, complex
        """
        if becp_a is None:
            becp_a = beta.conj() @ a.T
        if becp_b is None:
            becp_b = beta.conj() @ b.T
        return a.conj() @ b.T + becp_a.conj().T @ self.Q @ becp_b


def build_augmentation(data_controller) -> AugmentedOverlap | None:
    """The augmented overlap of the run, or ``None`` when the plain overlap is the right one.

    Parameters
    ----------
    data_controller : DataController
        Required arrays: ``species``, ``atoms``, ``tau``; attributes ``dft``, ``fpath``,
        ``ecutwfc``, ``omega``, ``alat``, ``dftSO``.

    Returns
    -------
    AugmentedOverlap or None
        ``None`` for VASP input, when every species is norm-conserving, or with spin-orbit
        coupling (the spinor augmentation is not implemented: the plain-overlap path with the
        band re-orthonormalisation is kept there, with a warning).
    """
    arry, attr = data_controller.data_dicts()
    if attr.get('dft', 'QE') != 'QE':
        return None
    upf_by_species = {
        name: UPF(os.path.join(attr['fpath'], pseudo)) for name, pseudo in arry['species']
    }
    if not any(is_augmented(upf) for upf in upf_by_species.values()):
        return None
    if attr.get('dftSO', False):
        if getattr(data_controller, 'rank', 0) == 0:
            print(
                'WARNING: ultrasoft / PAW pseudopotentials with spin-orbit coupling: the '
                'augmented overlap of the projections is not implemented, the bands are '
                're-orthonormalised with the plain overlap (approximate, breaks the point '
                'symmetry at some k-points). Use projwfc.x projections (read_atomic_proj_QE) '
                'or norm-conserving pseudopotentials.'
            )
        return None
    qmax = np.sqrt(float(attr['ecutwfc'])) + 1.0
    kb = KleinmanBylanderProjectors(
        upf_by_species,
        list(arry['atoms']),
        float(attr['omega']),
        qmax,
        require_norm_conserving=False,
    )
    alat = float(attr['alat'])
    return AugmentedOverlap(kb, np.asarray(arry['tau'], dtype=float) / alat, 2.0 * np.pi / alat)
