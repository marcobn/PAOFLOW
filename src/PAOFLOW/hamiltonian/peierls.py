r"""Orbital coupling of a uniform magnetic field to the PAO Hamiltonian: Peierls phases.

Every hopping :math:`\langle i,\mathbf 0|H|j,\mathbf R\rangle` acquires the phase of the vector potential
along the straight line between the orbital centres (electron charge :math:`-e`),

.. math::

    h_{ij}(\mathbf R) \to h_{ij}(\mathbf R)\,
    \exp\Bigl[\frac{ie}{\hbar}\int_{\boldsymbol\tau_i}^{\boldsymbol\tau_j + \mathbf R}\mathbf A\cdot d\mathbf l\Bigr].

The gauge is the Landau gauge of the plane :math:`(\mathbf a_p, \mathbf a_q)`,
:math:`(e/\hbar)\,\mathbf A\cdot d\mathbf r = 2\pi f\,s_p\,ds_q` in crystal coordinates, with :math:`f` the
flux through the cell :math:`\mathbf a_p\times\mathbf a_q` in units of :math:`\Phi_0 = h/e`; the field points
along the third lattice vector :math:`\mathbf a_r`. :math:`\mathbf A` does not depend on :math:`s_q`, so the
Hamiltonian stays periodic along :math:`\mathbf a_q`; along :math:`\mathbf a_p` the magnetic translation
adds :math:`-2\pi f R_p s_{q,j}` to the phase, which is consistent (and the Hamiltonian Hermitian) when
:math:`f` is an integer: the cell must be a magnetic supercell (``doubling_Hamiltonian``), or open along
:math:`\mathbf a_p` (a ribbon or a flake, ``cutting_Hamiltonian``), where any flux is allowed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .magnetic_field import FLUX_QUANTUM_TESLA_BOHR2
from .orbital_centres import orbital_centres_bohr

if TYPE_CHECKING:
    from ..DataController import DataController


def flux_quanta_per_cell(
    field_tesla: float, a_vectors: np.ndarray, alat: float, plane: tuple[int, int] = (0, 1)
) -> float:
    """Flux through the cell ``a_p x a_q`` of a field ``field_tesla`` along ``a_r``, in units of h/e.

    Parameters
    ----------
    field_tesla : float
        Field along the unit vector of :math:`\\mathbf a_r` (T).
    a_vectors : ndarray, shape ``(3, 3)``
        Lattice vectors (rows, alat units).
    alat : float
        Lattice parameter (bohr).
    plane : (int, int)
        The lattice vectors :math:`(p, q)` that span the plane of the orbits.

    Returns
    -------
    float
        :math:`f = B\\,V/(|\\mathbf a_r|\\,\\Phi_0)`, :math:`V` the signed cell volume of
        :math:`(\\mathbf a_p, \\mathbf a_q, \\mathbf a_r)`.
    """
    p, q = plane
    r = 3 - p - q
    vectors = np.asarray(a_vectors, dtype=float) * alat
    volume = float(np.dot(vectors[r], np.cross(vectors[p], vectors[q])))
    return float(field_tesla) * volume / (np.linalg.norm(vectors[r]) * FLUX_QUANTUM_TESLA_BOHR2)


def peierls_phases(
    data_controller: DataController, flux_quanta: float, plane: tuple[int, int] = (0, 1)
) -> None:
    """Multiply every hopping of ``HRs`` by its Peierls phase.

    Parameters
    ----------
    data_controller : DataController
        ``HRs``, ``a_vectors``, ``alat`` and the orbital centres
        (:func:`PAOFLOW.hamiltonian.orbital_centres.orbital_centres_bohr`).
    flux_quanta : float
        Flux through the cell ``a_p x a_q`` in units of :math:`\\Phi_0 = h/e`.
    plane : (int, int), optional
        The lattice vectors :math:`(p, q)` of the Landau gauge; the field is along the third.

    Returns
    -------
    None
        ``HRs`` modified in place.

    Raises
    ------
    ValueError
        If the cell is periodic along :math:`\\mathbf a_p` (more than one cell of ``HRs`` along it)
        and ``flux_quanta`` is not an integer.

    Notes
    -----
    ``HRs[..., n]`` holds :math:`h(\\mathbf R)` at :math:`\\mathbf R = -\\mathbf n` (PAOFLOW's Fourier
    convention). The phase of :math:`h_{ij}(\\mathbf R)` is

    .. math::

        2\\pi f\\Bigl[\\frac{s_{p,i} + s_{p,j} + R_p}{2}\\,(s_{q,j} + R_q - s_{q,i}) - R_p\\,s_{q,j}\\Bigr],

    the line integral of :math:`\\mathbf A` and the magnetic-translation factor. On an even grid the
    block at :math:`n = N/2` stands for :math:`\\pm N/2` together and takes the phase of :math:`R = -N/2`;
    model hopping lists leave it empty, DFT Hamiltonians hold only their weakest hoppings there.
    """
    arrays, attributes = data_controller.data_dicts()
    p, q = plane
    if p == q or not {p, q} <= {0, 1, 2}:
        raise ValueError(f'plane must hold two different lattice directions, got {plane}')
    hrs = np.array(arrays['HRs'], dtype=complex)
    grid = hrs.shape[2:5]
    flux = float(flux_quanta)
    if grid[p] > 1 and abs(flux - round(flux)) > 1e-9:
        raise ValueError(
            f'{flux:g} flux quanta per cell, periodic along a_{p + 1}: make a magnetic supercell '
            '(doubling_Hamiltonian) holding an integer number of quanta, or open the boundary along '
            f'a_{p + 1} (cutting_Hamiltonian)'
        )
    centres = orbital_centres_bohr(arrays, attributes) / float(attributes['alat'])
    fractional = centres @ np.linalg.inv(np.asarray(arrays['a_vectors'], dtype=float))
    sp, sq = fractional[:, p], fractional[:, q]
    for index in np.ndindex(*grid):
        signed = np.array([i if i <= n // 2 else i - n for i, n in zip(index, grid)])
        cell = -signed  # HRs[..., n] holds h(R) at R = -n
        rp, rq = cell[p], cell[q]
        phase = (
            2.0
            * np.pi
            * flux
            * (
                0.5 * (sp[:, None] + sp[None, :] + rp) * (sq[None, :] + rq - sq[:, None])
                - rp * sq[None, :]
            )
        )
        hrs[:, :, index[0], index[1], index[2], :] *= np.exp(1j * phase)[:, :, None]
    arrays['HRs'] = hrs
