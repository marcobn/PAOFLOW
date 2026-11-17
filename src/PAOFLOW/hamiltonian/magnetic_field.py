r"""Magnetic fields in the PAO Hamiltonian: the spinor basis and the on-site Zeeman coupling.

The on-site coupling of an electron to a uniform field :math:`\mathbf B` is

.. math::

    H_Z = \mu_B\,(\mathbf L + g_s\mathbf S)\cdot\mathbf B ,

with :math:`\mathbf L` and :math:`\mathbf S = \boldsymbol\sigma/2` in units of :math:`\hbar` and
:math:`g_s` the electron spin g-factor: the magnetic moment :math:`-\mu_B(\mathbf L + g_s\mathbf S)` lines
up with the field. :math:`\mathbf L` is the atomic orbital angular momentum of each PAO shell (the
intra-atomic, Van Vleck and orbital-Zeeman part); the coupling of the itinerant motion to the vector
potential is in :mod:`PAOFLOW.hamiltonian.peierls`, the orbital moment of the Bloch states in
:mod:`PAOFLOW.response.orbital_moment`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..DataController import DataController

BOHR_MAGNETON_EV_PER_TESLA = 5.7883818060e-5
ELECTRON_SPIN_G_FACTOR = 2.00231930436256
FLUX_QUANTUM_TESLA_BOHR2 = 4.135667696e-15 / 5.29177210903e-11**2  # h / e in T bohr^2

PAULI = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def spinor_basis(data_controller: DataController) -> None:
    """Rewrite ``HRs`` in block-ordered spinors (all orbitals spin up, then all spin down).

    Parameters
    ----------
    data_controller : DataController
        ``HRs`` of a spin-degenerate run (``nspin = 1`` without spin-orbit coupling) or of a
        collinear one (``nspin = 2``).

    Returns
    -------
    None
        ``HRs`` becomes ``diag(H, H)`` or ``diag(H_up, H_dn)`` with ``nspin = 1``; ``nawf`` and
        ``bnd`` double; ``Dnm`` and, if present, ``Hks`` and ``Lj`` are doubled the same way;
        ``Sj`` is :math:`\\sigma/2\\otimes 1`, ``orbital_centres`` the centres twice; ``dftSO`` and ``adhoc_SO`` are set (spinor bands
        hold one electron each, and the spin and orbital operators take the block order), and
        ``spinor_basis`` records the change.

    Raises
    ------
    ValueError
        If the Hamiltonian is already spinor (``dftSO``).

    Notes
    -----
    The doubling of ``adhoc_spin_orbit`` without the spin-orbit term: a magnetic field can then
    point in any direction. ``nelec`` is unchanged.
    """
    from .orbital_centres import orbital_centres_bohr

    arrays, attributes = data_controller.data_dicts()
    if attributes.get('dftSO', False):
        raise ValueError('the Hamiltonian is already in a spinor basis')
    try:
        centres = orbital_centres_bohr(arrays, attributes)
    except (KeyError, ValueError):
        centres = None
    hrs = np.asarray(arrays['HRs'])
    nawf, nspin = hrs.shape[0], hrs.shape[-1]
    doubled = np.zeros((2 * nawf, 2 * nawf, *hrs.shape[2:5], 1), dtype=complex)
    doubled[:nawf, :nawf, ..., 0] = hrs[..., 0]
    doubled[nawf:, nawf:, ..., 0] = hrs[..., nspin - 1]
    arrays['HRs'] = doubled
    if 'Hks' in arrays:
        arrays['Hks'] = np.fft.fftn(doubled, axes=(2, 3, 4))
    if 'Dnm' in arrays:
        arrays['Dnm'] = np.tile(np.asarray(arrays['Dnm']), (2, 2, 1))
    if 'Lj' in arrays:
        arrays['Lj'] = np.stack([np.kron(np.eye(2), np.asarray(arrays['Lj'])[a]) for a in range(3)])
    arrays['Sj'] = 0.5 * np.stack([np.kron(sigma, np.eye(nawf)) for sigma in PAULI])
    if centres is not None:
        arrays['orbital_centres'] = np.vstack([centres, centres])
    attributes['nawf'] = 2 * nawf
    if 'bnd' in attributes:
        attributes['bnd'] = 2 * int(attributes['bnd'])
    attributes['nspin'] = 1
    attributes['dftSO'] = attributes['adhoc_SO'] = True
    attributes['spinor_basis'] = f'from nspin = {nspin}'


def zeeman_matrix(
    arrays: dict,
    field_tesla: NDArray[np.float64],
    g_s: float = ELECTRON_SPIN_G_FACTOR,
    spin: bool = True,
    orbital: bool = True,
) -> NDArray[np.complex128]:
    """The on-site term :math:`\\mu_B(\\mathbf L + g_s\\mathbf S)\\cdot\\mathbf B` (eV), ``(nawf, nawf, nspin)``.

    Parameters
    ----------
    arrays : dict
        ``HRs`` (for the shape); ``Sj`` (spinor bases, ``nspin = 1``) and, with ``orbital``, ``Lj``,
        both ``(3, nawf, nawf)`` in units of :math:`\\hbar`. Collinear runs (``nspin = 2``) take
        ``Lj`` of one channel and only a field along z.
    field_tesla : array_like, shape ``(3,)``
        Cartesian field (T).
    g_s : float, optional
        Spin g-factor (the free-electron value by default).
    spin, orbital : bool, optional
        Include the spin and the orbital term.

    Returns
    -------
    ndarray, shape ``(nawf, nawf, nspin)``

    Raises
    ------
    ValueError
        For a transverse field on a collinear run, or a missing operator.
    """
    field = np.asarray(field_tesla, dtype=float).reshape(3)
    hrs = arrays['HRs']
    nawf, nspin = hrs.shape[0], hrs.shape[-1]
    energy = BOHR_MAGNETON_EV_PER_TESLA * field  # mu_B B (eV)
    term = np.zeros((nawf, nawf, nspin), dtype=complex)
    if orbital:
        if 'Lj' not in arrays:
            raise ValueError('the orbital term needs the orbital angular momentum Lj')
        orbital_operator = np.asarray(arrays['Lj'])
        if orbital_operator.shape != (3, nawf, nawf):
            raise ValueError(f'Lj has shape {orbital_operator.shape}, the Hamiltonian {nawf} orbitals')
    if nspin == 2:
        if np.any(field[:2] != 0.0):
            raise ValueError(
                'a collinear run (nspin = 2) takes only a field along z, its spin axis: call '
                'spinor_basis() first for a transverse field'
            )
        for channel, sign in ((0, 1.0), (1, -1.0)):
            if spin:
                term[:, :, channel] += sign * 0.5 * g_s * energy[2] * np.eye(nawf)
            if orbital:
                term[:, :, channel] += energy[2] * orbital_operator[2]
    else:
        if spin:
            if 'Sj' not in arrays:
                raise ValueError('the spin term needs the spin operator Sj (a spinor basis)')
            term[:, :, 0] += g_s * np.tensordot(energy, np.asarray(arrays['Sj']), axes=(0, 0))
        if orbital:
            term[:, :, 0] += np.tensordot(energy, orbital_operator, axes=(0, 0))
    return term


def add_onsite(data_controller: DataController, term: NDArray[np.complex128]) -> None:
    """Add an ``(nawf, nawf, nspin)`` term to the R = 0 block of ``HRs``."""
    arrays = data_controller.data_arrays
    hrs = arrays['HRs']
    shape = hrs.shape
    flat = np.reshape(np.asarray(hrs, dtype=complex), (shape[0], shape[1], -1, shape[-1]), order='C')
    flat[:, :, 0, :] += term
    arrays['HRs'] = np.reshape(flat, shape, order='C')


def onsite_magnetic_field(
    data_controller: DataController,
    field_tesla: NDArray[np.float64],
    g_s: float = ELECTRON_SPIN_G_FACTOR,
    spin: bool = True,
    orbital: bool = True,
) -> NDArray[np.complex128]:
    """Add :math:`\\mu_B(\\mathbf L + g_s\\mathbf S)\\cdot\\mathbf B` (:func:`zeeman_matrix`) to ``HRs``.

    Returns
    -------
    ndarray, shape ``(nawf, nawf, nspin)``
        The term added (eV), per spin channel.
    """
    term = zeeman_matrix(data_controller.data_arrays, field_tesla, g_s, spin, orbital)
    add_onsite(data_controller, term)
    return term
