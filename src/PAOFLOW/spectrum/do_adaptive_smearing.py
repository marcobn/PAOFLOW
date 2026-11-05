from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..DataController import DataController

import numpy as np


def adaptive_widths(
    vel: np.ndarray,
    afac: float,
    dk: float,
    pairs: bool = False,
    axis: int = 0,
    min_width: float | None = None,
    max_width: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Yates adaptive smearing widths from band-diagonal velocities.

    Parameters
    ----------
    vel : np.ndarray
        Band-diagonal velocities (real, or complex with zero imaginary part as
        the diagonal of a Hermitized ``pksp``).  The Cartesian direction is
        ``axis`` and the band index is the axis right after it, e.g.
        ``(3, m)`` at one k-point (``axis=0``) or the dense
        ``(npks, 3, nawf, nspin)`` layout (``axis=1``).
    afac : float
        Adaptive smearing prefactor :math:`\\alpha`.
    dk : float
        Mean k-point spacing :math:`\\delta k`.
    pairs : bool, optional
        Also return the interband widths.
    axis : int, optional
        Position of the Cartesian axis in ``vel``.
    min_width, max_width : float or None, optional
        Bounds (eV) applied to ``delta`` and ``delta2``, as in
        :func:`do_adaptive_smearing`; ``None`` skips a bound.  Not validated here.

    Returns
    -------
    delta : np.ndarray
        ``vel`` with the Cartesian axis removed:
        :math:`\\alpha\\,\\delta k\\,|v_n|`.
    delta2 : np.ndarray
        Only when ``pairs``: the band axis repeated,
        :math:`\\alpha\\,\\delta k\\,|v_n - v_m|`.

    Notes
    -----
    Shared by :func:`do_adaptive_smearing` (whole k-slice) and the sparse
    mesh pass (one k-point).  ``delta`` is the norm of the real part and
    ``delta2`` the norm of the complex difference, as the dense kernel
    always computed them.
    """
    from numpy.linalg import norm

    delta = np.ascontiguousarray(norm(np.real(vel), axis=axis))
    delta *= afac * dk
    _bound_widths(delta, min_width, max_width)
    if not pairs:
        return delta
    band_axis = axis + 1
    nband = vel.shape[band_axis]
    delta2 = np.zeros(delta.shape[:axis] + (nband,) + delta.shape[axis:], dtype=float)
    for n in range(nband):
        vel_n = np.take(vel, [n], axis=band_axis)
        delta2[(slice(None),) * axis + (n,)] = norm(vel_n - vel, axis=axis)
    delta2 *= afac * dk
    _bound_widths(delta2, min_width, max_width)
    return delta, delta2


def _bound_widths(widths: np.ndarray, min_width: float | None, max_width: float | None) -> None:
    if min_width is not None:
        np.maximum(widths, min_width, out=widths)
    if max_width is not None:
        np.minimum(widths, max_width, out=widths)


def do_adaptive_smearing(
    data_controller: DataController,
    smearing: str,
    afac: float | None,
    min_width: float | None = None,
    max_width: float | None = None,
) -> None:
    """Compute adaptive smearing widths for each k-point and band.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required array: ``pksp`` (shape ``(npks, 3, nawf, nawf, nspin)``) —
        the momentum matrix elements in the Bloch eigenstate basis.
        Required attributes: ``nawf``, ``nspin``, ``nkpnts``, ``omega``.
    smearing : str
        Smearing method identifier.  Pass ``'m-p'`` for Methfessel–Paxton;
        any other value selects the default prefactor.
    afac : float or None
        Adaptive smearing prefactor :math:`\\alpha`.  If ``None``, defaults
        to ``1.0`` for ``'m-p'`` smearing and ``0.7`` otherwise.
    min_width : float or None, optional
        Lower bound of every width (eV); ``None`` (default) keeps the raw
        widths.  The Yates width vanishes at the band extrema and is below the
        resolution of any energy grid on flat bands; a state narrower than the
        grid spacing of a later DOS, PDOS or transport call drops out of the
        sums, so a floor at or above that spacing restores the state count.
    max_width : float or None, optional
        Upper bound of every width (eV); ``None`` (default) leaves the widths
        uncapped.

    Returns
    -------
    None
        Adds the following keys to ``data_controller.data_arrays``:

        - ``deltakp`` : np.ndarray, shape ``(npks, nawf, nspin)`` —
          band-resolved adaptive smearing widths
          :math:`\\sigma_{nk} = \\alpha \\, |\\nabla_k E_n| \\, \\delta k`,
          bounded by ``min_width`` and ``max_width`` when given.
        - ``deltakp2`` : np.ndarray, shape ``(npks, nawf, nawf, nspin)`` —
          interband adaptive smearing widths proportional to
          :math:`|\\nabla_k E_n - \\nabla_k E_m|`, bounded the same way.

    Raises
    ------
    ValueError
        If ``min_width`` is negative or ``max_width`` is below ``min_width``
        (or below zero).

    Notes
    -----
    The mean k-point spacing is estimated as

    .. math::

        \\delta k = \\left(\\frac{8\\pi^3}{\\Omega \\, N_k}\\right)^{1/3}

    where :math:`\\Omega` is the unit-cell volume and :math:`N_k` is the total
    number of k-points.  The diagonal elements of ``pksp`` (proportional to
    the band velocities) are used as a proxy for :math:`\\nabla_k E_n`.  With
    both bounds given the stored widths are

    .. math::

        \\sigma_{nk} = \\min\\left(\\max\\left(\\alpha \\, |\\nabla_k E_n| \\,
        \\delta k, \\; w_{\\min}\\right), \\; w_{\\max}\\right)

    with :math:`w_{\\min}` = ``min_width`` and :math:`w_{\\max}` =
    ``max_width``; a bound left at ``None`` is skipped.  The floor plays the
    role of the minimum width of the electron-phonon deltas
    (:mod:`PAOFLOW.elphon.adaptive_smearing`): without it a state sitting at a
    band extremum has a vanishing width and is lost from every sum over an
    energy grid.

    Reference: J. R. Yates, X. Wang, D. Vanderbilt, I. Souza,
    Phys. Rev. B **75**, 195121 (2007).
    """
    arrays, attributes = data_controller.data_dicts()

    # ----------------------
    # adaptive smearing as in Yates et al. Phys. Rev. B 75, 195121 (2007).
    # ----------------------

    nawf = attributes['nawf']
    nkpnts = attributes['nkpnts']

    diag = np.diag_indices(nawf)

    dk = (8.0 * np.pi**3 / attributes['omega'] / (nkpnts)) ** (1.0 / 3.0)

    if afac == None:
        afac = 1.0 if smearing == 'm-p' else 0.7

    ## DEV: Try to make contiguinuity conditional. Requires benchmark testing
    # Source of the band group velocity nabla_k E_n used for the Yates width.
    # When the position-dipole term is folded into dHksp, gradient_and_momenta
    # stores the BARE eigenbasis velocity diagonal under 'velkp_bare': the
    # diagonal of the corrected 'pksp' is no longer the Hellmann-Feynman group
    # velocity (the interband commutator term leaks into it at near-degeneracies)
    # and would inflate the adaptive widths, producing a spurious epsilon spike
    # near omega -> 0. Prefer the bare diagonal whenever it is present (opt out
    # via attr['adaptive_smearing_bare_velocity'] = False).
    use_bare = attributes.get('adaptive_smearing_bare_velocity', True)
    if use_bare and 'velkp_bare' in arrays:
        pksaux = np.ascontiguousarray(arrays['velkp_bare'])
    else:
        pksaux = np.ascontiguousarray(arrays['pksp'][:, :, diag[0], diag[1]])

    if min_width is not None and min_width < 0.0:
        raise ValueError('min_width must be non-negative (got %g eV)' % min_width)
    if max_width is not None and max_width < (0.0 if min_width is None else min_width):
        raise ValueError(
            'max_width (%g eV) must not be below min_width (%s eV)' % (max_width, min_width)
        )
    deltakp, deltakp2 = adaptive_widths(
        pksaux, afac, dk, pairs=True, axis=1, min_width=min_width, max_width=max_width
    )
    pksaux = None

    arrays['deltakp'] = deltakp
    arrays['deltakp2'] = deltakp2
