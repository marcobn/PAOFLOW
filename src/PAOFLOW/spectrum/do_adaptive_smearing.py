from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..DataController import DataController


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
    from numpy.linalg import norm
    import numpy as np

    arrays, attributes = data_controller.data_dicts()

    # ----------------------
    # adaptive smearing as in Yates et al. Phys. Rev. B 75, 195121 (2007).
    # ----------------------

    nawf = attributes['nawf']
    nspin = attributes['nspin']
    nkpnts = attributes['nkpnts']
    npks = arrays['pksp'].shape[0]

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

    deltakp = np.zeros((npks, nawf, nspin), dtype=float)
    deltakp2 = np.zeros((npks, nawf, nawf, nspin), dtype=float)
    for n in range(nawf):
        deltakp[:, n] = norm(np.real(pksaux[:, :, n]), axis=1)
        for m in range(nawf):
            deltakp2[:, n, m, :] = norm(pksaux[:, :, n, :] - pksaux[:, :, m, :], axis=1)

    pksaux = None
    deltakp *= afac * dk
    deltakp2 *= afac * dk

    if min_width is not None and min_width < 0.0:
        raise ValueError('min_width must be non-negative (got %g eV)' % min_width)
    if max_width is not None and max_width < (0.0 if min_width is None else min_width):
        raise ValueError(
            'max_width (%g eV) must not be below min_width (%s eV)' % (max_width, min_width)
        )
    if min_width is not None:
        np.maximum(deltakp, min_width, out=deltakp)
        np.maximum(deltakp2, min_width, out=deltakp2)
    if max_width is not None:
        np.minimum(deltakp, max_width, out=deltakp)
        np.minimum(deltakp2, max_width, out=deltakp2)

    arrays['deltakp'] = deltakp
    arrays['deltakp2'] = deltakp2
