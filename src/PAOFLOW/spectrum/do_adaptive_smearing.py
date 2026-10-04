from __future__ import annotations

import numpy as np


def adaptive_widths(
    vel: np.ndarray, afac: float, dk: float, pairs: bool = False, axis: int = 0
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
    if not pairs:
        return delta
    band_axis = axis + 1
    nband = vel.shape[band_axis]
    delta2 = np.zeros(delta.shape[:axis] + (nband,) + delta.shape[axis:], dtype=float)
    for n in range(nband):
        vel_n = np.take(vel, [n], axis=band_axis)
        delta2[(slice(None),) * axis + (n,)] = norm(vel_n - vel, axis=axis)
    delta2 *= afac * dk
    return delta, delta2


def do_adaptive_smearing(data_controller, smearing, afac):
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
    afac : Optional[float]
        Adaptive smearing prefactor :math:`\\alpha`.  If ``None``, defaults
        to ``1.0`` for ``'m-p'`` smearing and ``0.7`` otherwise.

    Returns
    -------
    None
        Adds the following keys to ``data_controller.data_arrays``:

        - ``deltakp`` : np.ndarray, shape ``(npks, nawf, nspin)`` —
          band-resolved adaptive smearing widths
          :math:`\\sigma_{nk} = \\alpha \\, |\\nabla_k E_n| \\, \\delta k`.
        - ``deltakp2`` : np.ndarray, shape ``(npks, nawf, nawf, nspin)`` —
          interband adaptive smearing widths proportional to
          :math:`|\\nabla_k E_n - \\nabla_k E_m|`.

    Notes
    -----
    The mean k-point spacing is estimated as

    .. math::

        \\delta k = \\left(\\frac{8\\pi^3}{\\Omega \\, N_k}\\right)^{1/3}

    where :math:`\\Omega` is the unit-cell volume and :math:`N_k` is the total
    number of k-points.  The diagonal elements of ``pksp`` (proportional to
    the band velocities) are used as a proxy for :math:`\\nabla_k E_n`.

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
    # When the non-local velocity (NLV) correction is active, gradient_and_momenta
    # stores the BARE eigenbasis velocity diagonal under 'velkp_bare'. The
    # diagonal of the (NLV-corrected) 'pksp' is no longer the true group
    # velocity -- the interband position-commutator correction contaminates it
    # and inflates the adaptive widths, producing a spurious epsilon spike near
    # omega -> 0. Prefer the bare diagonal whenever it is present (opt out via
    # attr['adaptive_smearing_bare_velocity'] = False).
    use_bare = attributes.get('adaptive_smearing_bare_velocity', True)
    if use_bare and 'velkp_bare' in arrays:
        pksaux = np.ascontiguousarray(arrays['velkp_bare'])
    else:
        pksaux = np.ascontiguousarray(arrays['pksp'][:, :, diag[0], diag[1]])

    deltakp, deltakp2 = adaptive_widths(pksaux, afac, dk, pairs=True, axis=1)
    pksaux = None

    arrays['deltakp'] = deltakp
    arrays['deltakp2'] = deltakp2
