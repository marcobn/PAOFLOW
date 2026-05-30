from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..DataController import DataController


def do_adaptive_smearing(
    data_controller: DataController,
    smearing: str,
    afac: float | None,
) -> None:
    """Estimate adaptive broadening from the local band velocities.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding the local band velocities
        ``velkp(k, l, n, s)`` and the scalar metadata needed to set the
        smearing scale.
    smearing : str
        Smearing prescription used later in DOS and transport. Supported values
        are the same as in the dense workflow: ``'gauss'`` and ``'m-p'``.
    afac : float | None
        Dimensionless prefactor entering the adaptive width. If ``None``, the
        same default choice as the dense implementation is used.

    Returns
    -------
    None
        Stores ``deltakp(k, n, s)`` and ``deltakp2(k, n, m, s)`` in the runtime
        arrays.

    Notes
    -----
    The quantity being constructed is the k-dependent spectral width used to
    smooth discrete bands on a finite mesh. For each band,

    ``deltakp(k, n, s) = a_fac * dk * |v_n(k, s)|``

    and for band-to-band transitions,

    ``deltakp2(k, n, m, s) = a_fac * dk * |v_n(k, s) - v_m(k, s)|``.

    Here ``v_n`` is the band velocity extracted from the projected derivative of
    the Hamiltonian. In the dense workflow these widths are obtained after
    building the full momentum tensor ``pksp``. The sparse workflow does not
    need that much larger tensor for this step, so it works directly with the
    already available diagonal velocities and avoids reconstructing dense
    momentum matrices.

    Parallelization strategy:
        Each MPI rank evaluates the widths only for its own k-point slice. This
        matches the distributed layout used by the downstream DOS and transport
        routines, so no extra communication is introduced here.
    """
    arrays, attributes = data_controller.data_dicts()

    nawf = int(attributes['nawf'])
    nspin = int(attributes['nspin'])
    nkpnts = int(attributes['nkpnts'])
    npks = int(arrays['velkp'].shape[0])

    dk = (8.0 * np.pi**3 / attributes['omega'] / nkpnts) ** (1.0 / 3.0)

    if afac is None:
        afac = 1.0 if smearing == 'm-p' else 0.7

    velocities = np.ascontiguousarray(arrays['velkp'])
    deltakp = np.linalg.norm(velocities, axis=1)
    deltakp2 = np.empty((npks, nawf, nawf, nspin), dtype=float)

    for band in range(nawf):
        velocity_differences = velocities[:, :, band, :][:, :, None, :] - velocities
        deltakp2[:, band, :, :] = np.linalg.norm(velocity_differences, axis=1)

    arrays['deltakp'] = deltakp * afac * dk
    arrays['deltakp2'] = deltakp2 * afac * dk
