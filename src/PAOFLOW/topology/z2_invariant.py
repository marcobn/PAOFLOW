"""Z2 invariant of time-reversal-symmetric bands from the Wannier charge centres (Z2Pack)."""

from __future__ import annotations

from os.path import join
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

#: Default sampling of ``z2pack.surface.run`` (the mirror Chern defaults).
SURFACE_DEFAULTS = {'pos_tol': 1.0e-2, 'iterator': range(11, 61, 4), 'load': False}


def wcc_file_name(prefix: str, k3: float) -> str:
    """Name of the file of the Wannier charge centres of the plane ``k3``.

    Parameters
    ----------
    prefix : str
        Prefix of the files (``'z2_wcc'``).
    k3 : float
        Plane (crystal coordinate along the third reciprocal vector).

    Returns
    -------
    str
        ``'<prefix>_k3_<k3>.dat'``, for example ``'z2_wcc_k3_0.5.dat'``.
    """
    return '%s_k3_%g.dat' % (prefix, k3)


def do_z2_invariant(
    data_controller: DataController,
    nbnd_occ: int | str = 'auto',
    k3_planes: tuple[float, ...] = (0.0,),
    surface_kwargs: dict | None = None,
    hr_file: str = 'z2_HRs.dat',
    wcc_prefix: str = 'z2_wcc',
) -> dict:
    """Z2 index of the occupied bands on the time-reversal-invariant planes ``k3 = const``.

    Parameters
    ----------
    data_controller : DataController
        Provides ``HRs``, ``nelec``, ``dftSO`` / ``adhoc_SO`` and ``opath``.
    nbnd_occ : int or 'auto', optional
        Number of occupied bands; ``'auto'`` takes the number of electrons (one per band in a
        spinor basis).
    k3_planes : tuple of float, optional
        Planes of the third reciprocal coordinate: ``(0.0,)`` for a 2D layer (vacuum along
        ``c``), ``(0.0, 0.5)`` for the two invariant planes of a 3D crystal.
    surface_kwargs : dict, optional
        Overrides of the ``z2pack.surface.run`` sampling (:data:`SURFACE_DEFAULTS`).
    hr_file : str, optional
        File of the Hamiltonian that Z2Pack reads (written in ``opath``).
    wcc_prefix : str, optional
        Prefix of the files of the Wannier charge centres.

    Returns
    -------
    dict
        ``nocc``, ``planes`` (the ``k3`` values) and ``z2`` (one index per plane), the same on
        every rank. Rank 0 writes ``<wcc_prefix>_k3_<k3>.dat``: the pumping parameter ``t``
        (fraction of ``b2``) and the ``nocc`` charge centres (fractions of ``a1``) at each ``t``.

    Raises
    ------
    ValueError
        Without spin-orbit coupling (the index needs Kramers pairs).
    ImportError
        Without the optional ``z2pack`` and ``tbmodels`` packages.

    Notes
    -----
    On each plane the Wannier charge centres of the occupied bands, Wilson loops along
    :math:`\\mathbf{b}_1`, are followed as :math:`k_2` runs over half the zone; the Z2 index is
    the parity of the number of times they cross an arbitrary reference line
    (Soluyanov and Vanderbilt, Phys. Rev. B 83, 235401 (2011)). The surface
    :math:`(k_1, k_2) = (s/2, t)` of :mod:`PAOFLOW.topology.do_mirror_chern` is used.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    arry, attr = data_controller.data_dicts()
    if not (attr.get('dftSO', False) or attr.get('adhoc_SO', False)):
        raise ValueError('z2_invariant needs spin-orbit coupling (a fully relativistic save or '
                         'adhoc_spin_orbit)')  # fmt: skip
    nocc = int(round(attr['nelec'])) if nbnd_occ == 'auto' else int(nbnd_occ)
    planes = tuple(float(k3) for k3 in k3_planes)
    data_controller.write_HRs(hr_file)
    if rank != 0:
        return comm.bcast(None, root=0)

    import tbmodels
    import z2pack

    options = dict(SURFACE_DEFAULTS)
    options.update(surface_kwargs or {})
    model = tbmodels.Model.from_wannier_files(hr_file=join(attr['opath'], hr_file))
    system = z2pack.tb.System(model, bands=nocc)
    indices = []
    for k3 in planes:
        result = z2pack.surface.run(
            system=system, surface=lambda s, t, k3=k3: [s / 2, t, k3], **options
        )
        indices.append(int(z2pack.invariant.z2(result)))
        rows = [[t, *wcc] for t, wcc in zip(result.t, result.wcc)]
        np.savetxt(
            join(attr['opath'], wcc_file_name(wcc_prefix, k3)),
            np.array(rows),
            fmt='%14.8f',
            header='t (fraction of b2), Wannier charge centres (fraction of a1); k3 = %g, Z2 = %d'
            % (k3, indices[-1]),
        )
    return comm.bcast({'nocc': nocc, 'planes': list(planes), 'z2': indices}, root=0)
