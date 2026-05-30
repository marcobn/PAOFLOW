from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from ...DataController import DataController


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def use_topology(data_controller: DataController) -> bool:
    """Decide whether topology should run from the sparse real-space Hamiltonian.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with the arrays and attributes of the active run.

    Returns
    -------
    bool
        ``True`` when the sparse real-space Hamiltonian ``SparseHRs`` is
        available and dense ``HRs`` has intentionally not been built.

    Notes
    -----
    The dense topology implementation expects a dense real-space Hamiltonian.
    After the sparse boundary, the no-bridge workflow keeps only ``SparseHRs``.
    This check prevents an accidental dense reconstruction and selects the
    sparse-native topology route instead.
    """
    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None
    return bool(attributes.get('sparse', False)) and 'SparseHRs' in arrays and 'HRs' not in arrays


def _local_path_bounds(data_controller: DataController) -> tuple[int, int]:
    """Return the segment of the band path owned by the current MPI rank."""
    from ..communication import load_balancing

    arrays, _ = data_controller.data_dicts()
    assert arrays is not None
    start_kpoint, stop_kpoint = load_balancing(comm.Get_size(), rank, int(arrays['kq'].shape[1]))
    local_nkpnts = int(arrays['v_k'].shape[0])
    if stop_kpoint - start_kpoint != local_nkpnts:
        raise RuntimeError(
            'Sparse topology local k-point window does not match local eigenvector ownership.'
        )
    return int(start_kpoint), int(stop_kpoint)


def _compute_projected_momenta(
    data_controller: DataController,
) -> np.ndarray:
    """Project the path derivatives onto the local band basis.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with sparse ``H(R)`` data, path eigenvectors, and the
        metadata needed by the topology calculation.

    Returns
    -------
    numpy.ndarray
        Local projected operator matrices with shape
        ``(nkp_local, 3, bnd, bnd, nspin)``.

    Notes
    -----
    Along the chosen band path, topology needs the matrix elements
    ``<u_n(k)|dH/dk_l|u_m(k)>``. This helper obtains them directly from
    ``SparseHRs`` by evaluating only the local path segment, so it avoids both
    a dense ``HRs`` reconstruction and a dense global derivative tensor.
    """
    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    local_nkpnts = int(arrays['v_k'].shape[0])
    bnd = int(attributes['bnd'])
    nspin = int(attributes['nspin'])

    sparse_hrs = arrays['SparseHRs']
    r_cart = sparse_hrs.compute_R_cart(arrays['a_vectors'])
    start_kpoint, stop_kpoint = _local_path_bounds(data_controller)

    pks = np.zeros((local_nkpnts, 3, bnd, bnd, nspin), dtype=complex)

    for batch_start, batch_stop, ispin, dh_batch in sparse_hrs.iter_local_dHdk_batches(
        kgrid=arrays['kq'],
        r_cart=r_cart,
        alat=float(attributes['alat']),
        dnm=arrays['Dnm'],
        start_kpoint=start_kpoint,
        stop_kpoint=stop_kpoint,
        phase_sign=1.0,
        hermitianize=False,
    ):
        for batch_offset, ik_local in enumerate(range(batch_start, batch_stop)):
            vecs = arrays['v_k'][ik_local, :, :, ispin]
            vecs_h = np.conj(vecs.T)
            for direction in range(3):
                projected_operator = vecs_h.dot(dh_batch[batch_offset, direction]).dot(vecs)
                pks[ik_local, direction, :, :, ispin] = projected_operator[:bnd, :bnd]

    return pks


def _compute_effective_mass(
    data_controller: DataController,
    pks: np.ndarray,
) -> np.ndarray:
    """Evaluate the local inverse effective-mass tensor on the band path.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with sparse ``H(R)`` data and path eigensolutions.
    pks : numpy.ndarray
        Local first-derivative matrices in the band basis, with shape
        ``(nkp_local, 3, bnd, bnd, nspin)``.

    Returns
    -------
    numpy.ndarray
        Local inverse-mass tensor slice with shape
        ``(nkp_local, bnd, 3, 3, nspin)``.

    Notes
    -----
    The inverse effective mass contains a direct second-derivative term together
    with the usual interband correction. This helper mirrors the dense topology
    formula on the local path segment,

    ``m^{-1}_{n,lp}(k) ~ <n|d^2H/dk_l dk_p|n> + sum_{m != n} (...)``

    but it computes the needed operators from ``SparseHRs`` only for the local
    path window. No dense global second-derivative tensor is stored.

    Parallelization strategy:
        Each rank evaluates the required second derivatives only on its own path
        segment. Only the final compact tensor is gathered for output.
    """
    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    sparse_hrs = arrays['SparseHRs']
    r_cart = sparse_hrs.compute_R_cart(arrays['a_vectors'])
    start_kpoint, stop_kpoint = _local_path_bounds(data_controller)

    bnd = int(attributes['bnd'])
    nspin = int(attributes['nspin'])
    ipol = int(attributes['ipol'])
    jpol = int(attributes['jpol'])

    mkm1 = np.zeros((arrays['v_k'].shape[0], bnd, 3, 3, nspin), dtype=complex)

    for batch_start, batch_stop, ispin, _, d2h_batch in sparse_hrs.iter_local_d2Hdk2_batches(
        kgrid=arrays['kq'],
        r_cart=r_cart,
        alat=float(attributes['alat']),
        start_kpoint=start_kpoint,
        stop_kpoint=stop_kpoint,
        direction_pairs=((ipol, jpol),),
        phase_sign=1.0,
        hermitianize=False,
    ):
        for batch_offset, ik_local in enumerate(range(batch_start, batch_stop)):
            vecs = arrays['v_k'][ik_local, :, :, ispin]
            projected_second = np.conj(vecs.T).dot(d2h_batch[batch_offset, 0]).dot(vecs)[:bnd, :bnd]

            for n in range(bnd):
                for m in range(bnd):
                    if m != n:
                        mkm1[ik_local, n, ipol, jpol, ispin] += (
                            pks[ik_local, ipol, n, m, ispin] * pks[ik_local, jpol, m, n, ispin]
                            + pks[ik_local, jpol, n, m, ispin] * pks[ik_local, ipol, m, n, ispin]
                        ) / (
                            arrays['E_k'][ik_local, n, ispin]
                            - arrays['E_k'][ik_local, m, ispin]
                            + 1.0e-16
                        )
                    else:
                        mkm1[ik_local, n, ipol, jpol, ispin] += projected_second[n, n]

    return mkm1


def do_topology(data_controller: DataController) -> None:
    """Compute supported topology quantities from sparse ``H(R)`` data.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with sparse ``H(R)`` data, path eigenvalues, and path
        eigenvectors.

    Returns
    -------
    None
        Writes the same velocity, Berry-curvature, and effective-mass outputs
        as the dense topology path for the currently supported sparse features.

    Notes
    -----
    The topology workflow on a band path needs velocity-like and curvature-like
    matrix elements derived from the Hamiltonian. In dense mode these come from
    the dense real-space tensor ``HRs``. The sparse workflow instead evaluates
    the required derivative operators from ``SparseHRs`` only for the local path
    slice and keeps only compact band-space objects. This preserves the physics
    of the Berry and effective-mass calculations without rebuilding dense
    ``HRs``.

    Parallelization strategy:
        Each rank treats only its own segment of the band path and gathers only
        compact projected quantities or final output arrays for file writing.
        Sparse spin-Hall and Z2 topology remain separate future steps because
        they still need additional sparse spin-current and TRIM machinery.
    """
    from os.path import join

    from ..communication import gather_full
    from ..constants import LL

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    if not use_topology(data_controller):
        raise RuntimeError(
            'Sparse topology requires sparse no-bridge input: SparseHRs must exist '
            'and dense HRs must be absent.'
        )

    if bool(attributes.get('spin_Hall', False)):
        raise NotImplementedError(
            'Sparse topology currently supports Berry and eff_mass outputs only. '
            'spin_Hall/Z2 still requires dedicated sparse spin-current/TRIM handling.'
        )

    bnd = int(attributes['bnd'])
    nkpi = int(arrays['kq'].shape[1])
    nspin = int(attributes['nspin'])
    ipol = int(attributes['ipol'])
    jpol = int(attributes['jpol'])
    spol = int(attributes['spol'])

    pks = _compute_projected_momenta(data_controller)

    if bool(attributes.get('eff_mass', False)):
        mkm1 = _compute_effective_mass(data_controller, pks)
        mkm1 = gather_full(mkm1, int(attributes['npool']))
        if rank == 0:
            assert mkm1 is not None
            for ispin in range(nspin):
                with open(
                    join(
                        attributes['opath'],
                        'effmass_' + str(LL[ipol]) + str(LL[jpol]) + '_' + str(ispin) + '.dat',
                    ),
                    'w',
                ) as handle:
                    for ik in range(nkpi):
                        row = '%d\t' % ik
                        for value in np.real(mkm1[ik, :bnd, ipol, jpol, ispin]):
                            row += '% 3.5f\t' % value
                        row += '\n'
                        handle.write(row)

    berry = bool(attributes.get('Berry', False))
    om_zk = None
    if berry:
        deltab = 0.05
        om_zk = np.zeros((pks.shape[0], 1), dtype=float)
        om_znk = np.zeros((pks.shape[0], bnd), dtype=float)
        for ik_local in range(pks.shape[0]):
            for n in range(bnd):
                for m in range(bnd):
                    if m == n:
                        continue
                    om_znk[ik_local, n] += (
                        -1.0
                        * np.imag(
                            pks[ik_local, jpol, n, m, 0] * pks[ik_local, ipol, m, n, 0]
                            - pks[ik_local, ipol, n, m, 0] * pks[ik_local, jpol, m, n, 0]
                        )
                        / (
                            (arrays['E_k'][ik_local, m, 0] - arrays['E_k'][ik_local, n, 0]) ** 2
                            + deltab**2
                        )
                    )
            om_zk[ik_local] = np.sum(
                om_znk[ik_local, :] * (0.5 * (1 - np.sign(arrays['E_k'][ik_local, :bnd, 0])))
            )

    indices = (LL[spol], LL[ipol], LL[jpol])
    lrng = list(range(nkpi)) if rank == 0 else None

    pks_full = gather_full(pks, int(attributes['npool']))
    velk = np.zeros((nkpi, 3, bnd, nspin), dtype=float) if rank == 0 else None
    if rank == 0:
        assert pks_full is not None
        assert velk is not None
        for n in range(bnd):
            velk[:, :, n, :] = np.real(pks_full[:, :, n, n, :])
    for direction in range(3):
        band_values = velk[:, direction, :bnd, :] if rank == 0 and velk is not None else None
        data_controller.write_bands('velocity_' + str(direction), band_values)

    if berry:
        om_zk = gather_full(om_zk, int(attributes['npool']))
        if rank == 0:
            assert om_zk is not None
        omega_values = -om_zk[:, 0] if rank == 0 and om_zk is not None else None
        data_controller.write_file_row_col('Omega_%s_%s%s.dat' % indices, lrng, omega_values)
