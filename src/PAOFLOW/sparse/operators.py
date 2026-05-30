from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
from mpi4py import MPI

from ..perturb_split import perturb_split

if TYPE_CHECKING:
    from ..DataController import DataController


def _iter_streamed_derivative_batches(
    data_controller: DataController,
    directions: tuple[int, ...] | None = None,
):
    """Stream small batches of ``dH/dk`` from the sparse real-space Hamiltonian.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with ``SparseHRs``, the active k-grid, and the local
        eigensolutions.
    directions : tuple[int, ...] | None, optional
        Cartesian derivative directions to evaluate. If omitted, all three
        directions are produced.

    Yields
    ------
    tuple[int, int, int, numpy.ndarray]
        ``(batch_start, batch_stop, ispin, dh_batch)`` where ``dh_batch`` has
        shape ``(nbatch, 3, nawf, nawf)`` for the local rank-owned k-point
        window.

    Notes
    -----
    Several post-processing steps still need dense operator matrices in the band
    basis, but only for one small set of k-points at a time. This generator
    keeps ``SparseHRs`` as the source of truth and forms bounded batches of
    ``dH/dk`` only for the local MPI slice. In this way the sparse workflow
    avoids storing a dense global derivative tensor while still reusing the
    existing band-space algebra.
    """
    from ..communication import load_balancing
    from ..get_K_grid_fft import get_K_grid_fft
    from .gradient import _use_local_gradient

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    if not _use_local_gradient(arrays, attributes):
        raise RuntimeError(
            'Streamed sparse projected operators require sparse no-bridge input: '
            'SparseHRs must exist and dense Hksp must be absent.'
        )

    if 'kgrid' not in arrays:
        get_K_grid_fft(data_controller)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_kpoint, _ = load_balancing(size, rank, int(attributes['nkpnts']))
    local_nkpnts = int(arrays['v_k'].shape[0])
    stop_kpoint = min(int(start_kpoint) + local_nkpnts, int(attributes['nkpnts']))
    if stop_kpoint - int(start_kpoint) != local_nkpnts:
        raise RuntimeError(
            'Sparse projected-operator local slice does not match local eigenvector ownership.'
        )

    sparse_hrs = arrays['SparseHRs']
    r_cart = sparse_hrs.compute_R_cart(arrays['a_vectors'])
    yield from sparse_hrs.iter_local_dHdk_batches(
        kgrid=arrays['kgrid'],
        r_cart=r_cart,
        alat=float(attributes['alat']),
        dnm=arrays['Dnm'],
        start_kpoint=int(start_kpoint),
        stop_kpoint=int(stop_kpoint),
        directions=directions,
    )


def iter_projected_operators(
    data_controller: DataController,
    directions: Iterable[int],
    *,
    band_count: int | None = None,
):
    """Project streamed sparse derivatives onto the local band eigenstates.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with sparse no-bridge eigendata and ``SparseHRs``.
    directions : Iterable[int]
        Cartesian components of ``dH/dk`` to be projected.
    band_count : int | None, optional
        Number of bands retained in the projected matrices. ``None`` keeps the
        full band-space operator.

    Yields
    ------
    tuple[int, int, dict[int, numpy.ndarray]]
        ``(ik_local, ispin, projected_by_direction)`` for one local k-point,
        where each projected matrix has shape ``(band_count, band_count)``.

    Notes
    -----
    The matrix elements produced here are

    ``<u_n(k)| dH/dk_l |u_m(k)>``.

    They are the basic ingredients for velocities, dielectric spectra, and Hall
    responses. The sparse workflow builds them only temporarily, one local
    k-point at a time, so it never needs dense global ``dHksp`` or ``pksp``
    storage.
    """
    arrays, _ = data_controller.data_dicts()
    assert arrays is not None

    requested_directions = tuple(
        int(direction) for direction in np.unique(np.asarray(tuple(directions), dtype=int))
    )
    if not requested_directions:
        raise ValueError('Projected sparse operators require at least one direction.')

    band_limit = int(arrays['v_k'].shape[1]) if band_count is None else int(band_count)

    for batch_start, batch_stop, ispin, dh_batch in _iter_streamed_derivative_batches(
        data_controller,
        directions=requested_directions,
    ):
        for batch_offset, ik_local in enumerate(range(batch_start, batch_stop)):
            vecs = arrays['v_k'][ik_local, :, :, ispin]
            degen = arrays['degen'][ispin][ik_local]

            projected_by_direction: dict[int, np.ndarray] = {}
            for direction in requested_directions:
                operator = dh_batch[batch_offset, direction, :, :]
                projected_operator = perturb_split(operator, operator, vecs, degen)[0]
                projected_by_direction[direction] = projected_operator[:band_limit, :band_limit]

            yield ik_local, ispin, projected_by_direction


def projected_operator_diagonals(
    projected_by_direction: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Extract the diagonal velocity-like part of projected operators.

    Parameters
    ----------
    projected_by_direction : dict[int, numpy.ndarray]
        Projected band-space operators keyed by Cartesian direction.

    Returns
    -------
    dict[int, numpy.ndarray]
        Real diagonal entries for each requested direction.

    Notes
    -----
    For transport-like quantities only the diagonal elements
    ``Re <u_n|dH/dk_l|u_n>`` are needed. This helper makes that reduced contract
    explicit without introducing a second dense tensor representation.
    """
    return {
        direction: np.real(np.diag(projected_operator))
        for direction, projected_operator in projected_by_direction.items()
    }
