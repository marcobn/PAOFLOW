from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI

if TYPE_CHECKING:
    from ...DataController import DataController


def _use_local_gradient(arrays: dict, attributes: dict) -> bool:
    """Decide whether the gradient must stay local and sparse-aware.

    Parameters
    ----------
    arrays : dict
        Runtime arrays for the active calculation.
    attributes : dict
        Runtime scalar attributes for the active calculation.

    Returns
    -------
    bool
        ``True`` when the sparse workflow has ``SparseHRs`` available but has
        intentionally not materialized a dense ``Hksp`` tensor.

    Notes
    -----
    After sparse interpolation, the Hamiltonian exists only as sparse blocks or
    as the sparse real-space object. In that situation the derivative stage must
    work on the local k-point slice only; rebuilding a full dense derivative
    tensor would defeat the point of the sparse path.
    """
    return bool(attributes.get('sparse', False)) and 'SparseHRs' in arrays and 'Hksp' not in arrays


def do_gradient(data_controller: DataController) -> None:
    """Build the local first k-derivatives of the sparse Hamiltonian.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding the sparse real-space Hamiltonian, the active
        k-grid, and the orbital correction tensor ``Dnm``.

    Returns
    -------
    None
        Stores the rank-local derivative blocks in ``arrays['dHks_sparse']``.

    Notes
    -----
    The quantity being formed is ``dH/dk_l`` for ``l = x, y, z`` on the active
    k-mesh. In the sparse no-bridge workflow, the dense tensor ``Hksp`` is not
    present, so the derivative must be generated directly from ``SparseHRs``.
    Only the local k-point window is evaluated, which is sufficient for the
    downstream momentum step and avoids rebuilding dense global derivative
    arrays.

    Parallelization strategy:
        Each rank reuses the same contiguous k-point ownership as the local
        eigenvectors from ``pao_eigh`` and computes derivatives only for that
        slice.
    """
    from ..communication import load_balancing
    from ..get_K_grid_fft import get_K_grid_fft

    arry, attr = data_controller.data_dicts()

    sparse_hrs = arry['SparseHRs']
    R_cart = sparse_hrs.compute_R_cart(arry['a_vectors'])

    if 'kgrid' not in arry:
        get_K_grid_fft(data_controller)

    if not _use_local_gradient(arry, attr):
        raise RuntimeError(
            'do_gradient requires sparse no-bridge input: SparseHRs must '
            'exist and dense Hksp must be absent.'
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_kpoint, _ = load_balancing(size, rank, int(attr['nkpnts']))
    local_nkpnts = int(arry['v_k'].shape[0])
    stop_kpoint = min(int(start_kpoint) + local_nkpnts, int(attr['nkpnts']))
    if stop_kpoint - int(start_kpoint) != local_nkpnts:
        raise RuntimeError(
            'Sparse gradient local slice does not match local eigenvector ownership.'
        )

    arry['dHks_sparse'] = sparse_hrs.build_local_dHdk_blocks(
        kgrid=arry['kgrid'],
        r_cart=R_cart,
        alat=float(attr['alat']),
        dnm=arry['Dnm'],
        start_kpoint=int(start_kpoint),
        stop_kpoint=int(stop_kpoint),
    )
    if 'dHksp' in arry:
        del arry['dHksp']
