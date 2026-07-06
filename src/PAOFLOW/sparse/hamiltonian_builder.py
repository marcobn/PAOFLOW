"""Sparse PAO Hamiltonian construction.

The PAO Hamiltonian ``H(k) = A_c ε A_c^†`` is, by construction, a *dense*
projection outer product on the **coarse** QE k-grid — there is no way to build
it without that bounded per-k dense object (this is the documented, size-gated
input-stage exception).  The sparse backend reuses the dense coarse builder
(:func:`PAOFLOW.hamiltonian.do_build_pao_hamiltonian`) to obtain the coarse
real-space ``HRs``, then immediately *thresholds* it into a
:class:`~PAOFLOW.sparse.containers.SparseHamiltonian` and discards the dense
array.  From that point on no dense ``(nawf, nawf, nkpnts, nspin)`` tensor is
ever formed: the fine-grid ``H(k)`` is assembled matrix-free from the hopping
list.
"""

from .containers import SparseHamiltonian
from .stats import estimate_dense_grid_bytes

# Above this estimated dense coarse-grid footprint the bounded input-stage
# build is refused, directing the user to a (future) fully-sparse builder rather
# than silently allocating a very large dense intermediate.
_DENSE_COARSE_GATE_BYTES = 8 * 1024**3  # 8 GB


def build_coarse_hamiltonian(data_controller):
    """Build the dense coarse-grid PAO Hamiltonian ``Hks``/``HRs`` (bounded).

    Reuses the dense kernels verbatim.  The coarse grid is the QE k-mesh, whose
    dense footprint is small and is the *same* object the dense pipeline builds
    at this stage.  A size gate refuses pathologically large coarse grids.

    Parameters
    ----------
    data_controller : DataController
        Must provide the projection arrays (``U``, ``my_eigsmat``) and coarse
        grid attributes (``nawf``, ``nk1``, ``nk2``, ``nk3``, ``nspin``).

    Returns
    -------
    None
        Populates ``data_arrays['Hks']`` and ``data_arrays['HRs']`` (dense,
        coarse) in place, exactly as the dense path does.

    Raises
    ------
    MemoryError
        If the estimated dense coarse-grid Hamiltonian exceeds the size gate.
    """
    from ..hamiltonian.do_build_pao_hamiltonian import (
        do_build_pao_hamiltonian,
        do_Hks_to_HRs,
    )

    arry, attr = data_controller.data_dicts()

    nkpnts = attr['nk1'] * attr['nk2'] * attr['nk3']
    est = estimate_dense_grid_bytes(attr['nawf'], nkpnts, attr['nspin'])
    if est > _DENSE_COARSE_GATE_BYTES:
        raise MemoryError(
            'Sparse pao_hamiltonian: the dense coarse-grid PAO Hamiltonian '
            f'would need ~{est / 1024**3:.1f} GB, above the '
            f'{_DENSE_COARSE_GATE_BYTES / 1024**3:.0f} GB input-stage gate. '
            'The PAO projection outer product is intrinsically dense on the '
            'coarse grid; a fully matrix-free builder is not yet implemented.'
        )

    do_build_pao_hamiltonian(data_controller)
    do_Hks_to_HRs(data_controller)
    # The coarse Hks is not needed once HRs exists; free it to keep only one
    # dense coarse copy alive.
    if 'Hks' in arry:
        del arry['Hks']


def finalize_sparse_hamiltonian(data_controller, threshold):
    """Convert the dense coarse ``HRs`` into a sparse hopping list and free it.

    This is the boundary past which the pipeline is purely sparse.  The dense
    ``HRs`` array is deleted after conversion so that a dense and a sparse copy
    are never held simultaneously.

    Parameters
    ----------
    data_controller : DataController
        Must provide ``data_arrays['HRs']`` (dense coarse/doubled Hamiltonian),
        ``a_vectors``, and (optionally) ``Dnm``.
    threshold : float
        Real-space entries with ``abs(H) < threshold`` are discarded.

    Returns
    -------
    SparseHamiltonian
        Also stored under ``data_arrays['sparse_H']``.
    """
    from ..utils.get_R_grid_fft import get_R_grid_fft

    arry, attr = data_controller.data_dicts()

    if 'sparse_H' in arry and arry.get('sparse_H') is not None:
        return arry['sparse_H']

    if 'HRs' not in arry:
        raise KeyError(
            'finalize_sparse_hamiltonian: no dense HRs to convert. ' 'Call pao_hamiltonian() first.'
        )

    nawf, _, nk1, nk2, nk3, nspin = arry['HRs'].shape
    # Build the Cartesian real-space grid consistent with the HRs C-order flatten.
    get_R_grid_fft(data_controller, nk1, nk2, nk3)

    sparse_h = SparseHamiltonian.from_dense_HRs(arry['HRs'], arry['R'], attr['alat'], threshold)
    if 'Dnm' in arry:
        sparse_h.set_position_operator(arry['Dnm'])

    arry['sparse_H'] = sparse_h
    # Free the dense coarse/doubled Hamiltonian: from here everything is sparse.
    del arry['HRs']
    return sparse_h
