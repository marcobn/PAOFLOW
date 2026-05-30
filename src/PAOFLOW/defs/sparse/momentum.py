from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...DataController import DataController

from .operators import iter_projected_operators, projected_operator_diagonals


def do_momentum(data_controller: DataController) -> None:
    """Compute band velocities needed by the sparse transport workflow.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding the local derivative information, the
        eigenvectors, and the degeneracy metadata used to rotate operators into
        a consistent band basis.

    Returns
    -------
    None
        Stores ``velkp(k, l, n, s)`` with shape ``(nkp_local, 3, nawf, nspin)``.

    Notes
    -----
    The physically relevant output here is the diagonal band velocity,

    ``v_n^(l)(k, s) = Re <u_n(k, s)| dH/dk_l |u_n(k, s)>``.

    Dense PAOFLOW often keeps the full band-space momentum matrix
    ``pksp(k, l, n, m, s)`` because some properties need off-diagonal elements.
    The sparse no-bridge DOS and transport workflow does not. This routine
    therefore keeps only the diagonal velocities after the same degeneracy-aware
    rotation used by the dense code, and it avoids materializing the much larger
    dense ``pksp`` tensor.

    Parallelization strategy:
        Each rank computes velocities only for its own local k-point window.
        The resulting distributed layout is already the one used by adaptive
        smearing, DOS, and transport, so no extra MPI redistribution is needed.
    """
    from ..perturb_split import perturb_split

    arrays, _ = data_controller.data_dicts()
    assert arrays is not None

    sparse_gradient_blocks = arrays.get('dHks_sparse')
    use_streamed_sparse_derivatives = (
        sparse_gradient_blocks is None and 'SparseHRs' in arrays and 'Hksp' not in arrays
    )
    if sparse_gradient_blocks is None and not use_streamed_sparse_derivatives:
        nktot, _, nawf, _, nspin = arrays['dHksp'].shape
    else:
        nktot, nawf, _, nspin = arrays['v_k'].shape

    velocities = np.empty((nktot, 3, nawf, nspin), dtype=float)

    if use_streamed_sparse_derivatives:
        for ik, ispin, projected_by_direction in iter_projected_operators(
            data_controller,
            range(3),
        ):
            diagonal_by_direction = projected_operator_diagonals(projected_by_direction)
            for direction, diagonal in diagonal_by_direction.items():
                velocities[ik, direction, : diagonal.shape[0], ispin] = diagonal

        arrays['velkp'] = velocities
        arrays.pop('dHks_sparse', None)
        arrays.pop('dHksp', None)
        if 'pksp' in arrays:
            del arrays['pksp']
        return

    for ispin in range(nspin):
        for ik in range(nktot):
            for direction in range(3):
                if sparse_gradient_blocks is not None:
                    direction_operator = sparse_gradient_blocks[(ik, ispin)][direction].toarray()
                else:
                    direction_operator = arrays['dHksp'][ik, direction, :, :, ispin]

                momentum_matrix = perturb_split(
                    direction_operator,
                    direction_operator,
                    arrays['v_k'][ik, :, :, ispin],
                    arrays['degen'][ispin][ik],
                )[0]
                velocities[ik, direction, :, ispin] = np.real(np.diag(momentum_matrix))

    arrays['velkp'] = velocities
    arrays.pop('dHks_sparse', None)
    if 'pksp' in arrays:
        del arrays['pksp']
