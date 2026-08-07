from __future__ import annotations

import numpy as np

from PAOFLOW.DataController import DataController
from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.hamiltonian.fourier_par import fourier_transform_real_to_kspace
from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlock
from PAOFLOW.transport.parsers.parser_base import parse_index_array
from PAOFLOW.transport.partition.directions import direction_axis
from PAOFLOW.transport.utils.timing import timed_function


@timed_function('read_matrix')
def read_matrix(
    conductor_data: ConductorData,
    data_controller: DataController,
    ispin: int,
    transport_direction: str,
    opr: OperatorBlock,
) -> None:
    """Build a k-space operator block from the in-memory real-space Hamiltonian.

    Selects the real-space block requested by ``opr`` from the shared
    ``HRs``/``SRs`` arrays, slices it according to the row/column index metadata
    on ``opr.tag``, and stores the partial Fourier transform into ``opr``.

    Parameters
    ----------
    conductor_data : ConductorData
        Conductor input model. Only ``conductor_data.atomic_proj.do_overlap_transformation``
        is read, to decide whether overlap blocks are taken from ``SRs`` or replaced
        by an identity (on-site) / zero (off-site) block.
    data_controller : DataController
        Shared PAOFLOW data store. Required arrays: ``HRs`` (shape ``(nrtot, nawf, nawf)``),
        ``SRs`` (same shape, only when overlap is enabled), and ``ivr``
        (shape ``(nrtot, 3)``). Required attributes: ``nawf``, ``nspin``.
    ispin : int
        Spin index (0-based) selecting the spin channel. Must be non-negative
        when ``nspin == 2``.
    transport_direction : {'x', 'y', 'z'}
        Transport direction.
    opr : OperatorBlock
        Target operator block, mutated in place (see ``Returns``).

    Returns
    -------
    None
        Mutates ``opr``: sets ``opr.irows``, ``opr.icols``, ``opr.irows_sgm``,
        ``opr.icols_sgm`` from the parsed ``opr.tag`` metadata, and overwrites
        ``opr.H`` and ``opr.S`` with the k-resolved Hamiltonian and overlap blocks.

    Raises
    ------
    RuntimeError
        If ``opr`` has not been allocated.
    ValueError
        If ``nspin == 2`` and ``ispin`` is negative, if ``opr.name`` is not a
        recognized block label, or if the required 3D R-vector cannot be found
        in ``ivr``.

    Notes
    -----
    The data source is the in-memory ``HRs``/``SRs`` arrays produced by
    ``populate_real_space_hamiltonian``; nothing is read from disk. (The legacy
    ``.ham`` file is now a debug-only artifact and is not an input here.)

    Real-space blocks are selected by building the target 3D integer R-vector
    ``ivr_aux``: the component along ``transport_direction`` is fixed by the block
    label (``0`` for on-site blocks, ``1`` for coupling blocks, or an explicit
    ``ivr`` override from ``opr.tag``), while the remaining components come from
    ``opr.ivr_par``. The matching row of ``ivr`` selects the block from ``HRs``
    (and ``SRs`` when overlap is enabled). When overlap is disabled, the on-site
    block uses the identity at the zero R-vector and zeros elsewhere.

    Each selected block is sliced using the ``irows``/``icols`` index arrays
    parsed from ``opr.tag`` and inserted into the real-space tensors ``A``
    (Hamiltonian) and ``S`` (overlap), indexed over the parallel R-vector grid.
    A partial 2D Fourier transform in the directions orthogonal to the transport
    axis then yields the k-resolved operator block.
    """
    if not opr.allocated:
        raise RuntimeError('OperatorBlock is not allocated')

    arry, attr = data_controller.data_dicts()
    tag_attr = opr.tag
    label = opr.name.strip()

    # === Defaults and attribute parsing ===
    cols = tag_attr.get('cols', 'all').lower()
    rows = tag_attr.get('rows', 'all').lower()
    cols_sgm = tag_attr.get('cols_sgm', cols).lower()
    rows_sgm = tag_attr.get('rows_sgm', rows).lower()
    ivr_input = int(tag_attr.get('ivr', 0))
    ivr_from_input = 'ivr' in tag_attr

    dim1, dim2 = opr.dim1, opr.dim2
    transport_axis = direction_axis(transport_direction)

    # Convert "all" to full ranges
    if rows == 'all':
        rows = f'1-{dim1}'
    if cols == 'all':
        cols = f'1-{dim2}'
    if rows_sgm == 'all':
        rows_sgm = f'1-{dim1}'
    if cols_sgm == 'all':
        cols_sgm = f'1-{dim2}'

    # === File parsing ===

    nawf = attr['nawf']
    nspin = attr['nspin']
    do_overlap_transform = conductor_data.atomic_proj.do_overlap_transformation
    ivr = arry['ivr']
    nrtot = ivr.shape[0]
    irows = parse_index_array(rows, nawf)
    icols = parse_index_array(cols, nawf)
    irows_sgm = parse_index_array(rows_sgm, nawf)
    icols_sgm = parse_index_array(cols_sgm, nawf)

    opr.irows = irows
    opr.icols = icols
    opr.irows_sgm = irows_sgm
    opr.icols_sgm = icols_sgm

    if nspin == 2 and ispin < 0:
        raise ValueError('Unspecified ispin for spin-polarized case')
    ivr = ivr.T

    # Real-space arrays are indexed by the transverse R-vectors, which is
    # independent of the number of transverse k-points: a uniform mesh happens to
    # use nk_par == nr_par, but a surface k-path does not.
    nrtot_par = opr.ivr_par.shape[1]
    A = np.zeros((dim1, dim2, nrtot_par), dtype=complex)
    S = np.zeros((dim1, dim2, nrtot_par), dtype=complex)

    for ir_par in range(nrtot_par):
        ivr_aux = np.zeros(3, dtype=int)
        j = 0
        for i in range(3):
            if i + 1 == transport_axis:
                if label.lower() in {
                    'block_00c',
                    'block_00r',
                    'block_00l',
                    'block_t',
                    'block_e',
                    'block_b',
                    'block_eb',
                    'block_be',
                }:
                    ivr_aux[i] = 0
                elif label.lower() in {
                    'block_01r',
                    'block_01l',
                    'block_lc',
                    'block_cr',
                }:
                    ivr_aux[i] = 1
                else:
                    raise ValueError(f'Invalid block label {label}')
                if ivr_from_input:
                    ivr_aux[i] = ivr_input
            else:
                ivr_aux[i] = opr.ivr_par[j, ir_par]
                j += 1

        matches = [ir for ir in range(nrtot) if np.array_equal(ivr[:, ir], ivr_aux)]
        if not matches:
            raise ValueError(f'3D R-vector {ivr_aux} not found for ir_par={ir_par}')

        ind = matches[0]

        A_loc = arry['HRs'][ind, :, :]

        if do_overlap_transform:
            S_loc = arry['SRs'][ind, :, :]
        else:
            S_loc = np.zeros_like(A_loc)
            if label.lower() in {
                'block_00c',
                'block_00r',
                'block_00l',
                'block_t',
                'block_e',
                'block_b',
                'block_eb',
                'block_be',
            } and np.all(ivr_aux == 0):
                S_loc[:] = np.eye(nawf)

        A_loc_T = A_loc.T
        S_loc_T = S_loc.T
        for j in range(dim2):
            for i in range(dim1):
                if icols[j] < 0 or irows[i] < 0:
                    continue
                A[i, j, ir_par] = A_loc_T[irows[i], icols[j]]
                S[i, j, ir_par] = S_loc_T[irows[i], icols[j]]

    opr.H = fourier_transform_real_to_kspace(A, opr.wr_par, opr.table_par)
    opr.S = fourier_transform_real_to_kspace(S, opr.wr_par, opr.table_par)
