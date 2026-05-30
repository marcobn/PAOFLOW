#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import sparse

from PAOFLOW.sparse.utils import report_sparse_hr_stats

SparseHRKey = tuple[int, int, int, int]
SparseHRDict = dict[SparseHRKey, sparse.csr_matrix]


_R_CUT_REL_TOL = 64.0 * np.finfo(float).eps


def doubling_HRs(data_controller: Any) -> None:
    r"""Double real-space Hamiltonian blocks without dense Hamiltonian storage.

    The sparse real-space Hamiltonian is assumed to represent the PAO matrix
    blocks

    .. math::

        H_{ij}(R) = \langle \phi_{i0} | \hat{H} | \phi_{jR} \rangle,

    where :math:`R=(R_1,R_2,R_3)` labels a Bravais-lattice translation and
    :math:`i,j` label PAO basis functions in the reference cell. Doubling a
    lattice direction creates two translated copies of the PAO basis. For one
    doubling along direction :math:`\alpha`, each new sparse block has the form

    .. math::

        H'(R) =
        \begin{pmatrix}
        H(2R_\alpha) & H(2R_\alpha+1) \\
        H(2R_\alpha-1) & H(2R_\alpha)
        \end{pmatrix},

    with the components perpendicular to :math:`\alpha` unchanged. No
    physical-distance cutoff is applied during doubling. After each assembled
    sparse block is built, optional magnitude pruning may remove whole
    translations whose Hamiltonian amplitude is negligible compared with the
    largest assembled block.

    Parameters
    ----------
    data_controller : PAOFLOW DataController
        Object exposing PAOFLOW's ``data_dicts()`` method. The arrays
        dictionary must contain ``SparseHRs`` and the attributes dictionary must
        contain ``nx``, ``ny``, ``nz``, ``nk1``, ``nk2``, ``nk3``, ``nspin``,
        ``nawf`` and ``alat``.

    Returns
    -------
    None
        The arrays and attributes in ``data_controller`` are updated in place.
        ``arrays['SparseHRs']`` remains a sparse block dictionary and dense
        ``arrays['HRs']`` is removed when present. The transformation preserves
        the sparse block set implied by the dense doubling map before optional
        magnitude pruning.

    Raises
    ------
    KeyError
        If ``SparseHRs`` is missing.
    TypeError
        If a stored Hamiltonian block is not sparse.

    Notes
    -----
    This routine deliberately does not rebuild the dense displacement tensor
    ``Dnm``. For large sparse systems, storing :math:`D_{ij}` for all PAO pairs
    is an :math:`O(N^2)` dense operation. Sparse downstream routines should use
    PAO positions only for the nonzero Hamiltonian entries they consume.
    """
    arrays, attr = data_controller.data_dicts()

    if 'SparseHRs' not in arrays:
        raise KeyError("Sparse doubling requires arrays['SparseHRs']; dense HRs are not used.")

    nk1 = int(attr['nk1'])
    nk2 = int(attr['nk2'])
    nk3 = int(attr['nk3'])
    nspin = int(attr['nspin'])

    signed_to_grid, grid_to_signed, mins, maxs = _signed_grid_maps(nk1, nk2, nk3)
    blocks = _as_hr_dict(arrays['SparseHRs'], nspin=nspin)

    # Dense PAOFLOW doubling does not apply a physical-distance cutoff to
    # real-space Hamiltonian blocks.  The sparse implementation therefore
    # first assembles the literal sparse counterpart of dense doubling, then
    # optionally removes only whole assembled blocks whose sparse data norm is
    # negligible compared with the largest assembled block.

    for axis, repeats in enumerate((int(attr['nx']), int(attr['ny']), int(attr['nz']))):
        for _ in range(repeats):
            old_nawf = int(attr['nawf'])
            blocks = _double_HRs_along_axis(
                blocks,
                axis=axis,
                signed_to_grid=signed_to_grid,
                grid_to_signed=grid_to_signed,
                mins=mins,
                maxs=maxs,
                n=old_nawf,
                nspin=nspin,
                arrays=arrays,
                attr=attr,
            )
            blocks = _prune_blocks_by_magnitude(
                blocks,
                grid_to_signed=grid_to_signed,
                rel_tol=float(attr.get('sparse_doubling_block_rel_tol', 1.0e-7)),
                abs_tol=float(attr.get('sparse_doubling_block_abs_tol', 1e-5)),
                enabled=bool(attr.get('sparse_doubling_block_pruning', True)),
                keep_origin=bool(attr.get('sparse_doubling_keep_origin_block', True)),
            )
            arrays['SparseHRs'] = blocks
            arrays.pop('HRs', None)
            _double_basis_geometry(arrays, attr, axis)
            _double_attributes_and_arrays(arrays, attr)

            arrays['SparseHRs'] = blocks
            attr['sparse_doubling_uses_automatic_physical_rcut'] = False

    report_sparse_hr_stats(arrays, attr)


def _prune_blocks_by_magnitude(
    blocks: SparseHRDict,
    *,
    grid_to_signed: Mapping[tuple[int, int, int], tuple[int, int, int]],
    rel_tol: float,
    abs_tol: float,
    enabled: bool,
    keep_origin: bool,
) -> SparseHRDict:
    r"""Drop whole sparse Hamiltonian blocks with negligible amplitude.

    The assembled doubled Hamiltonian contains one sparse matrix block for each
    real-space translation :math:`R` and spin channel. This helper removes only
    entire translations whose sparse matrix norm is small,

    .. math::

        \|H(R)\|_F < \epsilon_{\mathrm{rel}}
        \max_{R',s}\|H_s(R')\|_F,

    unless the block contains at least one matrix element whose magnitude is
    above the absolute tolerance. The :math:`R=0` block is kept by default so
    that onsite energies and intra-cell hybridizations are never removed by a
    relative threshold.

    Parameters
    ----------
    blocks : dict
        Sparse real-space Hamiltonian blocks keyed by ``(i, j, k, spin)`` after
        one exact sparse doubling step.
    grid_to_signed : mapping
        Map from stored FFT-grid indices to signed real-space labels
        :math:`R=(R_1,R_2,R_3)`.
    rel_tol : float
        Relative Frobenius-norm tolerance. A block is kept when its Frobenius
        norm is at least ``rel_tol`` times the largest Frobenius norm among all
        assembled blocks. Set to ``0`` to disable the relative test.
    abs_tol : float
        Absolute element-magnitude tolerance. A block is kept when any stored
        Hamiltonian element has magnitude at least ``abs_tol``. Set to ``0`` to
        disable the absolute test.
    enabled : bool
        If ``False``, return the input blocks unchanged.
    keep_origin : bool
        If ``True``, always keep blocks with :math:`R=0`.

    Returns
    -------
    dict
        Sparse block dictionary after whole-block magnitude pruning.

    Notes
    -----
    The calculation uses only ``block.data`` from each sparse matrix. It does
    not form dense matrices and does not prune individual matrix elements.
    """
    if not enabled or not blocks:
        return blocks
    if rel_tol < 0.0:
        raise ValueError('sparse_doubling_block_rel_tol must be non-negative.')
    if abs_tol < 0.0:
        raise ValueError('sparse_doubling_block_abs_tol must be non-negative.')

    metrics: dict[SparseHRKey, tuple[float, float]] = {}
    max_frobenius = 0.0
    for key, block in blocks.items():
        frobenius, max_abs = _sparse_block_magnitude(block)
        metrics[key] = (frobenius, max_abs)
        if frobenius > max_frobenius:
            max_frobenius = frobenius

    if max_frobenius == 0.0:
        return {}

    rel_cutoff = rel_tol * max_frobenius
    use_abs_tol = abs_tol > 0.0
    pruned: SparseHRDict = {}
    dropped_blocks = 0
    dropped_nnz = 0

    for key, block in blocks.items():
        i, j, k, spin = key
        label = grid_to_signed.get((i, j, k))
        if keep_origin and label == (0, 0, 0):
            pruned[key] = block
            continue

        frobenius, max_abs = metrics[key]
        keep_by_relative_norm = rel_tol == 0.0 or frobenius >= rel_cutoff
        keep_by_absolute_element = use_abs_tol and max_abs >= abs_tol
        if keep_by_relative_norm or keep_by_absolute_element:
            pruned[key] = block
        else:
            dropped_blocks += 1
            dropped_nnz += int(block.nnz)

    return pruned


def _sparse_block_magnitude(block: sparse.spmatrix) -> tuple[float, float]:
    r"""Return sparse Frobenius and maximum-entry magnitudes for one block.

    Parameters
    ----------
    block : scipy.sparse.spmatrix
        Sparse matrix representing one real-space Hamiltonian block
        :math:`H_s(R)`.

    Returns
    -------
    frobenius : float
        Frobenius norm :math:`\sqrt{\sum_{ij}|H_{ij}(R)|^2}` evaluated from the
        stored sparse entries.
    max_abs : float
        Largest stored element magnitude :math:`\max_{ij}|H_{ij}(R)|`.
    """
    if not sparse.issparse(block):
        raise TypeError('Magnitude pruning requires sparse Hamiltonian blocks.')
    if block.nnz == 0:
        return 0.0, 0.0
    data = block.data
    frobenius = float(np.sqrt(np.vdot(data, data).real))
    max_abs = float(np.max(np.abs(data)))
    return frobenius, max_abs


def _automatic_physical_rcut(arrays: Mapping[str, Any], attr: Mapping[str, Any]) -> float:
    r"""Return the automatic minimum-image cutoff radius.

    The cutoff is the largest sphere guaranteed to fit inside the parallelepiped
    generated by the current lattice vectors when measured only by primitive
    vector lengths,

    .. math::

        R_{\mathrm{cut}} = \frac{1}{2}\min_\alpha |a_\alpha|.

    Here :math:`a_\alpha` are the physical lattice vectors, including the
    PAOFLOW lattice scale ``alat``. A real-space Hamiltonian block :math:`H(R)`
    is kept only if the physical translation vector lies within this radius.

    Parameters
    ----------
    arrays : mapping
        PAOFLOW arrays dictionary containing ``a_vectors``.
    attr : mapping
        PAOFLOW attributes dictionary containing ``alat``.

    Returns
    -------
    float
        Automatic physical cutoff radius in the same length units as
        ``alat * a_vectors``.
    """
    lattice = _physical_lattice_vectors(arrays, attr)
    lengths = np.linalg.norm(lattice, axis=1)
    positive = lengths[lengths > 0.0]
    if positive.size != 3:
        raise ValueError('Automatic Rcut requires three nonzero lattice vectors.')
    return 0.5 * float(np.min(positive))


def _physical_lattice_vectors(arrays: Mapping[str, Any], attr: Mapping[str, Any]) -> np.ndarray:
    r"""Return physical lattice vectors as rows of a small ``(3, 3)`` array.

    Parameters
    ----------
    arrays : mapping
        PAOFLOW arrays dictionary containing dimensionless lattice vectors
        ``a_vectors``.
    attr : mapping
        PAOFLOW attributes dictionary containing the lattice scale ``alat``.

    Returns
    -------
    numpy.ndarray
        Matrix whose row :math:`\alpha` is the physical primitive vector
        :math:`a_\alpha`.
    """
    if 'a_vectors' not in arrays:
        raise KeyError("Automatic physical Rcut requires arrays['a_vectors'].")
    if 'alat' not in attr:
        raise KeyError("Automatic physical Rcut requires attr['alat'].")
    lattice = np.asarray(arrays['a_vectors'], dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError("arrays['a_vectors'] must have shape (3, 3).")
    return float(attr['alat']) * lattice


def _translation_norm(
    label: tuple[int, int, int], arrays: Mapping[str, Any], attr: Mapping[str, Any]
) -> float:
    r"""Return the physical length of one real-space lattice translation.

    Parameters
    ----------
    label : tuple of int
        Signed lattice label :math:`(R_1,R_2,R_3)`.
    arrays : mapping
        PAOFLOW arrays dictionary containing ``a_vectors``.
    attr : mapping
        PAOFLOW attributes dictionary containing ``alat``.

    Returns
    -------
    float
        Norm of :math:`R_1 a_1 + R_2 a_2 + R_3 a_3`.
    """
    lattice = _physical_lattice_vectors(arrays, attr)
    vector = np.asarray(label, dtype=float) @ lattice
    return float(np.linalg.norm(vector))


def _within_physical_rcut(
    label: tuple[int, int, int], arrays: Mapping[str, Any], attr: Mapping[str, Any], rcut: float
) -> bool:
    r"""Test whether a signed real-space translation lies inside ``Rcut``.

    Parameters
    ----------
    label : tuple of int
        Signed real-space lattice label.
    arrays : mapping
        PAOFLOW arrays dictionary containing ``a_vectors``.
    attr : mapping
        PAOFLOW attributes dictionary containing ``alat``.
    rcut : float
        Physical cutoff radius.

    Returns
    -------
    bool
        ``True`` when the block at this translation is kept.
    """
    scale = max(1.0, abs(rcut))
    return _translation_norm(label, arrays, attr) <= rcut + _R_CUT_REL_TOL * scale


def _prune_blocks_by_physical_rcut(
    blocks: SparseHRDict,
    *,
    grid_to_signed: Mapping[tuple[int, int, int], tuple[int, int, int]],
    arrays: Mapping[str, Any],
    attr: Mapping[str, Any],
    rcut: float,
) -> SparseHRDict:
    r"""Drop whole sparse Hamiltonian blocks outside the physical cutoff.

    Parameters
    ----------
    blocks : dict
        Sparse real-space Hamiltonian blocks keyed by ``(i, j, k, spin)``.
    grid_to_signed : mapping
        Map from stored grid index to signed lattice label.
    arrays : mapping
        PAOFLOW arrays dictionary containing ``a_vectors``.
    attr : mapping
        PAOFLOW attributes dictionary containing ``alat``.
    rcut : float
        Physical cutoff radius.

    Returns
    -------
    dict
        Sparse block dictionary containing only translations within ``Rcut``.
    """
    pruned: SparseHRDict = {}
    for (i, j, k, spin), block in blocks.items():
        label = grid_to_signed.get((i, j, k))
        if label is None:
            continue
        if _within_physical_rcut(label, arrays, attr, rcut):
            pruned[(i, j, k, spin)] = block
    return pruned


def _signed_grid_maps(
    nk1: int, nk2: int, nk3: int
) -> tuple[
    dict[tuple[int, int, int], tuple[int, int, int]],
    dict[tuple[int, int, int], tuple[int, int, int]],
    np.ndarray,
    np.ndarray,
]:
    r"""Map signed real-space lattice labels and FFT-grid indices.

    PAOFLOW stores :math:`H(R)` on the FFT grid but the doubling construction is
    easier to express with signed lattice labels centered around the reference
    cell. This helper reproduces the integer relabeling used by the dense
    implementation while avoiding any Hamiltonian-sized dense object.

    Parameters
    ----------
    nk1, nk2, nk3 : int
        Number of real-space grid points along the three reciprocal-grid
        directions.

    Returns
    -------
    signed_to_grid : dict
        Map from signed lattice label ``(Rx, Ry, Rz)`` to stored grid index
        ``(i, j, k)``.
    grid_to_signed : dict
        Inverse map from stored grid index to signed lattice label.
    mins, maxs : numpy.ndarray
        Minimum and maximum signed labels in each lattice direction.
    """
    signed_to_grid: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    grid_to_signed: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    labels: list[tuple[int, int, int]] = []

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                rx = _signed_label(i, nk1)
                ry = _signed_label(j, nk2)
                rz = _signed_label(k, nk3)
                signed = (rx, ry, rz)
                grid = (i, j, k)
                signed_to_grid[signed] = grid
                grid_to_signed[grid] = signed
                labels.append(signed)

    label_array = np.asarray(labels, dtype=np.int64)
    return signed_to_grid, grid_to_signed, label_array.min(axis=0), label_array.max(axis=0)


def _signed_label(index: int, size: int) -> int:
    r"""Return PAOFLOW's signed lattice label for one FFT-grid index.

    Parameters
    ----------
    index : int
        Stored grid index along one lattice direction.
    size : int
        Number of grid points along the same direction.

    Returns
    -------
    int
        Signed lattice label used in the real-space Hamiltonian convention.
    """
    r = float(index) / float(size)
    if r >= 0.5:
        r -= 1.0
    r -= int(r)
    return int(-round(r * size, 0))


def _as_hr_dict(obj: Any, *, nspin: int) -> SparseHRDict:
    r"""Normalize supported sparse-Hamiltonian containers to a block dictionary.

    Parameters
    ----------
    obj : mapping or numpy.ndarray or object
        Sparse real-space Hamiltonian storage. Supported forms are mappings with
        ``(i, j, k, spin)`` keys, mappings with ``(i, j, k)`` keys whose values
        are spin-indexed containers, object arrays containing sparse blocks, or
        sparse-container objects exposing ``blocks`` keyed by ``(ir, spin)``
        together with ``nk1``, ``nk2``, and ``nk3``.
    nspin : int
        Number of spin channels.

    Returns
    -------
    dict
        Dictionary mapping ``(i, j, k, spin)`` to CSR matrices.

    Raises
    ------
    TypeError
        If a Hamiltonian block is not a SciPy sparse matrix.
    ValueError
        If a key has an unsupported form.
    """
    out: SparseHRDict = {}

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if len(key) == 4:
                i, j, k, spin = (int(v) for v in key)
                _insert_block(out, (i, j, k, spin), value)
            elif len(key) == 3:
                i, j, k = (int(v) for v in key)
                if sparse.issparse(value):
                    if nspin != 1:
                        raise ValueError(
                            'SparseHRs keys without spin are only unambiguous for nspin == 1.'
                        )
                    _insert_block(out, (i, j, k, 0), value)
                else:
                    for spin, block in enumerate(value):
                        _insert_block(out, (i, j, k, spin), block)
            else:
                raise ValueError(f'Unsupported SparseHRs key shape: {key!r}')
        return out

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.ndim == 4:
            for index in np.ndindex(obj.shape):
                block = obj[index]
                if block is not None:
                    _insert_block(out, tuple(int(v) for v in index), block)
            return out
        if obj.ndim == 3 and nspin == 1:
            for i, j, k in np.ndindex(obj.shape):
                block = obj[i, j, k]
                if block is not None:
                    _insert_block(out, (i, j, k, 0), block)
            return out

    # SparseHRs-style container: blocks keyed by flattened (ir, spin), plus
    # grid dimensions needed to recover (i, j, k).
    if (
        hasattr(obj, 'blocks')
        and hasattr(obj, 'nk1')
        and hasattr(obj, 'nk2')
        and hasattr(obj, 'nk3')
    ):
        blocks = getattr(obj, 'blocks')
        nk1 = int(getattr(obj, 'nk1'))
        nk2 = int(getattr(obj, 'nk2'))
        nk3 = int(getattr(obj, 'nk3'))

        if not isinstance(blocks, Mapping):
            raise TypeError('SparseHRs.blocks must be a mapping keyed by (ir, spin).')

        nrtot = nk1 * nk2 * nk3
        for key, value in blocks.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f'Unsupported SparseHRs.blocks key shape: {key!r}')
            ir, spin = (int(v) for v in key)
            if ir < 0 or ir >= nrtot:
                raise ValueError(f'SparseHRs block index ir={ir} out of range [0, {nrtot}).')
            if spin < 0 or spin >= nspin:
                raise ValueError(f'SparseHRs block spin={spin} out of range [0, {nspin}).')

            i = ir // (nk2 * nk3)
            rem = ir % (nk2 * nk3)
            j = rem // nk3
            k = rem % nk3
            _insert_block(out, (i, j, k, spin), value)
        return out

    raise TypeError(
        'Unsupported SparseHRs container. Expected a mapping, object array of sparse blocks, '
        'or a sparse container object with blocks and nk-grid attributes.'
    )


def _insert_block(out: SparseHRDict, key: SparseHRKey, block: Any) -> None:
    r"""Insert one Hamiltonian block after sparse validation.

    Parameters
    ----------
    out : dict
        Destination dictionary for sparse :math:`H(R)` blocks.
    key : tuple of int
        Grid and spin key ``(i, j, k, spin)``.
    block : scipy.sparse.spmatrix
        Sparse PAO Hamiltonian block.

    Returns
    -------
    None
        ``out`` is updated in place.
    """
    if not sparse.issparse(block):
        raise TypeError(f'SparseHRs block at {key!r} is not sparse.')
    csr = block.tocsr()
    csr.eliminate_zeros()
    if csr.nnz:
        out[key] = csr


def _double_HRs_along_axis(
    blocks: SparseHRDict,
    *,
    axis: int,
    signed_to_grid: Mapping[tuple[int, int, int], tuple[int, int, int]],
    grid_to_signed: Mapping[tuple[int, int, int], tuple[int, int, int]],
    mins: np.ndarray,
    maxs: np.ndarray,
    n: int,
    nspin: int,
    arrays: Mapping[str, Any],
    attr: Mapping[str, Any],
) -> SparseHRDict:
    r"""Double sparse :math:`H(R)` blocks along one lattice direction.

    Parameters
    ----------
    blocks : dict
        Current sparse real-space Hamiltonian blocks keyed by grid index and
        spin, ``(i, j, k, spin)``.
    axis : int
        Direction of the doubled lattice vector: ``0``, ``1`` or ``2``.
    signed_to_grid : mapping
        Map from signed lattice labels to stored grid indices.
    grid_to_signed : mapping
        Map from stored grid indices to signed lattice labels.
    mins, maxs : numpy.ndarray
        Minimum and maximum signed lattice labels in each direction.
    n : int
        Number of PAO basis functions before doubling.
    nspin : int
        Number of spin channels.
    arrays : mapping
        PAOFLOW arrays dictionary containing the current lattice vectors.
    attr : mapping
        PAOFLOW attributes dictionary containing the current lattice scale.

    Returns
    -------
    dict
        Doubled sparse Hamiltonian blocks keyed by grid index and spin.
    """
    doubled: SparseHRDict = {}

    for rx in range(int(mins[0]), int(maxs[0]) + 1):
        for ry in range(int(mins[1]), int(maxs[1]) + 1):
            for rz in range(int(mins[2]), int(maxs[2]) + 1):
                r_new = (rx, ry, rz)
                grid_new = signed_to_grid.get(r_new)
                if grid_new is None:
                    continue
                r_even = [rx, ry, rz]
                r_plus = [rx, ry, rz]
                r_minus = [rx, ry, rz]
                r_even[axis] = 2 * r_even[axis]
                r_plus[axis] = 2 * r_plus[axis] + 1
                r_minus[axis] = 2 * r_minus[axis] - 1

                grid_even = _grid_if_in_range(tuple(r_even), axis, signed_to_grid, mins, maxs)
                grid_plus = _grid_if_in_range(tuple(r_plus), axis, signed_to_grid, mins, maxs)
                grid_minus = _grid_if_in_range(tuple(r_minus), axis, signed_to_grid, mins, maxs)

                for spin in range(nspin):
                    h00 = _get_block(blocks, grid_even, spin)
                    h01 = _get_block(blocks, grid_plus, spin)
                    h10 = _get_block(blocks, grid_minus, spin)

                    if h00 is None and h01 is None and h10 is None:
                        continue

                    block = _assemble_doubled_block(h00, h01, h10, n)
                    if block.nnz:
                        doubled[(*grid_new, spin)] = block

    return doubled


def _grid_if_in_range(
    label: tuple[int, int, int],
    axis: int,
    signed_to_grid: Mapping[tuple[int, int, int], tuple[int, int, int]],
    mins: np.ndarray,
    maxs: np.ndarray,
) -> tuple[int, int, int] | None:
    r"""Return a stored grid index only when a signed label is valid.

    Parameters
    ----------
    label : tuple of int
        Signed lattice label :math:`R`.
    axis : int
        Direction whose range is being tested after the doubling map.
    signed_to_grid : mapping
        Map from signed lattice labels to FFT-grid indices.
    mins, maxs : numpy.ndarray
        Allowed signed-label bounds.

    Returns
    -------
    tuple of int or None
        Stored grid index, or ``None`` when the coupling lies outside the
        original real-space grid.
    """
    if label[axis] < mins[axis] or label[axis] > maxs[axis]:
        return None
    return signed_to_grid.get(label)


def _get_block(
    blocks: SparseHRDict, grid: tuple[int, int, int] | None, spin: int
) -> sparse.csr_matrix | None:
    r"""Fetch one sparse Hamiltonian block.

    Parameters
    ----------
    blocks : dict
        Sparse :math:`H(R)` blocks keyed by ``(i, j, k, spin)``.
    grid : tuple of int or None
        Real-space grid index. ``None`` denotes an absent block.
    spin : int
        Spin channel.

    Returns
    -------
    scipy.sparse.csr_matrix or None
        Requested sparse block, or ``None`` if no block is stored.
    """
    if grid is None:
        return None
    return blocks.get((*grid, spin))


def _assemble_doubled_block(
    h00: sparse.spmatrix | None,
    h01: sparse.spmatrix | None,
    h10: sparse.spmatrix | None,
    n: int,
) -> sparse.csr_matrix:
    r"""Assemble one doubled PAO Hamiltonian block sparsely.

    Parameters
    ----------
    h00, h01, h10 : scipy.sparse.spmatrix or None
        Sparse blocks entering the diagonal, upper-right and lower-left sectors
        of the doubled Hamiltonian. ``None`` represents an exactly zero block.
    n : int
        Number of PAO basis functions before doubling.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape ``(2*n, 2*n)``.
    """
    zero = sparse.csr_matrix((n, n), dtype=np.complex128)
    h00 = zero if h00 is None else h00.tocsr()
    h01 = zero if h01 is None else h01.tocsr()
    h10 = zero if h10 is None else h10.tocsr()

    block = sparse.bmat(((h00, h01), (h10, h00)), format='csr')
    block.eliminate_zeros()
    return block


def _double_basis_geometry(arrays: dict[str, Any], attr: dict[str, Any], axis: int) -> None:
    r"""Update basis positions and lattice vector after one cell doubling.

    Parameters
    ----------
    arrays : dict
        PAOFLOW arrays dictionary containing ``tau`` and ``a_vectors``.
    attr : dict
        PAOFLOW attributes dictionary containing ``alat``.
    axis : int
        Lattice direction whose primitive vector is doubled.

    Returns
    -------
    None
        ``tau`` and ``a_vectors`` are updated in place.
    """
    if 'tau' in arrays:
        shift = arrays['a_vectors'][axis, :] * attr['alat']
        arrays['tau'] = np.concatenate((arrays['tau'], arrays['tau'] + shift[None, :]), axis=0)

    arrays['a_vectors'][axis, :] = 2 * arrays['a_vectors'][axis, :]


def _double_attributes_and_arrays(arrays: dict[str, Any], attr: dict[str, Any]) -> None:
    r"""Update scalar counts and one-dimensional basis metadata after doubling.

    Parameters
    ----------
    arrays : dict
        PAOFLOW arrays dictionary.
    attr : dict
        PAOFLOW attributes dictionary.

    Returns
    -------
    None
        Metadata is updated in place. Dense :math:`N \times N` metadata that
        would scale quadratically with the PAO basis is removed rather than
        densely doubled.
    """
    attr['omega'] = attr['alat'] ** 3 * arrays['a_vectors'][0, :].dot(
        np.cross(arrays['a_vectors'][1, :], arrays['a_vectors'][2, :])
    )

    attr['nawf'] = 2 * attr['nawf']
    for name in ('natoms', 'nelec', 'nbnds', 'bnd'):
        if name in attr:
            attr[name] = 2 * attr[name]

    for name in ('naw', 'sh', 'nl', 'atoms'):
        if name in arrays:
            arrays[name] = np.concatenate((arrays[name], arrays[name]), axis=0)

    if attr.get('do_spin_orbit', False):
        for name in ('lambda_p', 'lambda_d', 'orb_pseudo'):
            if name in arrays:
                arrays[name] = np.concatenate((arrays[name], arrays[name]), axis=0)

    if 'Sj' in arrays:
        if sparse.issparse(arrays['Sj']):
            arrays['Sj'] = sparse.block_diag((arrays['Sj'], arrays['Sj']), format='csr')
        elif isinstance(arrays['Sj'], (list, tuple)) and all(
            sparse.issparse(s) for s in arrays['Sj']
        ):
            arrays['Sj'] = [sparse.block_diag((s, s), format='csr') for s in arrays['Sj']]
        else:
            raise TypeError(
                "Sparse doubling cannot safely double dense arrays['Sj']; "
                'store spin operators sparsely or build them after sparse doubling.'
            )

    if 'Dnm' in arrays:
        arrays.pop('Dnm')
        attr['Dnm_invalid_after_sparse_doubling'] = True
