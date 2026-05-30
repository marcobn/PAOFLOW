from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from scipy import sparse

from .get_hr import SparseHRs

if TYPE_CHECKING:
    from ...DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _raise_unsupported_wedge_case(reason: str) -> None:
    r"""Raise when sparse wedge expansion would require dense storage.

    Parameters
    ----------
    reason : str
        Symmetry case that is not yet implemented without materializing a dense
        Hamiltonian.

    Returns
    -------
    None
        This function always raises ``NotImplementedError``.

    Notes
    -----
    The sparse workflow stores each block :math:`H(k)` as a sparse matrix.
    Rebuilding the dense tensor :math:`H_{ij}(k)` is forbidden in this path.
    """
    raise NotImplementedError(
        'Pure sparse H(k) wedge expansion is not implemented for this case: '
        f'{reason}. The sparse path intentionally refuses the legacy dense adapter.'
    )


def _as_csr_operator(matrix: np.ndarray | sparse.spmatrix, *, name: str) -> sparse.csr_matrix:
    r"""Return a PAO symmetry operator as a CSR sparse matrix.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Matrix representation of a symmetry operator :math:`U(k)`.
    name : str
        Operator name used in diagnostics.

    Returns
    -------
    scipy.sparse.csr_matrix
        Square sparse operator with explicit zeros removed.

    Notes
    -----
    This does not densify a Hamiltonian block.  If the operator is
    mathematically dense, the product :math:`UHU^\dagger` may still become
    dense in practice; warnings elsewhere report that case.
    """
    op = matrix.tocsr(copy=True) if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    op.eliminate_zeros()
    if op.shape[0] != op.shape[1]:
        raise ValueError(f'{name} must be square, got shape {op.shape}.')
    return op


def _density(matrix: sparse.spmatrix) -> float:
    r"""Return :math:`\mathrm{nnz}(A)/(N_iN_j)` for a sparse matrix.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Sparse PAO operator or Hamiltonian block.

    Returns
    -------
    float
        Fraction of structurally stored nonzero entries.
    """
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 0.0
    return float(matrix.nnz) / float(matrix.shape[0] * matrix.shape[1])


def _warn_if_density_high(
    matrix: sparse.spmatrix,
    *,
    name: str,
    max_density: float,
    context: str,
) -> None:
    r"""Warn when a sparse PAO object is becoming effectively dense.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Matrix to inspect.
    name : str
        Physical object name.
    max_density : float
        Density threshold for the warning.
    context : str
        Symmetry or transform step being performed.

    Returns
    -------
    None
        Emits a warning but does not stop execution.
    """
    density = _density(matrix)
    if density > max_density:
        warnings.warn(
            f'{name} density is {density:.3%}, above configured sparse warning '
            f'threshold {max_density:.3%} during {context}. Continuing, but this '
            'operation may erase sparse memory savings.',
            RuntimeWarning,
            stacklevel=2,
        )


def _multiply_by_pattern(
    matrix: sparse.spmatrix, pattern: np.ndarray, *, name: str
) -> sparse.csr_matrix:
    r"""Multiply stored sparse entries by an element-wise sign or phase pattern.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Sparse Hamiltonian block with entries :math:`H_{ij}`.
    pattern : numpy.ndarray
        Dense metadata pattern :math:`P_{ij}` evaluated only at stored
        coordinates.
    name : str
        Pattern name used in diagnostics.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse block containing :math:`P_{ij}H_{ij}` only where ``matrix`` had
        stored entries.
    """
    coo = matrix.tocoo(copy=True)
    factors = np.asarray(pattern)[coo.row, coo.col]
    out = sparse.csr_matrix((coo.data * factors, (coo.row, coo.col)), shape=coo.shape)
    out.eliminate_zeros()
    return out


def _time_reversed_block(
    block: sparse.csr_matrix,
    *,
    spin_orbitit: bool,
    u_inv: np.ndarray,
    u_tr: np.ndarray | sparse.spmatrix | None,
) -> sparse.csr_matrix:
    r"""Construct the sparse time-reversed partner of one Hamiltonian block.

    Parameters
    ----------
    block : scipy.sparse.csr_matrix
        Hamiltonian block :math:`H(k)` in the PAO basis.
    spin_orbitit : bool
        Whether spin-orbit coupling is active.
    u_inv : numpy.ndarray
        PAOFLOW inversion sign pattern.
    u_tr : numpy.ndarray or scipy.sparse.spmatrix or None
        Spin-orbit time-reversal rotation.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse PAOFLOW time-reversal image of :math:`H(k)`.
    """
    if not spin_orbitit:
        out = block.conjugate().tocsr(copy=True)
        out.eliminate_zeros()
        return out

    if u_tr is None:
        raise ValueError('u_tr is required for spin-orbit time reversal.')

    u_tr_sp = _as_csr_operator(u_tr, name='u_tr')
    out = (u_tr_sp @ block @ u_tr_sp.getH()).tocsr()
    out = _multiply_by_pattern(out, u_inv, name='u_inv').conjugate().tocsr()
    out.eliminate_zeros()
    return out


def _apply_t_rev_blocks(
    wedge_blocks: list[sparse.csr_matrix],
    kp_red: np.ndarray,
    spin_orbit: bool,
    u_inv: np.ndarray,
    jchia: np.ndarray | None,
) -> tuple[list[sparse.csr_matrix], np.ndarray]:
    r"""Generate sparse time-reversed partners of wedge Hamiltonian blocks.

    Parameters
    ----------
    wedge_blocks : list[scipy.sparse.csr_matrix]
        Sparse Hamiltonian blocks on the symmetry wedge.
    kp_red : numpy.ndarray
        Reduced k-point list with shape ``(nkwedge, 3)``.
    spin_orbit : bool
        Whether spin-orbit coupling is active.
    u_inv : numpy.ndarray
        PAOFLOW time-reversal sign pattern.
    jchia : numpy.ndarray or None
        Spin-orbit angular-momentum data.

    Returns
    -------
    tuple[list[scipy.sparse.csr_matrix], numpy.ndarray]
        Wedge blocks and k-points after appending missing :math:`-k` partners.
    """
    from ..pao_sym import get_U_TR

    if len(wedge_blocks) == 1:
        return wedge_blocks, kp_red

    new_kp_list: list[np.ndarray] = []
    new_blocks: list[sparse.csr_matrix] = []
    u_tr = get_U_TR(jchia) if spin_orbit else None

    for ik, block in enumerate(wedge_blocks):
        new_kp = -kp_red[ik]
        if np.any(np.all(np.isclose(new_kp, kp_red), axis=1)):
            continue

        tr_block = _time_reversed_block(
            block.tocsr(), spin_orbitit=spin_orbit, u_inv=u_inv, u_tr=u_tr
        )
        new_kp_list.append(new_kp)
        new_blocks.append(tr_block)

    if not new_kp_list:
        return wedge_blocks, kp_red

    return wedge_blocks + new_blocks, np.vstack([kp_red, np.asarray(new_kp_list, dtype=float)])


def _enforce_t_rev_blocks(
    full_blocks: dict[int, sparse.csr_matrix],
    nk1: int,
    nk2: int,
    nk3: int,
    spin_orbit: bool,
    u_inv: np.ndarray,
    jchia: np.ndarray | None,
) -> dict[int, sparse.csr_matrix]:
    r"""Enforce time-reversal symmetry on the full sparse FFT grid.

    Parameters
    ----------
    full_blocks : dict[int, scipy.sparse.csr_matrix]
        Sparse Hamiltonian blocks on the full FFT grid.
    nk1, nk2, nk3 : int
        FFT-grid dimensions.
    spin_orbit : bool
        Whether spin-orbit coupling is active.
    u_inv : numpy.ndarray
        PAOFLOW time-reversal sign pattern.
    jchia : numpy.ndarray or None
        Spin-orbit angular-momentum data.

    Returns
    -------
    dict[int, scipy.sparse.csr_matrix]
        Full-grid sparse blocks after enforcing :math:`k \leftrightarrow -k`.
    """
    from ..pao_sym import get_U_TR

    u_tr = get_U_TR(jchia) if spin_orbit else None

    for i in range(int(nk1 / 2) + 1):
        for j in range(int(nk2 / 2) + 1):
            for k in range(int(nk3 / 2) + 1):
                iv = (nk1 - i) % nk1
                jv = (nk2 - j) % nk2
                kv = (nk3 - k) % nk3
                ik = k + j * nk3 + i * nk2 * nk3
                ivk = kv + jv * nk3 + iv * nk2 * nk3

                block = full_blocks[ik].tocsr()
                block_inv = full_blocks[ivk].tocsr()

                if not spin_orbit:
                    full_blocks[ik] = (0.5 * (block + block_inv.conjugate())).tocsr()
                    full_blocks[ivk] = (0.5 * (block_inv + block.conjugate())).tocsr()
                else:
                    full_blocks[ivk] = _time_reversed_block(
                        block, spin_orbitit=True, u_inv=u_inv, u_tr=u_tr
                    )

                full_blocks[ik].eliminate_zeros()
                full_blocks[ivk].eliminate_zeros()

    return full_blocks


def _expand_hks_wedge_serial(
    data_controller: DataController,
    sparse_blocks: dict[int, sparse.csr_matrix],
    nawf: int,
    nspin: int,
    nkpnts_wedge: int,
    nk1: int,
    nk2: int,
    nk3: int,
) -> tuple[dict[int, sparse.csr_matrix], int]:
    """Expand wedge-defined sparse ``H(k)`` blocks to the full FFT mesh.

    Notes
    -----
    A wedge calculation stores ``H(k)`` only on the irreducible part of the
    Brillouin-zone mesh. To perform the inverse FFT to ``H(R)``, the full FFT
    grid is needed. For the supported single-spin case this helper applies the
    symmetry operations directly to each sparse block, avoiding the older dense
    global ``H(k)`` reconstruction. Multi-spin and symmetrized-grid cases still
    use the dense adapter because their symmetry handling is more involved.
    """
    from numpy import linalg as LA

    from ..pao_sym import (
        add_U_TR,
        add_U_wyc,
        build_U_matrix,
        convert_wigner_d,
        correct_roundoff,
        correct_roundoff_kp,
        find_equiv_k,
        get_full_grid,
        get_inv_op,
        get_phase_shifts,
        get_U_k,
        get_wigner,
        get_wigner_so,
        map_equiv_atoms,
    )

    arrays, attr = data_controller.data_dicts()
    assert arrays is not None and attr is not None

    if nspin != 1:
        _raise_unsupported_wedge_case('nspin != 1')

    if attr['symmetrize']:
        _raise_unsupported_wedge_case('symmetrize=True')

    alat = float(attr['alat'])
    spin_orbit = bool(attr['dftSO'])
    mag_calc = bool(attr['dftMAG'])
    o1 = int(attr['ok1'])
    o2 = int(attr['ok2'])
    o3 = int(attr['ok3'])

    atom_pos = np.asarray(arrays['tau'], dtype=float) / alat
    atom_lab = arrays['atoms']
    equiv_atom = arrays['equiv_atom']
    kp_red = np.asarray(arrays['kpnts'], dtype=float)
    b_vectors = np.asarray(arrays['b_vectors'], dtype=float)
    a_vectors = np.asarray(arrays['a_vectors'], dtype=float)
    symop = np.asarray(arrays['sym_rot'], dtype=float)
    sym_tr = np.asarray(arrays['sym_TR'], dtype=bool)

    conv_a = LA.inv(a_vectors)
    atom_pos = atom_pos @ conv_a
    atom_pos = correct_roundoff(atom_pos)
    atom_pos = np.around(atom_pos, decimals=6)

    symop = correct_roundoff(symop)
    symop_cart = np.zeros_like(symop, dtype=float)
    inv_a_vectors = LA.inv(a_vectors)
    for isym in range(symop.shape[0]):
        symop_cart[isym] = inv_a_vectors @ symop[isym] @ a_vectors
    symop_cart = correct_roundoff(symop_cart, incl_hex=True, atol=1.0e-6)

    conv_b = LA.inv(b_vectors)
    kp_red = kp_red @ conv_b
    kp_red = correct_roundoff(kp_red)

    full_grid = get_full_grid(nk1, nk2, nk3, o1, o2, o3)
    kp_red = correct_roundoff_kp(kp_red, full_grid)

    jchia: list[float] = []
    shells: list[int] = []
    a_index: list[int] = []
    sh = arrays['shells']
    for atom_index, atom_label in enumerate(atom_lab):
        atom_shells: list[int] = []
        for shell in sh[atom_label]:
            atom_shells += [shell, shell] if shell == 0 and spin_orbit else [shell]
        shells += atom_shells
        a_index += [atom_index] * int(np.sum([2 * shell + 1 for shell in atom_shells]))
        if spin_orbit:
            jchia += arrays['jchia'][atom_label]

    shells_array = np.asarray(shells)
    a_index_array = np.asarray(a_index)
    jchia_array = np.asarray(jchia)
    u_inv = get_inv_op(shells_array)

    wedge_blocks = [sparse_blocks[ik].tocsr(copy=True) for ik in range(nkpnts_wedge)]
    if not (spin_orbit and mag_calc):
        wedge_blocks, kp_red = _apply_t_rev_blocks(
            wedge_blocks=wedge_blocks,
            kp_red=kp_red,
            spin_orbit=spin_orbit,
            u_inv=u_inv,
            jchia=jchia_array if spin_orbit else None,
        )

    if spin_orbit:
        wigner, inv_flag = get_wigner_so(symop_cart)
    else:
        wigner, inv_flag = get_wigner(symop_cart)
        wigner = convert_wigner_d(wigner)

    phase_shifts = get_phase_shifts(atom_pos, symop, equiv_atom)
    u_matrices = build_U_matrix(wigner, jchia_array if spin_orbit else shells_array)
    if np.any(sym_tr) and spin_orbit:
        u_matrices = add_U_TR(u_matrices, sym_tr, jchia_array)
    u_matrices = add_U_wyc(u_matrices, map_equiv_atoms(a_index_array, equiv_atom))

    new_k_ind, orig_k_ind, si_per_k = find_equiv_k(kp_red, symop, full_grid, sym_tr, check=True)

    expanded_blocks: dict[int, sparse.csr_matrix] = {}
    for nki, oki, isym in zip(new_k_ind, orig_k_ind, si_per_k):
        source_block = wedge_blocks[int(oki)]

        if int(isym) == 0:
            expanded_blocks[int(nki)] = source_block.tocsr(copy=True)
            continue

        u_k = get_U_k(
            kp_red[int(oki)], phase_shifts[int(isym)], a_index_array, u_matrices[int(isym)]
        )
        u_k_sp = _as_csr_operator(u_k, name='u_k')
        max_op_density = float(attr.get('sparse_symmetry_operator_max_density_warn', 0.05))
        max_block_density = float(attr.get('sparse_symmetry_block_max_density_warn', 0.25))
        context = (
            f'symmetry operation {int(isym)} mapping wedge k {int(oki)} to full-grid k {int(nki)}'
        )
        _warn_if_density_high(u_k_sp, name='u_k', max_density=max_op_density, context=context)

        transformed = (u_k_sp @ source_block @ u_k_sp.getH()).tocsr()

        if inv_flag[int(isym)]:
            transformed = _multiply_by_pattern(transformed, u_inv, name='u_inv')
        if sym_tr[int(isym)]:
            if spin_orbit:
                transformed = _multiply_by_pattern(transformed, u_inv, name='u_inv')
            transformed = transformed.conjugate().tocsr()

        transformed = (0.5 * (transformed + transformed.getH())).tocsr()
        transformed.eliminate_zeros()
        _warn_if_density_high(
            transformed, name='transformed H(k)', max_density=max_block_density, context=context
        )
        expanded_blocks[int(nki)] = transformed

    if not (spin_orbit and mag_calc):
        expanded_blocks = _enforce_t_rev_blocks(
            full_blocks=expanded_blocks,
            nk1=nk1,
            nk2=nk2,
            nk3=nk3,
            spin_orbit=spin_orbit,
            u_inv=u_inv,
            jchia=jchia_array if spin_orbit else None,
        )

    attr['nkpnts'] = int(full_grid.shape[0])
    return (
        {ik * nspin: expanded_blocks[ik] for ik in range(int(full_grid.shape[0]))},
        int(full_grid.shape[0]),
    )


def _fold_fft_index(index: int, size: int) -> int:
    r"""Map an FFT-grid index to the nearest signed lattice image.

    Parameters
    ----------
    index : int
        Grid index along one reciprocal direction.
    size : int
        Number of grid points along that direction.

    Returns
    -------
    int
        Signed image index. For example, on a grid of length ``8``, index
        ``7`` maps to ``-1``.
    """
    midpoint = size // 2
    if index > midpoint:
        return index - size
    return index


def _grid_tuple_from_index(index: int, nk2: int, nk3: int) -> tuple[int, int, int]:
    r"""Return the three-dimensional FFT-grid tuple for a flat index.

    Parameters
    ----------
    index : int
        Flat index using PAOFLOW ordering
        :math:`i = k + j N_3 + i_1 N_2 N_3`.
    nk2, nk3 : int
        FFT-grid dimensions for the second and third directions.

    Returns
    -------
    tuple[int, int, int]
        Grid tuple ``(i1, i2, i3)`` associated with the flat index.
    """
    i1 = index // (nk2 * nk3)
    rem = index - i1 * nk2 * nk3
    i2 = rem // nk3
    i3 = rem - i2 * nk3
    return int(i1), int(i2), int(i3)


def _cartesian_r_from_index(
    index: int,
    *,
    nk1: int,
    nk2: int,
    nk3: int,
    a_vectors: np.ndarray,
) -> np.ndarray:
    r"""Return the shortest real-space lattice vector for one FFT-grid index.

    Parameters
    ----------
    index : int
        Flat real-space index using PAOFLOW ordering.
    nk1, nk2, nk3 : int
        FFT-grid dimensions.
    a_vectors : numpy.ndarray
        Direct lattice vectors with shape ``(3, 3)``. The returned vector is
        :math:`R_1 a_1 + R_2 a_2 + R_3 a_3` using folded FFT image indices.

    Returns
    -------
    numpy.ndarray
        Cartesian real-space vector associated with the shortest periodic image
        of ``index``.
    """
    i1, i2, i3 = _grid_tuple_from_index(index, nk2, nk3)
    r1 = _fold_fft_index(i1, nk1)
    r2 = _fold_fft_index(i2, nk2)
    r3 = _fold_fft_index(i3, nk3)
    return r1 * a_vectors[0] + r2 * a_vectors[1] + r3 * a_vectors[2]


def _automatic_hr_distance_cutoff(
    *,
    nk1: int,
    nk2: int,
    nk3: int,
    a_vectors: np.ndarray,
) -> float:
    r"""Return the automatic real-space cutoff for sparse :math:`H(R)`.

    Parameters
    ----------
    nk1, nk2, nk3 : int
        FFT-grid dimensions of the reciprocal mesh.
    a_vectors : numpy.ndarray
        Direct lattice vectors with shape ``(3, 3)``.

    Returns
    -------
    float
        Distance cutoff
        :math:`R_c = \min(3a_{\min}, 0.25L_{\min})`, where
        :math:`a_{\min}` is the shortest primitive lattice-vector length and
        :math:`L_{\min}` is the shortest FFT-supercell length.
    """
    lattice_lengths = np.linalg.norm(a_vectors, axis=1)
    min_lattice_length = float(np.min(lattice_lengths))
    min_supercell_length = float(
        min(nk1 * lattice_lengths[0], nk2 * lattice_lengths[1], nk3 * lattice_lengths[2])
    )
    return min(3.0 * min_lattice_length, 0.25 * min_supercell_length)


def _selected_r_indices_from_attributes(
    attr: dict,
    *,
    nk1: int,
    nk2: int,
    nk3: int,
    a_vectors: np.ndarray,
) -> list[int]:
    r"""Choose which real-space lattice vectors are explicitly transformed.

    Parameters
    ----------
    attr : dict
        PAOFLOW runtime attributes. ``sparse_hr_selected_R_indices`` may hold
        explicit flat R indices. If no explicit list is provided, an automatic
        real-space distance cutoff is used.
    nk1, nk2, nk3 : int
        FFT-grid dimensions defining the real-space supercell.
    a_vectors : numpy.ndarray
        Direct lattice vectors with shape ``(3, 3)``.

    Returns
    -------
    list[int]
        Flat real-space indices to build. Omitted indices are represented by
        exactly zero sparse blocks in ``SparseHRs``.

    Notes
    -----
    The default cutoff is not user supplied. It keeps vectors satisfying
    :math:`|R| \le \min(3a_{\min}, 0.25L_{\min})`, where :math:`a_{\min}` is
    the shortest primitive lattice-vector length and :math:`L_{\min}` is the
    shortest FFT-supercell length. This favors short-ranged PAO Hamiltonians
    while bounding the cost of the sparse Fourier transform.
    """
    nrtot = nk1 * nk2 * nk3

    explicit_indices = attr.get('sparse_hr_selected_R_indices')
    if explicit_indices is not None:
        selected = sorted({int(ir) for ir in explicit_indices})
        invalid = [ir for ir in selected if ir < 0 or ir >= nrtot]
        if invalid:
            raise ValueError(
                f'sparse_hr_selected_R_indices contains out-of-range R indices: {invalid[:8]}'
            )
        return selected

    a_vectors = np.asarray(a_vectors, dtype=float)
    cutoff = _automatic_hr_distance_cutoff(nk1=nk1, nk2=nk2, nk3=nk3, a_vectors=a_vectors)

    selected: list[int] = []
    for ir in range(nrtot):
        r_cart = _cartesian_r_from_index(ir, nk1=nk1, nk2=nk2, nk3=nk3, a_vectors=a_vectors)
        if float(np.linalg.norm(r_cart)) <= cutoff:
            selected.append(ir)

    if not selected:
        raise ValueError(
            'Automatic sparse H(R) distance cutoff selected no R vectors. This should not happen '
            'because R=0 must be inside the cutoff.'
        )

    attr['sparse_hr_automatic_distance_cutoff'] = cutoff
    return selected


def _ifft_phase_for_indices(
    *,
    ik: int,
    r_indices: list[int],
    nk1: int,
    nk2: int,
    nk3: int,
    nrtot: int,
) -> list[np.complex128]:
    r"""Return inverse-FFT phase factors for one k-point and selected R vectors.

    Parameters
    ----------
    ik : int
        Flat k-point index in PAOFLOW ordering.
    r_indices : list[int]
        Flat real-space indices to evaluate.
    nk1, nk2, nk3 : int
        FFT-grid dimensions.
    nrtot : int
        Total number of k-points, :math:`N_k = N_1 N_2 N_3`.

    Returns
    -------
    list[numpy.complex128]
        Phase factors
        :math:`\exp[2\pi i(k_1R_1/N_1+k_2R_2/N_2+k_3R_3/N_3)]/N_k`.

    Notes
    -----
    The positive sign matches ``scipy.fftpack.ifftn``. Only one scalar per
    selected R vector is returned; no dense Fourier matrix is built.
    """
    k1, k2, k3 = _grid_tuple_from_index(ik, nk2, nk3)
    phases: list[np.complex128] = []
    norm = 1.0 / float(nrtot)
    for ir in r_indices:
        r1, r2, r3 = _grid_tuple_from_index(ir, nk2, nk3)
        phase_arg = (k1 * r1 / nk1) + (k2 * r2 / nk2) + (k3 * r3 / nk3)
        phases.append(np.complex128(norm * np.exp(2.0j * np.pi * phase_arg)))
    return phases


def _pruned_csr_from_coo_arrays(
    *,
    row_arrays: list[np.ndarray],
    col_arrays: list[np.ndarray],
    data_arrays: list[np.ndarray],
    shape: tuple[int, int],
    threshold: float,
) -> sparse.csr_matrix:
    r"""Build one pruned sparse real-space Hamiltonian block.

    Parameters
    ----------
    row_arrays, col_arrays, data_arrays : list[numpy.ndarray]
        Coordinate-form contributions to one :math:`H(R)` block. Duplicate PAO
        pairs :math:`(i,j)` are summed before pruning.
    shape : tuple[int, int]
        Matrix shape, normally ``(nawf, nawf)``.
    threshold : float
        Magnitude cutoff applied after the Fourier sum.

    Returns
    -------
    scipy.sparse.csr_matrix
        Pruned sparse real-space Hamiltonian block.

    Notes
    -----
    Pruning after duplicate summation preserves cancellations in the Fourier
    sum better than thresholding each individual k-point contribution.
    """
    if not data_arrays:
        return sparse.csr_matrix(shape, dtype=np.complex128)

    block = sparse.coo_matrix(
        (
            np.concatenate(data_arrays).astype(np.complex128, copy=False),
            (
                np.concatenate(row_arrays).astype(np.int32, copy=False),
                np.concatenate(col_arrays).astype(np.int32, copy=False),
            ),
        ),
        shape=shape,
        dtype=np.complex128,
    ).tocsr()
    block.sum_duplicates()

    if threshold > 0.0 and block.nnz:
        block.data[np.abs(block.data) < threshold] = 0.0

    block.eliminate_zeros()
    block.sum_duplicates()
    block.sort_indices()
    return block


def _build_hrs_from_hks(
    hks_global: dict[int, sparse.csr_matrix],
    *,
    nawf: int,
    nspin: int,
    nk1: int,
    nk2: int,
    nk3: int,
    threshold: float,
    attr: dict,
    arrays: dict,
) -> dict[tuple[int, int], sparse.csr_matrix]:
    r"""Transform sparse :math:`H(k)` blocks to sparse :math:`H(R)` blocks.

    Parameters
    ----------
    hks_global : dict[int, scipy.sparse.csr_matrix]
        Full-grid sparse Hamiltonian blocks keyed by ``ik * nspin + ispin``.
    nawf : int
        Number of PAO basis functions. Each block has shape ``(nawf, nawf)``.
    nspin : int
        Number of spin channels.
    nk1, nk2, nk3 : int
        FFT-grid dimensions.
    threshold : float
        Magnitude cutoff for pruning real-space matrix elements after summing
        all k-point contributions.
    attr : dict
        PAOFLOW attributes controlling the sparse transform. Supported keys are
        ``sparse_hr_selected_R_indices`` and ``sparse_hr_rchunk_size``. If no
        explicit R-index list is supplied, an automatic real-lattice distance
        cutoff is used.

    Returns
    -------
    dict[tuple[int, int], scipy.sparse.csr_matrix]
        Sparse real-space Hamiltonian blocks keyed by ``(ir, ispin)``.

    Notes
    -----
    The mathematical transform is

    .. math::

        H_{ij}(R) = \frac{1}{N_k}\sum_k e^{2\pi i k\cdot R} H_{ij}(k).

    This routine never builds dense spectra over all k-points or all PAO-pair
    coordinates. It streams sparse ``H(k)`` blocks into small chunks of selected
    real-space vectors, scales stored entries by scalar Fourier phases, sums
    duplicate PAO pairs, and immediately prunes small matrix elements.
    """
    nrtot = nk1 * nk2 * nk3
    a_vectors = np.asarray(arrays['a_vectors'], dtype=float)
    selected_r = _selected_r_indices_from_attributes(
        attr, nk1=nk1, nk2=nk2, nk3=nk3, a_vectors=a_vectors
    )
    selected_r_set = set(selected_r)
    rchunk_size = max(1, int(attr.get('sparse_hr_rchunk_size', 4)))
    hr_blocks: dict[tuple[int, int], sparse.csr_matrix] = {}

    if rank == 0 and len(selected_r) < nrtot:
        warnings.warn(
            f'Sparse H(k)->H(R) transform selected {len(selected_r)} of {nrtot} R vectors '
            f'using automatic distance cutoff {attr.get("sparse_hr_automatic_distance_cutoff", float("nan")):.6g}. '
            'Unselected H(R) blocks will be stored as exact sparse zero matrices.',
            RuntimeWarning,
            stacklevel=2,
        )

    empty_block = sparse.csr_matrix((nawf, nawf), dtype=np.complex128)

    for ispin in range(nspin):
        for ir in range(nrtot):
            if ir not in selected_r_set:
                hr_blocks[(ir, ispin)] = empty_block.copy()

        for start in range(0, len(selected_r), rchunk_size):
            r_chunk = selected_r[start : start + rchunk_size]
            rows_by_r: list[list[np.ndarray]] = [[] for _ in r_chunk]
            cols_by_r: list[list[np.ndarray]] = [[] for _ in r_chunk]
            vals_by_r: list[list[np.ndarray]] = [[] for _ in r_chunk]

            for ik in range(nrtot):
                block = hks_global[ik * nspin + ispin]
                if block.nnz == 0:
                    continue

                coo = block.tocoo(copy=False)
                phases = _ifft_phase_for_indices(
                    ik=ik,
                    r_indices=r_chunk,
                    nk1=nk1,
                    nk2=nk2,
                    nk3=nk3,
                    nrtot=nrtot,
                )

                for local_ir, phase in enumerate(phases):
                    rows_by_r[local_ir].append(coo.row)
                    cols_by_r[local_ir].append(coo.col)
                    vals_by_r[local_ir].append(coo.data * phase)

            for local_ir, ir in enumerate(r_chunk):
                hr_blocks[(ir, ispin)] = _pruned_csr_from_coo_arrays(
                    row_arrays=rows_by_r[local_ir],
                    col_arrays=cols_by_r[local_ir],
                    data_arrays=vals_by_r[local_ir],
                    shape=(nawf, nawf),
                    threshold=threshold,
                )

    return hr_blocks


def do_Hks_to_HRs(data_controller: DataController) -> None:
    """Build the sparse real-space Hamiltonian from sparse ``H(k)`` blocks.

    Parameters
    ----------
    data_controller : DataController
        Runtime container holding the distributed sparse ``H(k)`` blocks and the
        FFT-grid metadata.

    Returns
    -------
    None
        Stores the real-space Hamiltonian in ``arrays['SparseHRs']`` and updates
        the sparse-dispatch attributes used by later stages.

    Notes
    -----
    This routine performs the boundary step from reciprocal-space sparse blocks
    to the sparse real-space object ``SparseHRs``. The key physical relation is
    the inverse Fourier transform from ``H(k)`` to ``H(R)``. The sparse version
    avoids materializing a dense global ``H(k)`` tensor during that transform.
    For supported wedge cases it also expands the wedge blockwise before the
    transform, again without rebuilding a dense full-grid Hamiltonian.
    """

    arry, attr = data_controller.data_dicts()

    if 'Hks_sparse' not in arry:
        raise KeyError('Hks_sparse')

    nawf = int(attr['nawf'])
    nspin = int(attr['nspin'])
    nkpnts = int(attr['nkpnts'])
    nk1 = int(attr['nk1'])
    nk2 = int(attr['nk2'])
    nk3 = int(attr['nk3'])

    sparse_blocks = arry['Hks_sparse']

    if not isinstance(sparse_blocks, dict):
        raise TypeError('Hks_sparse must be dict[int, csr_matrix] in sparse H(k)->H(R) paths.')

    local_payload = sparse_blocks

    if size == 1:
        gathered_payloads = [local_payload]
    else:
        gathered_payloads = comm.gather(local_payload, root=0)

    if rank == 0:
        assert gathered_payloads is not None
        hks_global: dict[int, sparse.csr_matrix] = {}
        seen_blocks = set()

        for rank_payload in gathered_payloads:
            for block_idx, sparse_block in rank_payload.items():
                if block_idx in seen_blocks:
                    continue
                hks_global[int(block_idx)] = sparse_block.tocsr()
                seen_blocks.add(block_idx)

        total_blocks = nkpnts * nspin
        if len(seen_blocks) != total_blocks:
            missing = sorted(set(range(total_blocks)) - seen_blocks)
            raise ValueError(f'Missing sparse H(k) blocks for k->R transform: {missing[:8]}')

        if nkpnts != nk1 * nk2 * nk3:
            hks_global, nkpnts = _expand_hks_wedge_serial(
                data_controller=data_controller,
                sparse_blocks=hks_global,
                nawf=nawf,
                nspin=nspin,
                nkpnts_wedge=nkpnts,
                nk1=nk1,
                nk2=nk2,
                nk3=nk3,
            )

        sparse_threshold = attr['sparse_threshold']
        hr_blocks = _build_hrs_from_hks(
            hks_global,
            nawf=nawf,
            nspin=nspin,
            nk1=nk1,
            nk2=nk2,
            nk3=nk3,
            threshold=sparse_threshold,
            attr=attr,
            arrays=arry,
        )

        arry['SparseHRs'] = SparseHRs(
            nawf=nawf,
            nk1=nk1,
            nk2=nk2,
            nk3=nk3,
            nspin=nspin,
            blocks=hr_blocks,
        )

        attr['sparse_hr_boundary'] = 'sparse_object'
        attr['sparse_hr_interpolation_available'] = True
        attr['sparse_bands_interpolation_available'] = True
        attr['sparse_interpolated_hamiltonian_available'] = True

    # Subsequent sparse stages are collective MPI paths. Broadcast the sparse
    # H(R) container and the dispatch attributes once here so every rank takes
    # the same sparse orchestration branch after the H(k)->H(R) boundary.
    arry['SparseHRs'] = comm.bcast(arry.get('SparseHRs') if rank == 0 else None, root=0)

    for key in (
        'nkpnts',
        'sparse_hr_boundary',
        'sparse_hr_interpolation_available',
        'sparse_bands_interpolation_available',
        'sparse_interpolated_hamiltonian_available',
    ):
        attr[key] = comm.bcast(attr.get(key) if rank == 0 else None, root=0)
