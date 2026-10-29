"""Supercells of the PAO Hamiltonian with arbitrary repetitions.

:meth:`PAOFLOW.PAOFLOW.doubling_Hamiltonian` doubles the cell ``n`` times along a
direction, so it only reaches ``2**n`` repetitions, and it keeps the original
``nk1 x nk2 x nk3`` grid of lattice vectors at every step: the memory of ``HRs``
grows as ``(N nawf)**2 nk1 nk2 nk3`` although most of those supercell vectors hold no
hopping. The functions here build an ``n1 x n2 x n3`` supercell (any integers) in one
step, directly on the much smaller grid of supercell lattice vectors that can hold a
hopping of the primitive cell. Directions can be made finite (ribbons, slabs, wires)
without ever allocating their periodic images, and the range along a direction can be
limited (a transport principal layer only needs ``H(0)`` and ``H(1)``).

The public entry point is :meth:`PAOFLOW.PAOFLOW.supercell_Hamiltonian`, which calls
:func:`check_supercell_prerequisites`, :func:`build_supercell_real_space_hamiltonian`
(MPI rank 0 only), :func:`update_supercell_metadata` and
:func:`update_supercell_k_grid` (all ranks).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..DataController import DataController


# Arrays that exist only after later workflow steps (interpolation, eigenvectors on
# the dense grid, gradients, smearing). They belong to the primitive cell and cannot
# be carried over, so supercell_Hamiltonian must run before those steps.
LATER_STEP_ARRAYS = ('Hksp', 'E_k', 'v_k', 'dHksp', 'pksp', 'd2Hksp', 'deltakp', 'deltakp2')

# Arrays tied to the primitive cell and its DFT k grid (projections, DFT eigenvalues,
# symmetry map of the atoms). They have no meaning in the supercell and are dropped.
PRIMITIVE_CELL_ARRAYS = ('U', 'my_eigsmat', 'equiv_atom')

# Per-atom arrays: one entry per atom, repeated once per cell.
PER_ATOM_ARRAYS = ('atoms', 'naw', 'sh', 'nl', 'lambda_p', 'lambda_d', 'orb_pseudo', 'orb_atom')

# Per-shell lists of the spin operator, in the order of the atoms: repeated once per cell.
PER_SHELL_ARRAYS = ('sh_l', 'sh_j')

# Operators that act within each orbital of a cell, shape (3, nawf, nawf): block diagonal
# in the cells.
CELL_LOCAL_OPERATORS = ('Sj', 'Lj')


def lattice_vector_grid_map(grid_size: int) -> dict[int, tuple[int, float]]:
    """Grid index and weight of every lattice vector stored along one direction of ``HRs``.

    Parameters
    ----------
    grid_size : int
        Number of points of the real-space grid along this direction (``nk1``,
        ``nk2`` or ``nk3``).

    Returns
    -------
    dict[int, tuple[int, float]]
        For every lattice vector ``R`` (integer, in units of the lattice vector of this
        direction) the grid index that holds ``H(R)`` and the weight of that entry.

    Notes
    -----
    PAOFLOW stores ``H(R)`` at grid index ``i`` with ``R = -i`` folded into
    :math:`[-n/2, n/2)` (the convention of ``do_doubling``), so that

    .. math::

        H(\\mathbf{k}) = \\sum_{\\mathbf{R}} H(\\mathbf{R})\\,
        e^{2\\pi i \\mathbf{k}\\cdot\\mathbf{R}}
        = \\mathrm{FFT}[\\mathrm{HRs}](\\mathbf{k}).

    On an even grid the index :math:`n/2` holds the sum of the hoppings to
    :math:`R = +n/2` and :math:`R = -n/2`, which the Fourier transform of the DFT k
    grid cannot separate. It is assigned to both vectors with weight 1/2, which
    reproduces ``H(k)`` on the original grid and keeps the supercell Hamiltonian
    Hermitian.
    """
    vector_map = {}
    for grid_index in range(grid_size):
        # fractional position on the grid, folded into [-1/2, 1/2)
        fractional = float(grid_index) / float(grid_size)
        if fractional >= 0.5:
            fractional -= 1.0
        # grid index i holds H(R) with R = -i (sign of the Fourier transform)
        vector_map[int(-round(fractional * grid_size, 0))] = (grid_index, 1.0)

    # even grid: the zone-boundary entry is shared by R = +n/2 and R = -n/2
    if grid_size % 2 == 0:
        boundary_vector = grid_size // 2
        vector_map[boundary_vector] = vector_map[-boundary_vector] = (boundary_vector, 0.5)
    return vector_map


def supercell_vector_range(
    primitive_grid_shape: Sequence[int],
    repetitions: Sequence[int],
    finite_directions: Sequence[bool],
    max_range: Sequence[Optional[int]],
) -> list[int]:
    """Largest supercell lattice vector kept along each direction.

    Parameters
    ----------
    primitive_grid_shape : sequence of int, length 3
        ``(nk1, nk2, nk3)`` of the primitive ``HRs``.
    repetitions : sequence of int, length 3
        Number of primitive cells along each lattice vector, ``(n1, n2, n3)``.
    finite_directions : sequence of bool, length 3
        Directions without periodicity: only the supercell vector 0 is kept.
    max_range : sequence of int or None, length 3
        Optional upper bound on the supercell vectors kept along each direction.

    Returns
    -------
    list[int], length 3
        The supercell grid along direction ``i`` holds the vectors
        :math:`-R^{\\max}_i, \\dots, R^{\\max}_i`, i.e. ``2 R^max_i + 1`` points.

    Notes
    -----
    A primitive hopping :math:`H(\\mathbf{R})` with :math:`|R_i| \\le R^{p}_i` enters a
    supercell vector :math:`\\mathbf{R}'` only if
    :math:`|n_i R'_i| \\le R^{p}_i + n_i - 1`, i.e.
    :math:`|R'_i| \\le \\lceil R^{p}_i / n_i \\rceil`.
    """
    largest_vectors = []
    for direction in range(3):
        # largest primitive lattice vector along this direction
        primitive_largest = max(lattice_vector_grid_map(primitive_grid_shape[direction]))
        # supercell vectors that can still hold a primitive hopping
        largest = 0 if finite_directions[direction] else -(-primitive_largest // repetitions[direction])
        # optional user limit (e.g. 1 for a transport principal layer)
        if max_range[direction] is not None:
            largest = min(largest, int(max_range[direction]))
        largest_vectors.append(largest)
    return largest_vectors


def build_supercell_real_space_hamiltonian(
    hamiltonian_real_space: NDArray[np.complexfloating],
    repetitions: Sequence[int],
    finite_directions: Sequence[bool] = (False, False, False),
    max_range: Sequence[Optional[int]] = (None, None, None),
) -> NDArray[np.complex128]:
    """Real-space Hamiltonian of an ``n1 x n2 x n3`` supercell.

    Parameters
    ----------
    hamiltonian_real_space : NDArray[np.complexfloating], shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``
        PAOFLOW real-space Hamiltonian ``HRs`` of the primitive cell.
    repetitions : sequence of int, length 3
        Number of primitive cells along each lattice vector, ``(n1, n2, n3)``.
    finite_directions : sequence of bool, length 3, default ``(False, False, False)``
        Directions without periodicity (ribbon, slab, wire): only the supercell
        vector 0 is built, so the periodic images are never allocated.
    max_range : sequence of int or None, length 3, default ``(None, None, None)``
        Keep only the supercell vectors with :math:`|R_i| \\le` ``max_range[i]``
        (e.g. ``1`` along the transport direction of a principal layer).

    Returns
    -------
    NDArray[np.complex128], shape ``(N nawf, N nawf, m1, m2, m3, nspin)``
        Supercell ``HRs`` with ``N = n1 n2 n3`` and ``m_i = 2 R^max_i + 1`` (see
        :func:`supercell_vector_range`), in the same storage convention as the
        primitive ``HRs``.

    Notes
    -----
    The cells of the supercell are ordered with the first direction fastest,
    :math:`c = c_1 + n_1 (c_2 + n_2 c_3)` (the order of ``doubling_Hamiltonian``), and
    the orbitals of cell :math:`c` occupy rows ``c nawf ... (c + 1) nawf - 1``. Block
    :math:`(c, d)` of the supercell Hamiltonian at supercell vector
    :math:`\\mathbf{R}'` is the primitive hopping between the two cells,

    .. math::

        H^{\\rm sc}_{cd}(\\mathbf{R}') = H\\big(\\mathbf{n} \\circ \\mathbf{R}'
        + \\mathbf{c}_d - \\mathbf{c}_c\\big),

    with :math:`\\mathbf{n} \\circ \\mathbf{R}' = (n_1 R'_1, n_2 R'_2, n_3 R'_3)` and
    :math:`\\mathbf{c}_c` the position of cell :math:`c` in primitive lattice vectors.
    Blocks whose primitive vector lies outside the primitive grid are zero. Only the
    supercell vectors that can hold a hopping are stored, so the memory scales as
    :math:`N\\,\\mathrm{nawf}^2` per stored vector instead of the
    :math:`N^2\\,\\mathrm{nawf}^2\\,n_{k1} n_{k2} n_{k3}` of repeated doubling on the
    primitive grid.
    """
    nawf, nspin = hamiltonian_real_space.shape[0], hamiltonian_real_space.shape[5]
    repetitions = np.array(repetitions, dtype=int)
    number_of_cells = int(np.prod(repetitions))

    # where each primitive lattice vector is stored (index, weight), per direction
    primitive_maps = [lattice_vector_grid_map(hamiltonian_real_space.shape[2 + direction])
                      for direction in range(3)]

    # supercell vectors kept and their grid index (same R = -i storage convention)
    largest_vectors = supercell_vector_range(hamiltonian_real_space.shape[2:5], repetitions,
                                             finite_directions, max_range)
    supercell_maps = [{vector: (-vector) % (2 * largest + 1) for vector in range(-largest, largest + 1)}
                      for largest in largest_vectors]

    supercell_hamiltonian = np.zeros((number_of_cells * nawf, number_of_cells * nawf,
                                      *(2 * largest + 1 for largest in largest_vectors), nspin),
                                     dtype=complex)

    # positions of the cells inside the supercell, first direction fastest
    cell_positions = [(c1, c2, c3) for c3 in range(repetitions[2])
                      for c2 in range(repetitions[1]) for c1 in range(repetitions[0])]

    for supercell_vector in itertools.product(*[sorted(vector_map) for vector_map in supercell_maps]):
        supercell_index = tuple(supercell_maps[direction][supercell_vector[direction]] for direction in range(3))
        for row_cell, row_position in enumerate(cell_positions):
            for column_cell, column_position in enumerate(cell_positions):
                # primitive vector between cell 'row' at 0 and cell 'column' at R'
                primitive_vector = [repetitions[direction] * supercell_vector[direction]
                                    + column_position[direction] - row_position[direction]
                                    for direction in range(3)]
                # hoppings beyond the primitive grid are zero
                if not all(primitive_vector[direction] in primitive_maps[direction] for direction in range(3)):
                    continue
                primitive_index = tuple(primitive_maps[direction][primitive_vector[direction]][0]
                                        for direction in range(3))
                weight = np.prod([primitive_maps[direction][primitive_vector[direction]][1]
                                  for direction in range(3)])
                block = hamiltonian_real_space[:, :, primitive_index[0], primitive_index[1], primitive_index[2], :]
                supercell_hamiltonian[row_cell * nawf:(row_cell + 1) * nawf,
                                      column_cell * nawf:(column_cell + 1) * nawf,
                                      supercell_index[0], supercell_index[1], supercell_index[2], :] = \
                    block if weight == 1.0 else weight * block
    return supercell_hamiltonian


def check_supercell_prerequisites(data_controller: 'DataController') -> None:
    """Verify that the supercell can be built at this point of the workflow.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``tau``, ``a_vectors``.

    Returns
    -------
    None
        Nothing is modified.

    Raises
    ------
    RuntimeError
        If the Hamiltonian is missing, the basis is non-orthogonal (``Sks`` or
        ``SRs`` present) or the Hamiltonian was already interpolated or
        differentiated.
    """
    arry, attr = data_controller.data_dicts()

    # the real-space PAO Hamiltonian must exist (pao_hamiltonian has been called)
    if 'HRs' not in arry:
        raise RuntimeError('supercell_Hamiltonian: call pao_hamiltonian() first')

    # the overlaps of a non-orthogonal basis are not carried over
    if 'Sks' in arry or 'SRs' in arry:
        raise RuntimeError('supercell_Hamiltonian: not implemented for a non-orthogonal basis '
                           '(overlaps Sks/SRs present, e.g. acbn0 or save_overlaps)')

    # arrays of later steps belong to the primitive cell
    later_steps = [key for key in LATER_STEP_ARRAYS if key in arry]
    if later_steps:
        raise RuntimeError('supercell_Hamiltonian: call it before interpolated_hamiltonian() and the '
                           'property calculations (found %s)' % ', '.join(later_steps))

    # the orbital count must match the Hamiltonian
    if arry['HRs'].shape[0] != attr['nawf']:
        raise RuntimeError('supercell_Hamiltonian: HRs has %d orbitals but nawf is %d'
                           % (arry['HRs'].shape[0], attr['nawf']))


def update_supercell_metadata(
    data_controller: 'DataController',
    repetitions: Sequence[int],
    build_distance_matrix: bool = True,
) -> None:
    """Repeat the atoms, orbitals and operators of the primitive cell over the supercell.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``, still describing the
        primitive cell (``HRs`` may already be the supercell one).
        Arrays updated when present: ``tau`` (shape ``(natoms, 3)``), ``basis`` and
        ``atomic_basis`` (one entry per orbital), ``Dnm`` (shape ``(nawf, nawf, 3)``),
        ``Sj`` and ``Lj`` (shape ``(3, nawf, nawf)``), the per-atom arrays
        ``atoms``, ``naw``, ``sh``, ``nl``, ``lambda_p``, ``lambda_d``,
        ``orb_pseudo``, ``orb_atom``, the per-shell lists ``sh_l`` and ``sh_j``,
        ``a_vectors`` and ``b_vectors``. Arrays removed: ``U``, ``my_eigsmat``,
        ``equiv_atom``.
        Attributes updated: ``nawf``, ``natoms``, ``nelec``, ``nbnds``, ``bnd``,
        ``omega``.
    repetitions : sequence of int, length 3
        Number of primitive cells along each lattice vector, ``(n1, n2, n3)``.
    build_distance_matrix : bool, default True
        Rebuild ``Dnm`` for the supercell (``3 (N nawf)**2`` floats). If False it is
        removed, which saves memory when no velocities are needed (e.g. transport).

    Returns
    -------
    None
        All modifications are made in place. To be called on every MPI rank with the
        same arguments.

    Notes
    -----
    Cell :math:`c` (first direction fastest) is translated by
    :math:`\\mathbf{t}_c = c_1 \\mathbf{a}_1 + c_2 \\mathbf{a}_2 + c_3 \\mathbf{a}_3`.
    Positions and orbital lists are concatenated cell by cell, in the order of the
    Hamiltonian blocks. ``Dnm`` holds the distance between orbital centres,
    :math:`D_{nm} = \\boldsymbol{\\tau}_n - \\boldsymbol{\\tau}_m`, so it is rebuilt
    from the translated centres (it is not block diagonal: orbitals of different cells
    are at different distances). ``Sj`` and ``Lj`` act within each orbital and are
    block diagonal, :math:`S^{\\rm sc} = \\mathbb{1}_N \\otimes S`. The shell lists
    ``shells``, ``jchia`` and ``lchia`` are per species and need no change.
    """
    arry, attr = data_controller.data_dicts()
    repetitions = [int(n) for n in repetitions]
    number_of_cells = int(np.prod(repetitions))

    # translation of every cell (same units as tau, bohr), first direction fastest
    lattice_vectors = arry['a_vectors'] * attr['alat']
    cell_translations = [c1 * lattice_vectors[0] + c2 * lattice_vectors[1] + c3 * lattice_vectors[2]
                         for c3 in range(repetitions[2])
                         for c2 in range(repetitions[1])
                         for c1 in range(repetitions[0])]

    # atomic positions: all atoms of cell 0, then cell 1, ...
    arry['tau'] = np.concatenate([np.asarray(arry['tau'], dtype=float) + translation
                                  for translation in cell_translations])

    # per-orbital basis descriptions, with the orbital centre translated
    for basis_key in ('basis', 'atomic_basis'):
        if basis_key in arry and arry[basis_key] is not None:
            arry[basis_key] = [dict(orbital, tau=np.asarray(orbital['tau'], dtype=float) + translation)
                               for translation in cell_translations for orbital in arry[basis_key]]

    # orbital-centre distances: rebuilt from the translated centres (or dropped)
    if 'Dnm' in arry:
        if build_distance_matrix:
            # centres relative to orbital 0 (Dnm[n, 0] = tau_n - tau_0), cell by cell
            centres = np.concatenate([arry['Dnm'][:, 0, :] + translation for translation in cell_translations])
            arry['Dnm'] = centres[:, None, :] - centres[None, :, :]
        else:
            del arry['Dnm']

    # spin and orbital angular-momentum operators: block diagonal over the cells
    for operator in CELL_LOCAL_OPERATORS:
        if operator in arry:
            arry[operator] = np.array([np.kron(np.eye(number_of_cells), component)
                                       for component in arry[operator]])

    # per-atom and per-shell arrays: one copy per cell (lists stay lists)
    for key in PER_ATOM_ARRAYS + PER_SHELL_ARRAYS:
        if key in arry and arry[key] is not None:
            if isinstance(arry[key], np.ndarray):
                arry[key] = np.tile(arry[key], number_of_cells)
            else:
                arry[key] = list(arry[key]) * number_of_cells

    # quantities of the primitive cell and its DFT k grid that do not carry over
    for key in PRIMITIVE_CELL_ARRAYS:
        arry.pop(key, None)

    # lattice: a_i -> n_i a_i, reciprocal vectors (units of 2 pi/alat) and volume
    for direction, repetition in enumerate(repetitions):
        arry['a_vectors'][direction, :] = repetition * arry['a_vectors'][direction, :]
    arry['b_vectors'] = np.linalg.inv(arry['a_vectors']).T
    attr['omega'] = attr['alat'] ** 3 * arry['a_vectors'][0, :].dot(
        np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :]))

    # extensive counts: orbitals, atoms, electrons and bands scale with the number of cells
    for key in ('nawf', 'natoms', 'nelec', 'nbnds', 'bnd'):
        if attr.get(key) is not None:
            attr[key] = number_of_cells * attr[key]


def update_supercell_k_grid(data_controller: 'DataController') -> None:
    """Set the k grid that matches the supercell ``HRs`` and rebuild ``Hks`` from it.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` of the supercell (shape
        ``(nawf, nawf, m1, m2, m3, nspin)``), ``b_vectors`` of the supercell.

    Returns
    -------
    None
        Sets the attributes ``nk1``, ``nk2``, ``nk3``, ``nkpnts`` and the arrays
        ``kpnts`` (Cartesian, units of :math:`2\\pi/a`), ``kpnts_wght``, ``kgrid``
        and ``kq_wght``. If ``Hks`` is present on this rank (rank 0 after
        ``pao_hamiltonian``) it is replaced by the Fourier transform of the new
        ``HRs``.

    Notes
    -----
    The supercell ``HRs`` is the Fourier transform of ``H(k)`` on the
    :math:`m_1 \\times m_2 \\times m_3` grid :math:`\\mathbf{k} = (i/m_1, j/m_2, l/m_3)`
    (crystal coordinates of the supercell), so ``Hks = FFT[HRs]`` on that grid, as
    after ``pao_hamiltonian``. Finite directions have :math:`m_i = 1`.
    """
    from ..utils.get_K_grid_fft import get_K_grid_fft

    arry, attr = data_controller.data_dicts()

    # grid sizes are those of the supercell HRs
    grid1, grid2, grid3 = arry['HRs'].shape[2:5]
    attr['nk1'], attr['nk2'], attr['nk3'] = grid1, grid2, grid3
    attr['nkpnts'] = grid1 * grid2 * grid3

    # FFT k grid used by the interpolation and property steps
    get_K_grid_fft(data_controller)

    # explicit k points of the grid, in the order of numpy's fftn
    crystal_kpoints = np.array([[i / grid1, j / grid2, l / grid3]
                                for i in range(grid1) for j in range(grid2) for l in range(grid3)])
    arry['kpnts'] = crystal_kpoints @ arry['b_vectors']
    arry['kpnts_wght'] = np.full(attr['nkpnts'], 1.0 / attr['nkpnts'])

    # k-space Hamiltonian on that grid, where pao_hamiltonian left one
    if arry.get('Hks') is not None:
        arry['Hks'] = np.fft.fftn(arry['HRs'], axes=(2, 3, 4))
