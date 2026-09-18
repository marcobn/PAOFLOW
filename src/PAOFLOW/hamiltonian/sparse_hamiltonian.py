"""Sparse real-space PAO Hamiltonian: truncation, storage and restoration.

After the projection step the real-space Hamiltonian ``HRs`` is a dense array of
shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``.  Most of its content is numerical
noise: the PAO Hamiltonian decays quickly with the bond length, so only matrix
elements connecting orbitals on atoms that are close neighbours carry physical
weight.

This module truncates ``HRs`` to a sparse bond list using a *star* (neighbour
shell) criterion, stores it together with everything needed to reproduce a
PAOFLOW run, and rebuilds the dense array on load.

The bond that carries the matrix element :math:`H_{\\mu\\nu}(\\mathbf{R})` is

.. math::

    \\mathbf{d} = \\boldsymbol{\\tau}_B - \\mathbf{R} - \\boldsymbol{\\tau}_A

where :math:`\\mu` sits on atom :math:`A` and :math:`\\nu` on atom :math:`B`.
The sign of :math:`\\mathbf{R}` follows PAOFLOW's own transform
:math:`H(\\mathbf{k}) = \\sum_R H(\\mathbf{R})\\, e^{-2\\pi i \\mathbf{k}\\cdot\\mathbf{R}}`
(``Hks = fftn(HRs)``).  A bond is retained when :math:`|\\mathbf{d}|` lies within
the requested neighbour shell.

Every stored matrix element is tagged with the atoms, species and orbital labels
it connects, so the saved bundle doubles as a labelled dataset for fitting
Hamiltonian matrix elements directly.

The storage layout (flat COO arrays of row, column, lattice translation and
value) mirrors :class:`PAOFLOW.spectrum.sparse_bands.SparseEDTB`.  Note that
``SparseEDTB`` assembles :math:`H(\\mathbf{k})` with :math:`e^{+2\\pi i
\\mathbf{k}\\cdot\\mathbf{R}}`, so reusing these bonds there requires negating
``bond_translation``.
"""

from __future__ import annotations

import json

import numpy as np

#: Bumped whenever the on-disk layout changes in a backward-incompatible way.
SPARSE_HAMILTONIAN_FORMAT_VERSION = 1

#: Chemistry names of the real spherical harmonics in Quantum-ESPRESSO order,
#: indexed by ``m - 1`` for ``m = 1 .. 2l+1`` (see ``calc_ylmg``).
QE_ORBITAL_LABELS_BY_L = {
    0: ('s',),
    1: ('pz', 'px', 'py'),
    2: ('dz2', 'dzx', 'dyz', 'dx2-y2', 'dxy'),
    3: ('fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)'),
}

_UNSERIALIZABLE = object()


# ---------------------------------------------------------------------------
# Orbital -> atom / species / label map
# ---------------------------------------------------------------------------


def _orbital_label(l, m):
    """Return the chemistry name of the ``m``-th real harmonic of shell ``l``."""
    names = QE_ORBITAL_LABELS_BY_L.get(l)
    if names is None or not (1 <= m <= len(names)):
        return f'l{l}m{m}'
    return names[m - 1]


def build_orbital_basis_table(data_controller):
    """Map every PAO basis index onto the atom and orbital it represents.

    Targets the QE-PAOFLOW projection pipeline, where ``arry['basis']`` carries
    the authoritative per-orbital records (both ``projections`` and
    ``read_atomic_proj_QE`` populate it).  Orbital names follow the
    Quantum-ESPRESSO real-harmonic order, ``QE_ORBITAL_LABELS_BY_L``.

    Parameters
    ----------
    data_controller : DataController
        Must carry ``nawf`` and ``tau``.  The orbital identity is read from
        ``basis``, falling back to a species-keyed ``shells`` dictionary.

    Returns
    -------
    dict
        ``orbital_atom`` (int, atom index), ``orbital_species`` (str),
        ``orbital_l`` (int), ``orbital_m`` (int, 1-based within the shell),
        ``orbital_label`` (str, e.g. ``'px'``) and ``orbital_shell`` (str, e.g.
        ``'3P'``) — each of length ``nawf`` — plus ``atom_block_start`` and
        ``orbitals_per_atom``.

    Raises
    ------
    RuntimeError
        If the controller holds a tight-binding model rather than a QE
        projection, or if no description expands to exactly ``nawf`` orbitals.
    """
    arrays, attributes = data_controller.data_dicts()

    number_of_wavefunctions = int(attributes['nawf'])
    atomic_positions = np.asarray(arrays['tau'], dtype=float)
    number_of_atoms = atomic_positions.shape[0]
    atom_species = [str(s) for s in arrays['atoms']]

    orbital_atom = np.zeros(number_of_wavefunctions, dtype=np.int32)
    orbital_l = np.zeros(number_of_wavefunctions, dtype=np.int32)
    orbital_m = np.zeros(number_of_wavefunctions, dtype=np.int32)
    orbital_shell = []
    orbital_label = []

    basis = arrays.get('basis', None)
    shells = arrays.get('shells', None)

    #  Only the tight-binding builders set 'norbitals'.  Their p/d orbitals are
    #  ordered px,py,pz / dxy,... rather than QE's pz,px,py / dz2,..., and the
    #  ordering is not recoverable from what build_TB_model leaves behind, so
    #  refuse rather than emit plausible-looking but wrong labels.
    if 'norbitals' in arrays and basis is None:
        raise RuntimeError(
            'This DataController holds a tight-binding model, not a QE projection. '
            'The sparse Hamiltonian writer targets the QE-PAOFLOW projection pipeline, '
            'whose orbital ordering is taken from the projected basis.'
        )

    if basis is not None and len(basis) == number_of_wavefunctions:
        for index, record in enumerate(basis):
            separation = np.linalg.norm(atomic_positions - np.asarray(record['tau']), axis=1)
            orbital_atom[index] = int(np.argmin(separation))
            orbital_l[index] = int(record['l'])
            orbital_m[index] = int(record['m'])
            orbital_shell.append(str(record.get('label', '')))
            orbital_label.append(_orbital_label(int(record['l']), int(record['m'])))

    elif isinstance(shells, dict):
        index = 0
        for atom_index, species in enumerate(atom_species):
            for shell_l in shells[species]:
                for m in range(1, 2 * int(shell_l) + 2):
                    if index >= number_of_wavefunctions:
                        break
                    orbital_atom[index] = atom_index
                    orbital_l[index] = int(shell_l)
                    orbital_m[index] = m
                    orbital_shell.append('')
                    orbital_label.append(_orbital_label(int(shell_l), m))
                    index += 1
        if index != number_of_wavefunctions:
            raise RuntimeError(
                f"'shells' expands to {index} orbitals but nawf={number_of_wavefunctions}. "
                'Cannot build the orbital map.'
            )

    else:
        raise RuntimeError(
            "Cannot build the orbital map: the DataController carries neither 'basis' "
            "nor a species-keyed 'shells' description."
        )

    orbitals_per_atom = np.bincount(orbital_atom, minlength=number_of_atoms).astype(np.int32)
    atom_block_start = np.concatenate(([0], np.cumsum(orbitals_per_atom)[:-1])).astype(np.int32)

    return {
        'orbital_atom': orbital_atom,
        'orbital_species': np.array([atom_species[a] for a in orbital_atom], dtype=np.str_),
        'orbital_l': orbital_l,
        'orbital_m': orbital_m,
        'orbital_label': np.array(orbital_label, dtype=np.str_),
        'orbital_shell': np.array(orbital_shell, dtype=np.str_),
        'orbitals_per_atom': orbitals_per_atom,
        'atom_block_start': atom_block_start,
    }


# ---------------------------------------------------------------------------
# Real-space grid geometry
# ---------------------------------------------------------------------------


def _lattice_index_from_fft_cell(cell_index, grid_size):
    """Fold an FFT cell index into the symmetric range used by ``get_R_grid_fft``."""
    return cell_index if 2 * cell_index < grid_size else cell_index - grid_size


def _lattice_translations(grid_shape):
    """Return the integer lattice translation of every FFT cell.

    Returns
    -------
    np.ndarray, shape ``(nk1, nk2, nk3, 3)``, int
    """
    ranges = [
        np.array([_lattice_index_from_fft_cell(n, size) for n in range(size)], dtype=np.int32)
        for size in grid_shape
    ]
    translations = np.empty(tuple(grid_shape) + (3,), dtype=np.int32)
    translations[..., 0] = ranges[0][:, None, None]
    translations[..., 1] = ranges[1][None, :, None]
    translations[..., 2] = ranges[2][None, None, :]
    return translations


def aliasing_safe_radius(lattice_vectors, grid_shape):
    """Largest bond length representable on the FFT supercell without wrapping.

    Parameters
    ----------
    lattice_vectors : np.ndarray, shape ``(3, 3)``
        Primitive lattice vectors in Bohr (rows).
    grid_shape : sequence of int
        The ``(nk1, nk2, nk3)`` Monkhorst-Pack grid, i.e. the supercell repeat.

    Returns
    -------
    float
        Inradius of the supercell in Bohr.  Bonds longer than this fold back
        onto the same FFT cell as a shorter bond and cannot be distinguished.
    """
    supercell_vectors = np.asarray(lattice_vectors, dtype=float) * np.asarray(grid_shape)[:, None]
    dual_vectors = np.linalg.inv(supercell_vectors).T
    interplanar_spacing = 1.0 / np.linalg.norm(dual_vectors, axis=1)
    return 0.5 * float(np.min(interplanar_spacing))


def _bond_distance_matrix(atomic_positions, translation_cartesian, bond_sign=-1):
    """Distances ``|tau_B + s*R - tau_A|`` for every ordered atom pair.

    ``bond_sign`` (``s``) is needed because PAOFLOW carries two opposite
    conventions: a model Hamiltonian built by :mod:`PAOFLOW.models.models`
    stores the hopping to ``tau_B + R``, whereas ``HRs = ifftn(Hks)`` from the
    projection path stores it at ``-R``.  Band structures are insensitive to
    this (with time reversal the two give ``H(k)`` and ``H(-k)``), but the bond
    geometry is not.  Use :func:`detect_bond_sign` rather than guessing.
    """
    displacement = (
        atomic_positions[None, :, :]
        + bond_sign * translation_cartesian
        - atomic_positions[:, None, :]
    )
    return np.linalg.norm(displacement, axis=2)


def detect_bond_sign(hamiltonian_real_space, atomic_positions, lattice_vectors, orbital_atom):
    """Infer the sign of ``R`` in the bond vector from the Hamiltonian itself.

    A physical hopping decays with the bond length, so the correct sign is the
    one that concentrates the weight of :math:`|H(\\mathbf{R})|` at short bonds.

    Parameters
    ----------
    hamiltonian_real_space : np.ndarray
        ``HRs`` with shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``.
    atomic_positions : np.ndarray, shape ``(natoms, 3)``
        Cartesian positions in Bohr.
    lattice_vectors : np.ndarray, shape ``(3, 3)``
        Primitive lattice vectors in Bohr (rows).
    orbital_atom : np.ndarray, shape ``(nawf,)``
        Atom index of each orbital.

    Returns
    -------
    int
        ``+1`` or ``-1``: the value of ``s`` in ``d = tau_B + s*R - tau_A``.
    """
    grid_shape = hamiltonian_real_space.shape[2:5]
    translations = _lattice_translations(grid_shape)
    magnitude_all = np.abs(hamiltonian_real_space).max(axis=5)

    weighted_length = {1: 0.0, -1: 0.0}
    total_weight = 0.0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                magnitude = magnitude_all[:, :, i, j, k]
                weight = magnitude.sum()
                if weight == 0.0:
                    continue
                total_weight += weight
                shift = translations[i, j, k] @ lattice_vectors
                for sign in (1, -1):
                    distance = _bond_distance_matrix(atomic_positions, shift, sign)
                    weighted_length[sign] += float(
                        (magnitude * distance[orbital_atom[:, None], orbital_atom[None, :]]).sum()
                    )

    if total_weight == 0.0:
        return -1
    return 1 if weighted_length[1] <= weighted_length[-1] else -1


def compute_star_shells(
    atomic_positions,
    lattice_vectors,
    grid_shape,
    max_radius=None,
    distance_tol=1.0e-3,
):
    """Collect the distinct neighbour-shell distances (the *star*) of the lattice.

    Only distances that the FFT supercell can represent unambiguously are
    considered, so the returned shells are exactly those addressable in ``HRs``.

    Parameters
    ----------
    atomic_positions : np.ndarray, shape ``(natoms, 3)``
        Cartesian atomic positions in Bohr.
    lattice_vectors : np.ndarray, shape ``(3, 3)``
        Primitive lattice vectors in Bohr (rows).
    grid_shape : sequence of int
        ``(nk1, nk2, nk3)``.
    max_radius : float, optional
        Search cutoff in Bohr.  Defaults to :func:`aliasing_safe_radius`.
    distance_tol : float
        Distances closer than this are merged into a single shell (Bohr).

    Returns
    -------
    np.ndarray, 1-D, float
        Sorted shell distances, excluding the on-site distance (zero).
    """
    if max_radius is None:
        max_radius = aliasing_safe_radius(lattice_vectors, grid_shape)

    translations = _lattice_translations(grid_shape).reshape(-1, 3)
    candidate_distances = []
    for translation in translations:
        translation_cartesian = translation @ lattice_vectors
        distances = _bond_distance_matrix(atomic_positions, translation_cartesian)
        distances = distances[(distances > distance_tol) & (distances <= max_radius)]
        if distances.size:
            candidate_distances.append(np.unique(np.round(distances, 6)))

    if not candidate_distances:
        return np.empty(0, dtype=float)

    sorted_distances = np.unique(np.concatenate(candidate_distances))

    shells = [sorted_distances[0]]
    for distance in sorted_distances[1:]:
        if distance - shells[-1] > distance_tol:
            shells.append(distance)
    return np.array(shells, dtype=float)


def _assign_shell_order(distances, shell_distances, distance_tol):
    """Label each distance with its 1-based shell index (0 marks on-site)."""
    shell_order = np.zeros(distances.shape, dtype=np.int32)
    beyond_onsite = distances > distance_tol
    if shell_distances.size == 0 or not beyond_onsite.any():
        return shell_order

    selected = distances[beyond_onsite]
    insertion = np.searchsorted(shell_distances, selected)
    lower = np.clip(insertion - 1, 0, shell_distances.size - 1)
    upper = np.clip(insertion, 0, shell_distances.size - 1)
    distance_to_lower = np.abs(selected - shell_distances[lower])
    distance_to_upper = np.abs(shell_distances[upper] - selected)
    shell_order[beyond_onsite] = np.where(distance_to_upper < distance_to_lower, upper, lower) + 1
    return shell_order


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def sparsify_real_space_hamiltonian(
    data_controller,
    bond_order=3,
    r_cut=None,
    magnitude_tol=1.0e-8,
    distance_tol=1.0e-3,
    bond_sign='auto',
    verbose=False,
):
    """Truncate ``HRs`` to the bonds inside a neighbour-shell cutoff.

    Parameters
    ----------
    data_controller : DataController
        Must carry ``HRs`` (i.e. ``pao_hamiltonian`` has run), plus the geometry
        arrays ``a_vectors``, ``tau`` and ``atoms``.
    bond_order : int
        Keep bonds up to and including this neighbour shell.  ``1`` is nearest
        neighbours.  Ignored when ``r_cut`` is given.
    r_cut : float, optional
        Explicit bond-length cutoff in Bohr, overriding ``bond_order``.
    magnitude_tol : float
        Drop matrix elements whose magnitude is below this value in every spin
        channel.  Set to ``0`` to keep the full blocks.
    distance_tol : float
        Tolerance used to merge neighbour shells and to detect on-site terms.
    bond_sign : {'auto', 1, -1}
        Sign of ``R`` in the bond vector ``tau_B + s*R - tau_A``.  ``'auto'``
        infers it with :func:`detect_bond_sign`, which is the safe choice: the
        projection path and the model builders disagree on this convention.
    verbose : bool
        Report the retained fraction.

    Returns
    -------
    dict
        The bundle consumed by :func:`write_sparse_hamiltonian`.  Bond data is
        held in the flat COO arrays ``bond_row``, ``bond_col``,
        ``bond_translation``, ``bond_value``, ``bond_distance`` and
        ``bond_shell``.

    Notes
    -----
    A bond longer than :func:`aliasing_safe_radius` cannot be distinguished from
    its periodic image on the ``(nk1, nk2, nk3)`` grid; requesting such a cutoff
    raises rather than silently storing an aliased Hamiltonian.
    """
    arrays, attributes = data_controller.data_dicts()

    if 'HRs' not in arrays:
        raise RuntimeError("'HRs' is not available; run pao_hamiltonian() first.")

    hamiltonian_real_space = arrays['HRs']
    if hamiltonian_real_space.ndim != 6:
        raise RuntimeError(
            f'HRs must have shape (nawf, nawf, nk1, nk2, nk3, nspin); got '
            f'{hamiltonian_real_space.shape}.'
        )

    number_of_wavefunctions = hamiltonian_real_space.shape[0]
    grid_shape = hamiltonian_real_space.shape[2:5]
    number_of_spins = hamiltonian_real_space.shape[5]

    lattice_constant = float(attributes['alat'])
    lattice_vectors = np.asarray(arrays['a_vectors'], dtype=float) * lattice_constant
    atomic_positions = np.asarray(arrays['tau'], dtype=float)

    basis_table = build_orbital_basis_table(data_controller)
    orbital_atom = basis_table['orbital_atom']

    if bond_sign == 'auto':
        bond_sign = detect_bond_sign(
            hamiltonian_real_space, atomic_positions, lattice_vectors, orbital_atom
        )
    bond_sign = int(bond_sign)
    if bond_sign not in (1, -1):
        raise ValueError(f"bond_sign must be 'auto', 1 or -1; got {bond_sign}.")

    safe_radius = aliasing_safe_radius(lattice_vectors, grid_shape)
    shell_distances = compute_star_shells(
        atomic_positions,
        lattice_vectors,
        grid_shape,
        max_radius=safe_radius,
        distance_tol=distance_tol,
    )

    if r_cut is None:
        if bond_order is None or bond_order < 1:
            raise ValueError("Provide a positive 'bond_order' or an explicit 'r_cut'.")
        if bond_order > shell_distances.size:
            raise ValueError(
                f'bond_order={bond_order} exceeds the {shell_distances.size} neighbour '
                f'shells representable on the {tuple(grid_shape)} grid. Use a denser '
                'k-grid or a smaller bond_order.'
            )
        cutoff_radius = float(shell_distances[bond_order - 1]) + distance_tol
    else:
        cutoff_radius = float(r_cut)

    if cutoff_radius > safe_radius:
        raise ValueError(
            f'Cutoff {cutoff_radius:.4f} Bohr exceeds the aliasing-safe radius '
            f'{safe_radius:.4f} Bohr of the {tuple(grid_shape)} grid. Bonds beyond it '
            'wrap onto shorter ones and cannot be stored unambiguously.'
        )

    translations = _lattice_translations(grid_shape)

    row_chunks = []
    column_chunks = []
    translation_chunks = []
    value_chunks = []
    distance_chunks = []

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                translation = translations[i, j, k]
                translation_cartesian = translation @ lattice_vectors
                pair_distance = _bond_distance_matrix(
                    atomic_positions, translation_cartesian, bond_sign
                )

                atom_pair_inside_cutoff = pair_distance <= cutoff_radius
                if not atom_pair_inside_cutoff.any():
                    continue

                block = hamiltonian_real_space[:, :, i, j, k, :]
                retained = atom_pair_inside_cutoff[orbital_atom[:, None], orbital_atom[None, :]]
                if magnitude_tol > 0.0:
                    retained &= np.max(np.abs(block), axis=2) > magnitude_tol

                rows, columns = np.nonzero(retained)
                if rows.size == 0:
                    continue

                row_chunks.append(rows.astype(np.int32))
                column_chunks.append(columns.astype(np.int32))
                translation_chunks.append(np.broadcast_to(translation, (rows.size, 3)).copy())
                value_chunks.append(block[rows, columns, :])
                distance_chunks.append(pair_distance[orbital_atom[rows], orbital_atom[columns]])

    if row_chunks:
        bond_row = np.concatenate(row_chunks)
        bond_col = np.concatenate(column_chunks)
        bond_translation = np.concatenate(translation_chunks)
        bond_value = np.concatenate(value_chunks)
        bond_distance = np.concatenate(distance_chunks)
    else:
        bond_row = np.empty(0, dtype=np.int32)
        bond_col = np.empty(0, dtype=np.int32)
        bond_translation = np.empty((0, 3), dtype=np.int32)
        bond_value = np.empty((0, number_of_spins), dtype=complex)
        bond_distance = np.empty(0, dtype=float)

    bond_shell = _assign_shell_order(bond_distance, shell_distances, distance_tol)

    dense_element_count = hamiltonian_real_space.size // number_of_spins
    if verbose:
        density = bond_row.size / dense_element_count if dense_element_count else 0.0
        print(
            f'Sparse HRs: {bond_row.size} of {dense_element_count} elements retained '
            f'({100.0 * density:.3f} %), cutoff {cutoff_radius:.4f} Bohr '
            f'(shell {bond_order if r_cut is None else "custom"})'
        )

    bundle = {
        'format_version': SPARSE_HAMILTONIAN_FORMAT_VERSION,
        'bond_row': bond_row,
        'bond_col': bond_col,
        'bond_translation': bond_translation,
        'bond_value': bond_value,
        'bond_distance': bond_distance,
        'bond_shell': bond_shell,
        'shell_distances': shell_distances,
        'a_vectors': np.asarray(arrays['a_vectors'], dtype=float),
        'tau': atomic_positions,
        'grid_shape': np.asarray(grid_shape, dtype=np.int32),
        'nawf': number_of_wavefunctions,
        'nspin': number_of_spins,
        'alat': lattice_constant,
        'cutoff_radius': cutoff_radius,
        'bond_order': -1 if r_cut is not None else int(bond_order),
        'magnitude_tol': float(magnitude_tol),
        'distance_tol': float(distance_tol),
        'bond_sign': int(bond_sign),
        'aliasing_safe_radius': safe_radius,
        'dense_element_count': int(dense_element_count),
    }
    bundle.update(basis_table)

    #  Stored verbatim: its length unit follows the source (Bohr for a QE
    #  projection, alat for the model builders), so it must not be recomputed.
    orbital_offsets = arrays.get('Dnm')
    if orbital_offsets is None:
        centres = atomic_positions[orbital_atom]
        orbital_offsets = centres[:, None, :] - centres[None, :, :]
    bundle['Dnm'] = np.asarray(orbital_offsets, dtype=float)

    reciprocal_vectors = arrays.get('b_vectors')
    if reciprocal_vectors is None:
        # PAOFLOW stores b_vectors in units of 2*pi/alat, dual to a_vectors in alat units.
        reciprocal_vectors = np.linalg.inv(np.asarray(arrays['a_vectors'], dtype=float)).T
    bundle['b_vectors'] = np.asarray(reciprocal_vectors, dtype=float)

    return bundle


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _json_safe(value):
    """Recursively convert ``value`` to JSON-encodable data, or flag it as unusable."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple)):
        converted = [_json_safe(item) for item in value]
        return _UNSERIALIZABLE if any(c is _UNSERIALIZABLE for c in converted) else converted
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _UNSERIALIZABLE
            safe_item = _json_safe(item)
            if safe_item is _UNSERIALIZABLE:
                return _UNSERIALIZABLE
            converted[key] = safe_item
        return converted
    return _UNSERIALIZABLE


def _serializable_attributes(attributes):
    """Keep the scalar attributes that survive a JSON round trip."""
    return {
        key: safe
        for key, value in attributes.items()
        if (safe := _json_safe(value)) is not _UNSERIALIZABLE
    }


def write_sparse_hamiltonian(data_controller, bundle, fname):
    """Write a truncated Hamiltonian bundle to a compressed ``.npz`` archive.

    Parameters
    ----------
    data_controller : DataController
        Source of the run metadata stored alongside the bonds.
    bundle : dict
        Output of :func:`sparsify_real_space_hamiltonian`.
    fname : str
        Destination path.  Relative names resolve inside the output directory.

    Returns
    -------
    str
        The path that was written.

    Notes
    -----
    Metadata is stored as JSON text rather than a pickled object so that loading
    never needs ``allow_pickle`` and cannot execute code from an untrusted file.
    """
    from os.path import isabs, join

    arrays, attributes = data_controller.data_dicts()

    destination = fname if isabs(fname) else join(attributes['opath'], fname)

    payload = {key: value for key, value in bundle.items() if isinstance(value, np.ndarray)}

    metadata = {
        'format_version': SPARSE_HAMILTONIAN_FORMAT_VERSION,
        'nawf': int(bundle['nawf']),
        'nspin': int(bundle['nspin']),
        'alat': float(bundle['alat']),
        'cutoff_radius': float(bundle['cutoff_radius']),
        'bond_order': int(bundle['bond_order']),
        'magnitude_tol': float(bundle['magnitude_tol']),
        'distance_tol': float(bundle['distance_tol']),
        'bond_sign': int(bundle['bond_sign']),
        'aliasing_safe_radius': float(bundle['aliasing_safe_radius']),
        'dense_element_count': int(bundle['dense_element_count']),
        'atoms': [str(a) for a in arrays['atoms']],
        'attributes': _serializable_attributes(attributes),
    }
    for optional_key in ('species', 'shells'):
        safe_value = _json_safe(arrays.get(optional_key))
        if safe_value is not _UNSERIALIZABLE and safe_value is not None:
            metadata[optional_key] = safe_value

    payload['metadata_json'] = np.array(json.dumps(metadata))

    np.savez_compressed(destination, **payload)
    return destination


def read_sparse_hamiltonian(fname):
    """Read a bundle written by :func:`write_sparse_hamiltonian`.

    Parameters
    ----------
    fname : str
        Path to the ``.npz`` archive.

    Returns
    -------
    dict
        Arrays as stored, plus the decoded ``metadata`` dictionary and the
        scalars promoted to top level (``nawf``, ``nspin``, ``alat``, ...).

    Raises
    ------
    ValueError
        If the archive was written by an incompatible format version.
    """
    with np.load(fname, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files if key != 'metadata_json'}
        metadata = json.loads(str(archive['metadata_json']))

    if metadata['format_version'] != SPARSE_HAMILTONIAN_FORMAT_VERSION:
        raise ValueError(
            f'Unsupported sparse Hamiltonian format version {metadata["format_version"]}; '
            f'this PAOFLOW build reads version {SPARSE_HAMILTONIAN_FORMAT_VERSION}.'
        )

    bundle['metadata'] = metadata
    for key in (
        'nawf',
        'nspin',
        'alat',
        'cutoff_radius',
        'bond_order',
        'magnitude_tol',
        'distance_tol',
        'bond_sign',
        'aliasing_safe_radius',
        'dense_element_count',
    ):
        bundle[key] = metadata[key]
    return bundle


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------


def rebuild_HRs_from_sparse(bundle):
    """Scatter a sparse bond list back onto the dense FFT real-space grid.

    Parameters
    ----------
    bundle : dict
        Output of :func:`read_sparse_hamiltonian` or
        :func:`sparsify_real_space_hamiltonian`.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``, complex
        The truncated real-space Hamiltonian.  Elements outside the stored
        bonds are exactly zero.
    """
    grid_shape = tuple(int(n) for n in bundle['grid_shape'])
    number_of_wavefunctions = int(bundle['nawf'])
    number_of_spins = int(bundle['nspin'])

    hamiltonian_real_space = np.zeros(
        (number_of_wavefunctions, number_of_wavefunctions) + grid_shape + (number_of_spins,),
        dtype=complex,
    )

    translation = bundle['bond_translation']
    cell_i = np.mod(translation[:, 0], grid_shape[0])
    cell_j = np.mod(translation[:, 1], grid_shape[1])
    cell_k = np.mod(translation[:, 2], grid_shape[2])

    hamiltonian_real_space[bundle['bond_row'], bundle['bond_col'], cell_i, cell_j, cell_k, :] = (
        bundle['bond_value']
    )

    return hamiltonian_real_space


def restore_data_controller(data_controller, bundle):
    """Repopulate a ``DataController`` so the standard pipeline can run.

    Parameters
    ----------
    data_controller : DataController
        Target container; its arrays and attributes are overwritten with the
        saved run metadata and the rebuilt ``HRs``.
    bundle : dict
        Output of :func:`read_sparse_hamiltonian`.

    Returns
    -------
    None
        Populates ``HRs``, the geometry arrays, the orbital map and the real- and
        k-space FFT grids.

    Notes
    -----
    Attributes describing *where this session runs* (working and output paths,
    pool count, verbosity) are kept from the live controller when present, so a
    restored run writes to its own output directory rather than the one recorded
    in the archive.
    """
    from ..utils.get_K_grid_fft import get_K_grid_fft
    from ..utils.get_R_grid_fft import get_R_grid_fft

    arrays, attributes = data_controller.data_dicts()
    metadata = bundle['metadata']

    session_keys = (
        'workpath',
        'outputdir',
        'opath',
        'savedir',
        'fpath',
        'inputfile',
        'npool',
        'verbose',
        'mpisize',
        'abort_on_exception',
    )
    session_attributes = {key: attributes[key] for key in session_keys if key in attributes}

    attributes.update(metadata['attributes'])
    attributes.update(session_attributes)

    grid_shape = tuple(int(n) for n in bundle['grid_shape'])
    attributes['nawf'] = int(bundle['nawf'])
    attributes['nspin'] = int(bundle['nspin'])
    attributes['alat'] = float(bundle['alat'])
    attributes['nk1'], attributes['nk2'], attributes['nk3'] = grid_shape
    attributes['nkpnts'] = int(np.prod(grid_shape))
    attributes['natoms'] = int(bundle['tau'].shape[0])

    arrays['a_vectors'] = bundle['a_vectors']
    if 'b_vectors' in bundle:
        arrays['b_vectors'] = bundle['b_vectors']
    else:
        arrays['b_vectors'] = np.linalg.inv(bundle['a_vectors']).T
    arrays['tau'] = bundle['tau']
    arrays['atoms'] = list(metadata['atoms'])
    if 'species' in metadata:
        arrays['species'] = metadata['species']
    if 'shells' in metadata:
        arrays['shells'] = metadata['shells']

    for key in (
        'orbital_atom',
        'orbital_species',
        'orbital_l',
        'orbital_m',
        'orbital_label',
        'orbital_shell',
        'orbitals_per_atom',
        'atom_block_start',
        'Dnm',
    ):
        if key in bundle:
            arrays[key] = bundle[key]
    if 'orbitals_per_atom' in bundle:
        arrays['naw'] = np.asarray(bundle['orbitals_per_atom'], dtype=int)

    arrays['HRs'] = rebuild_HRs_from_sparse(bundle)

    get_R_grid_fft(data_controller, *grid_shape)
    get_K_grid_fft(data_controller)


# ---------------------------------------------------------------------------
# Labelled dataset view
# ---------------------------------------------------------------------------


def bond_table(bundle):
    """Expand the stored bonds into fully labelled records.

    Every retained matrix element becomes one row carrying the atoms, species and
    orbital names it connects — the form consumed by a matrix-element fit.

    Parameters
    ----------
    bundle : dict
        Output of :func:`read_sparse_hamiltonian` or
        :func:`sparsify_real_space_hamiltonian`.

    Returns
    -------
    dict of np.ndarray
        Column-oriented table of length ``n_bonds`` with keys ``atom_i``,
        ``atom_j``, ``species_i``, ``species_j``, ``orbital_i``, ``orbital_j``,
        ``l_i``, ``l_j``, ``translation``, ``bond_vector`` (Bohr), ``distance``
        (Bohr), ``shell`` and ``value`` (shape ``(n_bonds, nspin)``).
    """
    rows = bundle['bond_row']
    columns = bundle['bond_col']
    orbital_atom = bundle['orbital_atom']

    lattice_vectors = np.asarray(bundle['a_vectors'], dtype=float) * float(bundle['alat'])
    atomic_positions = np.asarray(bundle['tau'], dtype=float)

    atom_i = orbital_atom[rows]
    atom_j = orbital_atom[columns]
    bond_vector = (
        atomic_positions[atom_j]
        + int(bundle.get('bond_sign', -1)) * (bundle['bond_translation'] @ lattice_vectors)
        - atomic_positions[atom_i]
    )

    return {
        'atom_i': atom_i,
        'atom_j': atom_j,
        'species_i': bundle['orbital_species'][rows],
        'species_j': bundle['orbital_species'][columns],
        'orbital_i': bundle['orbital_label'][rows],
        'orbital_j': bundle['orbital_label'][columns],
        'l_i': bundle['orbital_l'][rows],
        'l_j': bundle['orbital_l'][columns],
        'translation': bundle['bond_translation'],
        'bond_vector': bond_vector,
        'distance': bundle['bond_distance'],
        'shell': bundle['bond_shell'],
        'value': bundle['bond_value'],
    }
