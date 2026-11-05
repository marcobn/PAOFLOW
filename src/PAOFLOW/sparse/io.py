"""Persistence and labelling of the base-cell bond list.

The bond list fully determines every downstream property, so a compressed ``.npz`` of it plus
the run metadata lets a later run skip the DFT read, projection and dense ``pao_hamiltonian``;
no dense ``HRs`` is rebuilt on load. Each bond also carries its atoms, species, orbitals,
vector, length and shell (:func:`bond_table`): a labelled dataset of PAO hoppings.

Only the base (pre-doubling) cell is written: doubling replicates orbitals, so the orbital map
would no longer match; reload and call ``doubling_Hamiltonian`` again (it is deterministic).
Metadata is JSON, not pickle, so loading never needs ``allow_pickle`` or runs untrusted code.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from .hamiltonian import SparseHamiltonian
from .shells import aliasing_safe_radius, assign_shell_order, compute_star_shells

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

#: Bumped whenever the on-disk layout changes in a backward-incompatible way.
#: Version 1 was the dense-rebuild format of the former
#: ``hamiltonian/sparse_hamiltonian.py``; it is not readable here.
SPARSE_HAMILTONIAN_FORMAT_VERSION = 2

#: Chemistry names of the real spherical harmonics in Quantum-ESPRESSO order,
#: indexed by ``m - 1`` for ``m = 1 .. 2l+1`` (see ``calc_ylmg``).
QE_ORBITAL_LABELS_BY_L = {
    0: ('s',),
    1: ('pz', 'px', 'py'),
    2: ('dz2', 'dzx', 'dyz', 'dx2-y2', 'dxy'),
    3: ('fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)'),
}

#: Data arrays that are not written: the dense Hamiltonian and everything
#: indexed by the DFT k-mesh, which a restarted run neither has nor needs.
#: ``Dnm`` is stored separately (it is part of the bond geometry), and the
#: projection ``basis`` records are replaced by the orbital table.
_SKIP_ARRAYS = frozenset(
    ('HRs', 'Hks', 'Dnm', 'basis', 'kpnts', 'kpnts_wght', 'my_eigsmat', 'kq_wght', 'kgrid')
)

#: Attributes describing where *this* session runs.  A restored run keeps
#: its own values rather than the ones recorded in the archive.
_SESSION_KEYS = (
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

#: Orbital-table entries stored in the archive and restored as data arrays.
_ORBITAL_KEYS = (
    'orbital_atom',
    'orbital_species',
    'orbital_l',
    'orbital_m',
    'orbital_label',
    'orbital_shell',
    'orbitals_per_atom',
    'atom_block_start',
)

_UNSERIALIZABLE = object()


# ---------------------------------------------------------------------------
# Orbital -> atom / species / label map
# ---------------------------------------------------------------------------


def _orbital_label(l: int, m: int) -> str:
    """Return the chemistry name of the ``m``-th real harmonic of shell ``l``."""
    names = QE_ORBITAL_LABELS_BY_L.get(l)
    if names is None or not (1 <= m <= len(names)):
        return f'l{l}m{m}'
    return names[m - 1]


def build_orbital_basis_table(data_controller: DataController) -> dict[str, np.ndarray]:
    """Map every PAO basis index onto the atom and orbital it represents.

    Parameters
    ----------
    data_controller : DataController
        Must carry ``nawf``, ``tau``, ``atoms`` and ``basis`` (set by ``projections`` and
        ``read_atomic_proj_QE``), or failing that a species-keyed ``shells`` dict.

    Returns
    -------
    dict
        Length-``nawf`` ``orbital_atom``, ``orbital_species``, ``orbital_l``, ``orbital_m``
        (1-based), ``orbital_label`` (``'px'``, QE order ``QE_ORBITAL_LABELS_BY_L``) and
        ``orbital_shell`` (``'3P'``), plus ``atom_block_start`` and ``orbitals_per_atom``.

    Raises
    ------
    RuntimeError
        For a tight-binding model (its px,py,pz / dxy,... order is not recoverable, so wrong
        labels are refused), or if no description expands to exactly ``nawf`` orbitals.
    """
    arrays, attributes = data_controller.data_dicts()

    nawf = int(attributes['nawf'])
    tau = np.asarray(arrays['tau'], dtype=float)
    atom_species = [str(s) for s in arrays['atoms']]

    orbital_atom = np.zeros(nawf, dtype=np.int32)
    orbital_l = np.zeros(nawf, dtype=np.int32)
    orbital_m = np.zeros(nawf, dtype=np.int32)
    orbital_shell = []
    orbital_label = []

    basis = arrays.get('basis', None)
    shells = arrays.get('shells', None)

    if 'norbitals' in arrays and basis is None:
        raise RuntimeError(
            'This DataController holds a tight-binding model, not a QE projection. '
            'The sparse Hamiltonian writer targets the QE-PAOFLOW projection pipeline, '
            'whose orbital ordering is taken from the projected basis.'
        )

    if basis is not None and len(basis) == nawf:
        for index, record in enumerate(basis):
            separation = np.linalg.norm(tau - np.asarray(record['tau']), axis=1)
            orbital_atom[index] = int(np.argmin(separation))
            orbital_l[index] = int(record['l'])
            orbital_m[index] = int(record['m'])
            orbital_shell.append(str(record.get('label', '')))
            orbital_label.append(_orbital_label(int(record['l']), int(record['m'])))

    elif isinstance(shells, dict):
        count = sum(2 * int(shell_l) + 1 for species in atom_species for shell_l in shells[species])
        if count != nawf:
            raise RuntimeError(
                f"'shells' expands to {count} orbitals but nawf={nawf}. "
                'Cannot build the orbital map.'
            )
        index = 0
        for atom_index, species in enumerate(atom_species):
            for shell_l in shells[species]:
                for m in range(1, 2 * int(shell_l) + 2):
                    orbital_atom[index] = atom_index
                    orbital_l[index] = int(shell_l)
                    orbital_m[index] = m
                    orbital_shell.append('')
                    orbital_label.append(_orbital_label(int(shell_l), m))
                    index += 1

    else:
        raise RuntimeError(
            "Cannot build the orbital map: the DataController carries neither 'basis' "
            "nor a species-keyed 'shells' description."
        )

    orbitals_per_atom = np.bincount(orbital_atom, minlength=tau.shape[0]).astype(np.int32)
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
# JSON helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively convert ``value`` to JSON-encodable data, or flag it as unusable."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
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


def _serializable(mapping: dict[str, Any]) -> dict[str, Any]:
    """Keep the entries of ``mapping`` that survive a JSON round trip."""
    return {
        key: safe
        for key, value in mapping.items()
        if (safe := _json_safe(value)) is not _UNSERIALIZABLE
    }


# ---------------------------------------------------------------------------
# Bond geometry
# ---------------------------------------------------------------------------


def _bond_geometry(
    H: SparseHamiltonian, tau: np.ndarray, orbital_atom: np.ndarray, distance_tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bond vector, length and shell of every stored bond, plus the shell list (Bohr).

    ``H_ij(R)`` links orbital ``i`` at home to ``j`` in the cell at ``-R``,
    ``d = tau_j - tau_i - alat*R``: the sign of PAOFLOW's ``H(k) = sum_R H(R) exp(-2 pi i k.R)``
    (``Hks = fftn(HRs)``). ``|d|`` is what the ``rcut`` / ``bond_order`` cutoff measures.
    """
    lattice = H.a_vectors * H.alat
    Rcart = H.R_int[H.ridx].astype(float) @ lattice
    vec = tau[orbital_atom[H.cols]] - tau[orbital_atom[H.rows]] - Rcart
    dist = np.linalg.norm(vec, axis=1)
    shells = compute_star_shells(tau, lattice, H.nk_grid, distance_tol=distance_tol)
    return vec, dist, assign_shell_order(dist, shells, distance_tol), shells


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------


def write_sparse_hamiltonian(
    data_controller: DataController,
    H: SparseHamiltonian,
    fname: str,
    distance_tol: float = 1.0e-3,
    Dnm: np.ndarray | None = None,
) -> str:
    """Write a base-cell bond list and its run metadata to a ``.npz`` archive.

    Parameters
    ----------
    data_controller : DataController
        Source of the geometry, orbital map and run attributes.
    H : SparseHamiltonian
        The base-cell bond list (as built by ``pao_hamiltonian``).
    fname : str
        Destination; relative names resolve inside the output directory.
    distance_tol : float, optional
        Shell-merging tolerance for the per-bond shell labels (Bohr).
    Dnm : np.ndarray or None, optional
        Orbital-centre offsets, default ``arrays['Dnm']`` (the sparse engine pops it at
        conversion); the dense gradient needs them after a dense restart.

    Returns
    -------
    str
        The path that was written.

    Raises
    ------
    RuntimeError
        If ``H`` has been doubled or compacted.
    """
    from os.path import isabs, join

    if H._doubled:
        raise RuntimeError(
            'write_sparse_hamiltonian: only the base (pre-doubling) cell can be saved. '
            'Doubling is deterministic: save before doubling_Hamiltonian() and double '
            'again after loading.'
        )
    H._require_bonds('write_sparse_hamiltonian')

    arrays, attributes = data_controller.data_dicts()
    destination = fname if isabs(fname) else join(attributes['opath'], fname)

    tau = np.asarray(arrays['tau'], dtype=float)
    table = build_orbital_basis_table(data_controller)
    vec, dist, shell, shell_distances = _bond_geometry(H, tau, table['orbital_atom'], distance_tol)

    payload = {
        # the container itself
        'R_int': H.R_int,
        'bond_row': H.rows,
        'bond_col': H.cols,
        'bond_ridx': H.ridx,
        'bond_value': H.vals,
        'bond_dnm': H.dnm,
        'nk_grid': np.asarray(H.nk_grid, dtype=np.int32),
        'a_vectors': H.a_vectors,
        # labelled-dataset view
        'bond_vector': vec,
        'bond_distance': dist,
        'bond_shell': shell,
        'shell_distances': shell_distances,
        'tau': tau,
    }
    payload.update(table)
    if Dnm is None:
        Dnm = arrays.get('Dnm')
    if Dnm is not None:
        payload['Dnm'] = np.asarray(Dnm, dtype=float)

    # remaining run arrays: plain numeric/string ndarrays go in as they are,
    # anything else only if it survives JSON
    json_arrays = {}
    for key, value in arrays.items():
        if key in _SKIP_ARRAYS or key in payload:
            continue
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                payload['arr__' + key] = value
        elif (safe := _json_safe(value)) is not _UNSERIALIZABLE:
            json_arrays[key] = safe

    report = dict(H.drop_report)
    metadata = {
        'format_version': SPARSE_HAMILTONIAN_FORMAT_VERSION,
        'nawf': H.nawf,
        'nspin': H.nspin,
        'alat': H.alat,
        'threshold': H.threshold,
        'cutoff_radius': report.get('rcut'),
        'bond_order': report.get('bond_order'),
        'distance_tol': float(distance_tol),
        'aliasing_safe_radius': aliasing_safe_radius(H.a_vectors * H.alat, H.nk_grid),
        'drop_report': _serializable(report),
        'arrays': json_arrays,
        'attributes': _serializable(attributes),
    }
    payload['metadata_json'] = np.array(json.dumps(metadata))

    np.savez_compressed(destination, **payload)
    return destination


def read_sparse_hamiltonian(fname: str) -> tuple[SparseHamiltonian, dict[str, Any]]:
    """Read a ``.npz`` archive written by :func:`write_sparse_hamiltonian`.

    Returns
    -------
    (H, bundle) : tuple
        The bond list, and a dict of the stored arrays, decoded ``metadata``, its scalars at
        top level (``nawf``, ``nspin``, ``alat``, ``cutoff_radius``, ...) and
        ``bond_translation`` (folded lattice triple per bond).

    Raises
    ------
    ValueError
        On an incompatible format version.
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
        'threshold',
        'cutoff_radius',
        'bond_order',
        'distance_tol',
        'aliasing_safe_radius',
    ):
        bundle[key] = metadata[key]
    bundle['bond_translation'] = bundle['R_int'][bundle['bond_ridx']]

    H = SparseHamiltonian(
        nawf=metadata['nawf'],
        nspin=metadata['nspin'],
        alat=metadata['alat'],
        a_vectors=bundle['a_vectors'],
        nk_grid=tuple(int(n) for n in bundle['nk_grid']),
        R_int=bundle['R_int'],
        rows=bundle['bond_row'],
        cols=bundle['bond_col'],
        ridx=bundle['bond_ridx'],
        vals=bundle['bond_value'],
        dnm=bundle['bond_dnm'],
        threshold=metadata['threshold'],
        drop_report=dict(metadata['drop_report']),
    )
    return H, bundle


def restore_data_controller(data_controller: DataController, bundle: dict[str, Any]) -> None:
    """Overwrite ``data_controller`` with the saved run state from ``bundle``.

    ``bundle`` is the second return of :func:`read_sparse_hamiltonian`. Leaves the state the
    sparse ``pao_hamiltonian`` leaves: geometry, orbital map, attributes and k grid set; no
    ``HRs``, ``Hks`` or ``Dnm`` (bonds carry ``dnm``). Session attributes (paths, pools,
    verbosity) stay from the live controller, so output goes to its own directory.
    """
    from ..utils.get_K_grid_fft import get_K_grid_fft

    arrays, attributes = data_controller.data_dicts()
    metadata = bundle['metadata']

    session = {key: attributes[key] for key in _SESSION_KEYS if key in attributes}
    attributes.update(metadata['attributes'])
    attributes.update(session)

    nk_grid = tuple(int(n) for n in bundle['nk_grid'])
    attributes['nawf'] = int(bundle['nawf'])
    attributes['nspin'] = int(bundle['nspin'])
    attributes['alat'] = float(bundle['alat'])
    attributes['nk1'], attributes['nk2'], attributes['nk3'] = nk_grid
    attributes['nkpnts'] = int(np.prod(nk_grid))
    attributes['natoms'] = int(bundle['tau'].shape[0])

    for key, value in bundle.items():
        if key.startswith('arr__'):
            arrays[key[len('arr__') :]] = value
    arrays.update(metadata['arrays'])
    arrays['a_vectors'] = np.array(bundle['a_vectors'], dtype=float)
    arrays['tau'] = np.array(bundle['tau'], dtype=float)
    for key in _ORBITAL_KEYS:
        arrays[key] = bundle[key]
    for key in ('HRs', 'Hks', 'Dnm'):
        arrays.pop(key, None)

    get_K_grid_fft(data_controller)


# ---------------------------------------------------------------------------
# Labelled dataset view
# ---------------------------------------------------------------------------


def bond_table(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    """Expand the bonds of a :func:`read_sparse_hamiltonian` ``bundle`` into labelled records.

    Returns
    -------
    dict of np.ndarray
        One row per retained element (length ``nnz``): ``atom_i/j``, ``species_i/j``,
        ``orbital_i/j``, ``shell_i/j`` (radial channel, ``'3S'``), ``l_i/j``, ``translation``,
        ``bond_vector`` and ``distance`` (Bohr), ``shell`` (neighbour shell, 0 on-site) and
        ``value`` (eV, as ``HRs``; shape ``(nnz, nspin)``).
    """
    rows = bundle['bond_row']
    cols = bundle['bond_col']
    orbital_atom = bundle['orbital_atom']
    return {
        'atom_i': orbital_atom[rows],
        'atom_j': orbital_atom[cols],
        'species_i': bundle['orbital_species'][rows],
        'species_j': bundle['orbital_species'][cols],
        'orbital_i': bundle['orbital_label'][rows],
        'orbital_j': bundle['orbital_label'][cols],
        'shell_i': bundle['orbital_shell'][rows],
        'shell_j': bundle['orbital_shell'][cols],
        'l_i': bundle['orbital_l'][rows],
        'l_j': bundle['orbital_l'][cols],
        'translation': bundle['bond_translation'],
        'bond_vector': bundle['bond_vector'],
        'distance': bundle['bond_distance'],
        'shell': bundle['bond_shell'],
        'value': bundle['bond_value'],
    }
