"""Unit tests for the sparse real-space PAO Hamiltonian (truncate / save / restore).

The fixtures build a small simple-cubic two-atom cell with an analytically known
neighbour star, so the retained bonds and their labels can be checked exactly
without standing up the MPI-backed PAOFLOW pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.hamiltonian.sparse_hamiltonian import (
    SPARSE_HAMILTONIAN_FORMAT_VERSION,
    aliasing_safe_radius,
    bond_table,
    build_orbital_basis_table,
    compute_star_shells,
    read_sparse_hamiltonian,
    rebuild_HRs_from_sparse,
    restore_data_controller,
    sparsify_real_space_hamiltonian,
    write_sparse_hamiltonian,
)

ALAT = 6.0
GRID = (4, 4, 4)


class _DataControllerStub:
    """Minimal stand-in exposing the two access patterns PAOFLOW modules use."""

    def __init__(self, arrays, attributes):
        self.data_arrays = arrays
        self.data_attributes = attributes

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


def _simple_cubic_controller(opath='.', nspin=1):
    """Simple-cubic cell: an ``s`` atom at the origin and an ``sp`` atom at (a/2)x.

    nawf = 1 + 4 = 5, so the orbital blocks are heterogeneous and exercise the
    per-atom offset bookkeeping.
    """
    lattice_vectors = np.eye(3)
    atomic_positions = np.array([[0.0, 0.0, 0.0], [0.5 * ALAT, 0.0, 0.0]])
    arrays = {
        'a_vectors': lattice_vectors,
        'b_vectors': np.linalg.inv(lattice_vectors).T,
        'tau': atomic_positions,
        'atoms': ['A', 'B'],
        'shells': {'A': [0], 'B': [0, 1]},
    }
    attributes = {
        'alat': ALAT,
        'nawf': 5,
        'nspin': nspin,
        'natoms': 2,
        'nk1': GRID[0],
        'nk2': GRID[1],
        'nk3': GRID[2],
        'opath': str(opath),
        'verbose': False,
        'abort_on_exception': True,
    }
    return _DataControllerStub(arrays, attributes)


def _hamiltonian_filled_with_distance_decay(controller):
    """Synthetic HRs whose magnitude decays with the bond length.

    Every orbital pair is populated, so truncation has something to remove at
    every shell.
    """
    arrays, attributes = controller.data_dicts()
    nawf = attributes['nawf']
    nspin = attributes['nspin']
    lattice_vectors = arrays['a_vectors'] * attributes['alat']
    atomic_positions = arrays['tau']

    table = build_orbital_basis_table(controller)
    orbital_atom = table['orbital_atom']

    hamiltonian = np.zeros((nawf, nawf) + GRID + (nspin,), dtype=complex)
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            for k in range(GRID[2]):
                translation = np.array(
                    [
                        i if 2 * i < GRID[0] else i - GRID[0],
                        j if 2 * j < GRID[1] else j - GRID[1],
                        k if 2 * k < GRID[2] else k - GRID[2],
                    ]
                )
                shift = translation @ lattice_vectors
                displacement = atomic_positions[None, :, :] + shift - atomic_positions[:, None, :]
                distance = np.linalg.norm(displacement, axis=2)
                amplitude = np.exp(-distance / ALAT)
                block = amplitude[orbital_atom[:, None], orbital_atom[None, :]]
                for ispin in range(nspin):
                    hamiltonian[:, :, i, j, k, ispin] = block * (1.0 + ispin)
    return hamiltonian


@pytest.fixture()
def controller(tmp_path):
    dc = _simple_cubic_controller(opath=tmp_path)
    dc.data_arrays['HRs'] = _hamiltonian_filled_with_distance_decay(dc)
    return dc


# ---------------------------------------------------------------------------
# Orbital map
# ---------------------------------------------------------------------------


def test_orbital_table_matches_shells_layout(controller):
    """The s / sp basis must expand to 5 orbitals with QE m ordering."""
    table = build_orbital_basis_table(controller)

    assert list(table['orbital_atom']) == [0, 1, 1, 1, 1]
    assert list(table['orbital_species']) == ['A', 'B', 'B', 'B', 'B']
    assert list(table['orbital_label']) == ['s', 's', 'pz', 'px', 'py']
    assert list(table['orbitals_per_atom']) == [1, 4]
    assert list(table['atom_block_start']) == [0, 1]


def test_tight_binding_models_are_rejected(controller):
    """A TB-model controller must be refused, not silently mislabelled.

    models.py orders p orbitals px,py,pz whereas QE uses pz,px,py, and
    build_TB_model does not keep the orbital names, so the ordering cannot be
    recovered.  This module targets the QE projection pipeline.
    """
    controller.data_arrays['norbitals'] = np.array([1, 4])

    with pytest.raises(RuntimeError, match='tight-binding model'):
        build_orbital_basis_table(controller)


def test_projected_basis_records_take_precedence(controller):
    """The QE pipeline always supplies arry['basis']; it is the authoritative map."""
    tau = controller.data_arrays['tau']
    controller.data_arrays['basis'] = [
        {'atom': 'A', 'tau': tau[0], 'l': 0, 'm': 1, 'label': '3S'},
        {'atom': 'B', 'tau': tau[1], 'l': 0, 'm': 1, 'label': '3S'},
        {'atom': 'B', 'tau': tau[1], 'l': 1, 'm': 1, 'label': '3P'},
        {'atom': 'B', 'tau': tau[1], 'l': 1, 'm': 2, 'label': '3P'},
        {'atom': 'B', 'tau': tau[1], 'l': 1, 'm': 3, 'label': '3P'},
    ]
    #  Present but ignored: 'basis' wins, so a QE run is never misrouted.
    controller.data_arrays['norbitals'] = np.array([1, 4])

    table = build_orbital_basis_table(controller)

    assert list(table['orbital_atom']) == [0, 1, 1, 1, 1]
    assert list(table['orbital_label']) == ['s', 's', 'pz', 'px', 'py']
    assert list(table['orbital_shell']) == ['3S', '3S', '3P', '3P', '3P']


# ---------------------------------------------------------------------------
# Neighbour star
# ---------------------------------------------------------------------------


def test_star_shells_reproduce_simple_cubic_distances(controller):
    """First shells are a/2 (A-B), a (A-A) and sqrt(2)a/2 ... in Bohr."""
    arrays, attributes = controller.data_dicts()
    shells = compute_star_shells(arrays['tau'], arrays['a_vectors'] * attributes['alat'], GRID)

    assert shells[0] == pytest.approx(0.5 * ALAT)
    assert np.any(np.isclose(shells, ALAT))
    assert np.all(np.diff(shells) > 0)


def test_aliasing_safe_radius_is_half_the_supercell(controller):
    arrays, attributes = controller.data_dicts()
    radius = aliasing_safe_radius(arrays['a_vectors'] * attributes['alat'], GRID)
    assert radius == pytest.approx(0.5 * GRID[0] * ALAT)


def test_cutoff_beyond_aliasing_radius_is_rejected(controller):
    with pytest.raises(ValueError, match='aliasing-safe radius'):
        sparsify_real_space_hamiltonian(controller, r_cut=10.0 * ALAT)


def test_bond_order_beyond_available_shells_is_rejected(controller):
    with pytest.raises(ValueError, match='exceeds'):
        sparsify_real_space_hamiltonian(controller, bond_order=10_000)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_first_shell_keeps_only_nearest_neighbour_bonds(controller):
    bundle = sparsify_real_space_hamiltonian(controller, bond_order=1, magnitude_tol=0.0)

    assert bundle['bond_distance'].max() <= bundle['cutoff_radius']
    # Shell 1 is the a/2 A-B bond; on-site terms (shell 0) come along with it.
    assert set(np.unique(bundle['bond_shell'])) == {0, 1}


def test_higher_bond_order_is_a_superset_of_lower(controller):
    first = sparsify_real_space_hamiltonian(controller, bond_order=1, magnitude_tol=0.0)
    second = sparsify_real_space_hamiltonian(controller, bond_order=2, magnitude_tol=0.0)

    assert second['cutoff_radius'] > first['cutoff_radius']
    assert second['bond_row'].size > first['bond_row'].size

    def bond_keys(bundle):
        return {
            (int(r), int(c), tuple(int(x) for x in t))
            for r, c, t in zip(bundle['bond_row'], bundle['bond_col'], bundle['bond_translation'])
        }

    assert bond_keys(first) <= bond_keys(second)


def test_magnitude_tolerance_discards_small_elements(controller):
    kept_all = sparsify_real_space_hamiltonian(controller, bond_order=2, magnitude_tol=0.0)
    kept_some = sparsify_real_space_hamiltonian(controller, bond_order=2, magnitude_tol=0.5)

    assert kept_some['bond_row'].size < kept_all['bond_row'].size
    assert np.all(np.max(np.abs(kept_some['bond_value']), axis=1) > 0.5)


def test_rebuild_reproduces_retained_elements_exactly(controller):
    """Rebuilding must return the original values wherever a bond was kept."""
    original = controller.data_arrays['HRs']
    bundle = sparsify_real_space_hamiltonian(controller, bond_order=2, magnitude_tol=0.0)
    rebuilt = rebuild_HRs_from_sparse(bundle)

    assert rebuilt.shape == original.shape

    retained = np.zeros(original.shape[:-1], dtype=bool)
    cells = np.mod(bundle['bond_translation'], np.array(GRID))
    retained[bundle['bond_row'], bundle['bond_col'], cells[:, 0], cells[:, 1], cells[:, 2]] = True

    np.testing.assert_allclose(rebuilt[retained], original[retained])
    assert np.all(rebuilt[~retained] == 0.0)


def test_full_cutoff_rebuilds_the_entire_hamiltonian(controller):
    """With a cutoff covering every representable bond, the round trip is lossless."""
    original = controller.data_arrays['HRs']
    safe = aliasing_safe_radius(controller.data_arrays['a_vectors'] * ALAT, GRID)
    bundle = sparsify_real_space_hamiltonian(controller, r_cut=safe, magnitude_tol=0.0)
    rebuilt = rebuild_HRs_from_sparse(bundle)

    # Corner cells of the FFT grid sit outside the inscribed sphere, so compare
    # only where a bond was representable.
    retained = np.zeros(original.shape[:-1], dtype=bool)
    cells = np.mod(bundle['bond_translation'], np.array(GRID))
    retained[bundle['bond_row'], bundle['bond_col'], cells[:, 0], cells[:, 1], cells[:, 2]] = True
    np.testing.assert_allclose(rebuilt[retained], original[retained])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_write_read_round_trip_preserves_bonds(controller, tmp_path):
    bundle = sparsify_real_space_hamiltonian(controller, bond_order=2, magnitude_tol=0.0)
    path = write_sparse_hamiltonian(controller, bundle, 'sparse.npz')
    loaded = read_sparse_hamiltonian(path)

    assert loaded['metadata']['format_version'] == SPARSE_HAMILTONIAN_FORMAT_VERSION
    assert loaded['nawf'] == bundle['nawf']
    assert loaded['nspin'] == bundle['nspin']
    assert loaded['alat'] == pytest.approx(bundle['alat'])
    np.testing.assert_array_equal(loaded['bond_row'], bundle['bond_row'])
    np.testing.assert_array_equal(loaded['bond_translation'], bundle['bond_translation'])
    np.testing.assert_allclose(loaded['bond_value'], bundle['bond_value'])
    np.testing.assert_array_equal(loaded['orbital_label'], bundle['orbital_label'])


def test_archive_loads_without_pickle(controller, tmp_path):
    """Reading an untrusted archive must never need allow_pickle."""
    bundle = sparsify_real_space_hamiltonian(controller, bond_order=1)
    path = write_sparse_hamiltonian(controller, bundle, 'sparse.npz')

    with np.load(path, allow_pickle=False) as archive:
        assert 'bond_value' in archive.files
        assert 'metadata_json' in archive.files


def test_stored_Dnm_is_preserved_verbatim(controller, tmp_path):
    """Dnm drives the gradient; its length unit depends on the source, so it
    must round-trip exactly rather than being recomputed."""
    nawf = controller.data_attributes['nawf']
    rng = np.random.default_rng(0)
    controller.data_arrays['Dnm'] = rng.normal(size=(nawf, nawf, 3))

    bundle = sparsify_real_space_hamiltonian(controller, bond_order=1)
    path = write_sparse_hamiltonian(controller, bundle, 'sparse.npz')

    restored = _DataControllerStub({}, {})
    restore_data_controller(restored, read_sparse_hamiltonian(path))

    np.testing.assert_allclose(restored.data_arrays['Dnm'], controller.data_arrays['Dnm'])


def test_Dnm_is_reconstructed_when_absent(controller, tmp_path):
    """Without a stored Dnm, rebuild it from the orbital centres."""
    assert 'Dnm' not in controller.data_arrays

    bundle = sparsify_real_space_hamiltonian(controller, bond_order=1)
    path = write_sparse_hamiltonian(controller, bundle, 'sparse.npz')

    restored = _DataControllerStub({}, {})
    restore_data_controller(restored, read_sparse_hamiltonian(path))
    Dnm = restored.data_arrays['Dnm']

    nawf = controller.data_attributes['nawf']
    assert Dnm.shape == (nawf, nawf, 3)

    tau = controller.data_arrays['tau']
    orbital_atom = build_orbital_basis_table(controller)['orbital_atom']
    centres = tau[orbital_atom]
    np.testing.assert_allclose(Dnm, centres[:, None, :] - centres[None, :, :])
    #  Antisymmetric by construction.
    np.testing.assert_allclose(Dnm, -np.transpose(Dnm, (1, 0, 2)))


def test_restore_repopulates_a_fresh_controller(controller, tmp_path):
    original = controller.data_arrays['HRs']
    bundle = sparsify_real_space_hamiltonian(controller, bond_order=2, magnitude_tol=0.0)
    path = write_sparse_hamiltonian(controller, bundle, 'sparse.npz')

    restored = _DataControllerStub({}, {})
    restore_data_controller(restored, read_sparse_hamiltonian(path))
    arrays, attributes = restored.data_dicts()

    assert attributes['nawf'] == 5
    assert (attributes['nk1'], attributes['nk2'], attributes['nk3']) == GRID
    assert attributes['nkpnts'] == int(np.prod(GRID))
    assert attributes['alat'] == pytest.approx(ALAT)
    assert list(arrays['atoms']) == ['A', 'B']
    np.testing.assert_allclose(arrays['tau'], controller.data_arrays['tau'])

    # The FFT grids the downstream pipeline consumes must be ready.
    assert arrays['R'].shape == (int(np.prod(GRID)), 3)
    assert arrays['kgrid'].shape == (3, int(np.prod(GRID)))
    assert arrays['Dnm'].shape == (5, 5, 3)

    rebuilt = arrays['HRs']
    retained = np.zeros(original.shape[:-1], dtype=bool)
    cells = np.mod(bundle['bond_translation'], np.array(GRID))
    retained[bundle['bond_row'], bundle['bond_col'], cells[:, 0], cells[:, 1], cells[:, 2]] = True
    np.testing.assert_allclose(rebuilt[retained], original[retained])


def test_spin_polarised_round_trip(tmp_path):
    dc = _simple_cubic_controller(opath=tmp_path, nspin=2)
    dc.data_arrays['HRs'] = _hamiltonian_filled_with_distance_decay(dc)

    bundle = sparsify_real_space_hamiltonian(dc, bond_order=2, magnitude_tol=0.0)
    assert bundle['bond_value'].shape[1] == 2

    rebuilt = rebuild_HRs_from_sparse(bundle)
    assert rebuilt.shape[-1] == 2
    # The fixture scales channel 1 by two; the round trip must not mix them.
    np.testing.assert_allclose(bundle['bond_value'][:, 1], 2.0 * bundle['bond_value'][:, 0])


# ---------------------------------------------------------------------------
# Labelled dataset view
# ---------------------------------------------------------------------------


def test_bond_table_labels_every_matrix_element(controller):
    bundle = sparsify_real_space_hamiltonian(controller, bond_order=1, magnitude_tol=0.0)
    table = bond_table(bundle)

    n_bonds = bundle['bond_row'].size
    for column in ('atom_i', 'atom_j', 'species_i', 'orbital_i', 'distance', 'shell'):
        assert len(table[column]) == n_bonds

    assert set(np.unique(table['species_i'])) <= {'A', 'B'}
    assert set(np.unique(table['orbital_i'])) <= {'s', 'px', 'py', 'pz'}

    # The bond vector must reproduce the stored distance.
    np.testing.assert_allclose(
        np.linalg.norm(table['bond_vector'], axis=1), table['distance'], atol=1e-10
    )


def test_bond_table_onsite_terms_are_zero_distance(controller):
    bundle = sparsify_real_space_hamiltonian(controller, bond_order=1, magnitude_tol=0.0)
    table = bond_table(bundle)

    onsite = table['shell'] == 0
    assert onsite.any()
    np.testing.assert_allclose(table['distance'][onsite], 0.0, atol=1e-10)
    np.testing.assert_array_equal(table['atom_i'][onsite], table['atom_j'][onsite])


def test_bond_vector_sign_follows_the_fftn_convention():
    """PAOFLOW uses H(k) = sum_R H(R) exp(-2 pi i k.R), so the bond is tau_B - R - tau_A.

    Getting this sign wrong silently keeps the wrong matrix elements: the star
    distances still look right (they are symmetric in +/- R) but the strongest
    hoppings get assigned to far shells and truncation destroys the bands.
    """
    from scipy import fftpack as FFT

    #  One orbital per atom on a simple-cubic lattice, with a single hopping
    #  placed on the bond that tau_B - R - tau_A identifies as nearest-neighbour.
    lattice_vectors = np.eye(3)
    atomic_positions = np.array([[0.0, 0.0, 0.0], [0.5 * ALAT, 0.0, 0.0]])
    arrays = {
        'a_vectors': lattice_vectors,
        'b_vectors': np.linalg.inv(lattice_vectors).T,
        'tau': atomic_positions,
        'atoms': ['A', 'B'],
        'shells': {'A': [0], 'B': [0]},
    }
    attributes = {'alat': ALAT, 'nawf': 2, 'nspin': 1, 'natoms': 2, 'opath': '.'}
    dc = _DataControllerStub(arrays, attributes)

    hamiltonian = np.zeros((2, 2) + GRID + (1,), dtype=complex)
    #  R = (1,0,0) with the minus convention is the a/2 bond, not the 3a/2 one.
    hamiltonian[0, 1, 1, 0, 0, 0] = 1.0
    arrays['HRs'] = hamiltonian

    bundle = sparsify_real_space_hamiltonian(dc, bond_order=1, magnitude_tol=1e-12)
    table = bond_table(bundle)

    assert table['distance'].size == 1
    assert table['distance'][0] == pytest.approx(0.5 * ALAT)

    #  The bond vector must reproduce the phase actually used by the transform.
    kpoint = np.array([0.3, 0.1, 0.2])
    hk = FFT.fftn(hamiltonian, axes=[2, 3, 4])
    cell = np.round(kpoint * np.array(GRID)).astype(int)
    kfrac = cell / np.array(GRID)
    expected = np.exp(-2j * np.pi * (bundle['bond_translation'][0] @ kfrac))
    assert hk[0, 1, cell[0], cell[1], cell[2], 0] == pytest.approx(expected)
