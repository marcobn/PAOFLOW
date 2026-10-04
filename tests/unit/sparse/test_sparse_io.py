"""Save / load of the base-cell bond list, the orbital map and the bond table."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import fftpack as FFT

from PAOFLOW.sparse.bridge import archive_Dnm, densify
from PAOFLOW.sparse.doubling import double_axis
from PAOFLOW.sparse.hamiltonian import SparseHamiltonian
from PAOFLOW.sparse.io import (
    SPARSE_HAMILTONIAN_FORMAT_VERSION,
    bond_table,
    build_orbital_basis_table,
    read_sparse_hamiltonian,
    restore_data_controller,
    write_sparse_hamiltonian,
)
from test_sparse_shells import ALAT, GRID, _DC, _simple_cubic


@pytest.fixture()
def dc(tmp_path):
    dc = _simple_cubic()
    dc.data_attributes.update(opath=str(tmp_path), verbose=False, abort_on_exception=True)
    return dc


def _saved(dc, **kwargs):
    H = SparseHamiltonian.from_data_controller(dc, 0.0, **kwargs)
    return H, write_sparse_hamiltonian(dc, H, 'sparse.npz')


# ---------------------------------------------------------------------------
# Orbital map
# ---------------------------------------------------------------------------


def test_orbital_table_matches_shells_layout(dc):
    table = build_orbital_basis_table(dc)
    assert list(table['orbital_atom']) == [0, 1, 1, 1, 1]
    assert list(table['orbital_species']) == ['A', 'B', 'B', 'B', 'B']
    assert list(table['orbital_label']) == ['s', 's', 'pz', 'px', 'py']
    assert list(table['orbitals_per_atom']) == [1, 4]
    assert list(table['atom_block_start']) == [0, 1]


def test_tight_binding_models_are_rejected(dc):
    """models.py orders p orbitals px,py,pz, QE pz,px,py; the TB ordering is
    not recoverable, so a TB controller is refused rather than mislabelled."""
    dc.data_arrays['norbitals'] = np.array([1, 4])
    with pytest.raises(RuntimeError, match='tight-binding model'):
        build_orbital_basis_table(dc)


def test_projected_basis_records_take_precedence(dc):
    tau = dc.data_arrays['tau']
    dc.data_arrays['basis'] = [
        {'atom': 'A', 'tau': tau[0], 'l': 0, 'm': 1, 'label': '3S'},
        {'atom': 'B', 'tau': tau[1], 'l': 0, 'm': 1, 'label': '3S'},
        {'atom': 'B', 'tau': tau[1], 'l': 1, 'm': 1, 'label': '3P'},
        {'atom': 'B', 'tau': tau[1], 'l': 1, 'm': 2, 'label': '3P'},
        {'atom': 'B', 'tau': tau[1], 'l': 1, 'm': 3, 'label': '3P'},
    ]
    dc.data_arrays['norbitals'] = np.array([1, 4])  # present but ignored
    table = build_orbital_basis_table(dc)
    assert list(table['orbital_label']) == ['s', 's', 'pz', 'px', 'py']
    assert list(table['orbital_shell']) == ['3S', '3S', '3P', '3P', '3P']


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_round_trip_reproduces_H_k_bit_for_bit(dc):
    H, path = _saved(dc, bond_order=2)
    loaded, bundle = read_sparse_hamiltonian(path)

    assert bundle['metadata']['format_version'] == SPARSE_HAMILTONIAN_FORMAT_VERSION
    assert loaded.nnz == H.nnz
    assert loaded.drop_report['bond_order'] == 2
    rng = np.random.default_rng(1)
    for k in rng.random((5, 3)):
        for sign in (-1, 1):
            a = H.assemble_hk(k, sign=sign)
            b = loaded.assemble_hk(k, sign=sign)
            np.testing.assert_array_equal(a.data, b.data)
            np.testing.assert_array_equal(a.indices, b.indices)


def test_archive_loads_without_pickle(dc):
    _, path = _saved(dc, bond_order=1)
    with np.load(path, allow_pickle=False) as archive:
        assert 'bond_value' in archive.files
        assert 'metadata_json' in archive.files


def test_doubled_or_compacted_lists_are_refused(dc):
    H = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=1)
    with pytest.raises(RuntimeError, match='base'):
        write_sparse_hamiltonian(dc, double_axis(H, 0), 'x.npz')
    with pytest.raises(RuntimeError, match='compact'):
        write_sparse_hamiltonian(dc, H.compact(), 'x.npz')


def test_stored_Dnm_is_preserved_verbatim(dc):
    """Dnm's length unit depends on the source, so it must round-trip exactly."""
    dc.data_arrays['Dnm'] = np.random.default_rng(0).normal(size=(5, 5, 3))
    _, path = _saved(dc)
    _, bundle = read_sparse_hamiltonian(path)
    np.testing.assert_array_equal(bundle['Dnm'], dc.data_arrays['Dnm'])


def test_restore_leaves_the_post_pao_hamiltonian_state(dc, tmp_path):
    _, path = _saved(dc, bond_order=2)
    _, bundle = read_sparse_hamiltonian(path)

    fresh = _DC({}, {'opath': 'elsewhere', 'npool': 3})
    restore_data_controller(fresh, bundle)
    arrays, attributes = fresh.data_dicts()

    assert attributes['nawf'] == 5
    assert (attributes['nk1'], attributes['nk2'], attributes['nk3']) == GRID
    assert attributes['nkpnts'] == int(np.prod(GRID))
    assert attributes['alat'] == pytest.approx(ALAT)
    # session keys come from the live controller, not the archive
    assert attributes['opath'] == 'elsewhere'
    assert attributes['npool'] == 3
    assert list(arrays['atoms']) == ['A', 'B']
    assert arrays['shells'] == {'A': [0], 'B': [0, 1]}
    np.testing.assert_allclose(arrays['tau'], dc.data_arrays['tau'])
    assert arrays['kgrid'].shape == (3, int(np.prod(GRID)))
    # the sparse pipeline never holds these after pao_hamiltonian
    for key in ('HRs', 'Hks', 'Dnm'):
        assert key not in arrays


# ---------------------------------------------------------------------------
# Dense <-> sparse bridge
# ---------------------------------------------------------------------------


def test_densify_leaves_the_dense_post_pao_hamiltonian_state(dc):
    H = SparseHamiltonian.from_data_controller(dc, bond_order=2)
    Dnm = dc.data_arrays['Dnm'].copy()
    fresh = _DC({}, {})
    densify(fresh, H, Dnm=Dnm)
    arrays = fresh.data_arrays

    np.testing.assert_array_equal(arrays['HRs'], H.to_dense_HRs())
    np.testing.assert_array_equal(arrays['Dnm'], Dnm)
    np.testing.assert_allclose(arrays['Hks'], np.fft.fftn(arrays['HRs'], axes=(2, 3, 4)))
    # dense pao_hamiltonian builds no real-space grid; do_gradient makes its
    # own at the current mesh, and do_berry_curvature would reuse a stale one
    for key in ('R', 'Rfft', 'idx', 'R_wght'):
        assert key not in arrays


def test_densify_refuses_a_doubled_list(dc):
    H = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=1)
    with pytest.raises(RuntimeError, match='doubling_Hamiltonian'):
        densify(_DC({}, {}), double_axis(H, 0))
    with pytest.raises(RuntimeError, match='compact'):
        densify(_DC({}, {}), H.compact())


def _restart_driver(tmp_path, name):
    from PAOFLOW.PAOFLOW import PAOFLOW

    return PAOFLOW(workpath=str(tmp_path), outputdir=name, restart=True, smearing='gauss')


def test_dense_save_then_dense_restart(dc, tmp_path):
    """Dense driver -> archive -> dense driver, with the saving run unchanged."""
    saver = _restart_driver(tmp_path, 'save')
    saver.data_controller.data_arrays = dc.data_arrays
    saver.data_controller.data_attributes = dc.data_attributes
    original = dc.data_arrays['HRs'].copy()
    saver.save_sparse_hamiltonian('s.npz', bond_order=2)
    np.testing.assert_array_equal(dc.data_arrays['HRs'], original)

    loader = _restart_driver(tmp_path, 'load')
    loader.load_sparse_hamiltonian(str(tmp_path / 's.npz'))
    arrays, attributes = loader.data_controller.data_dicts()

    H = SparseHamiltonian.from_data_controller(dc, bond_order=2)
    np.testing.assert_array_equal(arrays['HRs'], H.to_dense_HRs())
    np.testing.assert_array_equal(arrays['Dnm'], dc.data_arrays['Dnm'])
    assert attributes['opath'] == str(tmp_path / 'load')


def test_sparse_written_archive_restarts_densely_with_Dnm(dc, tmp_path):
    """A sparse run pops Dnm at conversion; the archive must still carry it,
    or the dense gradient fails after a dense restart."""
    Dnm = dc.data_arrays.pop('Dnm')
    H = SparseHamiltonian.from_data_controller(dc, bond_order=2)
    _, bundle = read_sparse_hamiltonian(write_sparse_hamiltonian(dc, H, 'a.npz', Dnm=Dnm))
    np.testing.assert_array_equal(bundle['Dnm'], Dnm)

    # an archive written without it gets the geometric offsets back exactly
    _, bundle = read_sparse_hamiltonian(write_sparse_hamiltonian(dc, H, 'b.npz'))
    assert 'Dnm' not in bundle
    np.testing.assert_allclose(archive_Dnm(bundle), Dnm, atol=1e-12)


def test_dense_save_refuses_both_cutoffs(dc, tmp_path):
    saver = _restart_driver(tmp_path, 'save')
    saver.data_controller.data_arrays = dc.data_arrays
    saver.data_controller.data_attributes = dc.data_attributes
    with pytest.raises(ValueError, match='not both'):
        saver.save_sparse_hamiltonian('s.npz', bond_order=1, rcut=3.0)


# ---------------------------------------------------------------------------
# Labelled dataset view
# ---------------------------------------------------------------------------


def test_bond_table_labels_every_matrix_element(dc):
    H, path = _saved(dc, bond_order=1)
    table = bond_table(read_sparse_hamiltonian(path)[1])

    for column in ('atom_i', 'atom_j', 'species_i', 'orbital_i', 'shell_i', 'distance', 'shell'):
        assert len(table[column]) == H.nnz
    assert set(np.unique(table['orbital_i'])) <= {'s', 'px', 'py', 'pz'}
    np.testing.assert_allclose(
        np.linalg.norm(table['bond_vector'], axis=1), table['distance'], atol=1e-10
    )
    assert set(np.unique(table['shell'])) == {0, 1}


def test_bond_table_onsite_terms_are_zero_distance(dc):
    _, path = _saved(dc, bond_order=1)
    table = bond_table(read_sparse_hamiltonian(path)[1])
    onsite = table['shell'] == 0
    assert onsite.any()
    np.testing.assert_allclose(table['distance'][onsite], 0.0, atol=1e-10)
    np.testing.assert_array_equal(table['atom_i'][onsite], table['atom_j'][onsite])


def test_bond_vector_sign_follows_the_fftn_convention(tmp_path):
    """H(k) = sum_R H(R) exp(-2 pi i k.R), so the bond is tau_j - tau_i - alat*R.

    With the wrong sign the star distances still look right (they are
    symmetric in +/- R) but the strongest hoppings land on far shells.
    """
    tau = np.array([[0.0, 0.0, 0.0], [0.5 * ALAT, 0.0, 0.0]])
    HRs = np.zeros((2, 2) + GRID + (1,), dtype=complex)
    HRs[0, 1, 1, 0, 0, 0] = 1.0  # R = (1,0,0): the a/2 bond, not the 3a/2 one
    dc = _DC(
        {
            'HRs': HRs,
            'a_vectors': np.eye(3),
            'tau': tau,
            'atoms': ['A', 'B'],
            'shells': {'A': [0], 'B': [0]},
            'Dnm': (tau[:, None, :] - tau[None, :, :]),
        },
        {'alat': ALAT, 'nawf': 2, 'nspin': 1, 'nk1': 4, 'nk2': 4, 'nk3': 4, 'opath': str(tmp_path)},
    )
    H = SparseHamiltonian.from_data_controller(dc, bond_order=1)
    _, bundle = read_sparse_hamiltonian(write_sparse_hamiltonian(dc, H, 's.npz'))
    table = bond_table(bundle)

    assert table['distance'].size == 1
    assert table['distance'][0] == pytest.approx(0.5 * ALAT)

    # the stored triple reproduces the phase of the transform itself
    cell = np.array([1, 0, 3])
    hk = FFT.fftn(HRs, axes=[2, 3, 4])[0, 1, cell[0], cell[1], cell[2], 0]
    assert hk == pytest.approx(np.exp(-2j * np.pi * (table['translation'][0] @ (cell / 4))))
    assert H.assemble_hk(cell / 4, sign=-1)[0, 1] == pytest.approx(hk)
