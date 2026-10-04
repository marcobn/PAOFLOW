"""Neighbour-shell (``bond_order``) cutoff and the dense round trip.

The fixtures build a small simple-cubic two-atom cell with an analytically
known neighbour star, so the retained bonds can be checked exactly without
standing up the MPI-backed PAOFLOW pipeline.  Dense arrays appear only as
test-side references.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fftpack import ifftn

from PAOFLOW.sparse.hamiltonian import SparseHamiltonian, _minus_R_index, folded_R_triples
from PAOFLOW.sparse.shells import aliasing_safe_radius, compute_star_shells, shell_cutoff

ALAT = 6.0
GRID = (4, 4, 4)


class _DC:
    def __init__(self, arrays, attributes):
        self.data_arrays = arrays
        self.data_attributes = attributes

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


def _simple_cubic(nspin=1, with_dnm=True):
    """``s`` atom at the origin, ``sp`` atom at (a/2) x: nawf = 5."""
    tau = np.array([[0.0, 0.0, 0.0], [0.5 * ALAT, 0.0, 0.0]])
    orbital_atom = np.array([0, 1, 1, 1, 1])
    arrays = {
        'a_vectors': np.eye(3),
        'b_vectors': np.eye(3),
        'tau': tau,
        'atoms': ['A', 'B'],
        'shells': {'A': [0], 'B': [0, 1]},
    }
    if with_dnm:
        centres = tau[orbital_atom]
        arrays['Dnm'] = centres[:, None, :] - centres[None, :, :]
    attributes = {
        'alat': ALAT,
        'nawf': 5,
        'nspin': nspin,
        'natoms': 2,
        'nk1': GRID[0],
        'nk2': GRID[1],
        'nk3': GRID[2],
    }
    dc = _DC(arrays, attributes)

    # H_ij(R) decays with the PAOFLOW bond length |alat*R + tau_i - tau_j|
    Rcart = folded_R_triples(*GRID).astype(float) * ALAT
    centres = tau[orbital_atom]
    dist = np.linalg.norm(
        (centres[:, None, :] - centres[None, :, :])[:, :, None, :] + Rcart[None, None], axis=3
    )
    HRs = np.exp(-dist / ALAT).reshape(5, 5, *GRID)[..., None] * np.arange(1, nspin + 1)
    arrays['HRs'] = HRs.astype(complex)
    return dc


def _lengths(dc, H):
    """Bond length of every stored bond, from the test-side geometry."""
    arry, attr = dc.data_dicts()
    Dnm = arry['Dnm']
    Rcart = H.R_int[H.ridx].astype(float) @ arry['a_vectors'] * attr['alat']
    return np.linalg.norm(Dnm[H.rows, H.cols] + Rcart, axis=1)


@pytest.fixture()
def dc():
    return _simple_cubic()


# ---------------------------------------------------------------------------
# Star geometry
# ---------------------------------------------------------------------------


def test_star_shells_reproduce_simple_cubic_distances(dc):
    """First shells are a/2 (A-B), then a (A-A, B-B), ... in Bohr."""
    shells = compute_star_shells(dc.data_arrays['tau'], np.eye(3) * ALAT, GRID)
    assert shells[0] == pytest.approx(0.5 * ALAT)
    assert np.any(np.isclose(shells, ALAT))
    assert np.all(np.diff(shells) > 0)
    assert shells[-1] <= aliasing_safe_radius(np.eye(3) * ALAT, GRID)


def test_aliasing_safe_radius_is_half_the_supercell():
    assert aliasing_safe_radius(np.eye(3) * ALAT, GRID) == pytest.approx(0.5 * GRID[0] * ALAT)


def test_bond_order_beyond_available_shells_is_rejected(dc):
    with pytest.raises(ValueError, match='exceeds'):
        shell_cutoff(dc, 10_000)


def test_rcut_and_bond_order_are_mutually_exclusive(dc):
    with pytest.raises(ValueError, match='not both'):
        SparseHamiltonian.from_data_controller(dc, 0.0, rcut=5.0, bond_order=1)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_bond_order_is_the_same_filter_as_rcut_at_the_shell(dc):
    """bond_order is only a way of choosing rcut."""
    rc, _, _ = shell_cutoff(dc, 2)
    by_shell = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=2)
    by_radius = SparseHamiltonian.from_data_controller(dc, 0.0, rcut=rc)
    np.testing.assert_array_equal(by_shell.rows, by_radius.rows)
    np.testing.assert_array_equal(by_shell.ridx, by_radius.ridx)
    assert by_shell.drop_report['bond_order'] == 2
    assert by_shell.drop_report['rcut'] == pytest.approx(rc)


def test_first_shell_keeps_only_onsite_and_nearest_neighbours(dc):
    H = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=1)
    d = _lengths(dc, H)
    assert d.max() == pytest.approx(0.5 * ALAT)
    assert set(np.unique(np.round(d, 6))) == {0.0, 0.5 * ALAT}


def test_higher_bond_order_is_a_superset_of_lower(dc):
    def keys(H):
        t = H.R_int[H.ridx]
        return {(int(r), int(c), *map(int, tt)) for r, c, tt in zip(H.rows, H.cols, t)}

    first = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=1)
    second = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=2)
    assert second.nnz > first.nnz
    assert keys(first) <= keys(second)


def test_threshold_is_exclusive_with_the_shell_cutoff(dc):
    """An element threshold would split degeneracies the shell cut keeps."""
    with pytest.raises(ValueError, match='threshold cannot be combined'):
        SparseHamiltonian.from_data_controller(dc, 0.5, bond_order=3)
    with pytest.raises(ValueError, match='threshold cannot be combined'):
        SparseHamiltonian.from_data_controller(dc, 0.5, rcut=2.0 * ALAT)


def test_shell_cutoff_defaults_to_no_element_threshold(dc):
    default = SparseHamiltonian.from_data_controller(dc, bond_order=3)
    explicit = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=3)
    assert default.threshold == 0.0
    np.testing.assert_array_equal(default.rows, explicit.rows)
    np.testing.assert_array_equal(default.ridx, explicit.ridx)


def test_threshold_alone_defaults_to_1e_3(dc):
    assert SparseHamiltonian.from_data_controller(dc).threshold == 1.0e-3


def test_geometry_falls_back_to_the_orbital_map_without_Dnm():
    """A controller without Dnm must cut on the true bond length, not |alat*R|."""
    with_dnm = SparseHamiltonian.from_data_controller(_simple_cubic(), 0.0, bond_order=1)
    without = SparseHamiltonian.from_data_controller(
        _simple_cubic(with_dnm=False), 0.0, bond_order=1
    )
    np.testing.assert_array_equal(with_dnm.rows, without.rows)
    np.testing.assert_array_equal(with_dnm.ridx, without.ridx)


def test_explicit_rcut_beyond_the_safe_radius_is_flagged(dc):
    safe = aliasing_safe_radius(np.eye(3) * ALAT, GRID)
    assert SparseHamiltonian.from_data_controller(dc, 0.0, rcut=1.5 * safe).drop_report['aliased']
    assert not SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=1).drop_report['aliased']


def test_outermost_shell_stays_hermitian_on_an_even_grid():
    """On the folded Nyquist plane a bond and its (j,i,-R) partner differ in
    length.  A plain |d| <= r_c mask (the former sparse_hamiltonian.py rule)
    keeps one and drops the other; the cutoff here keeps the pair."""
    rng = np.random.default_rng(3)
    nawf = 6
    tau = np.array([[0.0, 0.0, 0.0], [0.31, 0.17, 0.05]]) * ALAT
    orbital_atom = np.array([0, 0, 0, 1, 1, 1])
    centres = tau[orbital_atom]
    Hks = rng.standard_normal((nawf, nawf, *GRID)) + 1j * rng.standard_normal((nawf, nawf, *GRID))
    Hks = 0.5 * (Hks + Hks.conj().transpose(1, 0, 2, 3, 4))
    dc = _DC(
        {
            'HRs': ifftn(Hks, axes=(2, 3, 4))[..., None],
            'a_vectors': np.eye(3),
            'tau': tau,
            'Dnm': centres[:, None, :] - centres[None, :, :],
        },
        {'alat': ALAT, 'nawf': nawf, 'nspin': 1, 'nk1': GRID[0], 'nk2': GRID[1], 'nk3': GRID[2]},
    )
    _, shells, _ = shell_cutoff(dc, 1)
    order = len(shells)
    H = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=order)
    assert H.hermiticity_error() < 1e-12

    # counterfactual: the plain mask is not closed under the Hermitian pairing
    rc, _, _ = shell_cutoff(dc, order)
    R_int = folded_R_triples(*GRID)
    dist = np.linalg.norm(dc.data_arrays['Dnm'][:, :, None, :] + R_int[None, None] * ALAT, axis=3)
    plain = dist <= rc
    partner = plain.transpose(1, 0, 2)[:, :, _minus_R_index(R_int, GRID)]
    assert np.any(plain != partner)


# ---------------------------------------------------------------------------
# Dense round trip
# ---------------------------------------------------------------------------


def test_no_truncation_reproduces_HRs_exactly(dc):
    H = SparseHamiltonian.from_data_controller(dc, 0.0)
    np.testing.assert_array_equal(H.to_dense_HRs(), dc.data_arrays['HRs'])


def test_truncated_round_trip_is_exact_where_kept_and_zero_elsewhere(dc):
    H = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=2)
    dense = H.to_dense_HRs()
    kept = np.zeros(dense.shape[:-1], dtype=bool)
    cell = np.mod(H.R_int[H.ridx], GRID)
    kept[H.rows, H.cols, cell[:, 0], cell[:, 1], cell[:, 2]] = True
    np.testing.assert_array_equal(dense[kept], dc.data_arrays['HRs'][kept])
    assert np.all(dense[~kept] == 0.0)


def test_spin_channels_are_not_mixed():
    dc = _simple_cubic(nspin=2)
    H = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=2)
    assert H.to_dense_HRs().shape[-1] == 2
    np.testing.assert_allclose(H.vals[:, 1], 2.0 * H.vals[:, 0])


def test_to_dense_refuses_after_compact(dc):
    H = SparseHamiltonian.from_data_controller(dc, 0.0, bond_order=1).compact()
    with pytest.raises(RuntimeError, match='compact'):
        H.to_dense_HRs()
