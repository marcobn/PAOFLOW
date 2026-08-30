"""Real-space cutoff (``rcut``): mechanism tests.

``rcut`` is a second, physically different truncation axis from
``threshold`` — bond length rather than matrix-element magnitude — so it
needs its own guards.  These pin mechanisms (commutation with doubling,
Hermiticity closure, the eigenvalue bound), not output accuracy; the
accuracy of any particular cutoff value is an end-to-end question for the
comparison notebook.

Dense arrays appear only as test-side references.
"""

import numpy as np
from scipy.fftpack import ifftn

from PAOFLOW.hamiltonian.do_doubling import doubling_HRs
from PAOFLOW.sparse.doubling import double_axis
from PAOFLOW.sparse.hamiltonian import SparseHamiltonian, _minus_R_index, folded_R_triples


class _DC:
    def __init__(self, arrays, attributes):
        self.data_arrays = arrays
        self.data_attributes = attributes

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


NAWF, NK, ALAT = 6, (4, 4, 4), 10.26
A_VECTORS = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])


def _make_dc(rng, nx=0, ny=0, nz=0, spread=2.0):
    nk1, nk2, nk3 = NK
    Hks = rng.standard_normal((NAWF, NAWF, nk1, nk2, nk3, 1)) + 1j * rng.standard_normal(
        (NAWF, NAWF, nk1, nk2, nk3, 1)
    )
    Hks = 0.5 * (Hks + Hks.conj().transpose(1, 0, 2, 3, 4, 5))
    HRs = ifftn(Hks[..., 0], axes=(2, 3, 4))[..., None]
    tau_orb = rng.standard_normal((NAWF, 3)) * spread
    arrays = {
        'HRs': HRs,
        'a_vectors': A_VECTORS.copy(),
        'tau': rng.standard_normal((2, 3)),
        'Dnm': tau_orb[:, None, :] - tau_orb[None, :, :],
    }
    attrs = {
        'alat': ALAT,
        'nawf': NAWF,
        'nspin': 1,
        'nk1': nk1,
        'nk2': nk2,
        'nk3': nk3,
        'nx': nx,
        'ny': ny,
        'nz': nz,
    }
    return _DC(arrays, attrs)


def _bond_set(sph):
    triples = sph.R_int[sph.ridx]
    return {
        (int(sph.rows[b]), int(sph.cols[b]), *(int(x) for x in triples[b])) for b in range(sph.nnz)
    }


def _symmetric_distance(dc):
    """(nawf, nawf, nR) bond lengths, symmetrized over (i,j,R) -> (j,i,-R)."""
    arry, attr = dc.data_dicts()
    R_int = folded_R_triples(*NK)
    Rcart = (R_int.astype(float) @ arry['a_vectors']) * attr['alat']
    dist = np.linalg.norm(arry['Dnm'][:, :, None, :] + Rcart[None, None, :, :], axis=3)
    minus = _minus_R_index(R_int, NK)
    return np.minimum(dist, dist.transpose(1, 0, 2)[:, :, minus])


def test_rcut_drops_only_long_bonds():
    """Every kept bond is inside the cutoff and every dropped one is not
    (up to the magnitude threshold, which is off here)."""
    dc = _make_dc(np.random.default_rng(3))
    rcut = 18.0
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0, rcut=rcut)
    dist = _symmetric_distance(dc)
    triples = sph.R_int[sph.ridx]
    lut = {tuple(t): n for n, t in enumerate(folded_R_triples(*NK).tolist())}
    for b in range(sph.nnz):
        r = lut[tuple(int(x) for x in triples[b])]
        assert dist[int(sph.rows[b]), int(sph.cols[b]), r] <= rcut
    assert sph.nnz == int((dist <= rcut).sum())
    # and it actually bites
    full = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    assert 0 < sph.nnz < full.nnz


def test_rcut_preserves_hermiticity():
    """The kept set must be closed under (i,j,R) -> (j,i,-R).

    Off the Nyquist plane this is automatic, since
    |-R + tau_j - tau_i| = |R + tau_i - tau_j|.  On the folded Nyquist
    plane -R maps onto R itself and the two partners have *different*
    lengths, so an unsymmetrized mask would break Hermiticity there --
    which is what this pins.
    """
    for seed in (5, 6, 7):
        dc = _make_dc(np.random.default_rng(seed))
        for rcut in (12.0, 18.0, 25.0):
            sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0, rcut=rcut)
            assert sph.hermiticity_error() < 1e-12, (seed, rcut)
            bonds = _bond_set(sph)
            minus = _minus_R_index(folded_R_triples(*NK), NK)
            lut = {tuple(t): n for n, t in enumerate(folded_R_triples(*NK).tolist())}
            R_list = folded_R_triples(*NK)
            for i, j, m1, m2, m3 in bonds:
                mR = R_list[minus[lut[(m1, m2, m3)]]]
                assert (j, i, *(int(x) for x in mR)) in bonds
            # assembled H(k) is Hermitian at a random off-grid k
            Hk = sph.assemble_hk(np.random.default_rng(seed).standard_normal(3)).toarray()
            assert np.allclose(Hk, Hk.conj().T, atol=1e-12)


def test_rcut_commutes_with_doubling():
    """cutoff-then-double == double-of-cutoff, bond for bond.

    Doubling is a pure rearrangement of bond content, so applying the
    cutoff at the base cell and then doubling must reproduce exactly what
    the dense doubling kernel produces from the already-cut H(R).  This
    is the property that lets rcut live at the single sanctioned
    dense->sparse boundary.
    """
    seed = 51
    dc = _make_dc(np.random.default_rng(seed))
    rcut = 20.0
    early = double_axis(
        SparseHamiltonian.from_data_controller(dc, threshold=0.0, rcut=rcut),
        0,
    )

    # dense reference: zero the cut entries in HRs, then run the real kernel
    dc2 = _make_dc(np.random.default_rng(seed), nx=1)
    dist = _symmetric_distance(dc2)
    arry2, _ = dc2.data_dicts()
    nk1, nk2, nk3 = NK
    masked = arry2['HRs'].reshape(NAWF, NAWF, nk1 * nk2 * nk3, 1).copy()
    masked[dist > rcut, :] = 0.0
    arry2['HRs'] = masked.reshape(NAWF, NAWF, nk1, nk2, nk3, 1)
    doubling_HRs(dc2)
    late = SparseHamiltonian.from_data_controller(dc2, threshold=0.0)

    # the dense path keeps explicit zeros; compare against the nonzero set
    late_bonds = {b for b in _bond_set(late)}
    assert _bond_set(early) <= late_bonds
    triples_e = early.R_int[early.ridx]
    idx = {
        (int(early.rows[b]), int(early.cols[b]), *(int(x) for x in triples_e[b])): b
        for b in range(early.nnz)
    }
    triples_l = late.R_int[late.ridx]
    for b in range(late.nnz):
        key = (int(late.rows[b]), int(late.cols[b]), *(int(x) for x in triples_l[b]))
        if np.abs(late.vals[b]).max() == 0.0:
            continue  # explicit zero the sparse path never stored
        assert key in idx, key
        assert np.allclose(early.vals[idx[key]], late.vals[b], atol=1e-14)
        assert np.allclose(early.dnm[idx[key]], late.dnm[b], atol=1e-14)


def test_rcut_eig_bound_is_a_bound():
    """The reported eig_bound must cover the cutoff too, not just the
    magnitude threshold, since rcut is folded into the same keep mask."""
    dc = _make_dc(np.random.default_rng(23))
    exact = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    trunc = SparseHamiltonian.from_data_controller(dc, threshold=0.0, rcut=15.0)
    bound = trunc.drop_report['eig_bound']
    assert bound > 0.0
    assert trunc.drop_report['rcut'] == 15.0
    rng = np.random.default_rng(24)
    for _ in range(4):
        kfrac = rng.standard_normal(3)
        e_exact = np.linalg.eigvalsh(exact.assemble_hk(kfrac).toarray())
        e_trunc = np.linalg.eigvalsh(trunc.assemble_hk(kfrac).toarray())
        assert np.abs(e_exact - e_trunc).max() <= bound + 1e-12


def test_rcut_none_is_exactly_the_old_path():
    """rcut=None must not perturb anything, bit for bit."""
    dc = _make_dc(np.random.default_rng(31))
    a = SparseHamiltonian.from_data_controller(dc, threshold=1e-3)
    b = SparseHamiltonian.from_data_controller(dc, threshold=1e-3, rcut=None)
    assert a.nnz == b.nnz
    assert np.array_equal(a.rows, b.rows) and np.array_equal(a.cols, b.cols)
    assert (a.vals == b.vals).all()


def test_cutoff_must_be_applied_before_doubling():
    """The container records that it has been doubled, so a cutoff can
    never be applied to a cell whose dnm has already been zeroed on
    cross-replica blocks (after which the bond vector is unrecoverable)."""
    dc = _make_dc(np.random.default_rng(41))
    base = SparseHamiltonian.from_data_controller(dc, threshold=0.0, rcut=20.0)
    assert not base._doubled
    doubled = double_axis(base, 0)
    assert doubled._doubled
    assert doubled.hermitize()._doubled
