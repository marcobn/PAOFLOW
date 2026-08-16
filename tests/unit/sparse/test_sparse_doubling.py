"""Sparse doubling: exact bond-level parity with the dense doubling kernel.

The dense ``doubling_HRs`` (run here on small test matrices only) is the
reference: converting its doubled dense output to a bond list must yield
exactly the same bonds — values and per-bond ``dnm`` — as applying
``double_axis`` to the converted base cell.
"""

import numpy as np
from scipy.fftpack import ifftn

from PAOFLOW.hamiltonian.do_doubling import doubling_HRs
from PAOFLOW.sparse.doubling import double_axis
from PAOFLOW.sparse.hamiltonian import SparseHamiltonian


class _DC:
    def __init__(self, arrays, attributes):
        self.data_arrays = arrays
        self.data_attributes = attributes

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


NAWF, NK, ALAT = 6, (4, 4, 4), 10.26


def _make_dc(rng, nx=0, ny=0, nz=0):
    nk1, nk2, nk3 = NK
    Hks = rng.standard_normal((NAWF, NAWF, nk1, nk2, nk3, 1)) + 1j * rng.standard_normal(
        (NAWF, NAWF, nk1, nk2, nk3, 1)
    )
    Hks = 0.5 * (Hks + Hks.conj().transpose(1, 0, 2, 3, 4, 5))
    HRs = ifftn(Hks[..., 0], axes=(2, 3, 4))[..., None]

    a_vectors = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    tau = rng.standard_normal((2, 3))
    dnm = rng.standard_normal((NAWF, 3))
    arrays = {
        'HRs': HRs,
        'a_vectors': a_vectors.copy(),
        'tau': tau,
        'Dnm': dnm[:, None, :] - dnm[None, :, :],
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


def _bond_dict(sph):
    out = {}
    triples = sph.R_int[sph.ridx]
    for b in range(sph.nnz):
        key = (
            int(sph.rows[b]),
            int(sph.cols[b]),
            int(triples[b, 0]),
            int(triples[b, 1]),
            int(triples[b, 2]),
        )
        assert key not in out, 'duplicate bond %s' % (key,)
        out[key] = (sph.vals[b].copy(), sph.dnm[b].copy())
    return out


def _assert_same_bonds(sa, sb, atol=1e-13):
    da, db = _bond_dict(sa), _bond_dict(sb)
    assert set(da) == set(db)
    for key, (va, dnma) in da.items():
        vb, dnmb = db[key]
        assert np.allclose(va, vb, atol=atol), key
        assert np.allclose(dnma, dnmb, atol=atol), key


def _double_dense(rng_seed, nx, ny, nz):
    """Dense reference: doubling_HRs on a fresh DataController."""
    dc = _make_dc(np.random.default_rng(rng_seed), nx=nx, ny=ny, nz=nz)
    doubling_HRs(dc)
    return SparseHamiltonian.from_data_controller(dc, threshold=0.0)


def _double_sparse(rng_seed, axes):
    dc = _make_dc(np.random.default_rng(rng_seed))
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    for ax in axes:
        sph = double_axis(sph, ax)
    return sph


def test_single_doubling_each_axis_matches_dense():
    for seed, (axis, flags) in enumerate([(0, (1, 0, 0)), (1, (0, 1, 0)), (2, (0, 0, 1))]):
        dense = _double_dense(seed, *flags)
        sparse = _double_sparse(seed, [axis])
        assert sparse.nawf == 2 * NAWF
        _assert_same_bonds(sparse, dense)


def test_two_doublings_x_then_y_match_dense():
    dense = _double_dense(42, 1, 1, 0)
    sparse = _double_sparse(42, [0, 1])
    assert sparse.nawf == 4 * NAWF
    _assert_same_bonds(sparse, dense)


def test_nnz_doubles_and_a_vectors_scale():
    dc = _make_dc(np.random.default_rng(3))
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    doubled = double_axis(sph, 1)
    assert doubled.nnz == 2 * sph.nnz
    assert np.allclose(doubled.a_vectors[1], 2 * sph.a_vectors[1])
    assert np.allclose(doubled.a_vectors[0], sph.a_vectors[0])


def test_threshold_commutes_with_doubling():
    """double(threshold(H)) == threshold(double(H)) — doubling only
    rearranges and duplicates values."""
    seed = 51
    dc = _make_dc(np.random.default_rng(seed))
    thr = 0.02 * float(np.abs(dc.data_dicts()[0]['HRs']).max())

    early = double_axis(SparseHamiltonian.from_data_controller(dc, threshold=thr), 0)

    dc2 = _make_dc(np.random.default_rng(seed), nx=1)
    doubling_HRs(dc2)
    late = SparseHamiltonian.from_data_controller(dc2, threshold=thr)

    _assert_same_bonds(early, late)


def test_doubling_then_hermitize_is_hermitian():
    """The dense doubling kernel (replicated exactly) maps the Nyquist plane
    asymmetrically, so the raw doubled list is slightly non-Hermitian — the
    dense pipeline mops that up with per-k Hermitizations.  hermitize() must
    restore exact Hermiticity."""
    dc = _make_dc(np.random.default_rng(8))
    raw = double_axis(SparseHamiltonian.from_data_controller(dc, threshold=0.0), 2)
    herm = raw.hermitize()
    assert herm.hermiticity_error() < 1e-12
    rng = np.random.default_rng(9)
    Hk = herm.assemble_hk(rng.standard_normal(3), sign=-1).toarray()
    assert np.allclose(Hk, Hk.conj().T, atol=1e-12)


def test_hermitize_equals_per_k_hermitization():
    """Bond-level (B + B^dagger)/2 must equal (H(k) + H(k)^dagger)/2 at
    every k — the identity that justifies replacing the dense pipeline's
    per-k Hermitization."""
    dc = _make_dc(np.random.default_rng(21))
    raw = double_axis(SparseHamiltonian.from_data_controller(dc, threshold=0.0), 0)
    herm = raw.hermitize()
    rng = np.random.default_rng(22)
    for sign in (+1, -1):
        kfrac = rng.standard_normal(3)
        Hr = raw.assemble_hk(kfrac, sign=sign).toarray()
        Hh = herm.assemble_hk(kfrac, sign=sign).toarray()
        assert np.allclose(Hh, 0.5 * (Hr + Hr.conj().T), atol=1e-12)


def test_hermitize_is_noop_on_hermitian_base_cell():
    dc = _make_dc(np.random.default_rng(31))
    base = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    herm = base.hermitize()
    rng = np.random.default_rng(32)
    kfrac = rng.standard_normal(3)
    assert np.allclose(
        base.assemble_hk(kfrac).toarray(), herm.assemble_hk(kfrac).toarray(), atol=1e-12
    )
