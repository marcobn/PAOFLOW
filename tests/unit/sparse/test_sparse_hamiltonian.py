"""SparseHamiltonian container: assembly parity against dense references.

Dense arrays appear here only as test-side references — the container under
test never materializes them.  The references pin the exact conventions of
the dense pipeline: ``fftn``/``ifftn`` duality for the mesh (sign = -1),
``band_loop_H`` for the k-path (sign = +1, Cartesian k), and
``do_gradient``'s per-bond coefficient for dH/dk.
"""

import numpy as np
from scipy.fftpack import fftn, ifftn

from PAOFLOW.sparse.hamiltonian import SparseHamiltonian, folded_R_triples


class _DC:
    def __init__(self, arrays, attributes):
        self.data_arrays = arrays
        self.data_attributes = attributes

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


NAWF, NK = 6, (4, 4, 4)
ALAT = 10.26


def _make_dc(rng, nspin=1, with_dnm=True):
    """Random Hermitian H(k) on the grid, ifftn -> HRs (exact FFT duality)."""
    nk1, nk2, nk3 = NK
    Hks = rng.standard_normal((NAWF, NAWF, nk1, nk2, nk3, nspin)) + 1j * rng.standard_normal(
        (NAWF, NAWF, nk1, nk2, nk3, nspin)
    )
    Hks = 0.5 * (Hks + Hks.conj().transpose(1, 0, 2, 3, 4, 5))
    HRs = np.empty_like(Hks)
    for s in range(nspin):
        HRs[..., s] = ifftn(Hks[..., s], axes=(2, 3, 4))

    # fcc-like primitive vectors (rows, units of alat)
    a_vectors = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    arrays = {'HRs': HRs, 'a_vectors': a_vectors}
    if with_dnm:
        tau = rng.standard_normal((NAWF, 3))
        arrays['Dnm'] = tau[:, None, :] - tau[None, :, :]  # antisymmetric, like tau_n - tau_m
    attrs = {'alat': ALAT, 'nk1': nk1, 'nk2': nk2, 'nk3': nk3, 'nspin': nspin}
    return _DC(arrays, attrs), Hks


def test_folded_triples_match_R_grid_fft():
    from PAOFLOW.utils.get_R_grid_fft import get_R_grid_fft

    a_vectors = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    dc = _DC({'a_vectors': a_vectors}, {})
    get_R_grid_fft(dc, *NK)
    R_int = folded_R_triples(*NK)
    assert np.allclose(R_int.astype(float) @ a_vectors, dc.data_dicts()[0]['R'])


def test_assemble_matches_fftn_on_grid_points():
    rng = np.random.default_rng(7)
    dc, Hks = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    nk1, nk2, nk3 = NK
    for i, j, k in [(0, 0, 0), (1, 3, 2), (3, 1, 1)]:
        kfrac = np.array([i / nk1, j / nk2, k / nk3])
        # fftn of HRs recovers H(k) at grid points with the mesh (-) convention
        Hk = sph.assemble_hk(kfrac, sign=-1).toarray()
        assert np.allclose(Hk, Hks[:, :, i, j, k, 0], atol=1e-12)


def test_assemble_matches_zero_pad_interpolation_off_grid():
    """Off the original grid, assembly must match the dense pipeline's
    Hermiticity-preserving interpolation (zero_pad -> fftn on a finer grid),
    i.e. the Nyquist-split convention — not the naive folded sum."""
    from PAOFLOW.utils.zero_pad import zero_pad

    rng = np.random.default_rng(11)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    nk1, nk2, nk3 = NK
    nf1, nf2, nf3 = 2 * nk1, 2 * nk2, 2 * nk3
    HRs = dc.data_dicts()[0]['HRs']
    Hk_fine = np.empty((NAWF, NAWF, nf1, nf2, nf3), dtype=complex)
    for i in range(NAWF):
        for j in range(NAWF):
            padded = zero_pad(HRs[i, j, :, :, :, 0], nk1, nk2, nk3, nf1 - nk1, nf2 - nk2, nf3 - nk3)
            Hk_fine[i, j] = fftn(padded)
    for i, j, k in [(1, 0, 0), (3, 5, 7), (7, 2, 1)]:  # odd -> off the 4^3 grid
        kfrac = np.array([i / nf1, j / nf2, k / nf3])
        Hk = sph.assemble_hk(kfrac, sign=-1).toarray()
        assert np.allclose(Hk, Hk_fine[:, :, i, j, k], atol=1e-12)


def test_assembled_hk_is_hermitian_at_random_k():
    rng = np.random.default_rng(13)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    Hk = sph.assemble_hk(rng.standard_normal(3), sign=-1).toarray()
    assert np.allclose(Hk, Hk.conj().T, atol=1e-12)
    assert sph.hermiticity_error() < 1e-12


def test_band_loop_H_parity_cartesian_grid_k():
    """Parity with the dense band-path assembler at original-grid k-points
    (off the grid band_loop_H is non-Hermitian at the Nyquist shell and the
    two conventions legitimately differ by the split)."""
    from PAOFLOW.spectrum.do_bands import band_loop_H
    from PAOFLOW.utils.get_R_grid_fft import get_R_grid_fft

    rng = np.random.default_rng(17)
    dc, _ = _make_dc(rng)
    arrays, _ = dc.data_dicts()
    get_R_grid_fft(dc, *NK)
    b_vectors = np.linalg.inv(arrays['a_vectors']).T  # A . B^T = 1
    kfracs = np.array([[0.25, 0.5, 0.75], [0.0, 0.25, 0.5], [0.75, 0.75, 0.25]])
    kq = (kfracs @ b_vectors).T  # Cartesian, dense convention after rotation
    Haux = band_loop_H(dc, kq)

    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    for n in range(kq.shape[1]):
        Hk = sph.assemble_hk(kq[:, n], sign=+1, cart=True).toarray()
        assert np.allclose(Hk, Haux[:, :, n, 0], atol=1e-12)


def test_gradient_matches_hermitized_do_gradient_at_grid_k():
    """At grid k the split assembly must equal the dense pipeline's product:
    dH_l = sum_R 1j*(alat*Rcart_l + Dnm_l) * H(R) * e^{-2pi i k.m}, followed
    by the (dH + dH^dagger)/2 Hermitization done in
    PAOFLOW.gradient_and_momenta (the Nyquist-axis R components cancel in
    both representations)."""
    rng = np.random.default_rng(19)
    dc, _ = _make_dc(rng)
    arrays, _ = dc.data_dicts()
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)

    R_int = folded_R_triples(*NK)
    Rcart = R_int.astype(float) @ arrays['a_vectors']
    HRs = arrays['HRs']
    nk1, nk2, nk3 = NK
    for i, j, k in [(0, 0, 0), (2, 1, 3), (3, 3, 2)]:
        kfrac = np.array([i / nk1, j / nk2, k / nk3])
        hk, dhk = sph.assemble_hk_dhk(kfrac, sign=-1)

        flat = HRs[..., 0].reshape(NAWF, NAWF, -1)
        phase = np.exp(-2.0j * np.pi * (R_int @ kfrac))
        Hk_ref = np.tensordot(flat, phase, axes=([2], [0]))
        assert np.allclose(hk.toarray(), Hk_ref, atol=1e-12)
        for l in range(3):
            ref = 1j * (
                ALAT * np.tensordot(flat, Rcart[:, l] * phase, axes=([2], [0]))
                + arrays['Dnm'][:, :, l] * Hk_ref
            )
            ref = 0.5 * (ref + ref.conj().T)
            assert np.allclose(dhk[l].toarray(), ref, atol=1e-11)


def test_threshold_drop_report_bounds_eigenvalue_shift():
    rng = np.random.default_rng(23)
    dc, _ = _make_dc(rng)
    exact = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    thr = 0.05 * float(np.abs(dc.data_dicts()[0]['HRs']).max())
    trunc = SparseHamiltonian.from_data_controller(dc, threshold=thr)
    assert trunc.nnz < exact.nnz
    bound = trunc.drop_report['eig_bound']
    assert bound > 0.0
    for _ in range(3):
        kfrac = rng.standard_normal(3)
        e_exact = np.linalg.eigvalsh(exact.assemble_hk(kfrac).toarray())
        e_trunc = np.linalg.eigvalsh(trunc.assemble_hk(kfrac).toarray())
        assert np.abs(e_exact - e_trunc).max() <= bound + 1e-12


def test_nspin_two_channels_assemble_independently():
    rng = np.random.default_rng(29)
    dc, Hks = _make_dc(rng, nspin=2)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    i, j, k = 2, 0, 3
    kfrac = np.array([i / NK[0], j / NK[1], k / NK[2]])
    for s in range(2):
        Hk = sph.assemble_hk(kfrac, ispin=s, sign=-1).toarray()
        assert np.allclose(Hk, Hks[:, :, i, j, k, s], atol=1e-12)


def _reference_hermitize(self):
    """The pre-encoding implementation, kept here as the reference.

    hermitize() replaced this np.unique(..., axis=0) void-view lexsort
    with a single int64 key per bond.  The encoded key orders (row, col,
    R) identically, so the replacement has to be bit-identical -- not
    merely close: the bond order fixes the floating-point summation order
    inside the assembly plan's add.reduceat.
    """
    triples = self.R_int[self.ridx].astype(np.int64)
    mirrored = -triples
    for axis in range(3):
        nk = self.nk_grid[axis]
        if nk % 2 == 0:
            comp = mirrored[:, axis]
            mirrored[:, axis] = np.where(comp == nk // 2, -(nk // 2), comp)
    n = self.nnz
    key = np.empty((2 * n, 5), dtype=np.int64)
    key[:n, 0], key[:n, 1], key[:n, 2:] = self.rows, self.cols, triples
    key[n:, 0], key[n:, 1], key[n:, 2:] = self.cols, self.rows, mirrored
    vals = np.concatenate((0.5 * self.vals, 0.5 * np.conj(self.vals)))
    dnm = np.concatenate((self.dnm, -self.dnm))
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    vals_m = np.zeros((len(uniq), self.nspin), dtype=np.complex128)
    np.add.at(vals_m, inv, vals)
    dnm_m = np.zeros((len(uniq), 3))
    dnm_m[inv] = dnm
    R_uniq, ridx = np.unique(uniq[:, 2:], axis=0, return_inverse=True)
    return uniq[:, 0], uniq[:, 1], R_uniq, ridx.reshape(-1), vals_m, dnm_m


def test_encoded_hermitize_matches_reference():
    from PAOFLOW.sparse.doubling import double_axis

    rng = np.random.default_rng(101)
    dc, _ = _make_dc(rng)
    base = SparseHamiltonian.from_data_controller(dc, threshold=1e-3)
    raw = double_axis(double_axis(base, 0), 2)  # raw doubled list is non-Hermitian

    for sph in (base, raw):
        r, c, R, ri, v, d = _reference_hermitize(sph)
        got = sph.hermitize()
        assert np.array_equal(r.astype(np.int32), got.rows)
        assert np.array_equal(c.astype(np.int32), got.cols)
        assert np.array_equal(R.astype(np.int32), got.R_int)
        assert np.array_equal(ri.astype(np.int32), got.ridx)
        assert (v == got.vals).all(), 'values must be bit-identical, not merely close'
        assert (d == got.dnm).all()


def test_unique_R_matches_axis0_unique():
    from PAOFLOW.sparse.hamiltonian import unique_R

    rng = np.random.default_rng(103)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=1e-3)
    triples = sph.R_int[sph.ridx].astype(np.int64)
    R_ref, inv_ref = np.unique(triples, axis=0, return_inverse=True)
    R_got, inv_got = unique_R(triples, sph.nk_grid)
    assert np.array_equal(R_ref, R_got)
    assert np.array_equal(inv_ref.reshape(-1), inv_got)


def test_hermitize_is_noop_without_a_nyquist_asymmetry():
    """On an already-Hermitian bond list hermitize() must not move a bit."""
    rng = np.random.default_rng(107)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=1e-3)
    assert sph.hermiticity_error() < 1e-12
    once = sph.hermitize()
    twice = once.hermitize()
    assert np.array_equal(once.rows, twice.rows)
    assert np.array_equal(once.ridx, twice.ridx)
    assert np.abs(once.vals - twice.vals).max() < 1e-15


def test_compact_frees_bonds_and_preserves_assembly():
    rng = np.random.default_rng(109)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=1e-3)
    kfrac = rng.standard_normal(3)
    hk_before = sph.assemble_hk(kfrac).toarray()
    dhk_before = [m.toarray() for m in sph.assemble_hk_dhk(kfrac)[1]]
    nnz = sph.nnz

    sph.compact()
    assert sph.compacted and sph.nnz == nnz and sph.rows is None
    assert np.array_equal(sph.assemble_hk(kfrac).toarray(), hk_before)
    for a, b in zip(sph.assemble_hk_dhk(kfrac)[1], dhk_before):
        assert np.array_equal(a.toarray(), b)

    # mutation is refused loudly rather than working on stale data
    import pytest as _pytest

    from PAOFLOW.sparse.doubling import double_axis

    for call in (sph.hermitize, sph.hermiticity_error, lambda: double_axis(sph, 0)):
        with _pytest.raises(RuntimeError, match='compact'):
            call()


def test_pattern_reused_across_k():
    rng = np.random.default_rng(31)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    sph.assemble_hk(np.zeros(3))
    plan_before = sph.plan
    sph.assemble_hk(rng.standard_normal(3))
    assert sph.plan is plan_before


def test_project_doubling_bond_count_is_exact():
    """The projection must match what doubling actually produces, since a
    pre-flight gate that mis-sizes is worse than none."""
    from PAOFLOW.sparse.doubling import double_axis

    rng = np.random.default_rng(77)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)

    for nx, ny, nz in ((1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)):
        proj = sph.project_doubling(nx, ny, nz)
        assert proj['N'] == 2 ** (nx + ny + nz)
        assert proj['nawf'] == sph.nawf * proj['N']

        h = sph
        for axis, reps in ((0, nx), (1, ny), (2, nz)):
            for _ in range(reps):
                h = double_axis(h, axis)
        # exact: doubling replicates every bond twice per step
        assert h.nnz == proj['nnz']
        assert proj['dense_hk_bytes'] == 16 * proj['nawf'] ** 2


def test_project_doubling_steady_brackets_hermitized_size():
    """hermitize() unions the list with its mirror, so the real steady count
    lies in [nnz, 2*nnz] -- the projection must not sit below it."""
    from PAOFLOW.sparse.doubling import double_axis

    rng = np.random.default_rng(78)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.3)  # asymmetric drops

    proj = sph.project_doubling(1, 1, 1)
    h = sph
    for axis in (0, 1, 2):
        h = double_axis(h, axis)
    assert h.nnz == proj['nnz']
    final = h.hermitize().nnz
    assert proj['nnz'] <= final <= 2 * proj['nnz']

    _, plan_b, peak_b = sph.bytes_per_bond()
    assert peak_b > plan_b  # the transient, not the resting state, gates a run
    assert proj['peak_bytes'] == proj['nnz'] * peak_b


def test_project_doubling_zero_is_identity():
    rng = np.random.default_rng(79)
    dc, _ = _make_dc(rng)
    sph = SparseHamiltonian.from_data_controller(dc, threshold=0.0)
    proj = sph.project_doubling(0, 0, 0)
    assert proj['N'] == 1 and proj['nnz'] == sph.nnz and proj['nawf'] == sph.nawf
