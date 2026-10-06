"""Fused-mesh parity with the dense pipeline on the example01 base cell.

Runs the *real* dense pipeline (no doubling, nawf=18, cheap) —
``interpolated_hamiltonian(12,12,12)`` -> ``pao_eigh`` ->
``gradient_and_momenta`` -> ``adaptive_smearing`` — and compares
``E_k``/``velkp``/``deltakp`` index-wise against the sparse fused mesh at
threshold 0.  This is the strongest guard against phase-sign and
convention errors: an integrated observable can hide a k -> -k pairing
mistake, an index-wise comparison cannot.

k-points where the band at the ``bnd`` truncation boundary is degenerate
with the next one are excluded from the velocity comparison (the diagonal
there is gauge-dependent in both codes).

Requires the example01 QE data; skipped when absent.
"""

import os

import numpy as np
import pytest

EXAMPLE = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example01'
)
EXAMPLE = os.path.abspath(EXAMPLE)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(EXAMPLE, 'silicon.save')),
    reason='example01 QE data not available',
)


@pytest.fixture(
    scope='module',
    params=['sparse', 'dense'],
    ids=['hk_solver=sparse', 'hk_solver=dense'],
)
def dense_and_sparse(request, tmp_path_factory):
    """Parity must hold under *both* per-k kernels.

    The dispatch is convention-neutral by construction, and this is the
    cheapest proof of it: the same index-wise comparison against the real
    dense pipeline, once per kernel.  Note that at the base cell
    (nawf=18, bnd=9) 'auto' selects dense, so without forcing the kernel
    the 'sparse' path would no longer be covered here at all.
    """
    from PAOFLOW.PAOFLOW import PAOFLOW

    hk_solver = request.param
    out = str(tmp_path_factory.mktemp('mesh_parity_' + hk_solver))
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        p = PAOFLOW(
            savedir='silicon.save',
            outputdir=os.path.join(out, 'dense'),
            smearing='gauss',
            npool=1,
            verbose=False,
        )
        p.read_atomic_proj_QE()
        p.projectability()
        p.pao_hamiltonian()
        p.interpolated_hamiltonian(nfft1=12, nfft2=12, nfft3=12)
        p.pao_eigh()
        p.gradient_and_momenta()
        p.adaptive_smearing()
        d_arrays, d_attr = p.data_controller.data_dicts()

        q = PAOFLOW(
            savedir='silicon.save',
            outputdir=os.path.join(out, 'sparse'),
            smearing='gauss',
            npool=1,
            verbose=False,
            sparse={'threshold': 0.0, 'hk_solver': hk_solver},
        )
        q.read_atomic_proj_QE()
        q.projectability()
        q.pao_hamiltonian()
        q.interpolated_hamiltonian(nfft1=12, nfft2=12, nfft3=12)
        q.pao_eigh()
        q.gradient_and_momenta()
        q.adaptive_smearing()
        q.sparse._ensure_mesh()
        s_arrays, s_attr = q.data_controller.data_dicts()
    finally:
        os.chdir(cwd)
    return d_arrays, d_attr, s_arrays, s_attr


def test_eigenvalues_index_wise(dense_and_sparse):
    d_arrays, d_attr, s_arrays, _ = dense_and_sparse
    bnd = d_attr['bnd']
    dE = d_arrays['E_k'][:, :bnd, 0]
    sE = s_arrays['E_k'][:, :bnd, 0]
    assert dE.shape == sE.shape
    assert np.abs(dE - sE).max() < 1e-8


def _boundary_ok(d_arrays, bnd, gap=1e-4):
    """k-points whose band bnd-1 is NOT degenerate with band bnd."""
    E = d_arrays['E_k'][:, :, 0]
    return (E[:, bnd] - E[:, bnd - 1]) > gap


def _nondegenerate(d_arrays, bnd, gap=1e-6):
    """k-points with no near-exact degeneracy among the first bnd+1 bands.
    At (near-)exact degeneracies the perturb_split rotation is
    floating-point gauge-sensitive in both codes (and ARPACK's random
    start vector re-rolls the gauge every run), so strict index-wise
    parity is only demanded away from them."""
    E = d_arrays['E_k'][:, :, 0]
    return (np.diff(E[:, : bnd + 1], axis=1) > gap).all(axis=1)


def test_velocities_index_wise(dense_and_sparse):
    """Bulk of the mesh must agree to solver precision.  At k-points with
    near-exact internal degeneracies (gaps ~1e-10) the perturb_split
    rotation is floating-point gauge-sensitive in BOTH codes; those
    measure-zero points (about 10 of 1728 here) are only required to stay
    bounded — BZ integration absorbs them."""
    d_arrays, d_attr, s_arrays, _ = dense_and_sparse
    bnd = d_attr['bnd']
    strict = _nondegenerate(d_arrays, bnd)
    bounded = _boundary_ok(d_arrays, bnd) & ~strict
    assert strict.sum() > 100, 'test needs a meaningful strict set'
    # dense band-diagonal velocity: Re pksp[k, l, n, n]
    dv = np.real(np.einsum('klnn->kln', d_arrays['pksp'][:, :, :bnd, :bnd, 0]))
    sv = s_arrays['velkp'][:, :, :bnd, 0]
    err_k = np.abs(dv - sv).max(axis=(1, 2))
    scale = max(np.abs(dv).max(), 1.0)
    assert err_k[strict].max() < 1e-7 * scale, (
        'strict velocity parity failed: %.3e (scale %.3e)' % (err_k[strict].max(), scale)
    )
    assert err_k[bounded].max() < 1e-3, (
        'gauge-sensitive degenerate points exceeded bound: %.3e' % err_k[bounded].max()
    )


def test_adaptive_widths_index_wise(dense_and_sparse):
    d_arrays, d_attr, s_arrays, _ = dense_and_sparse
    bnd = d_attr['bnd']
    strict = _nondegenerate(d_arrays, bnd)
    bounded = _boundary_ok(d_arrays, bnd) & ~strict
    dd = d_arrays['deltakp'][:, :bnd, 0]
    sd = s_arrays['deltakp'][:, :bnd, 0]
    err_k = np.abs(dd - sd).max(axis=1)
    assert err_k[strict].max() < 1e-7 * max(dd.max(), 1.0)
    assert err_k[bounded].max() < 1e-3


def test_no_dense_tensors_in_sparse_run(dense_and_sparse):
    _, _, s_arrays, s_attr = dense_and_sparse
    for name in ('HRs', 'Hksp', 'dHksp', 'pksp', 'v_k', 'deltakp2'):
        assert name not in s_arrays, '%s must never exist in the sparse pipeline' % name


# ----------------------------------------------------------------------
# Band curvature and the Hall tensor (full-spectrum pass)
# ----------------------------------------------------------------------

HALL_ARGS = dict(emin=-5.0, emax=1.0, ne=101, do_hall=True)


def _run_example01(outdir, nfft, sparse=None, band_curvature=True, hall=True):
    from PAOFLOW.PAOFLOW import PAOFLOW

    p = PAOFLOW(
        savedir='silicon.save',
        outputdir=outdir,
        smearing='gauss',
        npool=1,
        verbose=False,
        sparse=sparse,
    )
    p.read_atomic_proj_QE()
    p.projectability()
    p.pao_hamiltonian()
    p.interpolated_hamiltonian(nfft1=nfft, nfft2=nfft, nfft3=nfft)
    p.pao_eigh()
    p.gradient_and_momenta(band_curvature=band_curvature)
    p.adaptive_smearing()
    if hall:
        p.transport(**HALL_ARGS)
        p.effective_mass()
        p.conductivity(cond_tensor=[[0, 0], [0, 1]], emin=-5.0, emax=1.0, ne=101)
        p.fermi_surface()
        p.doping(emin=-5.0, emax=1.0)
    return p


@pytest.fixture(scope='module')
def curvature_runs(tmp_path_factory):
    """Dense and sparse example01 on an interpolated 16^3 mesh, both with
    ``band_curvature=True`` and ``transport(do_hall=True)``.

    Interpolated rather than 12^3 (the DFT grid): on the base grid a bond
    with several Nyquist components averages a quadratic-in-R quantity over
    different images in the two codes (see
    ``test_second_derivatives_match_dense_d2Hd2k_on_interpolated_mesh``).
    Not parametrized over ``hk_solver``: the curvature's interband sum needs
    every state, so the pass always uses the dense kernel."""

    out = str(tmp_path_factory.mktemp('curvature'))
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        d = _run_example01(os.path.join(out, 'dense'), 16)
        s = _run_example01(os.path.join(out, 'sparse'), 16, sparse={'threshold': 0.0})
    finally:
        os.chdir(cwd)
    return d, s


def test_curvature_index_wise(curvature_runs):
    """``d2Ed2k`` is eigenvalue-like (second derivative of a band energy),
    so away from degeneracies it is gauge-invariant and must agree k-point
    by k-point (measured: 6e-12 relative on 4032 of 4096 points).

    The rest are k-points with a degenerate pair straddling the window edge
    ``bnd``.  ``get_degeneracies(E, bnd)`` leaves such a pair unrotated, so
    its velocity diagonal is whatever gauge the eigensolver returned
    (zheevd dense, zheevr here); and that diagonal decides, through
    ``get_degeneracies`` on the rotated velocities, whether bands far below
    with velocity zero by symmetry are treated as one velocity-degenerate
    block.  Their curvature then moves by O(10%) between any two LAPACK
    gauges, in the dense code as much as here, so those points are only
    required to be finite."""
    d, s = curvature_runs
    d_arrays, d_attr = d.data_controller.data_dicts()
    s_arrays, _ = s.data_controller.data_dicts()
    bnd = d_attr['bnd']
    dc = d_arrays['d2Ed2k'][:, :, :bnd, 0]
    sc = s_arrays['d2Ed2k'][:, :, :bnd, 0]
    assert dc.shape == sc.shape
    strict = _nondegenerate(d_arrays, bnd)
    assert strict.sum() > 0.9 * len(strict), 'test needs a meaningful strict set'
    scale = np.abs(dc).max()
    err_k = np.abs(dc - sc).max(axis=(0, 2))
    assert err_k[strict].max() < 1e-9 * scale, (
        'strict curvature parity failed: %.3e (scale %.3e)' % (err_k[strict].max(), scale)
    )
    assert np.isfinite(sc).all()


def _read(path):
    return np.loadtxt(path)


@pytest.mark.parametrize('stem', ['hall_trace_gauss_0', 'hall_gauss_0', 'nernst_gauss_0'])
def test_hall_and_nernst_files(curvature_runs, stem):
    """The Hall/Nernst files are BZ sums of d2Ed2k * v * v over the window
    bands.  The gauge-sensitive points of ``test_curvature_index_wise`` (64
    of 4096 here) enter with weight 1/nk, which leaves a relative difference
    of a few 1e-4 (measured: 2.6e-4 on the Nernst tensor), in the dense
    result as much as in this one."""
    d, s = curvature_runs
    dpath = os.path.join(d.data_controller.data_attributes['opath'], stem + '.dat')
    spath = os.path.join(s.data_controller.data_attributes['opath'], stem + '.dat')
    dd, ss = _read(dpath), _read(spath)
    assert dd.shape == ss.shape
    # one scale for the whole tensor: components that vanish by symmetry are
    # round-off in both codes and have no relative error of their own
    np.testing.assert_array_equal(dd[:, :2], ss[:, :2])
    scale = np.abs(dd[:, 2:]).max()
    rel = np.abs(dd[:, 2:] - ss[:, 2:]).max() / scale
    assert rel < 1e-3, '%s: relative error %.3e' % (stem, rel)


def test_curvature_pass_keeps_the_window(curvature_runs):
    """The pass solves every state but stores only the window."""
    d, s = curvature_runs
    s_arrays, s_attr = s.data_controller.data_dicts()
    bnd = s_attr['bnd']
    assert s_arrays['E_k'].shape[1] == bnd
    assert s_arrays['d2Ed2k'].shape == (6, s_arrays['E_k'].shape[0], bnd, 1)
    for name in ('Hksp', 'dHksp', 'pksp', 'v_k', 'deltakp2', 'd2Hksp'):
        assert name not in s_arrays


@pytest.mark.parametrize(
    'name', ['cond_xx_0.dat', 'cond_xy_0.dat', 'doping_p0.0.dat', 'dosdk_0.dat']
)
def test_band_diagonal_files(curvature_runs, name):
    """Conductivity, doping (and the DOS it writes) are BZ sums of
    band-diagonal quantities over the same window bands in both codes."""
    d, s = curvature_runs
    a = _read(os.path.join(d.data_controller.data_attributes['opath'], name))
    b = _read(os.path.join(s.data_controller.data_attributes['opath'], name))
    assert a.shape == b.shape
    assert np.abs(a - b).max() < 1e-9 * np.abs(a).max(), name


def test_fermi_surface_bands(curvature_runs):
    d, s = curvature_runs
    dpath = d.data_controller.data_attributes['opath']
    spath = s.data_controller.data_attributes['opath']
    names = sorted(
        f for f in os.listdir(dpath) if f.startswith('Fermi_surf_band_') and f.endswith('.npz')
    )
    assert names
    assert names == sorted(
        f for f in os.listdir(spath) if f.startswith('Fermi_surf_band_') and f.endswith('.npz')
    )
    for name in names:
        a = np.load(os.path.join(dpath, name))['nameband']
        b = np.load(os.path.join(spath, name))['nameband']
        assert np.abs(a - b).max() < 1e-10, name


def test_effective_masses_away_from_degeneracies(curvature_runs):
    """Masses invert the curvature, so only the strict k-points of
    ``test_curvature_index_wise`` are compared, to the file's 4 decimals
    scaled by the mass (a nearly flat band amplifies round-off)."""
    d, s = curvature_runs
    d_arrays, d_attr = d.data_controller.data_dicts()
    bnd = d_attr['bnd']
    name = 'effective_masses_0.dat'
    a = np.loadtxt(os.path.join(d_attr['opath'], name), skiprows=2)
    b = np.loadtxt(os.path.join(s.data_controller.data_attributes['opath'], name), skiprows=2)
    assert a.shape == b.shape
    np.testing.assert_array_equal(a[:, :3], b[:, :3])
    strict = np.repeat(_nondegenerate(d_arrays, bnd), bnd)
    rel = np.abs(a[strict, 4:] - b[strict, 4:]) / np.maximum(1.0, np.abs(a[strict, 4:]))
    assert rel.max() < 1e-2
    assert np.median(rel) < 1e-4
