"""Interband properties against the dense pipeline, on spin-orbit Fe (example04).

The interband kernels (textures, the Hall family, the dielectric tensor,
Rashba--Edelstein, linear response) run per k-point on eigenvectors, and
eigenvectors carry a gauge.  Away from degeneracies every quantity here is
gauge-invariant and must agree with dense to round-off; at degeneracies
the dense conventions themselves are gauge-sensitive, so the checks come
in two kinds:

- *mechanism*: the sparse per-k path (sparse-assembled ``dH/dk``, the same
  per-k kernel) fed the dense eigenvectors reproduces the dense per-k
  result to round-off, or the comparison excludes the k-points where a
  degenerate group straddles the window edge, the only gauge zone that
  remains (measured 3e-13 for the dielectric sums);
- *integrated*: the written files agree to a tolerance that bounds the
  dense convention's own gauge noise, with the measured value quoted.

Two dense conventions are worth knowing.  Bands above ``bnd`` are the
non-projectable states, shifted to one energy and therefore exactly
degenerate at every k; ``get_degeneracies(E, bnd)`` never rotates them, so
whatever enters through them (their velocities, hence adaptive widths up
to 1.6 eV, and their in-block Berry pairs, whose denominator is only
``delta**2``) is eigensolver-gauge noise.  Kernels that sum over all
``nawf`` bands inherit it; the sparse window kernels sum over ``bnd`` only.

Requires the example04 QE data; skipped when absent.
"""

import os

import numpy as np
import pytest

EXAMPLE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example04'
    )
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(EXAMPLE, 'fe.save')),
    reason='example04 QE data not available',
)

HALL = dict(do_ac=True, emin=-2.0, emax=2.0, ne=101)


def _properties(p):
    p.spin_texture(fermi_up=1.0, fermi_dw=-1.0)
    p.anomalous_Hall(a_tensor=[[0, 1]], **HALL)
    p.spin_Hall(s_tensor=[[0, 1, 2]], **HALL)
    p.dielectric_tensor(d_tensor='diag', ne=101)
    p.rashba_edelstein(ne=101, intra_band=True, ree_tensor=[[0, 1]])
    p.linear_response(response='shc', s_tensor=[[2, 0, 1]], a_tensor=[[0, 1]], esize=101)
    p.linear_response(
        response='cond', intraband=True, s_tensor=[[2, 0, 1]], a_tensor=[[0, 1]], esize=101
    )


def _driver(outdir, sparse=None):
    from PAOFLOW.PAOFLOW import PAOFLOW

    p = PAOFLOW(
        savedir='fe.save', outputdir=outdir, smearing='gauss', npool=1, verbose=False, sparse=sparse
    )
    p.read_atomic_proj_QE()
    p.projectability(pthr=0.95)
    p.pao_hamiltonian()
    p.spin_operator()
    return p


@pytest.fixture(scope='module')
def fe(tmp_path_factory):
    """Dense and sparse Fe at the DFT mesh; the sparse properties share one
    fused pass (full spectrum for the Hall and linear-response kernels)."""
    from PAOFLOW.sparse import SparseConfig

    out = str(tmp_path_factory.mktemp('fe'))
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        d = _driver(os.path.join(out, 'dense'))
        d.pao_eigh()
        d.gradient_and_momenta()
        d.adaptive_smearing()
        _properties(d)

        s = _driver(os.path.join(out, 'sparse'), sparse=SparseConfig(threshold=0.0))
        s.adaptive_smearing()
        with s.sparse.fused():
            _properties(s)
    finally:
        os.chdir(cwd)
    return d, s


def _files(p, name):
    return np.loadtxt(os.path.join(p.data_controller.data_attributes['opath'], name), comments='#')


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(a).max()


def test_one_fused_pass(fe):
    _, s = fe
    assert s.sparse._mesh_passes == 1


def test_spin_texture_away_from_degeneracies(fe):
    d, s = fe
    E = d.data_controller.data_arrays['E_k'][:, :, 0]
    ind = d.data_controller.data_arrays['ind_plot']
    assert ind == s.data_controller.data_arrays['ind_plot']
    for ib, band in enumerate(ind):
        name = 'spin_text_band_%d.npz' % ib
        a = np.load(os.path.join(d.data_controller.data_attributes['opath'], name))[
            'spinband'
        ].reshape(-1, 3)
        b = np.load(os.path.join(s.data_controller.data_attributes['opath'], name))[
            'spinband'
        ].reshape(-1, 3)
        gap = np.min(np.abs(np.delete(E, band, axis=1) - E[:, band : band + 1]), axis=1)
        clean = gap > 1e-5
        assert clean.sum() > 0.9 * len(clean)
        assert np.abs(a - b)[clean].max() < 1e-9, name


def test_berry_curvature_per_k_mechanism(fe):
    """Same eigenvectors, sparse-assembled dH/dk: the per-k Berry curvature of
    the anomalous Hall kernel equals the dense one."""
    from PAOFLOW.response.do_Hall import berry_curvature_k
    from PAOFLOW.utils.get_K_grid_fft import get_K_grid_fft_crystal
    from PAOFLOW.utils.perturb_split import perturb_split

    d, s = fe
    a, attr = d.data_controller.data_dicts()
    kf = get_K_grid_fft_crystal(attr['nk1'], attr['nk2'], attr['nk3'])
    for k in np.linspace(0, len(kf) - 1, 25).astype(int):
        V, E, dg = a['v_k'][k, :, :, 0], a['E_k'][k, :, 0], a['degen'][0][k]
        jd, pd = perturb_split(a['dHksp'][k, 0, :, :, 0], a['dHksp'][k, 1, :, :, 0], V, dg)
        _, dhk = s.sparse.H.assemble_hk_dhk(kf[k], sign=-1)
        js, ps = perturb_split(dhk[0], dhk[1], V, dg)
        ref = berry_curvature_k(E, jd, pd, attr['deltaH'])
        got = berry_curvature_k(E, js, ps, attr['deltaH'])
        assert np.abs(ref - got).max() < 1e-8 * max(1.0, np.abs(ref).max()), k


@pytest.mark.parametrize(
    'name, tol',
    [
        ('ahcEf_xy.dat', 2e-3),  # measured 6.5e-4
        ('shcEf_z_xy.dat', 2e-3),  # measured 4.2e-4
        ('MCDr_xy.dat', 5e-2),  # measured 8.1e-3: AC widths of the shifted block
        ('SCDr_z_xy.dat', 5e-2),  # measured 1.2e-2
    ],
)
def test_hall_files(fe, name, tol):
    """Integrated over the BZ; the remaining difference is the gauge of the
    non-projectable block (module docstring), in the dense result too."""
    d, s = fe
    a, b = _files(d, name), _files(s, name)
    assert a.shape == b.shape
    np.testing.assert_array_equal(a[:, 0], b[:, 0])
    assert _rel(a[:, 1], b[:, 1]) < tol


def test_dielectric_sums_away_from_the_window_edge(fe):
    """Per-k Kubo-Greenwood sums of the window, sparse vs dense, summed over
    the k-points without a degenerate pair at the window edge."""
    from PAOFLOW.response.do_epsilon import eps_accumulate, eps_occupations, eps_settings
    from PAOFLOW.sparse.kpoint import KPoint
    from PAOFLOW.sparse.solver import solve_lowest
    from PAOFLOW.utils.get_K_grid_fft import get_K_grid_fft_crystal

    d, s = fe
    a, attr = d.data_controller.data_dicts()
    bnd = attr['bnd']
    ene = np.linspace(1e-5, 10.0, 51)
    st = eps_settings(attr, ene, True)
    kf = get_K_grid_fft_crystal(attr['nk1'], attr['nk2'], attr['nk3'])
    dk = (8.0 * np.pi**3 / attr['omega'] / len(kf)) ** (1.0 / 3.0)
    Eall = a['E_k'][:, :, 0]
    clean = (Eall[:, bnd] - Eall[:, bnd - 1]) > 1e-4
    dense_sum = np.zeros(ene.size)
    sparse_sum = np.zeros(ene.size)
    for k in np.flatnonzero(clean)[::7]:
        Ek = a['E_k'][k : k + 1, :bnd, 0]
        fn, fnF = eps_occupations(Ek, st)
        P = a['pksp'][k : k + 1, 0, :bnd, :bnd, 0]
        dense_sum += eps_accumulate(
            Ek, fn, fnF, P, P, a['deltakp2'][k : k + 1, :bnd, :bnd, 0], ene, st
        )[0]
        hk, dhk = s.sparse.H.assemble_hk_dhk(kf[k], sign=-1)
        E, V = solve_lowest(hk, bnd, hk_solver='dense')
        kp = KPoint(s.sparse.H, 0, 0, kf[k], E, V, hk, dhk, None, None, bnd, 0.7, dk)
        fn, fnF = eps_occupations(E[None], st)
        sparse_sum += eps_accumulate(
            E[None], fn, fnF, kp.pksp[0][None], kp.pksp[0][None], kp.delta2[None], ene, st
        )[0]
    assert np.abs(dense_sum - sparse_sum).max() < 1e-10 * np.abs(dense_sum).max()


def test_dielectric_files(fe):
    """Integrated, including the 113 edge-degenerate k-points: epsilon_2 to
    a few 1e-9 of its (Drude-dominated) maximum, epsilon_1 to 1e-4."""
    d, s = fe
    for name, tol in (('epsi_xx.dat', 1e-7), ('epsr_xx.dat', 1e-3)):
        a, b = _files(d, name), _files(s, name)
        assert _rel(a[:, 1], b[:, 1]) < tol, name


def test_rashba_edelstein_intra_matches_the_window_truncated_dense_sum(fe):
    """The dense kernel sums the non-projectable bands too, whose gauge-noise
    widths dominate near ``shift``; truncated to the window it is the sparse
    result."""
    from PAOFLOW.response.do_rashba_edelstein import ree_intra_products_k
    from PAOFLOW.utils.constants import BOHR_RADIUS_CM, ELECTRONVOLT_SI, HBAR
    from PAOFLOW.utils.smearing import gaussian

    d, s = fe
    a, attr = d.data_controller.data_dicts()
    bnd = attr['bnd']
    S = a['Sj']
    products = [
        ree_intra_products_k(
            S[1], a['dHksp'][k, 0, :, :, 0], a['v_k'][k, :, :bnd, 0], a['degen'][0][k]
        )
        for k in range(a['E_k'].shape[0])
    ]
    sv = np.array([p[0] for p in products])
    vv = np.array([p[1] for p in products])
    ene = np.linspace(-2.0, 2.0, 101)
    E, delta = a['E_k'][:, :bnd, 0], a['deltakp'][:, :bnd, 0]
    acc = np.array([np.sum(np.real(gaussian(e, E, delta) * sv)) for e in ene])
    jc = np.array([np.sum(np.real(gaussian(e, E, delta) * vv)) for e in ene])
    chi = -HBAR * acc / (jc * ELECTRONVOLT_SI * BOHR_RADIUS_CM + 1e-30)
    got = _files(s, 'spin_reeEf_xy.dat')[:, 1]
    assert np.abs(chi - got).max() < 1e-5 * np.abs(chi).max()


@pytest.mark.parametrize(
    'name, tol',
    [('SHC_EVEN_z_xy.dat', 1e-3), ('Cond_xy.dat', 1e-5)],  # measured 4.4e-5, 5.6e-7
)
def test_linear_response_files(fe, name, tol):
    d, s = fe
    a, b = _files(d, name), _files(s, name)
    assert a.shape == b.shape
    assert _rel(a[:, 1], b[:, 1]) < tol


# ----------------------------------------------------------------------
# Band-path properties
# ----------------------------------------------------------------------


def _pad_HRs(p):
    """Test-only: zero-pad the dense HRs to twice the grid.

    The dense path kernels evaluate the raw base-grid HRs, whose folded
    Nyquist plane carries one sign of R; off the mesh that H(k) is
    non-Hermitian (0.2 eV here) and its bands differ by 47 meV from the
    Hermitian, Nyquist-split H(k) the bond list assembles, which is the
    zero-padded interpolant.  On a padded HRs the two coincide, so this is
    the dense reference with the sparse convention.
    """
    from PAOFLOW.utils.zero_pad import zero_pad

    arr, att = p.data_controller.data_dicts()
    HR = arr['HRs']
    n, nk, nspin = HR.shape[0], HR.shape[2:5], HR.shape[5]
    nf = tuple(2 * x for x in nk)
    HP = np.zeros((n, n) + nf + (nspin,), dtype=complex)
    for s in range(nspin):
        for i in range(n):
            for j in range(n):
                HP[i, j, ..., s] = zero_pad(HR[i, j, ..., s], *nk, *(f - x for f, x in zip(nf, nk)))
    arr['HRs'] = HP
    att['nk1'], att['nk2'], att['nk3'] = nf
    att['nkpnts'] = nf[0] * nf[1] * nf[2]


PATH = [
    ('berry_curvature', dict(spin_Hall=True, orbital_Hall=True, spol=2, ipol=0, jpol=1)),
    ('topology', dict(spol=2, ipol=0, jpol=1, eff_mass=True, Berry=True, spin_Hall=True)),
]


@pytest.fixture(scope='module', params=PATH, ids=[p[0] for p in PATH])
def fe_path(request, tmp_path_factory):
    from PAOFLOW.sparse import SparseConfig

    name, kwargs = request.param
    out = str(tmp_path_factory.mktemp('fe_path_' + name))
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        d = _driver(os.path.join(out, 'dense'))
        _pad_HRs(d)
        d.bands(ibrav=3, nk=100)
        getattr(d, name)(**kwargs)
        s = _driver(os.path.join(out, 'sparse'), sparse=SparseConfig(threshold=0.0))
        s.bands(ibrav=3, nk=100)
        getattr(s, name)(**kwargs)
    finally:
        os.chdir(cwd)
    return name, d, s


def test_path_outputs_away_from_degeneracies(fe_path):
    """Every path file, at the points with no degeneracy among the bands it
    covers: round-off (measured <= 6e-11 of each file's scale).  At the
    degenerate points the per-band curvatures and the 1e-16-regularized
    effective-mass denominators are gauge noise in both codes."""
    name, d, s = fe_path
    dpath = d.data_controller.data_attributes['opath']
    spath = s.data_controller.data_attributes['opath']
    E = np.loadtxt(os.path.join(dpath, 'bands_0.dat'))[:, 1:]
    clean = (np.diff(E, axis=1) > 1e-4).all(axis=1)
    bnd = d.data_controller.data_attributes['bnd']
    window_clean = (np.diff(E[:, : bnd + 1], axis=1) > 1e-4).all(axis=1)
    names = sorted(
        f
        for f in os.listdir(spath)
        if f.endswith('.dat') and f not in ('bands_0.dat', 'kpath_points.txt')
    )
    assert names
    for f in names:
        a = np.loadtxt(os.path.join(dpath, f))
        b = np.loadtxt(os.path.join(spath, f))
        assert a.shape == b.shape, f
        mask = clean if name == 'berry_curvature' else window_clean
        err = np.abs(a - b)
        err_k = err[:, 1:].max(axis=1) if err.ndim == 2 else err
        assert mask.sum() > 40
        assert err_k[mask].max() < 1e-9 * np.abs(a).max(), f
