"""Methods ported from dense-only in November 2026, against the dense pipeline.

``doped_transport``, ``kubo_response``, ``kubo_optical_response`` and ``arpes`` on Si
(``tests/data/silicon``); ``magnetic_field``, ``orbital_moment``, ``orbital_magnetization`` and the
spin-current Kubo response on spin-orbit Fe (example04, skipped when absent).

The Kubo comparisons give both codes the same explicit ``bands``, below the placeholder states at
the PAO shift: by default dense sums ``bnd`` bands at every k-point, placeholders included (their
basis is the eigensolver's), and sparse stops at the well-projected top.  Measured with the same
bands: 1e-12 relative on Fe, 1e-9 on Si; with the defaults the two differ by up to 0.8% (Fe).
"""

import os

import numpy as np
import pytest

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'silicon'))
FE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example04'
    )
)
needs_fe = pytest.mark.skipif(
    not os.path.isdir(os.path.join(FE, 'fe.save')), reason='example04 QE data not available'
)


def _driver(folder, savedir, outdir, sparse, **projectability):
    from PAOFLOW.PAOFLOW import PAOFLOW

    cwd = os.getcwd()
    os.chdir(folder)
    try:
        p = PAOFLOW(
            savedir=savedir,
            outputdir=outdir,
            smearing='gauss',
            npool=1,
            verbose=False,
            sparse=sparse,
            sparse_config={'hopping_threshold': 0.0} if sparse else None,
        )
        p.read_atomic_proj_QE()
        p.projectability(**projectability)
        p.pao_hamiltonian()
    finally:
        os.chdir(cwd)
    return p


def _mesh(dense, sparse):
    dense.pao_eigh()
    dense.gradient_and_momenta()
    dense.adaptive_smearing()
    sparse.adaptive_smearing()


def _rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(a)).max()


@pytest.fixture(scope='module')
def si(tmp_path_factory):
    out = tmp_path_factory.mktemp('si')
    dense = _driver(DATA, 'silicon.save', str(out / 'dense'), False)
    sparse = _driver(DATA, 'silicon.save', str(out / 'sparse'), True)
    return dense, sparse


@pytest.fixture(scope='module')
def si_mesh(si):
    _mesh(*si)
    return si


def test_arpes_runs_on_the_base_cell_and_keeps_the_bond_list(si):
    from PAOFLOW.arpes import ArpesExperiment

    dense, sparse = si
    experiment = ArpesExperiment(
        photon_ev=60.0, energy=(-6.0, 0.5, 40), kx=(-0.5, 0.5, 5), ky=(0.0, 0.0, 1),
        resolution=(0.05, 0.02),
    )  # fmt: skip
    H = sparse.sparse.H
    reference = dense.arpes(experiment).intensity
    assert np.abs(sparse.arpes(experiment).intensity - reference).max() < 1e-12
    assert sparse.sparse.H is H and 'HRs' not in sparse.data_controller.data_arrays


def test_magnetic_field_refuses_a_spin_degenerate_basis(si):
    with pytest.raises(NotImplementedError, match='(?s)spinor_basis.*dense-only'):
        si[1].magnetic_field([0.0, 0.0, 1.0])


def test_doped_transport(si_mesh):
    dense, sparse = si_mesh
    kwargs = dict(tmin=300.0, tmax=400.0, nt=2, ne=500)
    rows = sparse.doped_transport(-1.0e18, **kwargs)
    # measured 4e-9 (the degenerate velocities are paired differently, not their squares)
    assert _rel(dense.doped_transport(-1.0e18, **kwargs), rows) < 1e-6
    with pytest.raises(RuntimeError, match='cannot be queued'):
        with sparse.sparse.fused():
            sparse.doped_transport(-1.0e18, **kwargs)


def test_kubo_responses(si_mesh):
    dense, sparse = si_mesh
    # six bands: every one a real state at every k-point of this mesh
    bastin = dict(
        components=['xx'], terms=('total', 'surface', 'sea', 'intrinsic'), eta=0.05,
        emin=-3.0, emax=3.0, ne=21, bands=6, write=False,
    )  # fmt: skip
    optical = dict(components=['xx'], omega_max=8.0, nomega=41, bands=6, write=False)
    a, b = dense.kubo_response(**bastin)['xx'], sparse.kubo_response(**bastin)['xx']
    assert np.array_equal(a['mu'], b['mu'])
    assert _rel(a['total'], b['total']) < 1e-7  # measured 6e-10
    a = dense.kubo_optical_response(**optical)['xx']
    b = sparse.kubo_optical_response(**optical)['xx']
    assert _rel(a['total'], b['total']) < 1e-4  # measured 2e-6 (widths inside degenerate groups)
    with pytest.raises(NotImplementedError, match='give omega_max'):
        sparse.kubo_optical_response(components=['xx'])


@pytest.fixture(scope='module')
def fe(tmp_path_factory):
    out = tmp_path_factory.mktemp('fe')
    dense = _driver(FE, 'fe.save', str(out / 'dense'), False, pthr=0.95)
    sparse = _driver(FE, 'fe.save', str(out / 'sparse'), True, pthr=0.95)
    return dense, sparse


@needs_fe
def test_magnetic_field_matches_dense(fe):
    dense, sparse = fe
    field = [0.3, 0.0, 10.0]
    term = sparse.magnetic_field(field)
    assert np.abs(dense.magnetic_field(field) - term).max() < 1e-14
    arrays = sparse.data_controller.data_arrays
    assert 'HRs' not in arrays and isinstance(arrays['Sj'], list)
    assert (
        np.abs(sparse.sparse.H.to_dense_HRs() - dense.data_controller.data_arrays['HRs']).max()
        < 1e-12
    )
    # the field of the fixture stays in both runs; the dense mesh ignores it (stale Hks)
    sparse.magnetic_field([-x for x in field])
    dense.magnetic_field([-x for x in field])


@needs_fe
def test_orbital_moment_and_kubo_spin_current(fe):
    dense, sparse = fe
    dense.spin_operator()
    sparse.spin_operator()
    _mesh(dense, sparse)
    magnetization = dict(emin=-2.0, emax=2.0, ne=21)
    kubo = dict(
        observable='spin_current', components=['zxy'], terms=('total', 'intrinsic'), eta=0.05,
        emin=-2.0, emax=2.0, ne=21, bands=16, write=False,
    )  # fmt: skip
    dense.orbital_moment()
    with sparse.sparse.fused():
        sparse.orbital_moment()
        sparse.kubo_response(**kubo)
    d_arrays, s_arrays = dense.data_controller.data_arrays, sparse.data_controller.data_arrays
    real = s_arrays['E_k'][:, :, 0] < 100.0  # not the padding of the dropped placeholders
    nb = real.shape[1]
    for key in ('orbital_moment', 'berry_curvature_vector'):
        a, b = d_arrays[key][:, :nb, :, 0][real], s_arrays[key][:, :, :, 0][real]
        assert np.abs(a - b).max() < 1e-6 * np.abs(a).max(), key  # measured 1e-11
    _, a = dense.orbital_magnetization(**magnetization)
    _, b = sparse.orbital_magnetization(**magnetization)
    assert np.abs(a - b).max() < 1e-8  # measured 1e-12; |M_z| reaches 0.02 mu_B
    a, b = dense.kubo_response(**kubo)['z_xy'], sparse.kubo_response(**kubo)['z_xy']
    for term in ('total', 'intrinsic'):
        assert _rel(a[term], b[term]) < 1e-7, term  # measured 5e-12


def test_position_dipoles_reach_the_consumers_only(tmp_path):
    """The mesh pass hands its consumers ``dH/dk - i[H, d]`` and keeps the bare band velocities."""
    from PAOFLOW.sparse.properties import MeshProperty

    class Recorder(MeshProperty):
        method = 'recorder'

        def __init__(self, engine):
            super().__init__(engine)
            self.seen = {}

        def on_k(self, kp):
            if kp.ik in (5, 700):
                dhk = [kp.dhk[l] for l in range(3)]
                dhk = [a.toarray() if hasattr(a, 'toarray') else a for a in dhk]
                self.seen[kp.ik] = (kp.hk.toarray(), np.array(dhk))

    sparse = _driver(DATA, 'silicon.save', str(tmp_path / 'sparse'), True)
    engine = sparse.sparse
    bare = Recorder(engine)
    engine._execute([bare])
    velkp = sparse.data_controller.data_arrays['velkp'].copy()

    rng = np.random.default_rng(0)
    dipole = rng.standard_normal((3, 18, 18)) + 1j * rng.standard_normal((3, 18, 18))
    dipole = dipole + dipole.conj().transpose(0, 2, 1)
    engine._mesh_plan['dipoles'] = ({}, True)
    engine._dipoles_at = lambda kfrac, params: dipole
    dressed = Recorder(engine)
    engine._execute([dressed])
    assert np.array_equal(sparse.data_controller.data_arrays['velkp'], velkp)
    for ik, (hk, dhk) in dressed.seen.items():
        term = 1j * (hk @ dipole - dipole @ hk)
        assert np.abs(dhk - (bare.seen[ik][1] - term)).max() < 1e-12


def test_position_dipole_term_matches_dense(tmp_path, monkeypatch):
    """``dH/dk`` of two k-points with the dipoles of the real Si tables against the dense
    ``dHksp`` after ``position_dipole_correction`` (measured 5e-14; the term itself is 5-10 eV
    bohr).  The sparse run takes the tables of the dense one: building them is most of the time."""
    from PAOFLOW.sparse.mesh import _with_dipole_velocity
    from PAOFLOW.utils.get_K_grid_fft import get_K_grid_fft_crystal

    monkeypatch.chdir(DATA)  # the UPF files are read relative to the save folder
    tables = dict(level='intra-atomic', q_max=8.0, n_q=60)
    dense = _driver(DATA, 'silicon.save', str(tmp_path / 'dense'), False)
    dense.pao_eigh()
    dense.gradient_and_momenta(position_dipoles=False)
    assert dense.position_dipole_correction(**tables)
    sparse = _driver(DATA, 'silicon.save', str(tmp_path / 'sparse'), True)
    d_arrays, s_arrays = dense.data_controller.data_arrays, sparse.data_controller.data_arrays
    for key in ('_NL_pao_catalog', '_PD_tables', '_PD_on_site'):
        s_arrays[key] = d_arrays[key]
    assert sparse.position_dipole_correction(**tables)
    engine = sparse.sparse
    kfrac = get_K_grid_fft_crystal(12, 12, 12)
    for ik in (5, 700):
        hk, dhk = engine.H.assemble_derivatives(kfrac[ik], ispin=0, sign=-1, order=1)[:2]
        dipole = engine._dipoles_at(kfrac[ik], engine._mesh_plan['dipoles'][0])
        got = np.array(_with_dipole_velocity(hk, dhk, dipole))
        assert np.abs(got - d_arrays['dHksp'][ik, :, :, :, 0]).max() < 1e-10
