"""Engine parity from one bond-list archive, on the example01 base cell.

A dense run truncates its ``HRs`` with ``save_sparse_hamiltonian``; the
archive is then restarted twice, once on the dense driver (FFT + LAPACK on
the densified ``HRs``) and once on ``SparsePAOFLOW`` (per-k assembly from
the bonds).  Both see the *same* truncated model, so their eigenvalues
must agree to solver precision: this separates the engine from the
truncation, which ``test_sparse_mesh_parity`` cannot (it runs at
threshold 0).

Also covers ``SparsePAOFLOW.to_dense()``: the handed-off driver must be
the dense one and must reproduce the dense restart.

Requires the example01 QE data; skipped when absent.
"""

import os

import numpy as np
import pytest

EXAMPLE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example01'
    )
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(EXAMPLE, 'silicon.save')),
    reason='example01 QE data not available',
)

NFFT = 24  # example01 is on 12^3, so this exercises zero-padding vs bond assembly
THRESHOLD = 1e-3
BOND_ORDER = 3


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    from PAOFLOW.PAOFLOW import PAOFLOW
    from PAOFLOW.SparsePAOFLOW import SparsePAOFLOW

    out = str(tmp_path_factory.mktemp('bridge_parity'))
    archive = os.path.join(out, 'save', 'sparse_hamiltonian.npz')
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        saver = PAOFLOW(
            savedir='silicon.save', outputdir=os.path.join(out, 'save'), smearing='gauss'
        )
        saver.read_atomic_proj_QE()
        saver.projectability()
        saver.pao_hamiltonian()
        saver.save_sparse_hamiltonian(
            'sparse_hamiltonian.npz', threshold=THRESHOLD, bond_order=BOND_ORDER
        )

        dense = PAOFLOW(workpath=out, outputdir='dense', restart=True, smearing='gauss')
        dense.load_sparse_hamiltonian(archive)
        loaded_HRs = dense.data_controller.data_arrays['HRs'].copy()
        dense.interpolated_hamiltonian(NFFT, NFFT, NFFT)
        dense.pao_eigh()
        d_arrays, d_attr = dense.data_controller.data_dicts()

        sparse = SparsePAOFLOW(
            workpath=out, outputdir='sparse', restart=True, smearing='gauss', hk_solver='dense'
        )
        sparse.load_sparse_hamiltonian(archive)
        H = sparse.H
        sparse.interpolated_hamiltonian(NFFT, NFFT, NFFT)
        sparse.pao_eigh()
        sparse._ensure_mesh()
        s_arrays = sparse.data_controller.data_dicts()[0]

        handoff = SparsePAOFLOW(workpath=out, outputdir='handoff', restart=True, smearing='gauss')
        handoff.load_sparse_hamiltonian(archive)
        handed = handoff.to_dense()

        # the path the notebook takes: a SparsePAOFLOW-written archive (whose
        # controller popped Dnm) restarted densely, through the gradient
        writer = SparsePAOFLOW(
            savedir='silicon.save',
            outputdir=os.path.join(out, 'swrite'),
            smearing='gauss',
            threshold=THRESHOLD,
            bond_order=BOND_ORDER,
        )
        writer.read_atomic_proj_QE()
        writer.projectability()
        writer.pao_hamiltonian()
        writer.save_sparse_hamiltonian('sparse_hamiltonian.npz')
        from_sparse = PAOFLOW(workpath=out, outputdir='from_sparse', restart=True, smearing='gauss')
        from_sparse.load_sparse_hamiltonian(os.path.join(out, 'swrite', 'sparse_hamiltonian.npz'))
        from_sparse.interpolated_hamiltonian(NFFT, NFFT, NFFT)
        from_sparse.pao_eigh()
        from_sparse.gradient_and_momenta()
        f_arrays = from_sparse.data_controller.data_dicts()[0]

        late = SparsePAOFLOW(workpath=out, outputdir='late', restart=True, smearing='gauss')
        late.load_sparse_hamiltonian(archive)
        late.interpolated_hamiltonian(2 * NFFT, 2 * NFFT, 2 * NFFT)
        handed.interpolated_hamiltonian(NFFT, NFFT, NFFT)
        handed.pao_eigh()
        h_arrays = handed.data_controller.data_dicts()[0]
    finally:
        os.chdir(cwd)
    return dict(
        H=H,
        loaded_HRs=loaded_HRs,
        d_arrays=d_arrays,
        d_attr=d_attr,
        s_arrays=s_arrays,
        handoff=handoff,
        late=late,
        f_arrays=f_arrays,
        handed=handed,
        h_arrays=h_arrays,
    )


def test_dense_restart_holds_the_archived_model(runs):
    np.testing.assert_array_equal(runs['loaded_HRs'], runs['H'].to_dense_HRs())


def test_dense_restart_ran_the_dense_engine(runs):
    d_arrays, d_attr = runs['d_arrays'], runs['d_attr']
    assert 'Hksp' in d_arrays
    nk_local = d_arrays['Hksp'].shape[0]
    assert d_arrays['v_k'].shape == (nk_local, d_attr['nawf'], d_attr['nawf'], d_attr['nspin'])


def test_engines_agree_on_one_archive(runs):
    bnd = runs['d_attr']['bnd']
    dE = runs['d_arrays']['E_k'][:, :bnd, 0]
    sE = runs['s_arrays']['E_k'][:, :bnd, 0]
    assert dE.shape == sE.shape
    assert np.abs(dE - sE).max() < 1e-10


def test_to_dense_hands_over_the_dense_driver(runs):
    from PAOFLOW.PAOFLOW import PAOFLOW

    assert type(runs['handed']) is PAOFLOW
    assert runs['handed'].data_controller is runs['handoff'].data_controller
    bnd = runs['d_attr']['bnd']
    np.testing.assert_array_equal(
        runs['h_arrays']['E_k'][:, :bnd], runs['d_arrays']['E_k'][:, :bnd]
    )
    with pytest.raises(RuntimeError, match='to_dense'):
        runs['handoff'].bands(ibrav=2)


def test_to_dense_refuses_after_sparse_interpolation(runs):
    """Sparse interpolation only rewrites nk1..3; a base-grid HRs under a
    finer-mesh attribute set would be read inconsistently by the dense side."""
    with pytest.raises(RuntimeError, match='before interpolated_hamiltonian'):
        runs['late'].to_dense()


def test_sparse_written_archive_runs_the_dense_gradient(runs):
    """Same model whichever driver wrote the archive, and dHksp is formed."""
    bnd = runs['d_attr']['bnd']
    assert 'dHksp' in runs['f_arrays'] or 'pksp' in runs['f_arrays']
    np.testing.assert_array_equal(
        runs['f_arrays']['E_k'][:, :bnd], runs['d_arrays']['E_k'][:, :bnd]
    )
