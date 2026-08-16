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


@pytest.fixture(scope='module')
def dense_and_sparse(tmp_path_factory):
    from PAOFLOW.PAOFLOW import PAOFLOW
    from PAOFLOW.SparsePAOFLOW import SparsePAOFLOW

    out = str(tmp_path_factory.mktemp('mesh_parity'))
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

        q = SparsePAOFLOW(
            savedir='silicon.save',
            outputdir=os.path.join(out, 'sparse'),
            smearing='gauss',
            npool=1,
            verbose=False,
            threshold=0.0,
        )
        q.read_atomic_proj_QE()
        q.projectability()
        q.pao_hamiltonian()
        q.interpolated_hamiltonian(nfft1=12, nfft2=12, nfft3=12)
        q.pao_eigh()
        q.gradient_and_momenta()
        q.adaptive_smearing()
        q._ensure_mesh()
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
