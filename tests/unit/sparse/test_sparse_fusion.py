"""``pao.sparse.fused()``: several properties, one mesh pass.

The same example01 base-cell properties (DOS, PDOS, Boltzmann transport
with the Hall term, which needs the full-spectrum band curvature) are run
once inside a ``fused()`` block and once as plain calls.  The fused run must
make exactly one pass over the mesh, the unfused one two (PDOS streams in
the first, the Hall curvature needs the second), and every output file must
agree.  A block that raises must run nothing.

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

DOS = dict(emin=-12.0, emax=2.0, ne=200)
TRANSPORT = dict(emin=-5.0, emax=1.0, ne=101, do_hall=True)


def _driver(outdir):
    from PAOFLOW.PAOFLOW import PAOFLOW

    p = PAOFLOW(
        savedir='silicon.save',
        outputdir=outdir,
        smearing='gauss',
        npool=1,
        verbose=False,
        sparse=True,
        sparse_config={'hopping_threshold': 0.0},
    )
    p.read_atomic_proj_QE()
    p.projectability()
    p.pao_hamiltonian()
    return p


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    out = str(tmp_path_factory.mktemp('fusion'))
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        fused = _driver(os.path.join(out, 'fused'))
        with fused.sparse.fused():
            fused.dos(**DOS)
            fused.transport(**TRANSPORT)
            # nothing may run before the block exits
            assert fused.sparse._mesh_passes == 0
        plain = _driver(os.path.join(out, 'plain'))
        plain.dos(**DOS)
        plain.transport(**TRANSPORT)
    finally:
        os.chdir(cwd)
    return fused, plain


def test_fused_block_runs_one_pass(runs):
    fused, plain = runs
    assert fused.sparse._mesh_passes == 1
    assert plain.sparse._mesh_passes == 2
    assert fused.sparse._mesh_products == {'d2Ed2k'}


def test_fused_outputs_equal_unfused(runs):
    fused, plain = runs
    fdir = fused.data_controller.data_attributes['opath']
    pdir = plain.data_controller.data_attributes['opath']
    names = sorted(f for f in os.listdir(pdir) if f.endswith('.dat'))
    assert 'pdosdk_sum_0.dat' in names and 'dosdk_0.dat' in names
    assert 'hall_trace_gauss_0.dat' in names
    assert names == sorted(f for f in os.listdir(fdir) if f.endswith('.dat'))
    for name in names:
        a = np.loadtxt(os.path.join(fdir, name))
        b = np.loadtxt(os.path.join(pdir, name))
        # the passes solve different numbers of states with the same dense
        # kernel, so they agree to round-off, printed to 6 digits
        assert np.allclose(a, b, rtol=1e-5, atol=1e-12 * np.abs(b).max()), name


def test_context_holds_only_what_the_set_up_wrote(runs):
    """A queued property restores its own run data before it runs, and
    nothing else: in particular not arrays a later pass replaced."""
    fused, _ = runs
    arrays, attr = fused.data_controller.data_dicts()
    assert arrays['E_k'].shape[0] == attr['nkpnts']


def test_stale_arrays_are_not_restored(tmp_path):
    """bands() leaves a path E_k; the fused mesh pass replaces it, and the
    queued DOS must see the mesh (it once restored the path E_k)."""
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        p = _driver(str(tmp_path / 'stale'))
        p.bands(ibrav=2, nk=50)
        with p.sparse.fused():
            p.dos(emin=-12.0, emax=2.0, ne=50)
            for queued in p.sparse._fusing:
                assert 'E_k' not in queued._context[1]
                assert 'E_k' not in queued._context[0]
    finally:
        os.chdir(cwd)
    assert (
        p.data_controller.data_arrays['E_k'].shape[0] == p.data_controller.data_attributes['nkpnts']
    )


def test_raising_block_runs_nothing(tmp_path):
    from PAOFLOW.PAOFLOW import PAOFLOW

    p = PAOFLOW(workpath=str(tmp_path), outputdir='s', restart=True, sparse=True)
    p.sparse.H = object()  # past _require_H; nothing may touch it

    class Abort(Exception):
        pass

    with pytest.raises(Abort):
        with p.sparse.fused():
            p.dos(do_pdos=False)
            assert len(p.sparse._fusing) == 1
            raise Abort
    assert p.sparse._mesh_passes == 0
    assert p.sparse._fusing is None


def test_fused_blocks_do_not_nest(tmp_path):
    from PAOFLOW.PAOFLOW import PAOFLOW

    p = PAOFLOW(workpath=str(tmp_path), outputdir='s', restart=True, sparse=True)
    with pytest.raises(RuntimeError, match='nested'):
        with p.sparse.fused():
            with p.sparse.fused():
                pass
    assert p.sparse._fusing is None
