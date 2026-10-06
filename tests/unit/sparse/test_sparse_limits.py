"""The ``dense_n_max`` / ``dense_ratio`` options of ``sparse_config`` reach
the per-k dispatch of every pass.

On the example01 base cell (``nawf=18``, ``bnd=9``) the default dispatch
takes the dense kernel, since ``nev + guard = 13`` is past ``n/8``.  A
lowered ``dense_n_max`` must turn that into the loud refusal, and a raised
``dense_ratio`` must move the same solve onto ARPACK.  Requires the
example01 QE data; skipped when absent.
"""

import os

import pytest

EXAMPLE = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example01'
)
EXAMPLE = os.path.abspath(EXAMPLE)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(EXAMPLE, 'silicon.save')),
    reason='example01 QE data not available',
)


def _sparse_example01(outdir, config):
    from PAOFLOW.PAOFLOW import PAOFLOW

    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        p = PAOFLOW(
            savedir='silicon.save',
            outputdir=outdir,
            smearing='gauss',
            npool=1,
            verbose=False,
            sparse=True,
            sparse_config={'hopping_threshold': 0.0, **config},
        )
        p.read_atomic_proj_QE()
        p.projectability()
        p.pao_hamiltonian()
        p.interpolated_hamiltonian(nfft1=8, nfft2=8, nfft3=8)
    finally:
        os.chdir(cwd)
    return p


def _mesh_solver_line(p):
    with open(os.path.join(p.data_controller.data_attributes['opath'], 'sparse.log')) as f:
        text = f.read()
    mesh = text[text.index('Mesh pass') :]
    return next(line for line in mesh.splitlines() if line.startswith('H(k) solver'))


def test_defaults_take_the_dense_kernel(tmp_path):
    p = _sparse_example01(str(tmp_path / 'default'), {})
    p.sparse._ensure_mesh()
    assert 'dense' in _mesh_solver_line(p)


def test_raised_dense_ratio_moves_the_solve_to_arpack(tmp_path):
    p = _sparse_example01(str(tmp_path / 'ratio'), {'dense_ratio': 0.9})
    p.sparse._ensure_mesh()
    assert 'sparse' in _mesh_solver_line(p)


def test_lowered_dense_n_max_refuses(tmp_path):
    p = _sparse_example01(str(tmp_path / 'cap'), {'dense_n_max': 10})
    with pytest.raises(NotImplementedError, match='dense_n_max = 10'):
        p.sparse._ensure_mesh()


def test_lowered_dense_n_max_refuses_the_full_spectrum(tmp_path):
    p = _sparse_example01(str(tmp_path / 'full'), {'dense_n_max': 10, 'hk_solver': 'sparse'})
    p.gradient_and_momenta(band_curvature=True)
    with pytest.raises(NotImplementedError, match='dense_n_max = 10'):
        p.sparse._ensure_mesh()
