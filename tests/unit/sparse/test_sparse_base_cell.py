"""``@sparse_base_cell``: dense base-cell transforms on the bond list.

A sparse run scatters its base-cell bond list into a dense ``HRs``, runs
the dense method body, and converts the result back with the same
truncation.  At threshold 0 the round trip is exact, so the bond list
afterwards must be the dense pipeline's transformed ``HRs`` element for
element, and a writer must produce the dense file.

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


def _base_cell(outdir, sparse):
    from PAOFLOW.PAOFLOW import PAOFLOW

    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        p = PAOFLOW(
            savedir='silicon.save',
            outputdir=outdir,
            smearing='gauss',
            verbose=False,
            sparse=sparse,
            sparse_config={'threshold': 0.0} if sparse else None,
        )
        p.read_atomic_proj_QE()
        p.projectability()
        p.pao_hamiltonian()
    finally:
        os.chdir(cwd)
    return p


@pytest.mark.parametrize(
    'transform',
    [
        ('add_external_fields', dict(Efield=[0.0, 0.0, 0.05])),
        ('cutting_Hamiltonian', dict(z=True)),
    ],
    ids=['add_external_fields', 'cutting_Hamiltonian'],
)
def test_transform_matches_dense(tmp_path, transform):
    name, kwargs = transform
    dense = _base_cell(str(tmp_path / 'dense'), sparse=False)
    sparse = _base_cell(str(tmp_path / 'sparse'), sparse=True)
    getattr(dense, name)(**kwargs)
    getattr(sparse, name)(**kwargs)

    HRs = dense.data_controller.data_arrays['HRs']
    s_arrays, s_attr = sparse.data_controller.data_dicts()
    assert 'HRs' not in s_arrays and 'Hks' not in s_arrays
    assert sparse.sparse.H.nk_grid == HRs.shape[2:5]
    assert (s_attr['nk1'], s_attr['nk2'], s_attr['nk3']) == HRs.shape[2:5]
    assert np.abs(sparse.sparse.H.to_dense_HRs() - HRs).max() < 1e-12


def test_writer_keeps_the_bond_list(tmp_path):
    dense = _base_cell(str(tmp_path / 'dense'), sparse=False)
    sparse = _base_cell(str(tmp_path / 'sparse'), sparse=True)
    H = sparse.sparse.H
    dense.write_Hamiltonian('z2pack_hr.dat')
    sparse.write_Hamiltonian('z2pack_hr.dat')
    assert sparse.sparse.H is H
    assert 'HRs' not in sparse.data_controller.data_arrays
    # the same numbers; the round trip through the bond list only flips the
    # sign of some exact zeros (-0.0), which the text format shows
    with (
        open(tmp_path / 'dense' / 'z2pack_hr.dat') as fd,
        open(tmp_path / 'sparse' / 'z2pack_hr.dat') as fs,
    ):
        dense_lines, sparse_lines = fd.read().split('\n'), fs.read().split('\n')
    assert len(dense_lines) == len(sparse_lines)
    assert dense_lines[0] == sparse_lines[0]
    a = np.array([float(x) for line in dense_lines[1:] for x in line.split()])
    b = np.array([float(x) for line in sparse_lines[1:] for x in line.split()])
    assert a.shape == b.shape
    assert np.abs(a - b).max() < 1e-12


def test_transform_refuses_after_interpolation(tmp_path):
    """After interpolation the mesh no longer matches the base-cell H(R)."""
    sparse = _base_cell(str(tmp_path / 'sparse'), sparse=True)
    sparse.interpolated_hamiltonian(nfft1=4, nfft2=4, nfft3=4)
    with pytest.raises(RuntimeError, match='before doubling_Hamiltonian'):
        sparse.add_external_fields(Efield=[0.0, 0.0, 0.05])
