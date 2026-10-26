"""``sparse.log`` is a key/value file a user can read without the source.

Every line is a section title (with its underline), an indented
``key  value`` field, or a one-line ``WARNING:``; no line explains the
algorithm.  Requires the example01 QE data; skipped when absent.
"""

import os
import re

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

# a key padded to 22 columns and its value, or a longer key, two spaces, value
FIELD = re.compile(r'^  (\S.{21} \S|\S.{22,}?  +\S)')
WARNING = re.compile(r'^  WARNING: \S')
PROSE = ("attr['bnd']", 'Krylov', 'O(nk', 'bond list is simply', 'no new data', 'fused into')


def _log(outdir, config):
    from PAOFLOW.PAOFLOW import PAOFLOW

    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        p = PAOFLOW(
            savedir='silicon.save',
            outputdir=str(outdir),
            npool=1,
            verbose=True,
            sparse=True,
            sparse_config=config,
        )
        p.read_atomic_proj_QE()
        p.projectability(pthr=0.95)
        p.pao_hamiltonian()
        p.interpolated_hamiltonian(nfft1=6, nfft2=6, nfft3=6)
        p.pao_eigh()
        p.gradient_and_momenta()
        with p.sparse.fused():
            p.dos(emin=-12.0, emax=2.0, ne=50)
            p.transport(emin=-1.0, emax=1.0, ne=50)
        p.bands(ibrav=2, nk=10)
    finally:
        os.chdir(cwd)
    with open(os.path.join(p.data_controller.data_attributes['opath'], 'sparse.log')) as f:
        return f.read().splitlines()


@pytest.mark.parametrize(
    'config',
    [{'bond_order': 8}, {'hopping_threshold': 1e-4, 'nev': 12}, {'interior_window': {}}],
    ids=['bond_order', 'threshold+nev', 'interior'],
)
def test_every_line_is_a_title_a_field_or_a_warning(tmp_path, config):
    lines = _log(tmp_path, config)
    assert lines[0] == 'PAOFLOW sparse backend log'
    for i, line in enumerate(lines[1:], start=1):
        if not line:
            continue
        if set(line) == {'-'}:
            assert len(line) == len(lines[i - 1]), f'underline of {lines[i - 1]!r}'
            continue
        if not line.startswith(' '):
            # a section title: followed by its underline
            assert set(lines[i + 1]) == {'-'}, f'line {i} is neither a title nor a field: {line!r}'
            continue
        assert FIELD.match(line) or WARNING.match(line), f'line {i}: {line!r}'
    text = '\n'.join(lines)
    for phrase in PROSE:
        assert phrase not in text
    for title in ('Run', 'Sparse configuration', 'Hamiltonian (base cell)', 'Solve window'):
        assert any(line.startswith(title) for line in lines), title
