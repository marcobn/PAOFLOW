"""Band velocities of a doubled cell against the primitive cell, state by state.

A cell doubled along a1 and a2 is the same crystal, so its spectrum at a
supercell k-point K is the union of the primitive spectra at the four
folded k-points ``((K1 + i) / 2, (K2 + j) / 2, K3)``, i, j in {0, 1}, and
every state keeps its velocity vector.  Matching each supercell
``(E, v_x, v_y, v_z)`` to a folded primitive one checks two things on the
real example01 Hamiltonian:

- the cross-replica ``dnm`` of ``double_axis``.  With it zeroed, as the
  dense doubling does, velocities are off by tens of percent;
- the degenerate-group rule of ``_band_diagonal_velocities``.  Folding
  makes copies of different primitive states degenerate, and resolving the
  directions separately would mismatch their components.

Requires the example01 QE data; skipped when absent.
"""

import itertools
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


def _sparse_H(tmp_path, nx, ny, nz):
    from PAOFLOW.PAOFLOW import PAOFLOW

    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        p = PAOFLOW(
            savedir='silicon.save',
            outputdir=str(tmp_path / ('d%d%d%d' % (nx, ny, nz))),
            npool=1,
            verbose=False,
            sparse=True,
            # a cutoff inside the R grid: without one, the Nyquist-plane
            # bonds fold differently in the two cells away from the grid
            sparse_config={'bond_order': 36},
        )
        p.read_atomic_proj_QE()
        p.projectability()
        p.pao_hamiltonian()
        if nx + ny + nz:
            p.doubling_Hamiltonian(nx, ny, nz)
    finally:
        os.chdir(cwd)
    return p.sparse.H


def _states(H, k):
    """Rows (E, v_x, v_y, v_z) of every state of H at k."""
    from PAOFLOW.sparse.mesh import _band_diagonal_velocities

    hk, dhk = H.assemble_derivatives(k, sign=-1)
    E, V = np.linalg.eigh(hk.toarray())
    return np.c_[E, _band_diagonal_velocities(E, V, [d.toarray() for d in dhk]).T]


@pytest.fixture(scope='module')
def cells(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('doubled_velocities')
    return _sparse_H(tmp, 0, 0, 0), _sparse_H(tmp, 1, 1, 0)


@pytest.mark.parametrize(
    'K, degenerate',
    [([0.13, 0.31, 0.27], False), ([0.25, 0.5, 0.125], True)],
    ids=['generic', 'degenerate'],
)
def test_doubled_states_match_folded_primitive_states(cells, K, degenerate):
    from PAOFLOW.spectrum.do_eigh import get_degeneracies

    base, doubled = cells
    K = np.array(K)
    folded = np.vstack(
        [
            _states(base, np.array([(K[0] + i) / 2, (K[1] + j) / 2, K[2]]))
            for i, j in itertools.product((0, 1), repeat=2)
        ]
    )
    states = _states(doubled, K)
    assert np.abs(np.sort(states[:, 0]) - np.sort(folded[:, 0])).max() < 1e-10

    # Degenerate groups are formed at 5 decimals of E, so states up to ~1e-5 eV
    # apart are mixed: misses of ~1e-6 relative remain.  The per-direction
    # rule misses by up to 5e-3 relative here, zeroed cross-replica dnm by ~0.2.
    scale = np.abs(folded[:, 1:]).max()
    miss = max(np.abs(folded - row).max(axis=1).min() for row in states)
    assert miss < 1e-5 * scale, 'worst state miss %.3e (scale %.3e)' % (miss, scale)
    if degenerate:
        groups = get_degeneracies(states[None, :, :1], len(states))[0][0]
        assert len(groups) > 0, 'this K must exercise a degenerate group'
