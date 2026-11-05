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
    return np.c_[E, _band_diagonal_velocities(E, V, [d.toarray() for d in dhk])[0].T]


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


def _interband_spectrum(H, k):
    """``sum_{n<m} |p^l_nm|^2 g(w - (E_m - E_n); delta_nm)`` per direction ``l``,
    from the momenta and pair widths a mesh-pass KPoint hands the dielectric."""
    from PAOFLOW.sparse.kpoint import KPoint
    from PAOFLOW.sparse.mesh import _band_diagonal_velocities

    hk, dhk = H.assemble_derivatives(k, sign=-1)
    dhk = [d.toarray() for d in dhk]
    E, V = np.linalg.eigh(hk.toarray())
    vel, resolved = _band_diagonal_velocities(E, V, dhk)
    for D, R, _ in resolved:
        V[:, D] = V[:, D] @ R
    resolved = [(D, parts) for D, _, parts in resolved]
    kp = KPoint(H, 0, 0, k, E, V, hk, dhk, vel, None, len(E), 0.7, 0.05, resolved=resolved)
    width = np.maximum(kp.delta2, 0.02)
    dE = E[None, :] - E[:, None]
    keep = dE > 1e-3
    w = np.linspace(0.0, 12.0, 601)[:, None]
    g = np.exp(-(((w - dE[keep]) / width[keep]) ** 2)) / width[keep]
    return np.array([g @ (np.abs(kp.pksp[l]) ** 2)[keep] for l in range(3)])


@pytest.mark.parametrize(
    'K',
    [[0.13, 0.31, 0.27], [0.25, 0.5, 0.125], [0.5, 0.0, 0.0]],
    ids=['generic', 'degenerate', 'time-reversed copies'],
)
def test_doubled_interband_spectrum_matches_folded(cells, K):
    """The momenta and pair widths feed the dielectric: per direction, the
    doubled cell's transitions at K must be those of the four folded
    primitive k-points.  At K = (1/2, 0, 0) the folded k and -k put two
    copies of a non-commuting pair into one group; sorting each direction
    across the whole group, as perturb_split does, missed by 7-14% there."""
    base, doubled = cells
    K = np.array(K)
    folded = sum(
        _interband_spectrum(base, np.array([(K[0] + i) / 2, (K[1] + j) / 2, K[2]]))
        for i, j in itertools.product((0, 1), repeat=2)
    )
    err = np.abs(_interband_spectrum(doubled, K) - folded).max() / np.abs(folded).max()
    assert err < 1e-4, 'interband spectrum miss %.3e' % err
