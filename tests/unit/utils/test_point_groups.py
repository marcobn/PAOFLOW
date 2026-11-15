"""k-grid reduction and tensor symmetrisation of ``PAOFLOW.utils.point_groups``."""

import numpy as np
import pytest

from PAOFLOW.utils import point_groups as pg


def test_reduce_k_grid_td_cubic():
    ops = pg.point_group_operations('Td')  # cubic cell: Cartesian = fractional
    k_irr, weights, rep = pg.reduce_k_grid((6, 6, 6), ops)
    assert weights.sum() == 216 and rep.shape == (216,) and k_irr.shape[0] == weights.size
    assert set((24 / weights).astype(int)) <= {1, 2, 3, 4, 6, 8, 12, 24}
    assert np.all(24 % weights.astype(int) == 0)
    # every point's representative is in its orbit
    idx = pg.grid_indices((6, 6, 6))
    for i in (0, 7, 100, 215):
        images = {tuple((np.rint(W @ idx[i]).astype(int)) % 6) for W in ops}
        assert tuple(np.rint(k_irr[rep[i]] * 6).astype(int)) in images
    with pytest.raises(ValueError):
        pg.reduce_k_grid((6, 6, 6), [pg._rot('z', 45.0)])


def test_component_closure_and_symmetrisation():
    td = pg.point_group_operations('Td')
    closure = pg.component_closure([(0, 1, 2)], td)
    assert closure == sorted({(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)})
    rng = np.random.default_rng(0)
    values = rng.standard_normal((len(closure), 5))
    sym = pg.symmetrize_components(values, closure, td, [(0, 1, 2), (1, 2, 0)])
    assert np.allclose(sym[0], sym[1])  # xyz = yzx after symmetrisation
    assert np.allclose(sym[0], values.mean(axis=0))  # all six permutations averaged in T_d
    # a full-tensor symmetrisation agrees with the component one
    t = np.zeros((3, 3, 3, 5))
    for j, abc in enumerate(closure):
        t[abc] = values[j]
    full = np.array([pg.symmetrize(t[..., e], td) for e in range(5)])
    assert np.allclose(full[:, 0, 1, 2], sym[0])


def test_cartesian_and_fractional_conversions_round_trip():
    at = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])  # fcc
    bg = np.linalg.inv(at).T
    for R in pg.point_group_operations('Td'):
        W = pg.fractional_k_from_cartesian(R, bg)
        assert np.allclose(W, np.rint(W), atol=1e-10)  # a lattice symmetry of fcc
        assert np.allclose(pg.cartesian_from_fractional_k(W, bg), R)
    W = pg.fractional_k_from_cartesian(pg._rot('z', 45.0), bg)
    assert not np.allclose(W, np.rint(W), atol=1e-6)
