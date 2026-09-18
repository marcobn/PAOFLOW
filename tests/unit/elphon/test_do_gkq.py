"""Unit test for the real-space electron-phonon tensor assembly (no QE runtime)."""

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.elphon import do_gkq
from PAOFLOW.elphon.do_gkq import assemble_eph_tensor, enforce_acoustic_sum_rule


class _StubController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr
        self.rank = 0

    def data_dicts(self):
        return self._arry, self._attr


def _al(tmp_path):
    alat = 7.6326928726
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    arry = {
        'a_vectors': a_vectors,
        'tau': np.zeros((1, 3)),
        'atoms': ['Al'],
        'species': [('Al', 'Al.upf')],
    }
    attr = {
        'alat': alat,
        'natoms': 1,
        'opath': str(tmp_path),
        'fpath': str(tmp_path),
        'phonon_supercell_matrix': 2,
        'verbose': False,
    }
    return _StubController(arry, attr)


def test_assemble_eph_tensor_shape_and_hermiticity(tmp_path, monkeypatch):
    dc = _al(tmp_path)
    # One orbital per supercell atom; 2x2x2 supercell grid.
    n = 2
    rng = np.random.default_rng(0)

    def _herm_onsite(dv):
        # Make the R_e=0 block Hermitian so the assembled tensor inherits it.
        dv[:, :, 0, 0, 0, 0] = 0.5 * (dv[:, :, 0, 0, 0, 0] + dv[:, :, 0, 0, 0, 0].conj().T)
        return dv

    directional = []
    for alpha in range(3):
        vec = [0.0, 0.0, 0.0]
        vec[alpha] = 0.06
        dv = rng.standard_normal((8, 8, n, n, n, 1)) + 0j
        directional.append({'sc_atom': 0, 'displacement': vec, 'dV': _herm_onsite(dv)})
    dc._arry['elphon_dV'] = {'directional': directional}

    # Avoid reading a UPF: one orbital per supercell atom.
    monkeypatch.setattr(do_gkq, 'supercell_naw', lambda *a, **k: np.ones(8, dtype=int))

    out = assemble_eph_tensor(dc, configuration='standard')
    g = out['g_R']
    assert out['cart_index'] == [(0, 0), (0, 1), (0, 2)]
    assert g.shape == (3, 1, 1, 4, 4, 4, 2, 2, 2, 1)  # (a, i, j, R_e[4^3], R_p[2^3], spin)
    # On-site (R_e=0) block Hermitian for each Cartesian component / phonon cell.
    for al in range(3):
        for rp in [(0, 0, 0), (1, 0, 0)]:
            M = g[al][:, :, 0, 0, 0, rp[0], rp[1], rp[2], 0]
            assert np.linalg.norm(M - M.conj().T) < 1e-10


def test_enforce_acoustic_sum_rule_zeroes_rigid_shift():
    # g_R for two primitive atoms (natom*3 = 6 rows), 3x3x3 R_e, 2x2x2 R_p.
    rng = np.random.default_rng(1)
    natom = 2
    g = rng.standard_normal((natom * 3, 2, 2, 3, 3, 3, 2, 2, 2, 1)) + 0j
    cart_index = [(kappa, alpha) for kappa in range(natom) for alpha in range(3)]

    alphas = np.array([a for (_, a) in cart_index])
    # n_rp = 8

    # Residual before enforcement, per Cartesian direction, summed over kappa & R_p.
    expected_sq = 0.0
    for alpha in range(3):
        rows = np.where(alphas == alpha)[0]
        total = g[rows].sum(axis=(0, 6, 7, 8))
        expected_sq += float(np.vdot(total, total).real)

    g_out, residual = enforce_acoustic_sum_rule(g.copy(), cart_index)

    assert residual == pytest.approx(np.sqrt(expected_sq))
    # After enforcement the rigid-shift sum vanishes for every direction.
    for alpha in range(3):
        rows = np.where(alphas == alpha)[0]
        total = g_out[rows].sum(axis=(0, 6, 7, 8))
        assert np.linalg.norm(total) < 1e-10


def test_assemble_eph_tensor_reports_and_enforces_asr(tmp_path, monkeypatch):
    dc = _al(tmp_path)
    n = 2
    rng = np.random.default_rng(2)
    directional = []
    for alpha in range(3):
        vec = [0.0, 0.0, 0.0]
        vec[alpha] = 0.06
        dv = rng.standard_normal((8, 8, n, n, n, 1)) + 0j
        directional.append({'sc_atom': 0, 'displacement': vec, 'dV': dv})
    dc._arry['elphon_dV'] = {'directional': directional}
    monkeypatch.setattr(do_gkq, 'supercell_naw', lambda *a, **k: np.ones(8, dtype=int))

    out = assemble_eph_tensor(dc, configuration='standard', enforce_asr=True)
    assert out['asr_residual'] is not None and out['asr_residual'] > 0.0
    # Rigid-shift sum vanishes after enforcement (single atom -> sum over R_p).
    g = out['g_R']
    for alpha in range(3):
        total = g[alpha].sum(axis=(5, 6, 7))
        assert np.linalg.norm(total) < 1e-10

    dc._arry.pop('elphon_g_R', None)
    out0 = assemble_eph_tensor(dc, configuration='standard', enforce_asr=False)
    assert out0['asr_residual'] is None
